#!/usr/bin/env python
import argparse
import os
import time
import random
import shutil
import json
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from PIL import ImageFile

# Ensure truncated images are loadable
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Revised QCW model that no longer uses bounding-box logic
from MODELS.model_resnet_qcw import build_resnet_qcw, NUM_CLASSES, get_last_qcw_layer

# Revised dataset that physically crops/redacts images; returns (image, hl_label)
from MODELS.ConceptDataset_QCW import ConceptDataset

########################
# Global Constants
########################
LR_DECAY_EPOCH  = 30
LR_DECAY_FACTOR = 0.1
CROP_SIZE       = 224
RESIZE_SIZE     = 256
CW_ALIGN_FREQ   = 30   # how often (in mini-batches) we do concept alignment
PIN_MEMORY      = True

########################
# Argument Parser
########################
parser = argparse.ArgumentParser(description="Train Quantized Concept Whitening (QCW) - Revised")

# Main dataset
parser.add_argument("--data_dir", required=True,
    help="Path to main dataset containing train/val/test subfolders (ImageFolder structure).")

# Concept dataset
parser.add_argument("--concept_dir", required=True,
    help="Path to concept dataset with concept_train/, concept_val/ (optional), and bboxes.json.")
parser.add_argument("--bboxes", default="",
    help="Path to bboxes.json if not in concept_dir/bboxes.json")

# Which high-level concepts
parser.add_argument("--concepts", required=True,
    help="Comma-separated list of high-level concepts to use (e.g. 'wing,beak,general').")

# Logging prefix
parser.add_argument("--prefix", required=True,
    help="Prefix for logging & checkpoint saving")

# Which BN layers to replace with CW
parser.add_argument("--whitened_layers", default="5",
    help="Comma-separated BN layer indices to replace with QCW (e.g. '5' or '2,5')")

# Model depth & activation mode
parser.add_argument("--depth", type=int, default=18,
    help="ResNet depth (18 or 50).")
parser.add_argument("--act_mode", default="pool_max",
    help="Activation mode for QCW: 'mean','max','pos_mean','pool_max'")

# Training hyperparams
parser.add_argument("--epochs", type=int, default=100,
    help="Number of training epochs.")
parser.add_argument("--batch_size", type=int, default=64,
    help="Mini-batch size.")
parser.add_argument("--lr", type=float, default=0.1,
    help="Initial learning rate.")
parser.add_argument("--momentum", type=float, default=0.9,
    help="Momentum for SGD.")
parser.add_argument("--weight_decay", type=float, default=1e-4,
    help="Weight decay (L2 reg).")

# Checkpoint
parser.add_argument("--resume", default="", type=str,
    help="Path to checkpoint to resume from.")
parser.add_argument("--only_load_weights", action="store_true",
    help="If set, only load model weights from checkpoint (ignore epoch/optimizer).")

# Reproducibility
parser.add_argument("--seed", type=int, default=1234,
    help="Random seed.")
parser.add_argument("--workers", type=int, default=4,
    help="Number of data loading workers.")

# Feature toggles
parser.add_argument("--disable_subspaces", action="store_true",
    help="Disable subspace partitioning => one axis per concept.")
parser.add_argument("--use_free", action="store_true",
    help="Enable free unlabeled concept axes if the QCW layer supports it.")

# Not used in the dataset approach anymore, but left for code compatibility
parser.add_argument("--use_redaction", action="store_true",
    help="(Deprecated) was used for bounding-box redaction in the model, can ignore in revised approach.")

# Possibly used for weighting concept alignment if you implement it
parser.add_argument("--cw_loss_weight", type=float, default=1.0,
    help="Weight for QCW loss in alignment, if integrated into your alignment code.")


args = parser.parse_args()

########################
# Setup
########################
print("=========== ARGUMENTS ===========")
for k,v in sorted(vars(args).items()):
    print(f"{k}: {v}")
print("=================================")

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
cudnn.benchmark = True

writer = SummaryWriter(log_dir=os.path.join("runs", f"{args.prefix}_{int(time.time())}"))

########################
# Build Main Dataloaders
########################
def build_main_loaders(args):
    # transforms for train
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(CROP_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    # transforms for val/test
    val_transform = transforms.Compose([
        transforms.Resize(RESIZE_SIZE),
        transforms.CenterCrop(CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    train_dir = os.path.join(args.data_dir, "train")
    val_dir   = os.path.join(args.data_dir, "val")
    test_dir  = os.path.join(args.data_dir, "test")

    train_data = datasets.ImageFolder(train_dir, train_transform)
    val_data   = datasets.ImageFolder(val_dir,   val_transform)
    test_data  = datasets.ImageFolder(test_dir,  val_transform)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=PIN_MEMORY)
    val_loader   = DataLoader(val_data,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=PIN_MEMORY)
    test_loader  = DataLoader(test_data,  batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=PIN_MEMORY)

    return train_loader, val_loader, test_loader

train_loader, val_loader, test_loader = build_main_loaders(args)

########################
# Build Concept Dataset
########################
def build_concept_loaders(args):
    """
    Build a single concept loader for concept_train.
    The revised ConceptDataset physically crops or redacts images, so the model sees normal images.
    """
    concept_root = os.path.join(args.concept_dir, "concept_train")
    if not args.bboxes:
        args.bboxes = os.path.join(args.concept_dir, "bboxes.json")

    # Similar transforms as train
    concept_transform = transforms.Compose([
        transforms.RandomResizedCrop(CROP_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    # Example: "wing,beak,tail" => ["wing","beak","tail"]
    hl_list = [x.strip() for x in args.concepts.split(",")]

    # We'll define some argument in code for how we want to handle bounding boxes:
    # e.g. "crop" or "redact" or "none"
    # For demonstration, let's do "crop". If you want to pass as arg, do so as well.
    crop_mode = "crop"

    concept_dataset = ConceptDataset(
        root_dir=concept_root,
        bboxes_file=args.bboxes,
        high_level_filter=hl_list,
        transform=concept_transform,
        crop_mode=crop_mode
    )
    concept_loader = DataLoader(
        concept_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=PIN_MEMORY
    )

    return [concept_loader], concept_dataset

concept_loaders, concept_ds = build_concept_loaders(args)

print(f"[Data] #Main Train: {len(train_loader.dataset)}")
print(f"[Data] #Val:        {len(val_loader.dataset)}")
print(f"[Data] #Test:       {len(test_loader.dataset)}")
print(f"[Data] #Concept:    {len(concept_loaders[0].dataset)}")

########################
# Build QCW Model
########################
if not args.disable_subspaces:
    subspaces = concept_ds.subspace_mapping  # e.g. { "wing": [...], "beak": [...], ... }
else:
    # lumps each HL concept into dimension [0], or each concept => single axis
    subspaces = {hl: [0] for hl in concept_ds.subspace_mapping.keys()}

model = build_resnet_qcw(
    num_classes=NUM_CLASSES,
    depth=args.depth,
    whitened_layers=[int(x) for x in args.whitened_layers.split(",")],
    act_mode=args.act_mode,
    subspaces=subspaces,
    use_subspace=(not args.disable_subspaces),
    use_free=args.use_free,
    pretrained_model=None,
    vanilla_pretrain=True
)

########################
# Resume Checkpoint
########################
def maybe_resume_checkpoint(model, optimizer, args):
    start_epoch, best_prec = 0, 0.0
    if args.resume and os.path.isfile(args.resume):
        print(f"[Checkpoint] Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu")
        sd = ckpt.get("state_dict", ckpt)
        model.load_state_dict(sd, strict=False)
        if not args.only_load_weights:
            start_epoch = ckpt.get("epoch", 0)
            best_prec   = ckpt.get("best_prec1", 0.0)
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
        print(f"[Checkpoint] Resumed epoch={start_epoch}, best_prec={best_prec:.2f}")
    return start_epoch, best_prec

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                      weight_decay=args.weight_decay)
start_epoch, best_prec = maybe_resume_checkpoint(model, optimizer, args)
model = nn.DataParallel(model).cuda()

########################
# Train + Align
########################
def adjust_lr(optimizer, epoch, args):
    new_lr = args.lr * (LR_DECAY_FACTOR ** (epoch // LR_DECAY_EPOCH))
    for g in optimizer.param_groups:
        g["lr"] = new_lr
    return new_lr

def train_epoch(train_loader, concept_loaders, model, optimizer, epoch, args, writer):
    model.train()
    criterion = nn.CrossEntropyLoss().cuda()
    losses = AverageMeter()
    top1   = AverageMeter()
    top5   = AverageMeter()
    alignment_score = AverageMeter()  # For tracking the concept alignment score

    pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                desc=f"[Train] Epoch {epoch+1}", smoothing=0.02)

    for i, (imgs, lbls) in pbar:
        iteration = epoch * len(train_loader) + i
        imgs, lbls = imgs.cuda(), lbls.cuda()

        # Forward pass through the model
        outputs = model(imgs)

        loss = criterion(outputs, lbls)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        prec1, prec5 = accuracy_topk(outputs, lbls, (1, 5))
        losses.update(loss.item(), imgs.size(0))
        top1.update(prec1, imgs.size(0))
        top5.update(prec5, imgs.size(0))

        # Concept alignment every N batches
        if (i + 1) % CW_ALIGN_FREQ == 0 and len(concept_loaders) > 0:
            # Get the concept alignment score
            concept_acc = run_concept_alignment(model, concept_loaders[0])
            alignment_score.update(concept_acc)  # Update the alignment score

            # Log the alignment score in TensorBoard
            writer.add_scalar("CW/ConceptAlignment", alignment_score.avg, iteration)

        # Update the pbar with the new alignment score
        pbar.set_postfix({"Loss": f"{losses.avg:.3f}", "Top1": f"{top1.avg:.2f}", 
                          "Top5": f"{top5.avg:.2f}", "ConceptAlignment": f"{alignment_score.avg:.2f}"})

        writer.add_scalar("Train/Loss", losses.val, iteration)
        writer.add_scalar("Train/Top1", top1.val, iteration)
        writer.add_scalar("Train/Top5", top5.val, iteration)

def run_concept_alignment(model, concept_loader):
    """
    Evaluate alignment: 
      - For each concept sample, set the QCW mode=concept_idx
      - Forward pass
      - Hook final QCW layer => measure alignment by picking the axis with highest activation
    """
    model.eval()
    from types import SimpleNamespace
    hook_storage = SimpleNamespace(outputs=[])

    def forward_hook(module, input, output):
        hook_storage.outputs.append(output)

    last_qcw = get_last_qcw_layer(model.module)
    handle = last_qcw.register_forward_hook(forward_hook)

    correct=0
    total=0
    with torch.no_grad():
        for imgs, hl_label in concept_loader:
            # Suppose dataset returns hl_label as shape [B].
            imgs = imgs.cuda()
            hl_label = hl_label.cuda()

            # We'll just pick the first label or loop if you prefer
            # This is a simplification (old approach).
            model.module.change_mode(hl_label[0].item())

            out = model(imgs)
            if len(hook_storage.outputs)==0:
                continue

            featmap = hook_storage.outputs[0]  # shape [B, C, H, W]
            B, C, H, W = featmap.shape
            feat_avg = featmap.mean(dim=(2,3))  # shape [B, C]
            pred_axis = feat_avg.argmax(dim=1)
            correct += (pred_axis == hl_label).sum().item()
            total += B

            hook_storage.outputs.clear()

    handle.remove()
    model.module.update_rotation_matrix()
    model.module.change_mode(-1)
    model.train()

    if total > 0:
        return 100.0 * correct / total
    else:
        return 0.0

def validate(loader, model, epoch, writer, mode="Val"):
    model.eval()
    criterion = nn.CrossEntropyLoss().cuda()
    losses = AverageMeter()
    top1   = AverageMeter()
    top5   = AverageMeter()

    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.cuda(), lbls.cuda()
            outs = model(imgs)
            loss = criterion(outs, lbls)
            prec1, prec5 = accuracy_topk(outs, lbls, (1,5))
            losses.update(loss.item(), imgs.size(0))
            top1.update(prec1, imgs.size(0))
            top5.update(prec5, imgs.size(0))

    writer.add_scalar(f"{mode}/Loss", losses.avg, epoch)
    writer.add_scalar(f"{mode}/Top1", top1.avg, epoch)
    writer.add_scalar(f"{mode}/Top5", top5.avg, epoch)
    print(f"[{mode}] Epoch {epoch}: Loss={losses.avg:.3f}, Top1={top1.avg:.2f}, Top5={top5.avg:.2f}")
    return top1.avg

########################
# Save + Resume
########################
def save_checkpoint(state, is_best, prefix, outdir="./checkpoints"):
    os.makedirs(outdir, exist_ok=True)
    ckpt_path = os.path.join(outdir, f"{prefix}_checkpoint.pth")
    torch.save(state, ckpt_path)
    if is_best:
        best_path = os.path.join(outdir, f"{prefix}_best.pth")
        shutil.copyfile(ckpt_path, best_path)
        print(f"[Checkpoint] Best => {best_path}")

########################
# Utility Classes
########################
class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val=0; self.sum=0; self.count=0; self.avg=0
    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count+= n
        self.avg = self.sum/self.count if self.count>0 else 0

def accuracy_topk(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res=[]
    for k in topk:
        c_k = correct[:k].reshape(-1).float().sum(0,keepdim=True)
        res.append((c_k*100.0/batch_size).item())
    return tuple(res)

########################
# Main
########################
def main():
    global args
    start_epoch=0
    best_acc=0.0

    # The data, model, loaders are all built at script level above
    # We begin the main training loop:
    for epoch in range(start_epoch, args.epochs):
        lr_now = adjust_lr(optimizer, epoch, args)
        writer.add_scalar("LR", lr_now, epoch)

        train_epoch(train_loader, concept_loaders, model, optimizer, epoch, args, writer)
        val_acc = validate(val_loader, model, epoch, writer, mode="Val")

        is_best = (val_acc > best_acc)
        best_acc = max(val_acc, best_acc)

        cstate = {
            "epoch": epoch+1,
            "state_dict": model.state_dict(),
            "best_prec1": best_acc,
            "optimizer": optimizer.state_dict()
        }
        save_checkpoint(cstate, is_best, args.prefix)

    test_acc = validate(test_loader, model, args.epochs, writer, mode="Test")
    print(f"[Done] Best Val={best_acc:.2f}, Final Test={test_acc:.2f}")
    writer.close()

if __name__ == "__main__":
    main()
