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

# Make sure truncated images load
ImageFile.LOAD_TRUNCATED_IMAGES = True

from MODELS.model_resnet_qcw import build_resnet_qcw, NUM_CLASSES, get_last_qcw_layer
from MODELS.ConceptDataset_QCW import ConceptDataset

# If you prefer a custom dataset for the main classification, you can import it here
# from data.backbone_dataset import BackboneDataset
# or you can just use ImageFolder from torchvision.

########################
# Global Constants
########################
LR_DECAY_EPOCH = 30
LR_DECAY_FACTOR = 0.1
CROP_SIZE = 224
RESIZE_SIZE = 256
CW_ALIGN_FREQ = 30   # how often we do concept alignment
PIN_MEMORY = True

########################
# Arguments
########################
parser = argparse.ArgumentParser("Train Quantized Concept Whitening (QCW)")

# Main dataset directory
parser.add_argument("--data_dir", required=True,
    help="Path to main dataset containing train/val/test subfolders")

# Concept dataset directory
parser.add_argument("--concept_dir", required=True,
    help="Path to concept dataset with concept_train, concept_val, and bboxes.json")

# bboxes.json path (though we can also deduce it from concept_dir)
parser.add_argument("--bboxes", default="",
    help="Path to bboxes.json if not under concept_dir/bboxes.json")

parser.add_argument("--concepts", required=True,
    help="Comma-separated list of high-level concepts to use")

parser.add_argument("--prefix", required=True,
    help="Prefix for logging and checkpoint saving")

parser.add_argument("--whitened_layers", default="5",
    help="Comma-separated BN layer indices replaced by QCW")

parser.add_argument("--depth", type=int, default=18,
    help="ResNet depth (18 or 50)")

parser.add_argument("--act_mode", default="pool_max",
    help="CW activation mode: mean, max, pos_mean, pool_max")

parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--resume", default="", type=str,
    help="Path to checkpoint to resume from")
parser.add_argument("--only_load_weights", action="store_true",
    help="If set, only load model weights from checkpoint")
parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--workers", type=int, default=4)

# Feature toggles
parser.add_argument("--use_redaction", action="store_true",
    help="Enable bounding-box redaction in QCW layers")
parser.add_argument("--disable_subspaces", action="store_true",
    help="Disable subspace partitioning, i.e. one axis per HL concept")
parser.add_argument("--use_free", action="store_true",
    help="Enable free unlabeled concept axes if your QCW layer supports it.")
parser.add_argument("--cw_loss_weight", type=float, default=1.0,
    help="Weight for QCW loss (not fully integrated in this script, but can be used in alignment).")

args = parser.parse_args()

########################
# Setup
########################
print("=============== ARGUMENTS ===============")
for k,v in sorted(vars(args).items()):
    print(f"{k}: {v}")
print("=========================================")

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
cudnn.benchmark = True

writer = SummaryWriter(log_dir=os.path.join("runs", f"{args.prefix}_{int(time.time())}"))

########################
# Build Main DataLoaders
########################
def build_main_loaders(args):
    # standard train transform
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(CROP_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    # val/test transform
    test_transform = transforms.Compose([
        transforms.Resize(RESIZE_SIZE),
        transforms.CenterCrop(CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    train_dir = os.path.join(args.data_dir, "train")
    val_dir   = os.path.join(args.data_dir, "val")
    test_dir  = os.path.join(args.data_dir, "test")

    # If you wanted a custom dataset, you could do: "BackboneDataset(...)" instead
    train_data = datasets.ImageFolder(train_dir, transform=train_transform)
    val_data   = datasets.ImageFolder(val_dir, transform=test_transform)
    test_data  = datasets.ImageFolder(test_dir, transform=test_transform)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=PIN_MEMORY)
    val_loader   = DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=PIN_MEMORY)
    test_loader  = DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=PIN_MEMORY)
    return train_loader, val_loader, test_loader

train_loader, val_loader, test_loader = build_main_loaders(args)

########################
# Build Concept DataLoaders
########################
def build_concept_loaders(args):
    # concept_train: we get from e.g. concept_dir/concept_train
    # we can do concept_val too if we want, but let's keep it simple
    concept_root = os.path.join(args.concept_dir, "concept_train")
    if not args.bboxes:
        # default to concept_dir/bboxes.json
        args.bboxes = os.path.join(args.concept_dir,"bboxes.json")

    # same transforms as train?
    concept_transform = transforms.Compose([
        transforms.RandomResizedCrop(CROP_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225])
    ])

    # e.g. "wing,beak,tail" => list of HL concepts
    selected_hl = [x.strip() for x in args.concepts.split(",")]

    # your custom dataset reads bounding boxes from bboxes.json,
    # enumerates subfolders under concept_train, filters by HL
    concept_dataset = ConceptDataset(
        root_dir=concept_root,
        bboxes_file=args.bboxes,
        high_level_filter=selected_hl,
        transform=concept_transform
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

print(f"[Data] Main Train: {len(train_loader.dataset)} images")
print(f"[Data] Val: {len(val_loader.dataset)} images")
print(f"[Data] Test: {len(test_loader.dataset)} images")
print(f"[Data] Concept Train: {len(concept_loaders[0].dataset)} images")

########################
# Build Model
########################
# If subspaces are NOT disabled, we gather a subspace mapping from the concept dataset.
# Suppose your ConceptDataset has an attribute 'subspace_mapping' which is a dict
# like { "wing": [0,1], "beak": [2], "tail": [3,4,5] } etc
if not args.disable_subspaces:
    subspaces = concept_ds.subspace_mapping  # dynamic from dataset
else:
    # Each HL concept => [0]
    # or each concept => unique dimension => do your approach
    subspaces = {hl: [0] for hl in concept_ds.subspace_mapping.keys()}

model = build_resnet_qcw(
    num_classes=NUM_CLASSES,
    depth=args.depth,
    whitened_layers=[int(x) for x in args.whitened_layers.split(",")],
    act_mode=args.act_mode,
    subspaces=subspaces,
    use_redaction=args.use_redaction,
    use_subspace=(not args.disable_subspaces),
    use_free=args.use_free,
    pretrained_model=None,   # or pass in if you have a backbone
    vanilla_pretrain=True
)

########################
# Checkpoint Resume
########################
def maybe_resume_checkpoint(model, optimizer, args):
    start_epoch, best_prec = 0, 0.0
    if args.resume and os.path.isfile(args.resume):
        print(f"[Checkpoint] Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=False)
        if not args.only_load_weights:
            start_epoch = checkpoint.get("epoch", 0)
            best_prec = checkpoint.get("best_prec1", 0.0)
            if "optimizer" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer"])
        print(f"[Checkpoint] Resumed at epoch={start_epoch}, best_prec={best_prec:.2f}")
    return start_epoch, best_prec

optimizer = optim.SGD(model.parameters(), lr=args.lr,
                      momentum=args.momentum, weight_decay=args.weight_decay)
start_epoch, best_prec = maybe_resume_checkpoint(model, optimizer, args)

model = nn.DataParallel(model).cuda()

########################
# Training Function
########################
def adjust_lr(optimizer, epoch, args):
    new_lr = args.lr * (LR_DECAY_FACTOR ** (epoch // LR_DECAY_EPOCH))
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr
    return new_lr

def train_epoch(train_loader, concept_loaders, model, optimizer, epoch, args, writer):
    model.train()
    criterion = nn.CrossEntropyLoss().cuda()
    losses = AverageMeter()
    top1   = AverageMeter()
    top5   = AverageMeter()

    pbar = tqdm(enumerate(train_loader), total=len(train_loader), miniters=10,
                desc=f"[Train] Epoch {epoch+1}", smoothing=0.02)
    for i, (imgs, lbls) in pbar:
        iteration = epoch * len(train_loader) + i
        imgs, lbls = imgs.cuda(), lbls.cuda()

        # forward
        outputs = model(imgs, region=None)  # region=None => no redaction
        loss = criterion(outputs, lbls)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        prec1, prec5 = accuracy_topk(outputs, lbls, topk=(1,5))
        losses.update(loss.item(), imgs.size(0))
        top1.update(prec1, imgs.size(0))
        top5.update(prec5, imgs.size(0))

        # concept alignment every N batches
        if (i+1) % CW_ALIGN_FREQ == 0 and len(concept_loaders)>0:
            concept_acc = run_concept_alignment(model, concept_loaders[0])
            writer.add_scalar("CW/ConceptAlignment", concept_acc, iteration)

        pbar.set_postfix({"Loss":f"{losses.avg:.3f}","Top1":f"{top1.avg:.2f}"})
        writer.add_scalar("Train/Loss", losses.val, iteration)
        writer.add_scalar("Train/Top1", top1.val, iteration)
        writer.add_scalar("Train/Top5", top5.val, iteration)

########################
# Concept Alignment
########################
def run_concept_alignment(model, concept_loader):
    model.eval()
    from types import SimpleNamespace
    hook_storage = SimpleNamespace(outputs=[])

    def forward_hook(mod, inp, out):
        hook_storage.outputs.append(out)

    last_layer = get_last_qcw_layer(model.module)
    hook_handle = last_layer.register_forward_hook(forward_hook)

    concept_correct=0
    concept_total=0

    with torch.no_grad():
        for images, coords, hl_label in concept_loader:
            # Suppose concept dataset returns HL label as an int index
            # or you might store a dictionary mapping HL->index
            images, coords = images.cuda(), coords.cuda()
            model.module.change_mode(hl_label[0].item()) # for simplicity if batch=1
            model(images, region=coords)
            if len(hook_storage.outputs)==0:
                continue
            feat = hook_storage.outputs[0] # [B, C, H, W]
            B, C, H, W = feat.shape
            # simplest alignment measure => choose axis with highest average activation
            feat_avg = feat.mean(dim=(2,3))
            pred_axis = feat_avg.argmax(dim=1)
            concept_correct += (pred_axis==hl_label.cuda()).sum().item()
            concept_total   += B
            hook_storage.outputs.clear()

    hook_handle.remove()
    model.module.update_rotation_matrix()
    model.module.change_mode(-1)
    model.train()

    return 100.0*concept_correct/concept_total if concept_total>0 else 0.0

########################
# Validation + Test
########################
def validate(loader, model, epoch, writer, mode="Val"):
    model.eval()
    criterion = nn.CrossEntropyLoss().cuda()
    losses = AverageMeter()
    top1   = AverageMeter()
    top5   = AverageMeter()

    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.cuda(), lbls.cuda()
            outs = model(imgs, region=None)
            loss = criterion(outs, lbls)
            prec1, prec5 = accuracy_topk(outs, lbls, topk=(1,5))
            losses.update(loss.item(), imgs.size(0))
            top1.update(prec1, imgs.size(0))
            top5.update(prec5, imgs.size(0))

    writer.add_scalar(f"{mode}/Loss", losses.avg, epoch)
    writer.add_scalar(f"{mode}/Top1", top1.avg, epoch)
    writer.add_scalar(f"{mode}/Top5", top5.avg, epoch)

    print(f"[{mode}] Epoch {epoch}: Loss={losses.avg:.3f}, Top1={top1.avg:.2f}, Top5={top5.avg:.2f}")
    return top1.avg

########################
# Save / Resume
########################
def save_checkpoint(state, is_best, prefix, out_dir="./checkpoints"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{prefix}_checkpoint.pth")
    torch.save(state, path)
    if is_best:
        best_path = os.path.join(out_dir, f"{prefix}_best.pth")
        shutil.copyfile(path, best_path)
        print(f"[Checkpoint] Best model saved as {best_path}")

########################
# Meters + accuracy
########################
class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val=0; self.sum=0; self.count=0; self.avg=0
    def update(self, val, n=1):
        self.val=val
        self.sum+= val*n
        self.count+=n
        self.avg=self.sum/self.count if self.count else 0

def accuracy_topk(outputs, targets, topk=(1,)):
    maxk = max(topk)
    batch_size=targets.size(0)
    _, pred = outputs.topk(maxk,1,True,True)
    pred=pred.t()
    correct = pred.eq(targets.view(1,-1).expand_as(pred))
    res=[]
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0,keepdim=True)
        res.append((correct_k*100.0/batch_size).item())
    return tuple(res)

########################
# Main
########################
def main():
    global args
    # We have everything built above, so just do the training loop
    start_epoch=0
    best_acc=0.0

    # We already created model, train_loader, val_loader, test_loader, concept_loaders, etc.
    for epoch in range(start_epoch, args.epochs):
        lr_now=adjust_lr(optimizer, epoch, args)
        writer.add_scalar("LR",lr_now,epoch)

        train_epoch(train_loader, concept_loaders, model, optimizer, epoch, args, writer)
        val_acc = validate(val_loader, model, epoch, writer, mode="Val")

        is_best = (val_acc>best_acc)
        best_acc=max(val_acc, best_acc)
        checkpoint_state={
            "epoch":epoch+1,
            "state_dict":model.state_dict(),
            "best_prec1":best_acc,
            "optimizer":optimizer.state_dict()
        }
        save_checkpoint(checkpoint_state, is_best, args.prefix)

    test_acc = validate(test_loader, model, args.epochs, writer, mode="Test")
    print(f"[Done] Best Val {best_acc:.2f}, Final Test {test_acc:.2f}")
    writer.close()

if __name__=="__main__":
    main()
