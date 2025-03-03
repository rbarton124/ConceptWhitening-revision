import argparse
import os
import time
import random
import shutil
import numpy as np

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from MODELS.model_resnet import build_resnet_cw, build_resnet_bn
from PIL import ImageFile


# -------------------- GLOBAL CONSTANTS --------------------
LR_DECAY_EPOCH = 30
LR_DECAY_FACTOR = 0.1

CROP_SIZE = 224
RESIZE_SIZE = 256

TQDM_MINITERS = 10
TQDM_SMOOTHING = 0.02

ONE_BATCH_PER_CONCEPT = True  # Only 1 batch per concept loader during alignment
PIN_MEMORY = True

NUM_CLASSES = 200  # e.g. for CUB or Places
CW_ALIGN_FREQ = 30  # do concept alignment every N batches

ImageFile.LOAD_TRUNCATED_IMAGES = True


# -------------------- ARGUMENT PARSER ---------------------
parser = argparse.ArgumentParser(description='Minimal Concept Whitening Training Script')

parser.add_argument('--main_data', required=True,
                    help='Path to dataset with train/val/test subdirs.')
parser.add_argument('--concept_data', required=True,
                    help='Path to concept dataset (with concept_train).')

parser.add_argument('--whitened_layers', default='5',
                    help='Comma-separated indices of BN layers replaced by CW (e.g. "5" or "2,5").')
parser.add_argument('--act_mode', default='pool_max',
                    help='Activation mode for CW ("mean","max","pos_mean","pool_max").')
parser.add_argument('--depth', default=18, type=int,
                    help='ResNet depth (18 or 50).')

parser.add_argument('-j', '--workers', default=4, type=int,
                    help='Number of data loading workers.')
parser.add_argument('--epochs', default=100, type=int,
                    help='Total epochs to run.')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='Override start epoch if resuming.')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    help='Mini-batch size.')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='Initial learning rate.')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='SGD momentum.')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    help='Weight decay (L2 reg).')

parser.add_argument('--resume', default='', type=str,
                    help='Path to checkpoint to resume from.')
parser.add_argument('--only-load-weights', action='store_true',
                    help='Only load model weights (ignore epoch/optimizer state).')

parser.add_argument('--seed', type=int, default=1234,
                    help='Random seed.')
parser.add_argument('--prefix', type=str, required=True,
                    help='Prefix for logging & checkpoint saving.')

parser.add_argument('--concepts', type=str, required=True,
                    help='Comma-separated list of concept folder names.')


# -------------------- MAIN FUNCTION -----------------------
def main():
    args = parser.parse_args()

    # Print summary of arguments
    print("=============== ARGUMENTS ===============")
    for arg, val in sorted(vars(args).items()):
        print(f"{arg}: {val}")
    print("=========================================")

    # Basic setup
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    cudnn.benchmark = True

    # Detect GPUs automatically
    num_gpus = torch.cuda.device_count()
    print(f"[Info] Detected GPUs: {num_gpus}")

    # TensorBoard writer
    import time
    from torch.utils.tensorboard import SummaryWriter
    current_time = str(int(time.time()))
    writer = SummaryWriter(log_dir=os.path.join('runs', f"{args.prefix}_{current_time}"))

    # Build model
    model = build_model(args)
    print(f"[Model] Built ResNet with depth={args.depth}")

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # Optionally resume
    start_epoch, best_prec = maybe_resume_checkpoint(args, model, optimizer)

    # Wrap model in DataParallel if multiple GPUs
    if num_gpus > 1:
        model = nn.DataParallel(model).cuda()
    else:
        model = model.cuda()

    # Build DataLoaders
    train_loader, concept_loaders, val_loader, test_loader = setup_dataloaders(args)

    print(f"[Data] Train dataset size: {len(train_loader.dataset)}")
    for c_loader, c_name in zip(concept_loaders, args.concepts.split(',')):
        print(f"[Data] Concept '{c_name}' size: {len(c_loader.dataset)}")
    print(f"[Data] Val dataset size: {len(val_loader.dataset)}")
    print(f"[Data] Test dataset size: {len(test_loader.dataset)}")

    # Main training loop
    best_acc = best_prec
    print(f"[Training] Starting from epoch {start_epoch}, total {args.epochs} epochs.")
    for epoch in range(start_epoch, args.epochs):
        # Adjust LR
        lr_now = adjust_learning_rate(optimizer, epoch, args)
        writer.add_scalar('LR', lr_now, epoch)

        # Single-epoch train
        train_epoch(train_loader, concept_loaders, model, optimizer, epoch, args, writer)

        # Validation
        val_top1, val_top5 = validate(val_loader, model, epoch, writer, mode='Val')
        is_best = (val_top1 > best_acc)
        best_acc = max(val_top1, best_acc)

        # Save checkpoint
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, args.prefix)

    # Final test
    test_top1, test_top5 = validate(test_loader, model, args.epochs, writer, mode='Test')
    print(f"[Done] Best top-1 accuracy: {best_acc:.2f}, Final test top-1: {test_top1:.2f} (top-5: {test_top5:.2f})")

    writer.close()


# -------------------- BUILD MODEL -------------------------
def build_model(args):
    """Build a ResNet (18 or 50) with or without concept whitening."""
    # Convert whitened_layers to list of ints
    w_layers = [int(x) for x in args.whitened_layers.split(',')] if args.whitened_layers else []

    if len(w_layers) > 0:
        print("[build_model] Creating ResNet_CW.")
        model = build_resnet_cw(num_classes=NUM_CLASSES,
                                depth=args.depth,
                                whitened_layers=w_layers,
                                act_mode=args.act_mode)
    else:
        print("[build_model] Creating plain ResNet (BN only).")
        model = build_resnet_bn(num_classes=NUM_CLASSES, depth=args.depth)
    return model


# -------------------- DATA LOADERS ------------------------
def setup_dataloaders(args):
    """Returns (train_loader, concept_loaders, val_loader, test_loader)."""
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dir = os.path.join(args.main_data, 'train')
    val_dir   = os.path.join(args.main_data, 'val')
    test_dir  = os.path.join(args.main_data, 'test')
    concept_dir = os.path.join(args.concept_data, 'concept_train')

    # Train
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(CROP_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=PIN_MEMORY
    )

    # Concepts
    concept_loaders = []
    concept_names = args.concepts.split(',')
    for c_name in concept_names:
        c_path = os.path.join(concept_dir, c_name)
        c_dataset = datasets.ImageFolder(c_path, transform=train_transform)
        c_loader = torch.utils.data.DataLoader(
            c_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=PIN_MEMORY
        )
        concept_loaders.append(c_loader)

    # Val
    val_transform = transforms.Compose([
        transforms.Resize(RESIZE_SIZE),
        transforms.CenterCrop(CROP_SIZE),
        transforms.ToTensor(),
        normalize
    ])
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=PIN_MEMORY
    )

    # Test
    test_dataset = datasets.ImageFolder(test_dir, transform=val_transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=PIN_MEMORY
    )

    return train_loader, concept_loaders, val_loader, test_loader


# -------------------- TRAIN LOOP --------------------------
def train_epoch(train_loader, concept_loaders, model, optimizer, epoch, args, writer):
    """
    One epoch of training + concept alignment for a ResNet with CW layers.
    Tracks top-1 and top-5 accuracy, logs them to TensorBoard.
    """
    model.train()

    # Meters
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    criterion = nn.CrossEntropyLoss().cuda()

    end = time.time()
    pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                desc=f"[Train] Epoch {epoch+1}", smoothing=TQDM_SMOOTHING, miniters=TQDM_MINITERS)

    for i, (images, targets) in pbar:
        iteration = epoch * len(train_loader) + i

        # Periodically run concept alignment if concept loaders exist
        if (i + 1) % CW_ALIGN_FREQ == 0 and len(concept_loaders) > 0:
            concept_acc = run_concept_alignment(model, concept_loaders)
            writer.add_scalar("CW/ConceptAlignment(%)", concept_acc, iteration)

        # Measure data loading time
        data_time.update(time.time() - end)
        images, targets = images.cuda(), targets.cuda()

        # Forward
        logits = model(images)
        loss = criterion(logits, targets)

        # Accuracy
        prec1, prec5 = accuracy_topk_multi(logits, targets, topk=(1,5))
        losses.update(loss.item(), images.size(0))
        top1.update(prec1, images.size(0))
        top5.update(prec5, images.size(0))

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Measure elapsed
        batch_time.update(time.time() - end)
        end = time.time()

        # Log to TensorBoard
        writer.add_scalar('Train/Loss', losses.val, iteration)
        writer.add_scalar('Train/Top1Acc', top1.val, iteration)
        writer.add_scalar('Train/Top5Acc', top5.val, iteration)

        # TQDM status
        pbar.set_postfix({
            "Loss": f"{losses.val:.3f}",
            "Top1": f"{top1.val:.2f}",
            "Top5": f"{top5.val:.2f}"
        })


# -------------------- CONCEPT ALIGNMENT -------------------
def run_concept_alignment(model, concept_loaders):
    """
    1) Hook the *last BN/CW layer* to retrieve feature maps.
    2) For each concept, feed exactly ONE batch if available,
       compute the axis with highest average activation,
       check if it matches the concept index => concept_correct++.
    3) Update the rotation matrix, revert model to train mode, return alignment %.
    """
    # Switch to eval
    net = model.module if hasattr(model, 'module') else model
    net.eval()

    # We'll do hooking on the final BN (or last whitened layer).
    # For simplicity, we replicate the old approach of hooking:
    from types import SimpleNamespace
    hook_storage = SimpleNamespace(outputs=[])

    def forward_hook(module, inp, out):
        # out shape [B, C, H, W]
        hook_storage.outputs.append(out)

    # Identify the last BN/CW layer to hook
    last_bn = get_last_bn_layer(net.model)
    hook_handle = last_bn.register_forward_hook(forward_hook)

    concept_correct = 0
    concept_total = 0

    with torch.no_grad():
        # For each concept => set net.change_mode(concept_idx) => forward one batch
        for c_idx, c_loader in enumerate(concept_loaders):
            net.change_mode(c_idx)

            # Grab the first batch if available
            batch = next(iter(c_loader), None)
            if batch is None:
                continue

            imgs, _ = batch
            imgs = imgs.cuda()

            # forward
            net(imgs)
            if len(hook_storage.outputs) == 0:
                continue

            feat = hook_storage.outputs[0]  # [B, C, H, W]
            B, C, H, W = feat.shape
            feat_avg = feat.mean(dim=(2,3))  # [B, C]

            # predicted concept = dimension with highest activation
            pred_axis = feat_avg.argmax(dim=1)
            correct_mask = (pred_axis == c_idx)
            concept_correct += correct_mask.sum().item()
            concept_total   += B

            hook_storage.outputs.clear()

        # update rotation
        net.update_rotation_matrix()
        net.change_mode(-1)

    hook_handle.remove()
    net.train()

    if concept_total > 0:
        concept_acc = 100.0 * concept_correct / concept_total
    else:
        concept_acc = 0.0

    return concept_acc


def get_last_bn_layer(resnet_model):
    """
    Navigate resnet_model.layer1..layer4 to find the last BN (CW) block.
    This code is simplistic. If you want a specific layer index, adapt accordingly.
    """
    # layer4 is the last block. We'll check the last module in layer4
    blocks = resnet_model.layer4
    last_block = blocks[-1]
    # Typically the BN is 'bn1', or you might have a separate cw layer
    return last_block.bn1


# -------------------- VALIDATION --------------------------
def validate(loader, model, epoch, writer, mode='Val'):
    """
    Forward pass on a DataLoader, returning (top1, top5).
    Logs average stats to TensorBoard.
    """
    model.eval()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    criterion = nn.CrossEntropyLoss().cuda()

    with torch.no_grad():
        for i, (images, targets) in enumerate(loader):
            images, targets = images.cuda(), targets.cuda()

            logits = model(images)
            loss = criterion(logits, targets)

            prec1, prec5 = accuracy_topk_multi(logits, targets, topk=(1,5))
            losses.update(loss.item(), images.size(0))
            top1.update(prec1, images.size(0))
            top5.update(prec5, images.size(0))

    writer.add_scalar(f"{mode}/Loss", losses.avg, epoch)
    writer.add_scalar(f"{mode}/Top1Acc", top1.avg, epoch)
    writer.add_scalar(f"{mode}/Top5Acc", top5.avg, epoch)
    print(f"[{mode}] Epoch {epoch} => Loss: {losses.avg:.3f}, Top1: {top1.avg:.2f}, Top5: {top5.avg:.2f}")
    return top1.avg, top5.avg


# ------------------ CHECKPOINT LOGIC ----------------------
def maybe_resume_checkpoint(args, model, optimizer):
    """
    If args.resume is provided, load the checkpoint.
    If only_weights=True, just load model weights.
    Returns (start_epoch, best_prec).
    """
    start_epoch, best_prec = 0, 0
    if args.resume and os.path.isfile(args.resume):
        print(f"[Checkpoint] Loading from {args.resume} (only_weights={args.only_load_weights})")
        checkpoint = torch.load(args.resume, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)

        new_sd = {}
        for k, v in state_dict.items():
            nk = k.replace('module.', '')
            new_sd[nk] = v

        model.load_state_dict(new_sd, strict=False)

        if not args.only_load_weights:
            start_epoch = checkpoint.get('epoch', 0)
            best_prec = checkpoint.get('best_prec1', 0)
            opt_sd = checkpoint.get('optimizer', None)
            if opt_sd is not None:
                try:
                    optimizer.load_state_dict(opt_sd)
                except Exception as ex:
                    print(f"[Warning] Could not load optimizer state: {ex}")
        print(f"[Checkpoint] Resumed at epoch={start_epoch}, best_prec={best_prec:.2f}")
    return start_epoch, best_prec


def save_checkpoint(state, is_best, prefix, outdir='./checkpoints'):
    os.makedirs(outdir, exist_ok=True)
    filename = os.path.join(outdir, f"{prefix}_checkpoint.pth.tar")
    torch.save(state, filename)
    if is_best:
        best_name = os.path.join(outdir, f"{prefix}_best.pth.tar")
        shutil.copyfile(filename, best_name)
        print(f"[Checkpoint] Best model updated => {best_name}")


# ------------------ UTILITY CLASSES & FUNCS ---------------
class AverageMeter:
    """Tracks and updates mean and current value."""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

def adjust_learning_rate(optimizer, epoch, args):
    """Decays LR by LR_DECAY_FACTOR every LR_DECAY_EPOCH epochs."""
    new_lr = args.lr * (LR_DECAY_FACTOR ** (epoch // LR_DECAY_EPOCH))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    return new_lr

def accuracy_topk_multi(logits, targets, topk=(1,)):
    """
    Returns a tuple of accuracies for each top-k in `topk`.
    For instance if topk=(1,5), it returns (top1_acc, top5_acc).
    """
    maxk = max(topk)
    batch_size = targets.size(0)

    _, pred = logits.topk(maxk, 1, True, True)  # shape [B, maxk]
    pred = pred.t()  # shape [maxk, B]
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        # correct_k => how many were correct up to rank k
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    return tuple(res)  # e.g. (top1, top5)


if __name__ == '__main__':
    main()