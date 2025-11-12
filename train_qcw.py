import argparse
import re
import os
import time
import random
import shutil
import json
import math
import numpy as np
import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from collections import defaultdict
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from PIL import ImageFile

from types import SimpleNamespace

from MODELS.factory_qcw import build_qcw, get_last_qcw_layer
from MODELS.ConceptDataset_QCW import ConceptDataset

# Global Constants
NUM_CLASSES     = 200
CROP_SIZE       = 224
RESIZE_SIZE     = 256
PIN_MEMORY      = True

# Argument Parser
parser = argparse.ArgumentParser(description="Train Quantized Concept Whitening (QCW) - Revised")

# Required arguments
parser.add_argument("--data_dir", required=True, help="Path to main dataset containing train/val/test subfolders (ImageFolder structure).")
parser.add_argument("--concept_dir", required=True, help="Path to concept dataset with concept_train/, concept_val/ (optional), and bboxes.json.")
parser.add_argument("--bboxes", default="", help="Path to bboxes.json if not in concept_dir/bboxes.json")
parser.add_argument("--concepts", required=True, help="Comma-separated list of high-level concepts to use (e.g. 'wing,beak,general').")
parser.add_argument("--prefix", required=True, help="Prefix for logging & checkpoint saving")
# Model hyperparams
parser.add_argument("--whitened_layers", default="5", help="Comma-separated BN layer indices to replace with QCW (e.g. '5' or '2,5')")
parser.add_argument("--model", default="resnet", choices=["resnet", "densenet", "vgg16"], help="Which backbone to use.")
parser.add_argument("--depth", type=int, default=18, help="ResNet depth (18 or 50), or DenseNet depth (121 or 161). Ignored for VGG16.")
parser.add_argument("--act_mode", default="pool_max", help="Activation mode for QCW: 'mean','max','pos_mean','pool_max'")
# Training hyperparams
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
parser.add_argument("--batch_size", type=int, default=64, help="Mini-batch size.")
parser.add_argument("--lr", type=float, default=5e-4, help="Initial learning rate.")
parser.add_argument("--lr_decay_factor", type=float, default=0.1, help="Learning rate decay factor.")
parser.add_argument("--lr_decay_epoch", type=int, default=25, help="Learning rate decay epoch.")
parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD.")
parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay (L2 reg).")
# CW Training hyperparams
parser.add_argument("--batches_per_concept", type=int, default=1, help="Number of batches per subconcept for each alignment step.")
parser.add_argument("--cw_align_freq", type=int, default=40, help="How often (in mini-batches) we do concept alignment.")
parser.add_argument("--cw_lambda", type=float, default=0.05, help="Lambda parameter for QCW.")
parser.add_argument("--concept_image_mode", type=str, default="crop", choices=["crop","redact","blur","none"], help="How to handle concept images.")
# Checkpoint
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint to resume from.")
parser.add_argument("--only_load_weights", action="store_true", help="If set, only load model weights from checkpoint (ignore epoch/optimizer).")
# System
parser.add_argument("--seed", type=int, default=348129, help="Random seed.")
parser.add_argument("--workers", type=int, default=4, help="Number of data loading workers.")
parser.add_argument("--log_dir", type=str, default="runs", help="Directory to save logs.")
parser.add_argument("--checkpoint_dir", type=str, default="model_checkpoints", help="Directory to save checkpoints.")
# Feature toggles
parser.add_argument("--vanilla_pretrain", action="store_true", help="Train without Concept Whitening, i.e. vanilla ResNet.") # used to work, isnt working lately TODO: ACTUALLY FIX [issue was subspace null mapping issue]
parser.add_argument("--disable_subspaces", action="store_true", help="Disable subspace partitioning => one axis per concept.") # this logic is not fleshed out yet TODO: FIX
parser.add_argument("--use_free", action="store_true", help="Enable free unlabeled concept axes if the QCW layer supports it.") # doesn't do anything yet TODO: FIX
# Backbone nudge (CW-loss integration into main task)
parser.add_argument("--cw_nudge_alpha", type=float, default=0,
                    help="Weight for backbone nudge loss after alignment (0=disabled). Start with 1e-3.")
parser.add_argument("--cw_nudge_lambda", type=float, default=0.5,
                    help="Labeled:free balance in nudge loss (weights free term).")
parser.add_argument("--cw_nudge_tau", type=float, default=0.1,
                    help="Temperature for soft winner-takes-all (lower=sharper, higher=smoother).")
# Subconcept sampling (renamed for clarity)
parser.add_argument("--cw_nudge_labeled_subconcepts", type=int, default=4,
                    help="Number of labeled subconcepts to sample per nudge step. -1 = use ALL labeled subconcepts.")
parser.add_argument("--cw_nudge_free_subconcepts", type=int, default=4,
                    help="Number of free subconcepts to sample per nudge step. -1 = use ALL free subconcepts.")
# Batch sampling control (new)
parser.add_argument("--cw_nudge_batches_per_subconcept", type=int, default=1,
                    help="Number of batches to process per subconcept. -1 = use ALL batches from that loader.")
# Safety caps (prevent OOM)
parser.add_argument("--cw_nudge_max_images_per_batch", type=int, default=16,
                    help="Cap on batch size during nudge (0=no cap, use loader's batch size).")
parser.add_argument("--cw_nudge_max_total_batches", type=int, default=16,
                    help="Global cap on total batches processed during nudge (0=unlimited, across labeled+free).")
# Ergonomics / scheduling
parser.add_argument("--cw_nudge_shuffle", action="store_true",
                    help="Shuffle subconcepts before processing (recommended for variety).")
parser.add_argument("--cw_nudge_interleave", action="store_true",
                    help="Interleave labeled and free subconcepts during processing (helps memory/gradient mixing).")
parser.add_argument("--cw_nudge_disable_drip", action="store_true",
                    help="Disable inter-alignment drip nudges (only use pre-/post-align if enabled).")
parser.add_argument("--cw_nudge_pre_align", action="store_true",
                    help="Trigger a nudge immediately before each alignment step.")
parser.add_argument("--cw_nudge_post_align", action="store_true",
                    help="Trigger a nudge immediately after each alignment step (legacy behaviour).")
# Plan B: Optional slice-margin loss (for within-subspace separation)
parser.add_argument("--cw_nudge_beta", type=float, default=0.1,
                    help="Weight for slice-margin loss (Plan B, 0=disabled).")
parser.add_argument("--cw_nudge_gamma", type=float, default=0.2,
                    help="Margin for slice-margin loss (target should beat others by this amount).")
# Free concept sharpening / scaling
parser.add_argument("--cw_nudge_free_topk", type=int, default=0,
                    help="If >0, use top-K activations (K=min(topk,|S|)) instead of log-sum-exp for free concepts.")
parser.add_argument("--cw_nudge_free_scale", type=str, default="sqrt",
                    choices=["none", "sqrt", "log"],
                    help="Scaling mode for free-concept loss based on subspace size.")

args = parser.parse_args()

# Validate model and depth combinations
if args.model != "resnet" and args.depth not in [None, 121, 161]:
    args.depth = {"densenet": 161, "vgg16": None}[args.model]  # Force valid default
    print(f"Ignoring --depth for {args.model}, using {args.depth}")

if args.model == "resnet" and args.depth not in [18, 50]:
    raise ValueError(f"ResNet depth must be 18 or 50, got {args.depth}")

if args.model == "densenet" and args.depth not in [121, 161]:
    raise ValueError(f"DenseNet depth must be 121 or 161, got {args.depth}")

if args.vanilla_pretrain:
    print("Vanilla pretraining mode: Disabling Concept Whitening and related arguments.")
    args.whitened_layers = []
    args.batches_per_concept = None
    args.cw_align_freq = None
else:
    try:
        args.whitened_layers = [int(x) for x in args.whitened_layers.split(",")]
        # Validate whitened_layers based on model
        max_idx = {
            "resnet": 8 if args.depth == 18 else 16,
            "densenet": 5,
            "vgg16": 13
        }[args.model]
        
        invalid_indices = [x for x in args.whitened_layers if x < 1 or x > max_idx]
        if invalid_indices:
            raise ValueError(f"whitened_layers {invalid_indices} outside valid range 1..{max_idx} for {args.model}.")
            
    except ValueError as e:
        if "whitened_layers" in str(e):
            print(e)
            exit(1)
        print("Invalid whitened_layers format. Should be a comma-separated list of integers.")
        if args.whitened_layers == '':
            args.whitened_layers = []
            print("Setting whitened_layers to empty list. Please set vanilla_pretrain=True instead.")
        exit(1)

print("=========== ARGUMENTS ===========")
for k,v in sorted(vars(args).items()):
    print(f"{k}: {v}")
print("=================================")

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
cudnn.benchmark = True

writer = SummaryWriter(log_dir=os.path.join(args.log_dir, f"{args.prefix}_{int(time.time())}"))

################################################################################
# Backbone Nudge Helpers (CW-Loss Integration)
################################################################################

def reduce_axis_scores(featmap: torch.Tensor, act_mode: str = "pool_max") -> torch.Tensor:
    """
    Reduce spatial dimensions of CW layer output to get per-axis activation scores.
    Keeps computation graph for backprop.
    
    Args:
        featmap: [B, C, H, W] tensor from QCW layer (post-rotation)
        act_mode: How to reduce spatial dims ('mean', 'max', 'pos_mean', 'pool_max')
    
    Returns:
        [B, C] tensor of axis activation scores (with gradient)
    """
    if act_mode == "mean":
        return featmap.mean(dim=(2, 3))
    
    elif act_mode == "max":
        # Flatten spatial, take max
        return featmap.flatten(2).max(dim=2).values
    
    elif act_mode == "pos_mean":
        # Mean of positive activations only
        pos_mask = (featmap > 0).float()
        numerator = (featmap * pos_mask).sum(dim=(2, 3))
        denominator = pos_mask.sum(dim=(2, 3)).clamp(min=1e-6)
        return numerator / denominator
    
    elif act_mode == "pool_max":
        # Max-pool then spatial mean (default in QCW)
        pooled = F.max_pool2d(featmap, kernel_size=3, stride=3)
        return pooled.flatten(2).mean(dim=2)
    
    else:
        # Fallback to mean
        return featmap.mean(dim=(2, 3))


@contextlib.contextmanager
def freeze_norm_stats_for_qcw_nudge(model):
    """
    Context manager that freezes all normalization running statistics during
    the backbone nudge step. This prevents small concept batches from corrupting
    the running mean/variance used during main classification training.
    
    Mechanism:
    - Sets all BatchNorm layers to eval() mode (freezes their running stats)
    - Sets all QCW layer momentum to 0.0 (prevents running_mean/running_wm updates)
    - Keeps QCW layers in train() mode (so backprop through whitening still works)
    - Restores original states after the context
    
    Args:
        model: The DataParallel-wrapped model
    
    Yields:
        None (use as context manager)
    """
    # Store and freeze BatchNorm modules
    bn_modules = []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            bn_modules.append((m, m.training))
            m.eval()  # Freeze BN stats
    
    # Store and freeze QCW momentum (prevents running stat updates)
    saved_momentums = []
    if hasattr(model.module, 'cw_layers'):
        for cw_layer in model.module.cw_layers:
            saved_momentums.append(cw_layer.momentum)
            cw_layer.momentum = 0.0  # Freeze whitening stats
    
    try:
        yield
    finally:
        # Restore BatchNorm training states
        for m, was_training in bn_modules:
            if was_training:
                m.train()
        
        # Restore QCW momentum
        if hasattr(model.module, 'cw_layers'):
            for cw_layer, orig_momentum in zip(model.module.cw_layers, saved_momentums):
                cw_layer.momentum = orig_momentum


def compute_drip_points(align_freq: int):
    """Return unique mid-cycle steps (1-indexed) at which to fire drip nudges."""
    if align_freq is None or align_freq <= 1:
        return []
    raw_points = {align_freq * frac // 4 for frac in (1, 2, 3)}
    # Remove 0 and the alignment step itself
    valid_points = [p for p in raw_points if 0 < p < align_freq]
    return sorted(valid_points)


def sample_balanced_subconcepts(concept_triplets, target, shuffle=False):
    """Round-robin sample up to `target` items, balanced across HL groups."""
    if target < 0 or target >= len(concept_triplets):
        return list(concept_triplets)

    buckets = defaultdict(list)
    for item in concept_triplets:
        _, hl_name, _ = item
        buckets[hl_name].append(item)

    for hl_items in buckets.values():
        if shuffle:
            random.shuffle(hl_items)

    sampled = []
    hl_keys = list(buckets.keys())
    if shuffle:
        random.shuffle(hl_keys)

    while len(sampled) < target:
        progressed = False
        for hl in hl_keys:
            if buckets[hl]:
                sampled.append(buckets[hl].pop())
                progressed = True
                if len(sampled) >= target:
                    break
        if not progressed:
            break
    return sampled

# Build Main Dataloaders
def build_main_loaders(args):
    # train transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(CROP_SIZE, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), shear=5, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # val transforms
    val_transform = transforms.Compose([
        transforms.Resize(RESIZE_SIZE),
        transforms.CenterCrop(CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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

# Build Concept Dataset
def build_concept_loaders(args):
    """
    Build concept loaders for concept_train:
    1. Main loader with all subconcepts
    2. Individual loaders for each subconcept for alignment
    
    ConceptDataset handles image cropping/redaction directly.
    """
    # skip concept loaders for vanilla pretraining
    concept_root = os.path.join(args.concept_dir, "concept_train")
    if not args.bboxes:
        args.bboxes = os.path.join(args.concept_dir, "bboxes.json")

    # concept transforms (similar to train)
    concept_transform = transforms.Compose([
    transforms.RandomResizedCrop(CROP_SIZE, scale=(0.8, 1.0), ratio=(0.9, 1.1), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
    transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

    hl_list = [x.strip() for x in args.concepts.split(",")]

    crop_mode = args.concept_image_mode

    # create main concept dataset
    concept_dataset = ConceptDataset(
        root_dir=concept_root,
        bboxes_file=args.bboxes,
        high_level_filter=hl_list,
        transform=concept_transform,
        crop_mode=crop_mode
    )

    # main concept loader (shuffled)
    main_concept_loader = DataLoader(
        concept_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=PIN_MEMORY
    )

    subconcept_loaders = []
    
    # group by subconcept
    subconcept_samples = {}
    for idx, (_, _, hl_name, sc_name) in enumerate(concept_dataset.samples):
        # Get the subconcept label from the subconcept name
        sc_label = concept_dataset.sc2idx[sc_name]
        
        if sc_label not in subconcept_samples:
            subconcept_samples[sc_label] = []
        subconcept_samples[sc_label].append(idx)
    
    # create dataset for each subconcept
    for sc_label, indices in subconcept_samples.items():
        # filtered dataset via PyTorch Subset
        sc_dataset = torch.utils.data.Subset(concept_dataset, indices)
        
        # loader for this subconcept
        sc_loader = DataLoader(
            sc_dataset,
            batch_size=min(args.batch_size, len(indices)),  # for small subconcepts
            shuffle=True,
            num_workers=args.workers,
            pin_memory=PIN_MEMORY
        )
        subconcept_loaders.append((sc_label, sc_loader))
    
    print(f"Created {len(subconcept_loaders)} subconcept-specific loaders")

    # add main loader with label -1
    concept_loaders = [(-1, main_concept_loader)] + subconcept_loaders
    
    # return both loaders
    return concept_loaders, concept_dataset

print(f"[Data] #Main Train: {len(train_loader.dataset)}")
print(f"[Data] #Val:        {len(val_loader.dataset)}")
print(f"[Data] #Test:       {len(test_loader.dataset)}")

concept_loaders, concept_ds, subconcept_loaders, subspaces = None, None, None, None # init vars for vanilla pretraining
# build concept loaders if needed
if not args.vanilla_pretrain:
    concept_loaders, concept_ds = build_concept_loaders(args)
    subconcept_loaders = concept_loaders[1:]

    print(f"[Data] #Concept:    {len(concept_loaders[0][1].dataset)}")
    print(f"[Data] #Subconcept Loaders: {len(concept_loaders) - 1}")

    # print concept dataset details
    print("\n===== CONCEPT DATASET DETAILS =====")
    print(f"Total number of samples: {len(concept_ds.samples)}")
    print(f"Total number of subconcepts: {concept_ds.get_num_subconcepts()}")
    print(f"Total number of high-level concepts: {concept_ds.get_num_high_level()}")

    # subconcept names and indices
    print("\nSubconcept names and indices:")
    for sc_name in concept_ds.get_subconcept_names():
        sc_idx = concept_ds.sc2idx[sc_name]
        print(f"  [{sc_idx}] {sc_name}")

    # high-level concepts and their subspaces
    print("\nHigh-level concepts and their subconcept indices:")
    for hl_name in concept_ds.get_hl_names():
        subspace = concept_ds.subspace_mapping.get(hl_name, [])
        print(f"  {hl_name}: {subspace}")

    subconcept_loaders = sorted(subconcept_loaders, key=lambda x: x[0])
    print("\nSubconcept-specific loaders:")
    for sc_label, loader in subconcept_loaders:
        sc_name = [name for name, idx in concept_ds.sc2idx.items() if idx == sc_label][0]
        print(f"  Subconcept '{sc_name}' [{sc_label}]: {len(loader.dataset)} samples")

    random.shuffle(subconcept_loaders)

    print("\nShuffled Subconcept-specific loaders:")
    for sc_label, loader in subconcept_loaders:
        sc_name = [name for name, idx in concept_ds.sc2idx.items() if idx == sc_label][0]
        print(f"  Subconcept '{sc_name}' [{sc_label}]: {len(loader.dataset)} samples")

    # Build QCW Model
    if not args.disable_subspaces: # use subspace mapping
        subspaces = concept_ds.subspace_mapping  # e.g. { "wing": [...], "beak": [...], ... }
    else:
        # one axis per concept
        subspaces = {hl: [0] for hl in concept_ds.subspace_mapping.keys()}

else:
    concept_loaders   = []
    concept_ds        = None
    subspaces         = {}

subspace_mapping = subspaces
print(f"Subspace mapping: {subspace_mapping}")

model = build_qcw(
    model_type=args.model,
    num_classes=NUM_CLASSES,
    depth=args.depth,
    whitened_layers=args.whitened_layers,
    act_mode=args.act_mode,
    subspaces=subspace_mapping,
    use_subspace=(not args.disable_subspaces),
    use_free=args.use_free,
    cw_lambda=args.cw_lambda,
    pretrained_model=None,
    vanilla_pretrain=args.vanilla_pretrain
)

def maybe_resume_checkpoint(model, optimizer, args):
    """
    Resume from checkpoint, handling ResNet, DenseNet, VGG, or QCW formats.
    Renames layers as needed and loads optimizer state if requested.
    Returns: (start_epoch, best_prec)
    """
    start_epoch, best_prec = 0, 0.0
    if not args.resume or not os.path.isfile(args.resume):
        print("[Checkpoint] No checkpoint found or provided.")
        return start_epoch, best_prec

    print(f"[Checkpoint] Resuming from {args.resume}")
    ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
    raw_sd = ckpt.get("state_dict", ckpt)

    model_sd = model.state_dict()
    renamed_sd = {}

    def rename_key(old_key):
        # Remove DataParallel 'module.' prefix
        if old_key.startswith("module."):
            old_key = old_key[len("module."):]

        # Match old DenseNet-style conv.norm naming
        old_key = re.sub(r"\.norm\.(\d+)", lambda m: f".norm{m.group(1)}", old_key)
        old_key = re.sub(r"\.conv\.(\d+)", lambda m: f".conv{m.group(1)}", old_key)

        # Optional backbone prefix
        if not old_key.startswith("backbone.") and ("backbone." + old_key in model_sd):
            return "backbone." + old_key
        if old_key.startswith("fc.") and ("backbone.fc" + old_key[2:] in model_sd):
            return "backbone." + old_key
        if old_key.startswith("classifier.") and ("backbone.classifier" + old_key[10:] in model_sd):
            return "backbone.classifier" + old_key[10:]

        return old_key

    matched_keys, skipped_keys = [], []

    for ckpt_k, ckpt_v in raw_sd.items():
        new_k = rename_key(ckpt_k)
        if new_k in model_sd:
            if ckpt_v.shape == model_sd[new_k].shape:
                renamed_sd[new_k] = ckpt_v
                matched_keys.append(f"{ckpt_k} -> {new_k}")
            else:
                skipped_keys.append(f"{ckpt_k}: shape {ckpt_v.shape} != {model_sd[new_k].shape}")
        else:
            skipped_keys.append(f"{ckpt_k}: no match found in model")

    print("Loading model...")
    result = model.load_state_dict(renamed_sd, strict=False)
    print("Missing keys:", result.missing_keys)
    print("Unexpected keys:", result.unexpected_keys)

    # print("[Checkpoint] Matched keys:")
    # for mk in matched_keys:
    #     print("   ", mk)
    print("[Checkpoint] Skipped keys from checkpoint:")
    for sk in skipped_keys:
        print("   ", sk)

    if isinstance(ckpt, dict):
        if not args.only_load_weights:
            start_epoch = ckpt.get("epoch", 0)
            if "optimizer" in ckpt:
                opt_sd = ckpt["optimizer"]
                try:
                    optimizer.load_state_dict(opt_sd)
                    for param in optimizer.state.values():
                        if isinstance(param, dict) and "momentum_buffer" in param:
                            buf = param["momentum_buffer"]
                            if buf is not None and buf.device != torch.device("cuda"):
                                param["momentum_buffer"] = buf.cuda()
                except Exception as e:
                    print(f"[Warning] Could not load optimizer state: {e}")
        print(f"[Checkpoint] Resumed epoch={start_epoch}, best_prec={best_prec:.2f}")

    return start_epoch, best_prec


optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
start_epoch, best_prec = maybe_resume_checkpoint(model, optimizer, args)
model = nn.DataParallel(model).cuda()

# Train + Align
def adjust_lr(optimizer, epoch, args):
    new_lr = args.lr * (args.lr_decay_factor ** (epoch // args.lr_decay_epoch))
    for g in optimizer.param_groups:
        g["lr"] = new_lr
    return new_lr

def train_epoch(train_loader, concept_loaders, model, optimizer, epoch, args, writer):
    model.train()
    criterion = nn.CrossEntropyLoss().cuda()

    losses = AverageMeter()
    top1   = AverageMeter()
    top5   = AverageMeter()
    alignment_score = AverageMeter()
    concept_loss = 0.0

    concept_enabled = (
        not args.vanilla_pretrain
        and args.cw_align_freq is not None
        and concept_loaders is not None
        and len(concept_loaders) > 1
    )
    drip_points = compute_drip_points(args.cw_align_freq) if concept_enabled else []
    steps_since_alignment = 0

    pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                desc=f"[Train] Epoch {epoch+1}", smoothing=0.02)

    for i, (imgs, lbls) in pbar:
        iteration = epoch * len(train_loader) + i
        imgs, lbls = imgs.cuda(), lbls.cuda()

        # classification forward/backward
        outputs = model(imgs)
        loss = criterion(outputs, lbls)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update classification metrics
        prec1, prec5 = accuracy_topk(outputs, lbls, (1, 5))
        losses.update(loss.item(), imgs.size(0))
        top1.update(prec1, imgs.size(0))
        top5.update(prec5, imgs.size(0))

        if concept_enabled:
            steps_since_alignment += 1
            if (
                args.cw_nudge_alpha > 0.0
                and not args.cw_nudge_disable_drip
                and steps_since_alignment in drip_points
            ):
                run_backbone_nudge(
                    model, optimizer, concept_ds, concept_loaders[1:],
                    args, writer, iteration, stage="drip"
                )

        # concept alignment
        if concept_enabled and steps_since_alignment >= args.cw_align_freq:

            if args.cw_nudge_alpha > 0.0 and args.cw_nudge_pre_align:
                run_backbone_nudge(
                    model, optimizer, concept_ds, concept_loaders[1:],
                    args, writer, iteration, stage="pre"
                )

            alignment_metrics = align_concepts(
                model,
                concept_loaders[1:],  # sub-concept loaders
                concept_ds,
                args.batches_per_concept,
                args.cw_lambda,
                act_mode=args.act_mode
            )

            # get alignment metrics
            global_top1   = alignment_metrics["global_top1"]
            subspace_top1 = alignment_metrics["subspace_top1"]
            global_top5   = alignment_metrics["global_top5"]
            concept_loss  = alignment_metrics["concept_loss"]
            axis_consistency = alignment_metrics["axis_consistency"]
            axis_purity = alignment_metrics["axis_purity"]
            act_strength_ratio = alignment_metrics["act_strength_ratio"]

            # log to tensorboard
            writer.add_scalar("CW/Alignment/GlobalTop1",   global_top1,   iteration)
            writer.add_scalar("CW/Alignment/SubspaceTop1", subspace_top1, iteration)
            writer.add_scalar("CW/Alignment/GlobalTop5",   global_top5,   iteration)
            writer.add_scalar("CW/Alignment/ConceptLoss",  concept_loss,  iteration)
            writer.add_scalar("CW/FreeConcept/AxisConsistency", axis_consistency, iteration)
            writer.add_scalar("CW/FreeConcept/AxisPurity", axis_purity,  iteration)
            writer.add_scalar("CW/FreeConcept/ActStrengthRatio", act_strength_ratio, iteration)

            alignment_score.update(0.3 * global_top5 + 0.65 * subspace_top1 + 0.05 * global_top1)

            if args.cw_nudge_alpha > 0.0 and args.cw_nudge_post_align:
                run_backbone_nudge(
                    model, optimizer, concept_ds, concept_loaders[1:],
                    args, writer, iteration, stage="post"
                )

            steps_since_alignment = 0

        # update progress bar
        postfix_dict = {
            "Loss": f"{losses.avg:.3f}",
            "Top1": f"{top1.avg:.2f}",
            "Top5": f"{top5.avg:.2f}"
        }
        if not args.vanilla_pretrain:
            postfix_dict["Align"]  = f"{alignment_score.avg:.2f}"
            postfix_dict["CWLoss"] = f"{concept_loss:.2f}"
        pbar.set_postfix(postfix_dict)

        writer.add_scalar("Train/Loss", losses.val, iteration)
        writer.add_scalar("Train/Top1", top1.val, iteration)
        writer.add_scalar("Train/Top5", top5.val, iteration)

def align_concepts(model, subconcept_loaders, concept_dataset, batches_per_concept, lambda_, act_mode="pool_max"):
    """
    Two-phase concept alignment:
      1. Push labeled sub-concepts to their axes
      2. Apply winner-takes-all for free sub-concepts
    Then update rotation matrix and compute metrics.
    
    Returns metrics dict with alignment scores and concept loss.
    """
    model.eval()

    # separate labeled vs free subconcepts
    sc_to_hl_name = concept_dataset.get_subconcept_to_hl_name_mapping()
    hl_subconcepts = defaultdict(lambda: {"labeled": [], "free": []})

    label_to_scname = {}
    for (sc_label, loader) in subconcept_loaders:
        for name, idx in concept_dataset.sc2idx.items():
            if idx == sc_label:
                label_to_scname[sc_label] = name
                break

    for (sc_label, loader) in subconcept_loaders:
        hl_name = sc_to_hl_name.get(sc_label, "general")
        sc_name = label_to_scname[sc_label]
        if concept_dataset.is_free_subconcept_name(sc_name):
            hl_subconcepts[hl_name]["free"].append((sc_label, loader))
        else:
            hl_subconcepts[hl_name]["labeled"].append((sc_label, loader))

    # reset concept loss
    for cw_layer in model.module.cw_layers:
        cw_layer.reset_concept_loss()

    # labeled subconcepts - direct axis push
    with torch.no_grad():
        for hl_name, groups in hl_subconcepts.items():
            for (sc_label, sc_loader) in groups["labeled"]:
                model.module.change_mode(sc_label)  # set target axis
                loader_iter = iter(sc_loader)
                for _ in range(batches_per_concept):
                    batch_data = next(loader_iter, None)
                    if not batch_data or not len(batch_data[0]):
                        break
                    imgs = batch_data[0].cuda()
                    for img in imgs:
                        _ = model(img.unsqueeze(0))

    # free subconcepts - scaled by lambda
    for cw_layer in model.module.cw_layers:
        cw_layer.set_subspace_scaling(lambda_)

    with torch.no_grad():
        for hl_name, groups in hl_subconcepts.items():
            free_list = groups["free"]
            if not free_list:
                continue

            # set active subspace
            for cw_layer in model.module.cw_layers:
                cw_layer.clear_subspace()
                cw_layer.set_subspace(hl_name)

            model.module.change_mode(-1)  # subspace-based alignment

            for (sc_label, sc_loader) in free_list:
                loader_iter = iter(sc_loader)
                for _ in range(batches_per_concept):
                    batch_data = next(loader_iter, None)
                    if not batch_data or not len(batch_data[0]):
                        break
                    imgs = batch_data[0].cuda()
                    for img in imgs:
                        _ = model(img.unsqueeze(0))

    # update rotation matrix (FIXED: removed duplicate call)
    model.module.update_rotation_matrix()
    model.module.change_mode(-1)
    for cw_layer in model.module.cw_layers:
        cw_layer.clear_subspace()
        cw_layer.set_subspace_scaling(1.0)

    # get concept loss
    total_cw_loss = 0.0
    for cw_layer in model.module.cw_layers:
        total_cw_loss += cw_layer.get_concept_loss()
    avg_cw_loss = total_cw_loss / max(1, len(model.module.cw_layers))

    # compute all metrics
    metrics = compute_alignment_metrics(
        model, subconcept_loaders, concept_dataset,
        batches_per_concept, label_to_scname, act_mode=act_mode
    )
    return {
        "global_top1":  metrics["global_top1"],
        "subspace_top1": metrics["subspace_top1"],
        "global_top5":  metrics["global_top5"],
        "concept_loss": avg_cw_loss,
        "axis_consistency": metrics["axis_consistency"],
        "axis_purity": metrics["axis_purity"],
        "act_strength_ratio": metrics["act_strength_ratio"]
    }


def run_backbone_nudge(model, optimizer, concept_dataset, subconcept_loaders,
                       args, writer=None, iteration=None, stage="post"):
    """Run a backbone nudge stage (drip / pre / post) with purity-aware losses."""
    if args.cw_nudge_alpha <= 0.0 or not subconcept_loaders:
        return

    last_cw = get_last_qcw_layer(model)
    if last_cw is None:
        return

    was_training = model.training
    model.train()

    sc_to_hl_name = concept_dataset.get_subconcept_to_hl_name_mapping()
    idx_to_scname = {idx: name for name, idx in concept_dataset.sc2idx.items()}

    labeled_concepts, free_concepts = [], []
    for (sc_label, loader) in subconcept_loaders:
        sc_name = idx_to_scname.get(sc_label, "")
        hl_name = sc_to_hl_name.get(sc_label, "general")
        triplet = (sc_label, hl_name, loader)
        if concept_dataset.is_free_subconcept_name(sc_name):
            free_concepts.append(triplet)
        else:
            labeled_concepts.append(triplet)

    labeled_concepts = sample_balanced_subconcepts(
        labeled_concepts,
        args.cw_nudge_labeled_subconcepts,
        shuffle=args.cw_nudge_shuffle
    )
    free_concepts = sample_balanced_subconcepts(
        free_concepts,
        args.cw_nudge_free_subconcepts,
        shuffle=args.cw_nudge_shuffle
    )

    if args.cw_nudge_interleave and labeled_concepts and free_concepts:
        max_len = max(len(labeled_concepts), len(free_concepts))
        concepts_to_process = []
        for idx in range(max_len):
            if idx < len(labeled_concepts):
                concepts_to_process.append(("labeled", labeled_concepts[idx]))
            if idx < len(free_concepts):
                concepts_to_process.append(("free", free_concepts[idx]))
    else:
        concepts_to_process = [("labeled", item) for item in labeled_concepts]
        concepts_to_process.extend(("free", item) for item in free_concepts)

    if not concepts_to_process:
        if not was_training:
            model.eval()
        return

    hook_storage = []

    def hook_fn(module, inp, out):
        scores = reduce_axis_scores(out.clone(), act_mode=args.act_mode)
        hook_storage.append(scores)

    handle = last_cw.register_forward_hook(hook_fn)

    total_batches_processed = 0
    labeled_losses = []
    free_losses = []
    margin_losses = []
    grad_norm = 0.0

    def free_scale(num_axes):
        if num_axes <= 1:
            return 1.0
        mode = args.cw_nudge_free_scale
        if mode == "sqrt":
            return 1.0 / math.sqrt(num_axes)
        if mode == "log":
            return 1.0 / math.log(num_axes + 1.0)
        return 1.0

    try:
        with freeze_norm_stats_for_qcw_nudge(model):
            optimizer.zero_grad()

            for concept_type, concept_data in concepts_to_process:
                if args.cw_nudge_max_total_batches > 0 and total_batches_processed >= args.cw_nudge_max_total_batches:
                    break

                sc_label, hl_name, loader = concept_data
                num_batches = args.cw_nudge_batches_per_subconcept
                if num_batches == -1:
                    num_batches = len(loader)
                if args.cw_nudge_max_total_batches > 0:
                    remaining = args.cw_nudge_max_total_batches - total_batches_processed
                    num_batches = min(num_batches, remaining)
                if num_batches <= 0:
                    continue

                loader_iter = iter(loader)
                for _ in range(num_batches):
                    if args.cw_nudge_max_total_batches > 0 and total_batches_processed >= args.cw_nudge_max_total_batches:
                        break
                    batch_data = next(loader_iter, None)
                    if batch_data is None or len(batch_data[0]) == 0:
                        break

                    imgs = batch_data[0].cuda(non_blocking=True)
                    if args.cw_nudge_max_images_per_batch > 0:
                        imgs = imgs[:args.cw_nudge_max_images_per_batch]
                    if imgs.numel() == 0:
                        continue

                    hook_storage.clear()
                    _ = model(imgs)
                    if not hook_storage:
                        continue

                    # Gather scores from DP replicas
                    device = hook_storage[0].device
                    axis_scores = torch.cat(
                        [scores.to(device) for scores in hook_storage], dim=0
                    )
                    hook_storage.clear()

                    if concept_type == "labeled":
                        target_axis = sc_label
                        main_loss = -axis_scores[:, target_axis].mean()
                        margin_loss = 0.0
                        if args.cw_nudge_beta > 0.0:
                            subspace_axes = concept_dataset.subspace_mapping.get(hl_name, [])
                            if subspace_axes and target_axis in subspace_axes:
                                others = [ax for ax in subspace_axes if ax != target_axis]
                                if others:
                                    top_other = axis_scores[:, others].max(dim=1).values
                                    margin_loss = (args.cw_nudge_gamma - axis_scores[:, target_axis] + top_other).clamp(min=0).mean()
                        loss_lab_mb = main_loss + args.cw_nudge_beta * margin_loss
                        total_loss = args.cw_nudge_alpha * loss_lab_mb
                        total_loss.backward()
                        labeled_losses.append(float(loss_lab_mb.detach().cpu()))
                        if margin_loss != 0.0:
                            margin_losses.append(float(margin_loss.detach().cpu()))

                    else:  # free concept
                        subspace_axes = concept_dataset.subspace_mapping.get(hl_name, [])
                        if not subspace_axes:
                            continue
                        scores_slice = axis_scores[:, subspace_axes]
                        if args.cw_nudge_free_topk > 0:
                            k = min(args.cw_nudge_free_topk, scores_slice.shape[1])
                            top_vals = scores_slice.topk(k=k, dim=1).values
                            base_loss = -top_vals.mean()
                        else:
                            tau = max(args.cw_nudge_tau, 1e-4)
                            base_loss = -tau * torch.logsumexp(scores_slice / tau, dim=1).mean()
                        scaled_loss = free_scale(len(subspace_axes)) * base_loss
                        total_loss = args.cw_nudge_alpha * args.cw_nudge_lambda * scaled_loss
                        total_loss.backward()
                        free_losses.append(float(scaled_loss.detach().cpu()))

                    total_batches_processed += 1

            if total_batches_processed > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

    finally:
        handle.remove()
        if not was_training:
            model.eval()

    if writer is not None and iteration is not None:
        stage_tag = stage.capitalize()
        if labeled_losses:
            writer.add_scalar(f"CW/Nudge/{stage_tag}/Loss_Labeled", sum(labeled_losses) / len(labeled_losses), iteration)
        if free_losses:
            writer.add_scalar(f"CW/Nudge/{stage_tag}/Loss_Free", sum(free_losses) / len(free_losses), iteration)
        if margin_losses:
            writer.add_scalar(f"CW/Nudge/{stage_tag}/Loss_Margin", sum(margin_losses) / len(margin_losses), iteration)
        if labeled_losses or free_losses:
            avg_lab = sum(labeled_losses) / len(labeled_losses) if labeled_losses else 0.0
            avg_free = sum(free_losses) / len(free_losses) if free_losses else 0.0
            total_loss_value = args.cw_nudge_alpha * (avg_lab + args.cw_nudge_lambda * avg_free)
            writer.add_scalar(f"CW/Nudge/{stage_tag}/Loss_Total", total_loss_value, iteration)
        grad_norm_value = 0.0
        if total_batches_processed > 0:
            grad_norm_value = grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
        writer.add_scalar(f"CW/Nudge/{stage_tag}/TotalBatches", total_batches_processed, iteration)
        writer.add_scalar(f"CW/Nudge/{stage_tag}/GradNorm", grad_norm_value, iteration)


def compute_alignment_metrics(model, subconcept_loaders, concept_dataset, batches_per_concept, label_to_scname, act_mode="pool_max"):
    """
    Calculate alignment metrics for both labeled and free subconcepts.
    
    For labeled subconcepts: global_top1, global_top5, subspace_top1
    For free subconcepts: axis_consistency, axis_purity, act_strength_ratio
    
    Returns dict with all metrics.
    """
    labeled_top1_correct = 0
    labeled_top5_correct = 0
    labeled_subspace_correct = 0
    labeled_total = 0


    axis_usage_by_sc = defaultdict(lambda: defaultdict(int))

    # free concepts
    free_img_count = defaultdict(int)
    free_chosen_act_sum = defaultdict(float)

    # labeled concepts
    labeled_axis_act_sum = defaultdict(float)
    labeled_axis_count = defaultdict(int)

    sc_to_hl = concept_dataset.get_subconcept_to_hl_name_mapping()
    subspace_map = concept_dataset.subspace_mapping

    # get last CW layer
    if not model.module.cw_layers:
        # no CW layer, return zeros
        return {
            "global_top1": 0.0,
            "subspace_top1": 0.0,
            "global_top5": 0.0,
            "axis_consistency": 0.0,
            "axis_purity": 0.0,
            "act_strength_ratio": 0.0
        }

    last_cw = model.module.cw_layers[-1]

    hook_storage = []
    def forward_hook(module, inp, out):
        hook_storage.append(out)

    handle = last_cw.register_forward_hook(forward_hook)
    model.eval()

    with torch.no_grad():
        # collect activations for each subconcept
        for (sc_label, loader) in subconcept_loaders:
            sc_name = label_to_scname[sc_label]
            is_free = concept_dataset.is_free_subconcept_name(sc_name)
            hl_name = sc_to_hl.get(sc_label, "general")
            subspace_axes = subspace_map.get(hl_name, [])

            loader_iter = iter(loader)
            for _ in range(batches_per_concept):
                batch_data = next(loader_iter, None)
                if not batch_data or not len(batch_data[0]):
                    break

                imgs = batch_data[0].cuda()
                for img in imgs:
                    _ = model(img.unsqueeze(0))
                    if not hook_storage:
                        continue

                    featmap = hook_storage[0]
                    hook_storage.clear()

                    axis_scores = reduce_axis_scores(featmap, act_mode=act_mode).squeeze(0)
                    if axis_scores.numel() == 0:
                        continue
                    top1_axis = axis_scores.argmax().item()
                    k = min(5, axis_scores.numel())
                    top5_axes = axis_scores.topk(k).indices.tolist() if k > 0 else []

                    # track top-1 axis usage
                    axis_usage_by_sc[sc_label][top1_axis] += 1

                    if is_free:
                        # free concept stats
                        free_img_count[sc_label] += 1
                        free_chosen_act_sum[sc_label] += float(axis_scores[top1_axis].item())
                    else:
                        # labeled concept stats
                        if top1_axis == sc_label:
                            labeled_top1_correct += 1
                        if sc_label in top5_axes:
                            labeled_top5_correct += 1
                        if subspace_axes:
                            local_sub_acts = axis_scores[subspace_axes]
                            local_winner_idx = local_sub_acts.argmax().item()
                            global_winner_axis = subspace_axes[local_winner_idx]
                            if global_winner_axis == sc_label:
                                labeled_subspace_correct += 1
                        labeled_total += 1

                        # track activation on target axis
                        if sc_label < len(axis_scores):
                            labeled_axis_act_sum[sc_label] += float(axis_scores[sc_label].item())
                            labeled_axis_count[sc_label] += 1

    handle.remove()
    model.train()

    # labeled concept metrics
    if labeled_total > 0:
        labeled_top1_pct = 100.0 * labeled_top1_correct / labeled_total
        labeled_top5_pct = 100.0 * labeled_top5_correct / labeled_total
        labeled_subspace_pct = 100.0 * labeled_subspace_correct / labeled_total
    else:
        labeled_top1_pct = labeled_top5_pct = labeled_subspace_pct = 0.0

    # free concept metrics
    sum_consistency = 0.0
    sum_purity = 0.0
    sum_act_ratio = 0.0
    free_count = 0

    for sc_label, axis_counts in axis_usage_by_sc.items():
        sc_name = label_to_scname[sc_label]
        if not concept_dataset.is_free_subconcept_name(sc_name):
            continue

        total_images = free_img_count[sc_label]
        if total_images <= 0:
            continue  # skip empty concepts

        free_count += 1

        # axis consistency
        best_axis, best_count = max(axis_counts.items(), key=lambda x: x[1])
        axis_consistency = 100.0 * best_count / float(total_images)

        # axis purity
        hl_name = sc_to_hl[sc_label]
        used_by_others = 0
        for (other_label, _) in subconcept_loaders:
            if other_label == sc_label:
                continue
            other_hl = sc_to_hl.get(other_label, "general")
            if other_hl == hl_name:
                used_by_others += axis_usage_by_sc[other_label].get(best_axis, 0)

        purity_den = best_count + used_by_others
        axis_purity = 100.0 * best_count / purity_den if purity_den > 0 else 100.0

        # activation strength ratio
        free_avg_act = free_chosen_act_sum[sc_label] / float(total_images)

        # get labeled activations in same high-level concept
        sum_labeled_act = 0.0
        count_labeled_act = 0
        for (lab_label, _) in subconcept_loaders:
            if lab_label == sc_label:
                continue
            if sc_to_hl.get(lab_label, "general") == hl_name:
                if labeled_axis_count[lab_label] > 0:
                    lab_mean = labeled_axis_act_sum[lab_label] / labeled_axis_count[lab_label]
                    sum_labeled_act += lab_mean
                    count_labeled_act += 1

        if count_labeled_act > 0:
            labeled_mean = sum_labeled_act / float(count_labeled_act)
            if labeled_mean < 1e-9:
                act_strength_ratio = 100.0
            else:
                act_strength_ratio = 100.0 * (free_avg_act / labeled_mean)
        else:
            act_strength_ratio = -1.0

        # accumulate metrics
        sum_consistency += axis_consistency
        sum_purity += axis_purity
        sum_act_ratio += act_strength_ratio

    if free_count > 0:
        axis_consistency_avg = sum_consistency / free_count
        axis_purity_avg = sum_purity / free_count
        act_strength_ratio_avg = sum_act_ratio / free_count
    else:
        axis_consistency_avg = 0.0
        axis_purity_avg = 0.0
        act_strength_ratio_avg = 0.0
    # return all metrics
    return {
        "global_top1": labeled_top1_pct,
        "subspace_top1": labeled_subspace_pct,
        "global_top5": labeled_top5_pct,
        "axis_consistency": axis_consistency_avg,
        "axis_purity": axis_purity_avg,
        "act_strength_ratio": act_strength_ratio_avg
    }


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
    print(f"[{mode}] Epoch {epoch+1}: Loss={losses.avg:.3f}, Top1={top1.avg:.2f}, Top5={top5.avg:.2f}")
    return top1.avg

# Save + Resume
def save_checkpoint(state, is_best, prefix, outdir=args.checkpoint_dir):
    os.makedirs(outdir, exist_ok=True)
    ckpt_path = os.path.join(outdir, f"{prefix}_checkpoint.pth")
    torch.save(state, ckpt_path)
    if is_best:
        best_path = os.path.join(outdir, f"{prefix}_best.pth")
        shutil.copyfile(ckpt_path, best_path)
        print(f"[Checkpoint] Best => {best_path}")

# Utility Classes
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

# Main Loop
def main():
    global args
    best_acc = best_prec

    for epoch in range(start_epoch, start_epoch + args.epochs):
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

    test_acc = validate(test_loader, model, start_epoch + args.epochs - 1, writer, mode="Test")
    print(f"[Done] Best Val={best_acc:.2f}, Final Test={test_acc:.2f}")
    writer.close()

if __name__ == "__main__":
    main()