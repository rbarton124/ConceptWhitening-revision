import argparse
import re
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

from collections import defaultdict
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from PIL import ImageFile

from types import SimpleNamespace
from contextlib import contextmanager

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
parser.add_argument("--vanilla_pretrain", action="store_true", help="Train without Concept Whitening, i.e. vanilla ResNet.")
parser.add_argument("--disable_subspaces", action="store_true", help="Disable subspace partitioning => one axis per concept.")
parser.add_argument("--use_free", action="store_true", help="Enable free unlabeled concept axes if the QCW layer supports it.")
# Alignment Nudge hyperparameters
parser.add_argument("--enable_nudge", action="store_true", help="Enable differentiable concept alignment nudge at alignment steps")
parser.add_argument("--nudge_alpha", type=float, default=1e-3, help="Weight for CW alignment nudge loss (applied at alignment steps)")
parser.add_argument("--nudge_margin", type=float, default=0.2, help="Margin for labeled concept separation within subspace")
parser.add_argument("--nudge_tau", type=float, default=0.1, help="Temperature for soft winner-takes-all (lower=harder)")
parser.add_argument("--nudge_min_activation", type=float, default=0.0, help="Minimum activation threshold for target concepts (0=disabled)")
# Label-free regularization
parser.add_argument("--use_block_diag", action="store_true", help="Enable block-diagonal covariance penalty on CE batches")
parser.add_argument("--block_diag_weight", type=float, default=1e-4, help="Weight for block-diagonal penalty")
# Advanced options
parser.add_argument("--anneal_nudge_alpha", action="store_true", help="Anneal nudge_alpha inversely to learning rate")
parser.add_argument("--nudge_grad_clip", type=float, default=5.0, help="Gradient clipping max norm for nudge step")

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

cross_subspace_mask = None
if not args.vanilla_pretrain and args.use_block_diag:
    if hasattr(model, 'cw_layers') and model.cw_layers:
        last_cw = model.cw_layers[-1]
        num_channels = last_cw.num_channels
        
        print(f"\n[Block-Diag] Building cross-subspace mask for {num_channels} channels...")
        cross_subspace_mask = build_cross_subspace_mask(num_channels, subspace_mapping)
        
        # Log mask statistics
        total_elements = num_channels ** 2
        cross_elements = cross_subspace_mask.sum().item()
        block_elements = total_elements - cross_elements
        print(f"[Block-Diag] Cross-subspace pairs: {int(cross_elements)} / {total_elements} "
              f"({100*cross_elements/total_elements:.1f}%)")
        print(f"[Block-Diag] Within-subspace pairs: {int(block_elements)} / {total_elements} "
              f"({100*block_elements/total_elements:.1f}%)")
        print(f"[Block-Diag] Penalty weight: {args.block_diag_weight}\n")
    else:
        print("[Warning] --use_block_diag=True but no CW layers found. Disabling block-diag penalty.")
        args.use_block_diag = False

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

# Move cross-subspace mask to CUDA if it was built
if cross_subspace_mask is not None:
    cross_subspace_mask = cross_subspace_mask.cuda()
    print(f"[Block-Diag] Mask moved to CUDA")

# Pre-Training Diagnostics and Validation
print("\n" + "="*70)
print(" "*20 + "PRE-TRAINING DIAGNOSTICS")
print("="*70)

if not args.vanilla_pretrain and model.module.cw_layers:
    print(f"\n[QCW Layers] Found {len(model.module.cw_layers)} QCW layer(s)")
    for idx, cw in enumerate(model.module.cw_layers):
        print(f"  Layer {idx}: {cw.num_channels} channels, "
              f"activation_mode={cw.activation_mode}, "
              f"cw_lambda={cw.cw_lambda:.4f}")
        print(f"    Subspace map: {list(cw.subspace_map.keys())}")
        
        # Check orthogonality
        Q = cw.running_rot[0]  # [C, C]
        orth_error = torch.norm(Q.t() @ Q - torch.eye(Q.size(0), device=Q.device)).item()
        print(f"    Initial orthogonality error: {orth_error:.2e} (should be ~0)")
    
    print(f"\n[Nudge Config]")
    print(f"  Enabled: {args.enable_nudge}")
    if args.enable_nudge:
        print(f"  Alpha: {args.nudge_alpha:.2e}")
        print(f"  Margin: {args.nudge_margin}")
        print(f"  Tau: {args.nudge_tau}")
        print(f"  Min activation: {args.nudge_min_activation}")
        print(f"  Grad clip: {args.nudge_grad_clip}")
        print(f"  Annealing: {args.anneal_nudge_alpha}")
    
    print(f"\n[Block-Diag Config]")
    print(f"  Enabled: {args.use_block_diag}")
    if args.use_block_diag:
        print(f"  Weight: {args.block_diag_weight:.2e}")
        print(f"  Mask shape: {cross_subspace_mask.shape if cross_subspace_mask is not None else 'N/A'}")
    
    print(f"\n[Alignment Config]")
    print(f"  Frequency: every {args.cw_align_freq} batches")
    print(f"  Batches per concept: {args.batches_per_concept}")
    print(f"  Lambda (free concepts): {args.cw_lambda}")
    
    # Test hook functionality with a dummy forward
    print(f"\n[Hook Test] Testing get_last_qcw_axis_scores with dummy batch...")
    try:
        dummy_imgs = torch.randn(4, 3, 224, 224).cuda()
        with torch.no_grad():
            a_test = get_last_qcw_axis_scores(model, dummy_imgs, reduce_mode="mean")
        print(f"  ✓ Hook successful! Output shape: {a_test.shape} (expected [4, {cw.num_channels}])")
        print(f"  ✓ Activation range: [{a_test.min():.3f}, {a_test.max():.3f}]")
    except Exception as e:
        print(f"  ✗ Hook test failed: {e}")
        print(f"  This may cause issues during training. Please investigate.")
    
    # Test frozen_norm_stats
    print(f"\n[Frozen Stats Test] Testing frozen_norm_stats context manager...")
    try:
        if model.module.cw_layers:
            cw = model.module.cw_layers[0]
            mean_before = cw.running_mean.clone()
            momentum_before = cw.momentum
            
            with frozen_norm_stats(model):
                _ = model(dummy_imgs)
                momentum_during = cw.momentum
            
            mean_after = cw.running_mean
            momentum_after = cw.momentum
            
            mean_changed = not torch.allclose(mean_before, mean_after)
            print(f"  Momentum: {momentum_before:.3f} → {momentum_during:.3f} (during) → {momentum_after:.3f} (after)")
            print(f"  Running mean changed: {mean_changed} (should be False)")
            print(f"  ✓ frozen_norm_stats working correctly!" if not mean_changed else "  ✗ Stats were updated!")
    except Exception as e:
        print(f"  ✗ Frozen stats test failed: {e}")

print("="*70 + "\n")

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

    pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                desc=f"[Train] Epoch {epoch+1}", smoothing=0.02)

    for i, (imgs, lbls) in pbar:
        iteration = epoch * len(train_loader) + i
        imgs, lbls = imgs.cuda(), lbls.cuda()

        # ========================================
        # Classification forward/backward/step
        # ========================================
        optimizer.zero_grad()
        outputs = model(imgs)
        ce_loss = criterion(outputs, lbls)
        
        # Optional block-diagonal penalty
        total_loss = ce_loss
        if (hasattr(args, 'use_block_diag') and args.use_block_diag and 
            'cross_subspace_mask' in globals() and cross_subspace_mask is not None):
            try:
                a_ce = get_last_qcw_axis_scores(model, imgs, reduce_mode="mean")
                bd_penalty = block_diag_penalty_fast(a_ce, cross_subspace_mask)
                total_loss = ce_loss + args.block_diag_weight * bd_penalty
                
                # Log occasionally (every 20 iterations)
                if i % 20 == 0:
                    writer.add_scalar("CW/BlockDiag/Penalty", bd_penalty.item(), iteration)
                    writer.add_scalar("CW/BlockDiag/WeightedPenalty", 
                                    (args.block_diag_weight * bd_penalty).item(), iteration)
            except Exception as e:
                # If block-diag fails, fall back to just CE loss
                print(f"[Warning] Block-diag penalty failed: {e}")
                total_loss = ce_loss
        
        total_loss.backward()
        optimizer.step()

        # Update classification metrics
        prec1, prec5 = accuracy_topk(outputs, lbls, (1, 5))
        losses.update(total_loss.item(), imgs.size(0))
        top1.update(prec1, imgs.size(0))
        top5.update(prec5, imgs.size(0))

        # ========================================
        # Concept Alignment (with optional nudge)
        # ========================================
        if (args.cw_align_freq is not None 
            and (i + 1) % args.cw_align_freq == 0 
            and len(concept_loaders) > 1):
            
            # Compute effective nudge_alpha (with optional annealing)
            effective_nudge_alpha = args.nudge_alpha
            if args.anneal_nudge_alpha:
                current_lr = optimizer.param_groups[0]['lr']
                lr_ratio = args.lr / current_lr if current_lr > 0 else 1.0
                effective_nudge_alpha = args.nudge_alpha * (lr_ratio ** 0.5)
            
            alignment_metrics = align_concepts(
                model,
                optimizer,  # Pass optimizer for nudge step
                concept_loaders[1:],  # sub-concept loaders
                concept_ds,
                args.batches_per_concept,
                args.cw_lambda,
                enable_nudge=args.enable_nudge,
                nudge_alpha=effective_nudge_alpha,
                nudge_margin=args.nudge_margin,
                nudge_tau=args.nudge_tau,
                nudge_min_activation=args.nudge_min_activation,
                nudge_grad_clip=args.nudge_grad_clip
            )

            # Extract metrics
            global_top1   = alignment_metrics["global_top1"]
            subspace_top1 = alignment_metrics["subspace_top1"]
            global_top5   = alignment_metrics["global_top5"]
            concept_loss  = alignment_metrics["concept_loss"]
            axis_consistency = alignment_metrics["axis_consistency"]
            axis_purity = alignment_metrics["axis_purity"]
            act_strength_ratio = alignment_metrics["act_strength_ratio"]
            labeled_nudge_loss = alignment_metrics["labeled_nudge_loss"]
            free_nudge_loss = alignment_metrics["free_nudge_loss"]
            total_nudge_loss = alignment_metrics["total_nudge_loss"]

            # Log to tensorboard
            writer.add_scalar("CW/Alignment/GlobalTop1",   global_top1,   iteration)
            writer.add_scalar("CW/Alignment/SubspaceTop1", subspace_top1, iteration)
            writer.add_scalar("CW/Alignment/GlobalTop5",   global_top5,   iteration)
            writer.add_scalar("CW/Alignment/ConceptLoss",  concept_loss,  iteration)
            writer.add_scalar("CW/FreeConcept/AxisConsistency", axis_consistency, iteration)
            writer.add_scalar("CW/FreeConcept/AxisPurity", axis_purity,  iteration)
            writer.add_scalar("CW/FreeConcept/ActStrengthRatio", act_strength_ratio, iteration)
            
            # Log nudge losses
            if args.enable_nudge:
                writer.add_scalar("CW/Nudge/LabeledLoss", labeled_nudge_loss, iteration)
                writer.add_scalar("CW/Nudge/FreeLoss", free_nudge_loss, iteration)
                writer.add_scalar("CW/Nudge/TotalLoss", total_nudge_loss, iteration)
                writer.add_scalar("CW/Nudge/EffectiveAlpha", effective_nudge_alpha, iteration)

            # Updated alignment score (emphasize subspace_top1)
            alignment_score.update(0.65 * subspace_top1 + 0.30 * global_top5 + 0.05 * axis_consistency)

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

def align_concepts(model, optimizer, subconcept_loaders, concept_dataset, batches_per_concept, lambda_,
                   enable_nudge=False, nudge_alpha=1e-3, nudge_margin=0.2, nudge_tau=0.1, 
                   nudge_min_activation=0.0, nudge_grad_clip=5.0):
    """
    Three-phase concept alignment:
      1. No-grad accumulation for Cayley update (existing)
      2. Cayley rotation matrix update (existing, FIXED: single update)
      3. Differentiable alignment nudge with frozen stats (NEW, optional)
    Then compute metrics.
    
    Returns: metrics dict with alignment scores, concept loss, and nudge losses
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


    # Update rotation matrix (just once now [not sure about this yet])
    model.module.update_rotation_matrix()
    model.module.change_mode(-1)
    for cw_layer in model.module.cw_layers:
        cw_layer.clear_subspace()
        cw_layer.set_subspace_scaling(1.0)

    # Get concept loss from no-grad tracking
    total_cw_loss = 0.0
    for cw_layer in model.module.cw_layers:
        total_cw_loss += cw_layer.get_concept_loss()
    avg_cw_loss = total_cw_loss / max(1, len(model.module.cw_layers))
    
    # Differentiable alignment nudge
    labeled_nudge_loss_val = 0.0
    free_nudge_loss_val = 0.0
    total_nudge_loss_val = 0.0
    
    if enable_nudge:
        # Verify QCW layers are in correct state (not accumulating)
        for cw in model.module.cw_layers:
            if cw.mode >= 0:
                print("[ERROR] QCW layer has mode>=0 during nudge! Resetting to -1.")
                cw.mode = -1
            if cw.active_subspace is not None:
                print("[ERROR] QCW layer has active_subspace set during nudge! Clearing.")
                cw.clear_subspace()
        
        model.train()  # Enable autograd
        optimizer.zero_grad()  # Fresh gradients
        
        sc_to_hl = concept_dataset.get_subconcept_to_hl_name_mapping()
        subspace_map = concept_dataset.subspace_mapping
        
        labeled_nudge_loss = torch.tensor(0.0, device=next(model.parameters()).device, requires_grad=True)
        free_nudge_loss = torch.tensor(0.0, device=next(model.parameters()).device, requires_grad=True)
        num_labeled_terms = 0
        num_free_terms = 0
        
        with frozen_norm_stats(model):
            # === Labeled Concepts: Margin Loss ===
            for hl_name, groups in hl_subconcepts.items():
                for (sc_label, sc_loader) in groups["labeled"]:
                    loader_iter = iter(sc_loader)
                    for _ in range(min(batches_per_concept, len(sc_loader))):
                        batch_data = next(loader_iter, None)
                        if not batch_data or not len(batch_data[0]):
                            break
                        
                        c_imgs = batch_data[0].cuda(non_blocking=True)
                        if c_imgs.size(0) == 0:
                            continue
                        
                        # Get axis scores (differentiable)
                        a = get_last_qcw_axis_scores(model, c_imgs, reduce_mode="mean")
                        sc_lbls = torch.tensor([sc_label] * a.size(0), 
                                              dtype=torch.long, device=a.device)
                        
                        loss_lab = labeled_margin_loss(a, sc_lbls, sc_to_hl, subspace_map, 
                                                       margin=nudge_margin, 
                                                       min_activation=nudge_min_activation)
                        labeled_nudge_loss = labeled_nudge_loss + loss_lab
                        num_labeled_terms += 1
            
            # === Free Concepts: Soft Winner-Takes-All ===
            for hl_name, groups in hl_subconcepts.items():
                free_list = groups["free"]
                if not free_list:
                    continue
                
                for (sc_label, sc_loader) in free_list:
                    loader_iter = iter(sc_loader)
                    for _ in range(min(batches_per_concept, len(sc_loader))):
                        batch_data = next(loader_iter, None)
                        if not batch_data or not len(batch_data[0]):
                            break
                        
                        c_imgs = batch_data[0].cuda(non_blocking=True)
                        if c_imgs.size(0) == 0:
                            continue
                        
                        # Get axis scores (differentiable)
                        a = get_last_qcw_axis_scores(model, c_imgs, reduce_mode="mean")
                        hl_lbls = [hl_name] * a.size(0)
                        
                        loss_free = soft_wta_hl_loss(a, hl_lbls, subspace_map, tau=nudge_tau)
                        free_nudge_loss = free_nudge_loss + loss_free
                        num_free_terms += 1
        
        # Average losses
        if num_labeled_terms > 0:
            labeled_nudge_loss = labeled_nudge_loss / float(num_labeled_terms)
        if num_free_terms > 0:
            free_nudge_loss = free_nudge_loss / float(num_free_terms)
        
        total_nudge_loss = labeled_nudge_loss + free_nudge_loss
        
        # Backward and step (only if we computed any losses)
        if num_labeled_terms + num_free_terms > 0:
            scaled_loss = nudge_alpha * total_nudge_loss
            scaled_loss.backward()
            
            # Gradient clipping for safety
            if nudge_grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=nudge_grad_clip)
            else:
                grad_norm = sum(p.grad.norm()**2 for p in model.parameters() if p.grad is not None)**0.5
            
            # Check for invalid gradients
            has_invalid = any(
                p.grad is not None and not torch.isfinite(p.grad).all()
                for p in model.parameters()
            )
            
            if not has_invalid:
                optimizer.step()
            else:
                print("[Warning] Invalid gradients in CW nudge, skipping optimizer step")
            
            optimizer.zero_grad()
            
            # Store for logging
            labeled_nudge_loss_val = labeled_nudge_loss.item() if num_labeled_terms > 0 else 0.0
            free_nudge_loss_val = free_nudge_loss.item() if num_free_terms > 0 else 0.0
            total_nudge_loss_val = total_nudge_loss.item()
        
        model.eval()
    
    # Compute alignment metrics
    metrics = compute_alignment_metrics(
        model, subconcept_loaders, concept_dataset,
        batches_per_concept, label_to_scname
    )
    
    # Add nudge losses to metrics
    metrics["concept_loss"] = avg_cw_loss
    metrics["labeled_nudge_loss"] = labeled_nudge_loss_val
    metrics["free_nudge_loss"] = free_nudge_loss_val
    metrics["total_nudge_loss"] = total_nudge_loss_val
    
    return metrics



def compute_alignment_metrics(model, subconcept_loaders, concept_dataset, batches_per_concept, label_to_scname):
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

                    featmap = hook_storage[0]  # shape [1, C, H, W]
                    hook_storage.clear()

                    # spatial average
                    feat_avg = featmap.mean(dim=(2,3)).squeeze(0)  # shape [C]
                    top1_axis = feat_avg.argmax().item()
                    top5_axes = feat_avg.topk(5)[1].tolist()

                    # track top-1 axis usage
                    axis_usage_by_sc[sc_label][top1_axis] += 1

                    if is_free:
                        # free concept stats
                        free_img_count[sc_label] += 1
                        free_chosen_act_sum[sc_label] += float(feat_avg[top1_axis].item())
                    else:
                        # labeled concept stats
                        if top1_axis == sc_label:
                            labeled_top1_correct += 1
                        if sc_label in top5_axes:
                            labeled_top5_correct += 1
                        if subspace_axes:
                            local_sub_acts = feat_avg[subspace_axes]
                            local_winner_idx = local_sub_acts.argmax().item()
                            global_winner_axis = subspace_axes[local_winner_idx]
                            if global_winner_axis == sc_label:
                                labeled_subspace_correct += 1
                        labeled_total += 1

                        # track activation on target axis
                        if sc_label < len(feat_avg):
                            labeled_axis_act_sum[sc_label] += float(feat_avg[sc_label].item())
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


# CW-LOSS INTEGRATION INFRASTRUCTURE

@contextmanager
def frozen_norm_stats(model):
    """
    Context manager that freezes running statistics in BatchNorm and IterNormRotation
    layers while still allowing gradients to flow for backpropagation.
    
    This enables differentiable forward passes on concept data without corrupting
    the running statistics that were learned on the main classification dataset.
    
    Key behaviors:
    - BatchNorm layers: Set to eval() mode (use running stats, don't update them)
    - IterNormRotation layers: Set momentum=0.0 (prevents running stat updates)
    - Gradients: Still flow normally (training=True for autograd)
    
    Usage:
        with frozen_norm_stats(model):
            outputs = model(concept_images)
            loss = some_loss(outputs)
            loss.backward()  # Gradients flow, but stats don't update
    """
    # Determine if model is wrapped in DataParallel
    if isinstance(model, nn.DataParallel):
        actual_model = model.module
    else:
        actual_model = model
    
    # Store original training state
    was_training = model.training
    
    # Freeze BatchNorm layers
    bn_states = []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
            bn_states.append((m, m.training))
            m.eval()  # Use running stats but don't update them
    
    # Freeze IterNormRotation layers
    cw_layers = getattr(actual_model, 'cw_layers', [])
    saved_momentums = []
    for cw in cw_layers:
        saved_momentums.append(cw.momentum)
        cw.momentum = 0.0  # Prevent running stat updates while keeping training=True
    
    try:
        yield  # Code inside the `with` block runs here
    finally:
        # Restore BatchNorm states
        for m, was_train in bn_states:
            m.train(was_train)
        
        # Restore IterNormRotation momentums
        for cw, original_momentum in zip(cw_layers, saved_momentums):
            cw.momentum = original_momentum
        
        # Restore overall model training state
        model.train(was_training)


def get_last_qcw_axis_scores(model, imgs, reduce_mode="mean"):
    """
    Extract per-axis activation scores from the last QCW layer with DataParallel support.
    
    Critical: When using DataParallel, the forward hook fires once PER REPLICA,
    so we must gather outputs from all replicas and concatenate them.
    
    Args:
        model: QCW model (possibly wrapped in DataParallel)
        imgs: [B, 3, H, W] input images (already on cuda)
        reduce_mode: How to reduce spatial dimensions ('mean', 'max', 'pool_max')
    
    Returns:
        a: [B, C] activation scores with gradients attached
    """
    if isinstance(model, nn.DataParallel):
        actual_model = model.module
    else:
        actual_model = model
    
    if not actual_model.cw_layers:
        raise ValueError("Model has no QCW layers")
    
    last_cw = actual_model.cw_layers[-1]
    
    # Hook to capture output (will fire once per replica in DataParallel)
    captured = []
    def hook(module, input, output):
        captured.append(output)
    
    handle = last_cw.register_forward_hook(hook)
    
    try:
        outputs = model(imgs)  # Full forward pass
    finally:
        handle.remove()
    
    if not captured:
        raise RuntimeError("Failed to capture QCW layer output via hook")
    
    # If DataParallel, captured will have one tensor per GPU
    # We must concatenate them to get the full batch
    if len(captured) > 1:
        # Multiple replicas - concatenate
        feat = torch.cat([t.to(imgs.device) for t in captured], dim=0)
    else:
        feat = captured[0]
    
    # Verify we got the full batch
    if feat.size(0) != imgs.size(0):
        print(f"[Warning] Hook captured {feat.size(0)} samples but expected {imgs.size(0)}. "
              f"This may indicate a DataParallel gathering issue.")
    
    # Reduce spatial dimensions: [B, C, H, W] -> [B, C]
    if reduce_mode == "mean":
        a = feat.mean(dim=(2, 3))
    elif reduce_mode == "max":
        a = feat.flatten(2).max(dim=2)[0]
    elif reduce_mode == "pool_max":
        # Match IterNormRotation._reduce_activation logic
        pooled, _ = nn.functional.max_pool2d(feat, kernel_size=3, stride=3, return_indices=True)
        a = pooled.flatten(2).mean(dim=2)
    else:
        a = feat.mean(dim=(2, 3))
    
    return a  # [B, C] with gradients


def labeled_margin_loss(a, sc_labels, sc_to_hl, subspace_map, margin=0.2, min_activation=0.0):
    """
    Margin loss: target concept axis should beat competitors in its subspace by `margin`.
    
    This encourages clean separation within each high-level concept subspace,
    matching the QCW objective for labeled subconcepts.
    
    Args:
        a: [B, C] activation scores (from rotated QCW output)
        sc_labels: [B] or list of target axis indices
        sc_to_hl: dict mapping subconcept_idx -> high_level_name
        subspace_map: dict mapping high_level_name -> list of axis indices
        margin: Minimum gap between target and best competitor (default 0.2)
        min_activation: Minimum absolute activation for target (0=disabled)
    
    Returns:
        scalar loss (average over batch)
    """
    device = a.device
    loss = torch.tensor(0.0, device=device, requires_grad=True)
    B, C = a.shape
    valid_samples = 0
    
    for i in range(B):
        j = int(sc_labels[i])  # Target axis index
        
        # Bounds check
        if j >= C:
            continue
        
        # Get high-level concept and subspace
        hl_name = sc_to_hl.get(j)
        if hl_name is None:
            continue
        
        S = subspace_map.get(hl_name, [])
        if not S or j not in S:
            continue
        
        # Find competitors in same subspace (excluding target)
        competitors = [k for k in S if k != j and k < C]
        if not competitors:
            # Single-axis subspace: only apply min_activation if set
            if min_activation > 0:
                loss = loss + torch.clamp(min_activation - a[i, j], min=0.0)
            valid_samples += 1
            continue
        
        a_target = a[i, j]
        a_comp = a[i, competitors]
        max_comp = a_comp.max()
        
        # Margin violation: target should beat best competitor by at least `margin`
        margin_term = torch.clamp(margin + max_comp - a_target, min=0.0)
        loss = loss + margin_term
        
        # Optional: encourage minimum absolute activation
        if min_activation > 0:
            min_term = torch.clamp(min_activation - a_target, min=0.0)
            loss = loss + min_term
        
        valid_samples += 1
    
    if valid_samples > 0:
        return loss / float(valid_samples)
    else:
        return torch.tensor(0.0, device=device, requires_grad=True)


def soft_wta_hl_loss(a, hl_labels, subspace_map, tau=0.1):
    """
    Soft winner-takes-all within high-level subspace using LogSumExp.
    
    This encourages at least one axis in the subspace to activate strongly,
    matching the QCW objective for free (unlabeled) subconcepts.
    
    Math: Maximizes max_{j in S} a[i,j] via smooth approximation:
          -tau * log(sum_{j in S} exp(a[i,j] / tau)) ≈ -max_{j in S} a[i,j]
    
    Args:
        a: [B, C] activation scores
        hl_labels: list of length B with high-level concept names (strings)
        subspace_map: dict mapping high_level_name -> list of axis indices
        tau: Temperature for softmax approximation (smaller = harder max, default 0.1)
    
    Returns:
        scalar loss (average over batch)
    """
    device = a.device
    loss = torch.tensor(0.0, device=device, requires_grad=True)
    B, C = a.shape
    valid_samples = 0
    
    for i in range(B):
        hl_name = hl_labels[i]
        S = subspace_map.get(hl_name, [])
        
        # Filter to valid indices
        S_valid = [j for j in S if j < C]
        if not S_valid:
            continue
        
        # Get activations for this subspace
        a_S = a[i, S_valid]
        
        # LogSumExp: -tau * log(sum(exp(a_S / tau)))
        # As tau -> 0, this approaches -max(a_S)
        lse = tau * torch.logsumexp(a_S / tau, dim=0)
        loss = loss - lse  # Negative because we want to maximize
        valid_samples += 1
    
    if valid_samples > 0:
        return loss / float(valid_samples)
    else:
        return torch.tensor(0.0, device=device, requires_grad=True)


def build_cross_subspace_mask(num_channels, subspace_map):
    """
    Precompute boolean mask for cross-subspace (i,j) pairs.
    
    This mask identifies which pairs of axes belong to DIFFERENT high-level concepts,
    used for the block-diagonal covariance penalty.
    
    Args:
        num_channels: Total number of channels (C) in the QCW layer
        subspace_map: dict {hl_name -> [axis_indices]}
    
    Returns:
        [C, C] boolean tensor (True for cross-subspace pairs)
    """
    mask = torch.zeros(num_channels, num_channels, dtype=torch.bool)
    
    subspaces_list = list(subspace_map.values())
    
    # Mark all (i,j) pairs where i and j are in different subspaces
    for s1_idx, S1 in enumerate(subspaces_list):
        for s2_idx, S2 in enumerate(subspaces_list):
            if s1_idx >= s2_idx:  # Skip same subspace and avoid double-counting
                continue
            
            # Mark all cross-subspace pairs
            for i in S1:
                for j in S2:
                    if i < num_channels and j < num_channels:
                        mask[i, j] = True
                        mask[j, i] = True  # Symmetric
    
    return mask


def block_diag_penalty_fast(a, cross_subspace_mask):
    """
    Penalize covariance between axes in different subspaces.
    
    Encourages the latent space to have block-diagonal covariance structure,
    where axes within a subspace can correlate but axes across subspaces cannot.
    
    Args:
        a: [B, C] activation scores
        cross_subspace_mask: [C, C] boolean mask (from build_cross_subspace_mask)
    
    Returns:
        scalar penalty (mean squared cross-subspace covariance)
    """
    # Center activations
    a_centered = a - a.mean(dim=0, keepdim=True)
    
    # Compute covariance matrix
    Sigma = (a_centered.t() @ a_centered) / a.size(0)  # [C, C]
    
    # Penalize off-block-diagonal elements
    penalty = (Sigma[cross_subspace_mask] ** 2).mean()
    
    return penalty


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