import argparse
import re
import os
import time
import random
import shutil

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

from MODELS.factory_qcw import build_qcw
from MODELS.ConceptDataset_QCW import ConceptDataset

# Optional: tolerate truncated JPEGs if dataset has any
ImageFile.LOAD_TRUNCATED_IMAGES = True

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

args = parser.parse_args()

# Validate model and depth combinations
if args.model != "resnet" and args.depth not in [None, 121, 161]:
    args.depth = {"densenet": 161, "vgg16": None}[args.model]
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
        args.whitened_layers = [int(x) for x in args.whitened_layers.split(",") if x.strip()]
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
        exit(1)

print("=========== ARGUMENTS ===========")
for k, v in sorted(vars(args).items()):
    print(f"{k}: {v}")
print("=================================")

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
cudnn.benchmark = True

writer = SummaryWriter(log_dir=os.path.join(args.log_dir, f"{args.prefix}_{int(time.time())}"))

################################################################################
# Helpers
################################################################################

def reduce_axis_scores(featmap: torch.Tensor, act_mode: str = "pool_max") -> torch.Tensor:
    """
    Reduce spatial dimensions of CW layer output to per-axis activation scores.
    Used by compute_alignment_metrics to match the pooling used inside IterNormRotation.
    """
    if act_mode == "mean":
        return featmap.mean(dim=(2, 3))
    elif act_mode == "max":
        return featmap.flatten(2).max(dim=2).values
    elif act_mode == "pos_mean":
        pos_mask = (featmap > 0).to(featmap.dtype)
        numerator = (featmap * pos_mask).sum(dim=(2, 3))
        denominator = pos_mask.sum(dim=(2, 3)).clamp(min=1e-6)
        return numerator / denominator
    elif act_mode == "pool_max":
        pooled = F.max_pool2d(featmap, kernel_size=3, stride=3)
        return pooled.flatten(2).mean(dim=2)
    else:
        return featmap.mean(dim=(2, 3))


################################################################################
# Build Main Dataloaders
################################################################################
def build_main_loaders(args):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(CROP_SIZE, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), shear=5,
                                interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

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


################################################################################
# Build Concept Dataset + Loaders
################################################################################
def build_concept_loaders(args):
    concept_root = os.path.join(args.concept_dir, "concept_train")
    if not args.bboxes:
        args.bboxes = os.path.join(args.concept_dir, "bboxes.json")

    concept_transform = transforms.Compose([
        transforms.RandomResizedCrop(CROP_SIZE, scale=(0.8, 1.0), ratio=(0.9, 1.1),
                                     interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05),
                                interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    hl_list = [x.strip() for x in args.concepts.split(",") if x.strip()]
    crop_mode = args.concept_image_mode

    concept_dataset = ConceptDataset(
        root_dir=concept_root,
        bboxes_file=args.bboxes,
        high_level_filter=hl_list,
        transform=concept_transform,
        crop_mode=crop_mode
    )

    main_concept_loader = DataLoader(
        concept_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=PIN_MEMORY
    )

    subconcept_loaders = []
    subconcept_samples = defaultdict(list)

    for idx, (_, _, hl_name, sc_name) in enumerate(concept_dataset.samples):
        sc_label = concept_dataset.sc2idx[sc_name]
        subconcept_samples[sc_label].append(idx)

    for sc_label, indices in subconcept_samples.items():
        sc_dataset = torch.utils.data.Subset(concept_dataset, indices)
        sc_loader = DataLoader(
            sc_dataset,
            batch_size=min(args.batch_size, len(indices)),
            shuffle=True,
            num_workers=args.workers,
            pin_memory=PIN_MEMORY
        )
        subconcept_loaders.append((sc_label, sc_loader))

    print(f"Created {len(subconcept_loaders)} subconcept-specific loaders")
    concept_loaders = [(-1, main_concept_loader)] + subconcept_loaders
    return concept_loaders, concept_dataset


print(f"[Data] #Main Train: {len(train_loader.dataset)}")
print(f"[Data] #Val:        {len(val_loader.dataset)}")
print(f"[Data] #Test:       {len(test_loader.dataset)}")

concept_loaders, concept_ds, subconcept_loaders, subspaces = None, None, None, None
if not args.vanilla_pretrain:
    concept_loaders, concept_ds = build_concept_loaders(args)
    subconcept_loaders = concept_loaders[1:]

    print(f"[Data] #Concept:    {len(concept_loaders[0][1].dataset)}")
    print(f"[Data] #Subconcept Loaders: {len(concept_loaders) - 1}")

    print("\n===== CONCEPT DATASET DETAILS =====")
    print(f"Total number of samples: {len(concept_ds.samples)}")
    print(f"Total number of subconcepts: {concept_ds.get_num_subconcepts()}")
    print(f"Total number of high-level concepts: {concept_ds.get_num_high_level()}")

    print("\nSubconcept names and indices:")
    for sc_name in concept_ds.get_subconcept_names():
        sc_idx = concept_ds.sc2idx[sc_name]
        print(f"  [{sc_idx}] {sc_name}")

    print("\nHigh-level concepts and their subspaces:")
    for hl_name in concept_ds.get_hl_names():
        subspace = concept_ds.subspace_mapping.get(hl_name, [])
        print(f"  {hl_name}: {subspace}")

    subconcept_loaders = sorted(subconcept_loaders, key=lambda x: x[0])
    print("\nSubconcept-specific loaders:")
    for sc_label, loader in subconcept_loaders:
        sc_name = concept_ds.idx2sc.get(sc_label, str(sc_label))
        print(f"  Subconcept '{sc_name}' [{sc_label}]: {len(loader.dataset)} samples")

    random.shuffle(subconcept_loaders)

    print("\nShuffled Subconcept-specific loaders:")
    for sc_label, loader in subconcept_loaders:
        sc_name = concept_ds.idx2sc.get(sc_label, str(sc_label))
        print(f"  Subconcept '{sc_name}' [{sc_label}]: {len(loader.dataset)} samples")

    if not args.disable_subspaces:
        subspaces = concept_ds.subspace_mapping
    else:
        subspaces = {hl: [0] for hl in concept_ds.subspace_mapping.keys()}

else:
    concept_loaders = []
    concept_ds = None
    subspaces = {}

subspace_mapping = subspaces
print(f"Subspace mapping: {subspace_mapping}")


################################################################################
# Build QCW model
################################################################################
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


def _move_optimizer_state_to_cuda(opt: optim.Optimizer):
    """Move any tensor optimizer state to CUDA (for resuming)."""
    for st in opt.state.values():
        if isinstance(st, dict):
            for k, v in st.items():
                if torch.is_tensor(v) and v.device.type != "cuda":
                    st[k] = v.cuda(non_blocking=True)


def maybe_resume_checkpoint(model, optimizer, args):
    """
    Resume from checkpoint, handling ResNet, DenseNet, VGG, or QCW formats.
    Loads optimizer state if requested.
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
        if old_key.startswith("module."):
            old_key = old_key[len("module."):]

        old_key = re.sub(r"\.norm\.(\d+)", lambda m: f".norm{m.group(1)}", old_key)
        old_key = re.sub(r"\.conv\.(\d+)", lambda m: f".conv{m.group(1)}", old_key)

        if not old_key.startswith("backbone.") and ("backbone." + old_key in model_sd):
            return "backbone." + old_key
        if old_key.startswith("fc.") and ("backbone.fc" + old_key[2:] in model_sd):
            return "backbone." + old_key
        if old_key.startswith("classifier.") and ("backbone.classifier" + old_key[10:] in model_sd):
            return "backbone.classifier" + old_key[10:]

        return old_key

    skipped_keys = []

    for ckpt_k, ckpt_v in raw_sd.items():
        new_k = rename_key(ckpt_k)
        if new_k in model_sd and ckpt_v.shape == model_sd[new_k].shape:
            renamed_sd[new_k] = ckpt_v
        else:
            if new_k in model_sd:
                skipped_keys.append(f"{ckpt_k}: shape {getattr(ckpt_v, 'shape', None)} != {model_sd[new_k].shape}")
            else:
                skipped_keys.append(f"{ckpt_k}: no match found in model")

    print("Loading model...")
    result = model.load_state_dict(renamed_sd, strict=False)
    print("Missing keys:", result.missing_keys)
    print("Unexpected keys:", result.unexpected_keys)

    print("[Checkpoint] Skipped keys from checkpoint:")
    for sk in skipped_keys[:50]:
        print("   ", sk)
    if len(skipped_keys) > 50:
        print(f"   ... and {len(skipped_keys)-50} more")

    if isinstance(ckpt, dict):
        best_prec = float(ckpt.get("best_prec1", best_prec))
        if not args.only_load_weights:
            start_epoch = int(ckpt.get("epoch", 0))

            if "optimizer" in ckpt and optimizer is not None:
                try:
                    optimizer.load_state_dict(ckpt["optimizer"])
                    _move_optimizer_state_to_cuda(optimizer)
                except Exception as e:
                    print(f"[Warning] Could not load optimizer state: {e}")

        print(f"[Checkpoint] Resumed epoch={start_epoch}, best_prec={best_prec:.2f}")

    return start_epoch, best_prec


################################################################################
# Build optimizer
################################################################################

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
start_epoch, best_prec = maybe_resume_checkpoint(model, optimizer, args)

# Wrap DataParallel AFTER loading weights/states
model = nn.DataParallel(model).cuda()


################################################################################
# Train + Align
################################################################################
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
        imgs, lbls = imgs.cuda(non_blocking=True), lbls.cuda(non_blocking=True)

        # classification forward/backward
        outputs = model(imgs)
        loss = criterion(outputs, lbls)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Update classification metrics
        prec1, prec5 = accuracy_topk(outputs, lbls, (1, 5))
        losses.update(loss.item(), imgs.size(0))
        top1.update(prec1, imgs.size(0))
        top5.update(prec5, imgs.size(0))

        # concept alignment
        if (args.cw_align_freq is not None
            and (i + 1) % args.cw_align_freq == 0
            and concept_loaders is not None
            and len(concept_loaders) > 1):

            alignment_metrics = align_concepts(
                model,
                concept_loaders[1:],
                concept_ds,
                args.batches_per_concept,
                args.cw_lambda,
                act_mode=args.act_mode
            )

            global_top1   = alignment_metrics["global_top1"]
            subspace_top1 = alignment_metrics["subspace_top1"]
            global_top5   = alignment_metrics["global_top5"]
            concept_loss  = alignment_metrics["concept_loss"]
            axis_consistency = alignment_metrics["axis_consistency"]
            axis_purity = alignment_metrics["axis_purity"]
            act_strength_ratio = alignment_metrics["act_strength_ratio"]

            writer.add_scalar("CW/Alignment/GlobalTop1",   global_top1,   iteration)
            writer.add_scalar("CW/Alignment/SubspaceTop1", subspace_top1, iteration)
            writer.add_scalar("CW/Alignment/GlobalTop5",   global_top5,   iteration)
            writer.add_scalar("CW/Alignment/ConceptLoss",  concept_loss,  iteration)
            writer.add_scalar("CW/FreeConcept/AxisConsistency", axis_consistency, iteration)
            writer.add_scalar("CW/FreeConcept/AxisPurity", axis_purity,  iteration)
            writer.add_scalar("CW/FreeConcept/ActStrengthRatio", act_strength_ratio, iteration)

            alignment_score.update(0.3 * global_top5 + 0.65 * subspace_top1 + 0.05 * global_top1)

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
    model.eval()

    sc_to_hl_name = concept_dataset.get_subconcept_to_hl_name_mapping()
    hl_subconcepts = defaultdict(lambda: {"labeled": [], "free": []})

    label_to_scname = {}
    for (sc_label, loader) in subconcept_loaders:
        label_to_scname[sc_label] = concept_dataset.idx2sc.get(sc_label, None)

    for (sc_label, loader) in subconcept_loaders:
        hl_name = sc_to_hl_name.get(sc_label, "general")
        sc_name = label_to_scname[sc_label]
        if sc_name is None:
            continue
        if concept_dataset.is_free_subconcept_name(sc_name):
            hl_subconcepts[hl_name]["free"].append((sc_label, loader))
        else:
            hl_subconcepts[hl_name]["labeled"].append((sc_label, loader))

    for cw_layer in model.module.cw_layers:
        cw_layer.reset_concept_loss()

    # labeled pass
    with torch.no_grad():
        for hl_name, groups in hl_subconcepts.items():
            for (sc_label, sc_loader) in groups["labeled"]:
                model.module.change_mode(sc_label)
                loader_iter = iter(sc_loader)
                for _ in range(batches_per_concept):
                    batch_data = next(loader_iter, None)
                    if not batch_data or not len(batch_data[0]):
                        break
                    imgs = batch_data[0].cuda(non_blocking=True)
                    for img in imgs:
                        model(img.unsqueeze(0))

    # free pass
    for cw_layer in model.module.cw_layers:
        cw_layer.set_subspace_scaling(lambda_)

    with torch.no_grad():
        for hl_name, groups in hl_subconcepts.items():
            free_list = groups["free"]
            if not free_list:
                continue

            for cw_layer in model.module.cw_layers:
                cw_layer.clear_subspace()
                cw_layer.set_subspace(hl_name)

            model.module.change_mode(-1)

            for (sc_label, sc_loader) in free_list:
                loader_iter = iter(sc_loader)
                for _ in range(batches_per_concept):
                    batch_data = next(loader_iter, None)
                    if not batch_data or not len(batch_data[0]):
                        break
                    imgs = batch_data[0].cuda(non_blocking=True)
                    for img in imgs:
                        model(img.unsqueeze(0))

    model.module.update_rotation_matrix()
    model.module.change_mode(-1)
    for cw_layer in model.module.cw_layers:
        cw_layer.clear_subspace()
        cw_layer.set_subspace_scaling(1.0)

    total_cw_loss = 0.0
    for cw_layer in model.module.cw_layers:
        total_cw_loss += cw_layer.get_concept_loss()
    avg_cw_loss = total_cw_loss / max(1, len(model.module.cw_layers))

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


def compute_alignment_metrics(model, subconcept_loaders, concept_dataset, batches_per_concept, label_to_scname, act_mode="pool_max"):
    labeled_top1_correct = 0
    labeled_top5_correct = 0
    labeled_subspace_correct = 0
    labeled_total = 0

    axis_usage_by_sc = defaultdict(lambda: defaultdict(int))

    free_img_count = defaultdict(int)
    free_chosen_act_sum = defaultdict(float)

    labeled_axis_act_sum = defaultdict(float)
    labeled_axis_count = defaultdict(int)

    sc_to_hl = concept_dataset.get_subconcept_to_hl_name_mapping()
    subspace_map = concept_dataset.subspace_mapping

    if not model.module.cw_layers:
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
        for (sc_label, loader) in subconcept_loaders:
            sc_name = label_to_scname.get(sc_label, None)
            if sc_name is None:
                continue
            is_free = concept_dataset.is_free_subconcept_name(sc_name)
            hl_name = sc_to_hl.get(sc_label, "general")
            subspace_axes = subspace_map.get(hl_name, [])

            loader_iter = iter(loader)
            for _ in range(batches_per_concept):
                batch_data = next(loader_iter, None)
                if not batch_data or not len(batch_data[0]):
                    break

                imgs = batch_data[0].cuda(non_blocking=True)
                for img in imgs:
                    model(img.unsqueeze(0))
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

                    axis_usage_by_sc[sc_label][top1_axis] += 1

                    if is_free:
                        free_img_count[sc_label] += 1
                        free_chosen_act_sum[sc_label] += float(axis_scores[top1_axis].item())
                    else:
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

                        if sc_label < len(axis_scores):
                            labeled_axis_act_sum[sc_label] += float(axis_scores[sc_label].item())
                            labeled_axis_count[sc_label] += 1

    handle.remove()
    model.train()

    if labeled_total > 0:
        labeled_top1_pct = 100.0 * labeled_top1_correct / labeled_total
        labeled_top5_pct = 100.0 * labeled_top5_correct / labeled_total
        labeled_subspace_pct = 100.0 * labeled_subspace_correct / labeled_total
    else:
        labeled_top1_pct = labeled_top5_pct = labeled_subspace_pct = 0.0

    sum_consistency = 0.0
    sum_purity = 0.0
    sum_act_ratio = 0.0
    free_count = 0

    for sc_label, axis_counts in axis_usage_by_sc.items():
        sc_name = label_to_scname.get(sc_label, "")
        if not concept_dataset.is_free_subconcept_name(sc_name):
            continue

        total_images = free_img_count[sc_label]
        if total_images <= 0:
            continue

        free_count += 1

        best_axis, best_count = max(axis_counts.items(), key=lambda x: x[1])
        axis_consistency = 100.0 * best_count / float(total_images)

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

        free_avg_act = free_chosen_act_sum[sc_label] / float(total_images)

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
            act_strength_ratio = 100.0 if labeled_mean < 1e-9 else 100.0 * (free_avg_act / labeled_mean)
        else:
            act_strength_ratio = -1.0

        sum_consistency += axis_consistency
        sum_purity += axis_purity
        sum_act_ratio += act_strength_ratio

    if free_count > 0:
        axis_consistency_avg = sum_consistency / free_count
        axis_purity_avg = sum_purity / free_count
        act_strength_ratio_avg = sum_act_ratio / free_count
    else:
        axis_consistency_avg = axis_purity_avg = act_strength_ratio_avg = 0.0

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
            imgs, lbls = imgs.cuda(non_blocking=True), lbls.cuda(non_blocking=True)
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


def save_checkpoint(state, is_best, prefix, outdir=args.checkpoint_dir):
    os.makedirs(outdir, exist_ok=True)
    ckpt_path = os.path.join(outdir, f"{prefix}_checkpoint.pth")
    torch.save(state, ckpt_path)
    if is_best:
        best_path = os.path.join(outdir, f"{prefix}_best.pth")
        shutil.copyfile(ckpt_path, best_path)
        print(f"[Checkpoint] Best => {best_path}")


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
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(cstate, is_best, args.prefix)

    test_acc = validate(test_loader, model, start_epoch + args.epochs - 1, writer, mode="Test")
    print(f"[Done] Best Val={best_acc:.2f}, Final Test={test_acc:.2f}")
    writer.close()


if __name__ == "__main__":
    main()
