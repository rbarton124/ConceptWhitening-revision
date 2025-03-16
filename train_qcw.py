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

ImageFile.LOAD_TRUNCATED_IMAGES = True # ensure truncated images are loadable JIC

from MODELS.model_resnet_qcw import build_resnet_qcw, NUM_CLASSES, get_last_qcw_layer
from MODELS.ConceptDataset_QCW import ConceptDataset

########################
# Global Constants
########################
LR_DECAY_EPOCH  = 30
LR_DECAY_FACTOR = 0.1
CROP_SIZE       = 224
RESIZE_SIZE     = 256
CW_ALIGN_FREQ   = 20   # how often (in mini-batches) we do concept alignment
PIN_MEMORY      = True
ALIGNMENT_BATCHES_PER_STEP = 2

########################
# Argument Parser
########################
parser = argparse.ArgumentParser(description="Train Quantized Concept Whitening (QCW) - Revised")
parser.add_argument("--data_dir", required=True, help="Path to main dataset containing train/val/test subfolders (ImageFolder structure).")
parser.add_argument("--concept_dir", required=True, help="Path to concept dataset with concept_train/, concept_val/ (optional), and bboxes.json.")
parser.add_argument("--bboxes", default="", help="Path to bboxes.json if not in concept_dir/bboxes.json")
parser.add_argument("--concepts", required=True, help="Comma-separated list of high-level concepts to use (e.g. 'wing,beak,general').")
parser.add_argument("--prefix", required=True, help="Prefix for logging & checkpoint saving")
parser.add_argument("--whitened_layers", default="5", help="Comma-separated BN layer indices to replace with QCW (e.g. '5' or '2,5')")
parser.add_argument("--depth", type=int, default=18, help="ResNet depth (18 or 50).")
parser.add_argument("--act_mode", default="pool_max", help="Activation mode for QCW: 'mean','max','pos_mean','pool_max'")
# Training hyperparams
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
parser.add_argument("--batch_size", type=int, default=64, help="Mini-batch size.")
parser.add_argument("--lr", type=float, default=0.1, help="Initial learning rate.")
parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD.")
parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay (L2 reg).")
# Checkpoint
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint to resume from.")
parser.add_argument("--only_load_weights", action="store_true", help="If set, only load model weights from checkpoint (ignore epoch/optimizer).")

parser.add_argument("--seed", type=int, default=1234, help="Random seed.")
parser.add_argument("--workers", type=int, default=4, help="Number of data loading workers.")
# Feature toggles
parser.add_argument("--disable_subspaces", action="store_true", help="Disable subspace partitioning => one axis per concept.")
parser.add_argument("--use_free", action="store_true", help="Enable free unlabeled concept axes if the QCW layer supports it.")
# Still need to add this logic in!
parser.add_argument("--cw_loss_weight", type=float, default=1.0, help="Weight for QCW loss in alignment, if integrated alignment code.")

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
    Build concept loaders for concept_train:
    1. A main loader with all subconcepts (for general use)
    2. Individual loaders for each subconcept (for proper alignment)
    
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

    crop_mode = "crop" # redaction mode for dataset, clean this up later

    # Main concept dataset with all subconcepts
    concept_dataset = ConceptDataset(
        root_dir=concept_root,
        bboxes_file=args.bboxes,
        high_level_filter=hl_list,
        transform=concept_transform,
        crop_mode=crop_mode
    )
    
    # Main loader with all subconcepts (shuffled)
    main_concept_loader = DataLoader(
        concept_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=PIN_MEMORY
    )
    
    # Now create per-subconcept loaders to ensure each subconcept is represented
    # during concept alignment (similar to old QCW implementation)
    subconcept_loaders = []
    
    # Group samples by subconcept
    subconcept_samples = {}
    for idx, (_, _, hl_name, sc_name) in enumerate(concept_dataset.samples):
        # Get the subconcept label from the subconcept name
        sc_label = concept_dataset.sc2idx[sc_name]
        
        if sc_label not in subconcept_samples:
            subconcept_samples[sc_label] = []
        subconcept_samples[sc_label].append(idx)
    
    # Create a filtered dataset for each subconcept
    for sc_label, indices in subconcept_samples.items():
        # Use PyTorch's Subset to create a filtered dataset
        sc_dataset = torch.utils.data.Subset(concept_dataset, indices)
        
        # Create a loader for this subconcept
        sc_loader = DataLoader(
            sc_dataset,
            batch_size=min(args.batch_size, len(indices)),  # Handle small subconcepts
            shuffle=True,
            num_workers=args.workers,
            pin_memory=PIN_MEMORY
        )
        subconcept_loaders.append((sc_label, sc_loader))
    
    print(f"Created {len(subconcept_loaders)} subconcept-specific loaders")
    
    # Return both the main loader and subconcept-specific loaders
    return [main_concept_loader] + [loader for _, loader in subconcept_loaders], concept_dataset

concept_loaders, concept_ds = build_concept_loaders(args)

print(f"[Data] #Main Train: {len(train_loader.dataset)}")
print(f"[Data] #Val:        {len(val_loader.dataset)}")
print(f"[Data] #Test:       {len(test_loader.dataset)}")
print(f"[Data] #Concept:    {len(concept_loaders[0].dataset)}")
print(f"[Data] #Subconcept Loaders: {len(concept_loaders) - 1}")

# Print detailed information about the concept dataset
print("\n===== CONCEPT DATASET DETAILS =====")
print(f"Total number of samples: {len(concept_ds.samples)}")
print(f"Total number of subconcepts: {concept_ds.get_num_subconcepts()}")
print(f"Total number of high-level concepts: {concept_ds.get_num_high_level()}")

# Print subconcept names and their indices
print("\nSubconcept names and indices:")
for sc_name in concept_ds.get_subconcept_names():
    sc_idx = concept_ds.sc2idx[sc_name]
    print(f"  [{sc_idx}] {sc_name}")

# Print high-level concept names and their subspaces
print("\nHigh-level concepts and their subconcept indices:")
for hl_name in concept_ds.get_hl_names():
    subspace = concept_ds.subspace_mapping.get(hl_name, [])
    print(f"  {hl_name}: {subspace}")

# Print information about each subconcept-specific loader
print("\nSubconcept-specific loaders:")
for i, loader in enumerate(concept_loaders[1:], 1):
    print(f"  Loader {i}: {len(loader.dataset)} samples")

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
    """
    Attempt to resume from a checkpoint (either a standard ResNet-18
    or our custom QCW checkpoint). We'll rename layers if needed,
    skip any missing or extra keys, and optionally load optimizer state.
    Returns: (start_epoch, best_prec)
    """
    start_epoch, best_prec = 0, 0.0
    if not args.resume or not os.path.isfile(args.resume):
        return start_epoch, best_prec

    print(f"[Checkpoint] Resuming from {args.resume}")
    ckpt = torch.load(args.resume, map_location="cpu")

    # If the checkpoint is from our QCW code, we typically have a dictionary. If not, it might be a raw state_dict (standard torchvision model)
    raw_sd = ckpt.get("state_dict", ckpt)  # either the 'state_dict' sub-dict or the entire ckpt

    # 3) We'll build a new dictionary that maps the checkpoint keys to our model's keys
    #    Also if the checkpoint has "module.xxx", remove the "module." prefix.
    #    We'll skip keys that obviously belong to BN layers if we've replaced them with IterNorm
    #    or we skip keys that reference "running_rot" or "sum_G" if the model doesn't have them, etc.

    model_sd = model.state_dict()  # the current model’s parameter dict
    renamed_sd = {}

    def rename_key(old_key):
        # Remove any leading "module." if it exists
        if old_key.startswith("module."):
            old_key = old_key[len("module."):]
        # If old_key doesn't start with "backbone." and the model expects "backbone.*",
        # we can prepend "backbone." for main layers.  But only if that matches model keys
        if not old_key.startswith("backbone.") and ("backbone."+old_key in model_sd):
            return "backbone."+old_key
        # Similarly, if we have "fc.weight" in standard resnet, we want "backbone.fc.weight" in QCW
        # However, for resnet-18 standard checkpoint, "fc.weight" is the final linear layer
        if old_key.startswith("fc.") and ("backbone.fc"+old_key[2:] in model_sd):
            return "backbone."+old_key
        # We rely on partial load + strict=False logic to skip mismatched key.
        return old_key  # fallback if no special rename
    
    matched_keys = []
    skipped_keys = []
    
    for ckpt_k, ckpt_v in raw_sd.items():
        new_k = rename_key(ckpt_k)
        # if new_k is in the model's dict and shape matches, we keep it
        if new_k in model_sd:
            if ckpt_v.shape == model_sd[new_k].shape:
                # good shape => we keep it
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

    print("[Checkpoint] Matched keys:")
    for mk in matched_keys:
        print("   ", mk)
    print("[Checkpoint] Skipped keys from checkpoint:")
    for sk in skipped_keys:
        print("   ", sk)

    # possibly recover epoch/best_prec/optimizer if it’s a QCW style checkpoint
    if isinstance(ckpt, dict):
        if not args.only_load_weights:
            start_epoch = ckpt.get("epoch", 0)
            best_prec = ckpt.get("best_prec1", 0.0)
            if "optimizer" in ckpt:
                opt_sd = ckpt["optimizer"]
                try:
                    optimizer.load_state_dict(opt_sd)
                    # Ensure param states are on correct device
                    for param in optimizer.state.values():
                        if isinstance(param, dict) and "momentum_buffer" in param:
                            buf = param["momentum_buffer"]
                            if buf is not None and buf.device != torch.device("cuda"):
                                param["momentum_buffer"] = buf.cuda()
                except Exception as e:
                    print(f"[Warning] Could not load optimizer state: {e}")

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

        # Concept alignment every N batches using subconcept-specific loaders
        if (i + 1) % CW_ALIGN_FREQ == 0 and len(concept_loaders) > 1:  # Ensure we have subconcept loaders
            # Skip the first loader (which is the main loader with all concepts)
            # and use subconcept-specific loaders (starting from index 1)
            alignment_metrics = run_concept_alignment_all(model, concept_loaders[1:], concept_ds.subspace_mapping)
            
            # Update metrics
            global_top1 = alignment_metrics['global_top1']
            subspace_top1 = alignment_metrics['subspace_top1']
            global_top5 = alignment_metrics['global_top5']
            
            # Use the subspace_top1 as the main alignment score for backward compatibility
            alignment_score.update(subspace_top1)
            
            # Log all metrics
            writer.add_scalar("CW/Alignment/GlobalTop1", global_top1, iteration)
            writer.add_scalar("CW/Alignment/SubspaceTop1", subspace_top1, iteration)
            writer.add_scalar("CW/Alignment/GlobalTop5", global_top5, iteration)

        # Update the pbar with the new alignment score
        pbar.set_postfix({"Loss": f"{losses.avg:.3f}", "Top1": f"{top1.avg:.2f}", 
                          "Top5": f"{top5.avg:.2f}", "ConceptAlignment": f"{alignment_score.avg:.2f}"})

        writer.add_scalar("Train/Loss", losses.val, iteration)
        writer.add_scalar("Train/Top1", top1.val, iteration)
        writer.add_scalar("Train/Top5", top5.val, iteration)

def run_concept_alignment_all(model, subconcept_loaders, subspace_mapping):
    """
    Run alignment checks for all subconcept loaders and calculate multiple metrics:
    
    Args:
        model: The QCW model
        subconcept_loaders: List of subconcept-specific dataloaders
        subspace_mapping: Dict mapping high-level concepts to lists of subconcept indices
        
    Returns:
        Dict with alignment metrics
    """
    model.eval()
    # Create inverted mapping from subconcept index to high-level concept
    sc_to_hl = {}
    for hl, sc_indices in subspace_mapping.items():
        for sc_idx in sc_indices:
            sc_to_hl[sc_idx] = hl
    
    from types import SimpleNamespace
    hook_storage = SimpleNamespace(outputs=[])

    def forward_hook(module, input, output):
        hook_storage.outputs.append(output)

    last_qcw = get_last_qcw_layer(model.module)
    handle = last_qcw.register_forward_hook(forward_hook)
    
    # Initialize counters for our metrics
    global_top1_correct = 0
    subspace_top1_correct = 0
    global_top5_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        # For each subconcept-specific loader
        for subconcept_loader in subconcept_loaders:
            try:
                # Get a batch from this loader with error handling
                batch_iter = iter(subconcept_loader)
                batch_data = next(batch_iter, None)
                
                if batch_data is None or len(batch_data) < 2:
                    print(f"Warning: Empty or invalid batch from subconcept loader")
                    continue
                    
                imgs, sc_labels = batch_data
                
                if not len(imgs):
                    continue
                    
                # All images in this loader have the same subconcept
                sc_label = sc_labels[0].item()
                
                # Get the high-level concept this subconcept belongs to
                hl = sc_to_hl.get(sc_label, None)
                if hl is None:
                    # Skip if we don't have subspace mapping for this subconcept
                    print(f"Warning: No high-level concept found for subconcept {sc_label}")
                    continue
            except Exception as e:
                print(f"Error processing subconcept loader: {str(e)}")
                continue
            
            # Get the subspace (all subconcepts under this high-level concept)
            subspace = subspace_mapping.get(hl, [])
            
            # Set mode to this subconcept's index for alignment
            model.module.change_mode(sc_label)
            
            # Process images
            imgs = imgs.cuda()
            
            # Process each image
            for i in range(len(imgs)):
                # Set the specific subconcept index
                last_qcw.set_subconcept(sc_label)
                
                # Forward pass
                out = model(imgs[i:i+1])
                
                if len(hook_storage.outputs) == 0:
                    continue
                
                # Extract activations
                featmap = hook_storage.outputs[0]  # shape [1, C, H, W]
                feat_avg = featmap.mean(dim=(2,3)).squeeze()  # shape [C]
                
                # Global Top-1: Is the highest activated axis the correct subconcept?
                global_pred = feat_avg.argmax().item()
                if global_pred == sc_label:
                    global_top1_correct += 1
                
                # Global Top-5: Is the correct subconcept among the top 5 activated axes?
                top5_values, top5_indices = feat_avg.topk(5)
                if sc_label in top5_indices:
                    global_top5_correct += 1
                
                # Subspace Top-1: Within the subspace, is the highest activated axis the correct subconcept?
                if len(subspace) > 0:
                    # Extract activations only for axes in this subspace
                    subspace_activations = feat_avg[subspace]
                    subspace_pred_local = subspace_activations.argmax().item()
                    # Convert local index to global axis index
                    predicted_global_axis = subspace[subspace_pred_local]
                    if predicted_global_axis == sc_label:
                        subspace_top1_correct += 1
                
                total_samples += 1
                hook_storage.outputs.clear()
    
    # Cleanup
    handle.remove()
    model.module.update_rotation_matrix()
    model.module.change_mode(-1)
    model.train()
    
    # Calculate percentages
    if total_samples > 0:
        global_top1_pct = 100.0 * global_top1_correct / total_samples
        subspace_top1_pct = 100.0 * subspace_top1_correct / total_samples
        global_top5_pct = 100.0 * global_top5_correct / total_samples
    else:
        global_top1_pct = 0.0
        subspace_top1_pct = 0.0
        global_top5_pct = 0.0
    
    return {
        'global_top1': global_top1_pct,
        'subspace_top1': subspace_top1_pct,
        'global_top5': global_top5_pct
    }

def run_concept_alignment(model, concept_loader):
    """
    Legacy alignment function for single-loader compatibility.
    Uses Global Top-1 accuracy (correctly predicting a subconcept as the highest activated axis).
    
    This function is maintained for backward compatibility, but the run_concept_alignment_all
    function should be preferred for more comprehensive alignment metrics.
    """
    model.eval()
    from types import SimpleNamespace
    hook_storage = SimpleNamespace(outputs=[])

    def forward_hook(module, input, output):
        hook_storage.outputs.append(output)

    last_qcw = get_last_qcw_layer(model.module)
    handle = last_qcw.register_forward_hook(forward_hook)

    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, sc_labels in concept_loader:
            # Dataset now returns sc_label (subconcept index)
            imgs = imgs.cuda()
            sc_labels = sc_labels.cuda()

            # Set subconcept_idx for precise alignment - loop through batch for proper alignment
            for i in range(len(sc_labels)):
                # Get the subconcept index for this image
                sc_label = sc_labels[i].item()
                
                # Set the mode to this subconcept
                model.module.change_mode(sc_label)
                last_qcw.set_subconcept(sc_label)
                
                # Process a single image
                out = model(imgs[i:i+1])
                
                if len(hook_storage.outputs) == 0:
                    continue

                featmap = hook_storage.outputs[0]  # shape [1, C, H, W]
                feat_avg = featmap.mean(dim=(2,3))  # shape [1, C]
                pred_axis = feat_avg.argmax(dim=1)[0]  # get the predicted axis
                
                # Compare with subconcept label for correct axis alignment
                correct += (pred_axis == sc_label).item()
                total += 1
                
                hook_storage.outputs.clear()

    handle.remove()
    model.module.update_rotation_matrix()
    model.module.change_mode(-1)
    model.train()
    
    # Return the accuracy of subconcept alignment
    return (correct / max(1, total)) * 100.0 if total > 0 else 0.0

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