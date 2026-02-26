"""
Train Concept Bottleneck Model (CBM) on Places365 main dataset with COCO concepts.

Pipeline:
  Stage 1: Train X → C (concept predictor)
  Stage 2: Train C → Y (label predictor conditioned on concepts)

This adapts ConceptBottleneck to your dataset structure:
  - Main dataset: Places365 (365 classes)
  - Concept dataset: COCO (supercategories as concepts)
  - Data format: ImageFolder + concept labels

Example usage (with 2 supercategories -> auto-discover all subconcepts):
  python train_places365_coco_cbm.py \\
    --data_dir /path/to/Places365_COCO_QCW/main_dataset \\
    --concept_dir /path/to/Places365_COCO_QCW/concept_dataset \\
    --concepts "animal,food" \\
    --auto_discover_subconcepts \\
    --prefix PLACES365_COCO_CBM_resnet18 \\
    --stage 1
  # This will auto-discover dog, cat, bird, apple, cake, etc. as concepts
    
  python train_places365_coco_cbm.py \\
    --data_dir /path/to/Places365_COCO_QCW/main_dataset \\
    --concept_dir /path/to/Places365_COCO_QCW/concept_dataset \\
    --concepts "animal,food" \\
    --auto_discover_subconcepts \\
    --prefix PLACES365_COCO_CBM_resnet18 \\
    --stage 2 \\
    --concept_model model_checkpoints/PLACES365_COCO_CBM_resnet18_concepts_best.pth

Example usage (without auto-discovery, use supercategories directly):
  python train_places365_coco_cbm.py \\
    --data_dir /path/to/Places365_COCO_QCW/main_dataset \\
    --concept_dir /path/to/Places365_COCO_QCW/concept_dataset \\
    --concepts "animal,food" \\
    --prefix PLACES365_COCO_CBM_resnet18 \\
    --stage 1
"""

import argparse
import os
import time
import random
import json
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from PIL import ImageFile, Image

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Constants
CROP_SIZE = 224
RESIZE_SIZE = 256
PIN_MEMORY = True

# Arguments
parser = argparse.ArgumentParser(description="Train Concept Bottleneck Model on Places365 + COCO")

# Required
parser.add_argument("--data_dir", required=True, help="Main dataset path (train/val/test)")
parser.add_argument("--concept_dir", required=True, help="Concept dataset path (concept_train/concept_val)")
parser.add_argument("--concepts", required=True, help="Comma-separated concept names")
parser.add_argument("--prefix", required=True, help="Model name prefix")

# Stage selection
parser.add_argument("--stage", type=int, default=1, choices=[1, 2],
                   help="1=train concept predictor, 2=train label predictor with concepts")
parser.add_argument("--concept_model", default="",
                   help="Path to trained concept model (required for stage 2)")

# Model hyperparams
parser.add_argument("--depth", type=int, default=18, choices=[18, 50],
                   help="ResNet depth")
parser.add_argument("--pretrained", action="store_true",
                   help="Use ImageNet pretrained weights")

# Training hyperparams
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--momentum", type=float, default=0.9)

# CBM-specific
parser.add_argument("--concept_image_mode", default="crop", 
                   choices=["crop", "redact", "blur", "none"])
parser.add_argument("--auto_discover_subconcepts", action="store_true",
                   help="Auto-discover subconcepts from supercategories")
parser.add_argument("--use_concepts_during_training", action="store_true",
                   help="Whether to use ground-truth concepts during stage 1 training")

# System
parser.add_argument("--seed", type=int, default=348129)
parser.add_argument("--workers", type=int, default=4)
parser.add_argument("--log_dir", default="runs")
parser.add_argument("--checkpoint_dir", default="model_checkpoints")

args = parser.parse_args()

# Setup
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
cudnn.benchmark = True

print("=========== ARGUMENTS ===========")
for k, v in sorted(vars(args).items()):
    print(f"{k}: {v}")
print("=================================")

# Utility function to discover subconcepts
def discover_concepts_from_supercategories(concept_base_dir, supercategories, split="train"):
    """
    Auto-discover all subconcepts from supercategory folders.
    
    Input structure:
      concept_train/
        ├── animal/
        │   ├── dog/
        │   ├── cat/
        │   └── bird/
        └── food/
            ├── apple/
            └── cake/
    
    If supercategories=["animal", "food"], returns:
      ["animal/dog", "animal/cat", "animal/bird", "food/apple", "food/cake"]
    """
    concept_dir = os.path.join(concept_base_dir, f"concept_{split}")
    discovered_concepts = []
    
    for super_cat in supercategories:
        super_path = os.path.join(concept_dir, super_cat)
        if not os.path.isdir(super_path):
            print(f"Warning: Supercategory path not found: {super_path}")
            continue
        
        # Find all direct subdirectories (subcategories)
        try:
            subdirs = sorted([d for d in os.listdir(super_path)
                            if os.path.isdir(os.path.join(super_path, d))])
            for subdir in subdirs:
                discovered_concepts.append(f"{super_cat}/{subdir}")
                print(f"  Discovered: {super_cat}/{subdir}")
        except Exception as e:
            print(f"Error reading {super_path}: {e}")
    
    if not discovered_concepts:
        print("WARNING: No subconcepts discovered! Falling back to supercategories.")
        discovered_concepts = supercategories
    
    return discovered_concepts


# Dataset dimensions
NUM_CLASSES = 365  # Places365

# Load concepts and create mapping
supercategories_list = [x.strip() for x in args.concepts.split(",")]

if args.auto_discover_subconcepts:
    print("[Setup] Auto-discovering subconcepts from supercategories...")
    concepts_list = discover_concepts_from_supercategories(
        args.concept_dir, supercategories_list, split="train"
    )
else:
    concepts_list = supercategories_list

NUM_CONCEPTS = len(concepts_list)

print(f"[Setup] NUM_CLASSES={NUM_CLASSES}, NUM_CONCEPTS={NUM_CONCEPTS}")
print(f"[Setup] Concepts: {concepts_list}")


# ============================================================================#
# Data Loading
# ============================================================================#
class ConceptDatasetForCBM(Dataset):
    """Load concept images with concept labels.
    
    Handles hierarchical structure:
      concept_train/
        ├── animal/
        │   ├── dog/
        │   │   ├── image1.jpg
        │   │   └── ...
        │   └── cat/
        │       └── ...
        └── food/
            └── ...
    
    Images are loaded recursively from all subdirectories under each concept.
    """
    def __init__(self, concept_dir, concepts_list, split="train", transform=None, 
                 crop_mode="crop", bboxes_file=None):
        self.concept_dir = os.path.join(concept_dir, f"concept_{split}")
        self.concepts_list = concepts_list
        self.transform = transform
        self.crop_mode = crop_mode
        
        # Load bboxes if available
        self.bboxes = {}
        if bboxes_file and os.path.exists(bboxes_file):
            with open(bboxes_file, 'r') as f:
                self.bboxes = json.load(f)
        
        # Build samples: (image_path, concept_idx, concept_name)
        # Recursively search for images in subdirectories
        self.samples = []
        for concept_idx, concept_name in enumerate(self.concepts_list):
            concept_path = os.path.join(self.concept_dir, concept_name)
            if not os.path.isdir(concept_path):
                print(f"Warning: Concept path not found: {concept_path}")
                continue
            
            # Recursively find all images in this concept directory
            for root, dirs, files in os.walk(concept_path):
                for img_file in files:
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(root, img_file)
                        self.samples.append((img_path, concept_idx, concept_name))
        
        print(f"[ConceptDatasetForCBM] Loaded {len(self.samples)} concept images for split={split}")


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, concept_idx, concept_name = self.samples[idx]
        
        img = Image.open(img_path).convert("RGB")
        
        # Apply cropping if bboxes available and crop_mode enabled
        if self.crop_mode == "crop" and img_path in self.bboxes:
            bbox = self.bboxes[img_path]
            x1, y1, x2, y2 = bbox
            x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), int(x2), int(y2)
            img = img.crop((x1, y1, x2, y2))
        
        if self.transform:
            img = self.transform(img)
        
        # Return class index directly (single-label classification)
        return img, concept_idx


class MainDatasetForCBM(Dataset):
    """Load main dataset (Places365) without concept labels - for stage 1 concept prediction."""
    def __init__(self, data_dir, split="train", transform=None):
        self.data_source = datasets.ImageFolder(os.path.join(data_dir, split), transform=transform)
        self.split = split
    
    def __len__(self):
        return len(self.data_source)
    
    def __getitem__(self, idx):
        img, label = self.data_source[idx]
        return img, label


class MainWithConceptsDataset(Dataset):
    """
    Load main dataset + concept predictions for stage 2.
    Pairs main images with predicted/pseudo concepts.
    """
    def __init__(self, data_dir, split="train", transform=None, concept_model=None):
        self.main_dataset = MainDatasetForCBM(data_dir, split, transform)
        self.concept_model = concept_model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __len__(self):
        return len(self.main_dataset)
    
    def __getitem__(self, idx):
        img, label = self.main_dataset[idx]
        
        # Get concept prediction if model provided
        if self.concept_model is not None:
            with torch.no_grad():
                img_batch = img.unsqueeze(0).to(self.device) if isinstance(img, torch.Tensor) else img
                concept_pred = self.concept_model(img_batch)
                concept_pred = concept_pred.squeeze(0).cpu()
        else:
            concept_pred = None
        
        return img, label, concept_pred


# ============================================================================#
# Model Architecture
# ============================================================================#
def get_resnet(depth=18, num_classes=1000, pretrained=False):
    """Get ResNet backbone."""
    import torchvision.models as models
    if depth == 18:
        model = models.resnet18(pretrained=pretrained)
    elif depth == 50:
        model = models.resnet50(pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported depth: {depth}")
    
    # Replace final layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


class ConceptPredictor(nn.Module):
    """X → C: Predict concepts from images (single-label classification)."""
    def __init__(self, depth=18, num_concepts=10, pretrained=False):
        super().__init__()
        self.backbone = get_resnet(depth, num_concepts, pretrained=pretrained)
    
    def forward(self, x):
        # Returns class logits (no activation, loss handles it)
        return self.backbone(x)


class LabelPredictorWithConcepts(nn.Module):
    """C → Y: Predict labels from concepts."""
    def __init__(self, num_concepts=10, num_classes=365, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(num_concepts, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, concepts):
        """
        Args:
            concepts: [B, num_concepts] predictions or binary labels
        Returns:
            logits: [B, num_classes]
        """
        x = self.fc1(concepts)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# ============================================================================#
# Utilities
# ============================================================================#
class AverageMeter:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0


def accuracy_topk(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        c_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append((c_k * 100.0 / batch_size).item())
    return tuple(res)


def save_checkpoint(state, is_best, prefix, outdir):
    os.makedirs(outdir, exist_ok=True)
    ckpt_path = os.path.join(outdir, f"{prefix}_checkpoint.pth")
    torch.save(state, ckpt_path)
    if is_best:
        best_path = os.path.join(outdir, f"{prefix}_best.pth")
        import shutil
        shutil.copyfile(ckpt_path, best_path)
        print(f"[Checkpoint] Best model → {best_path}")


# ============================================================================#
# Stage 1: Train Concept Predictor (X → C)
# ============================================================================#
def train_stage1():
    print("\n" + "="*60)
    print("STAGE 1: Training Concept Predictor (X → C)")
    print("="*60)
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(CROP_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(RESIZE_SIZE),
        transforms.CenterCrop(CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Data loaders
    concept_data_train = ConceptDatasetForCBM(
        args.concept_dir, concepts_list, split="train", 
        transform=train_transform, crop_mode=args.concept_image_mode,
        bboxes_file=os.path.join(args.concept_dir, "bboxes.json")
    )
    concept_data_val = ConceptDatasetForCBM(
        args.concept_dir, concepts_list, split="val", 
        transform=val_transform, crop_mode=args.concept_image_mode,
        bboxes_file=os.path.join(args.concept_dir, "bboxes.json")
    )
    
    train_loader = DataLoader(concept_data_train, batch_size=args.batch_size, 
                             shuffle=True, num_workers=args.workers, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(concept_data_val, batch_size=args.batch_size, 
                           shuffle=False, num_workers=args.workers, pin_memory=PIN_MEMORY)
    
    print(f"[Data] Train: {len(train_loader.dataset)} images")
    print(f"[Data] Val: {len(val_loader.dataset)} images")
    
    # Model
    model = ConceptPredictor(args.depth, NUM_CONCEPTS, pretrained=args.pretrained).cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, 
                         weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    criterion = nn.CrossEntropyLoss()  # Single-label classification
    
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, f"{args.prefix}_concepts_{int(time.time())}"))
    
    best_val_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = AverageMeter()
        
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"[S1-Train] Epoch {epoch+1}")
        for i, (imgs, concept_idxs) in pbar:
            imgs = imgs.cuda()
            concept_idxs = concept_idxs.cuda().long()
            
            outputs = model(imgs)
            loss = criterion(outputs, concept_idxs)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss.update(loss.item(), imgs.size(0))
            pbar.set_postfix({"Loss": f"{train_loss.avg:.4f}"})
            
            writer.add_scalar("S1/Train/Loss", train_loss.val, epoch * len(train_loader) + i)
        
        # Validate
        model.eval()
        val_loss = AverageMeter()
        concept_acc = AverageMeter()
        
        with torch.no_grad():
            for imgs, concept_idxs in val_loader:
                imgs = imgs.cuda()
                concept_idxs = concept_idxs.cuda().long()
                
                outputs = model(imgs)
                loss = criterion(outputs, concept_idxs)
                
                # Concept accuracy (top-1)
                _, pred = outputs.topk(1, dim=1, largest=True)
                acc = pred.eq(concept_idxs.view(-1, 1)).float().mean()
                
                val_loss.update(loss.item(), imgs.size(0))
                concept_acc.update(acc.item(), imgs.size(0))
        
        scheduler.step()
        
        print(f"[S1] Epoch {epoch+1}: Train Loss={train_loss.avg:.4f}, Val Loss={val_loss.avg:.4f}, Concept Acc={concept_acc.avg*100:.2f}%")
        
        writer.add_scalar("S1/Val/Loss", val_loss.avg, epoch)
        writer.add_scalar("S1/Val/ConceptAcc", concept_acc.avg * 100, epoch)
        
        is_best = val_loss.avg < best_val_loss
        if is_best:
            best_val_loss = val_loss.avg
            best_epoch = epoch
        
        save_checkpoint({
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_loss": best_val_loss
        }, is_best, f"{args.prefix}_concepts", args.checkpoint_dir)
    
    writer.close()
    print(f"\n[S1] Training complete. Best epoch: {best_epoch+1}, Best val loss: {best_val_loss:.4f}")


# ============================================================================#
# Stage 2: Train Label Predictor (C → Y)
# ============================================================================#
def train_stage2():
    print("\n" + "="*60)
    print("STAGE 2: Training Label Predictor (C → Y)")
    print("="*60)
    
    if not args.concept_model or not os.path.exists(args.concept_model):
        print(f"ERROR: Concept model not found at {args.concept_model}")
        print("Please provide --concept_model with path to trained concept predictor")
        exit(1)
    
    # Load trained concept model
    print(f"[S2] Loading concept model from {args.concept_model}")
    concept_model = ConceptPredictor(args.depth, NUM_CONCEPTS, 
                                    pretrained=args.pretrained).cuda()
    ckpt = torch.load(args.concept_model, map_location="cuda", weights_only=False)
    concept_model.load_state_dict(ckpt["state_dict"])
    concept_model.eval()
    for param in concept_model.parameters():
        param.requires_grad = False
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(CROP_SIZE, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(RESIZE_SIZE),
        transforms.CenterCrop(CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Pre-compute concept predictions for main dataset
    print("[S2] Pre-computing concept predictions for main dataset...")
    
    def get_concept_predictions(data_dir, split, transform):
        """Return (image_paths, labels, concept_preds)."""
        main_loader = DataLoader(
            MainDatasetForCBM(data_dir, split, transform),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=PIN_MEMORY
        )
        
        all_concept_preds = []
        all_labels = []
        all_paths = []
        
        with torch.no_grad():
            for imgs, labels in tqdm(main_loader, desc=f"Computing concepts for {split}"):
                imgs = imgs.cuda()
                concept_logits = concept_model(imgs)
                # For single-label classification, use softmax to get concept probabilities
                concept_preds = torch.softmax(concept_logits, dim=1).cpu()
                
                all_concept_preds.append(concept_preds)
                all_labels.append(labels)
        
        return torch.cat(all_concept_preds), torch.cat(all_labels)
    
    train_concepts, train_labels = get_concept_predictions(args.data_dir, "train", train_transform)
    val_concepts, val_labels = get_concept_predictions(args.data_dir, "val", val_transform)
    test_concepts, test_labels = get_concept_predictions(args.data_dir, "test", val_transform)
    
    print(f"[S2-Data] Train: {train_concepts.shape}, Val: {val_concepts.shape}, Test: {test_concepts.shape}")
    
    # Simple tensor dataset for stage 2
    class TensorDataset:
        def __init__(self, concepts, labels):
            self.concepts = concepts.float()
            self.labels = labels.long()
        
        def __len__(self):
            return len(self.labels)
        
        def __getitem__(self, idx):
            return self.concepts[idx], self.labels[idx]
    
    train_data = TensorDataset(train_concepts, train_labels)
    val_data = TensorDataset(val_concepts, val_labels)
    test_data = TensorDataset(test_concepts, test_labels)
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                             num_workers=0, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
                           num_workers=0, pin_memory=PIN_MEMORY)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=PIN_MEMORY)
    
    # Model
    label_model = LabelPredictorWithConcepts(NUM_CONCEPTS, NUM_CLASSES).cuda()
    optimizer = optim.SGD(label_model.parameters(), lr=args.lr, momentum=args.momentum,
                         weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, f"{args.prefix}_labels_{int(time.time())}"))
    
    best_val_acc = 0.0
    best_epoch = 0
    
    for epoch in range(args.epochs):
        # Train
        label_model.train()
        train_loss = AverageMeter()
        train_top1 = AverageMeter()
        
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"[S2-Train] Epoch {epoch+1}")
        for i, (concepts, labels) in pbar:
            concepts = concepts.cuda()
            labels = labels.cuda()
            
            outputs = label_model(concepts)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            prec1, _ = accuracy_topk(outputs, labels, (1, 5))
            train_loss.update(loss.item(), concepts.size(0))
            train_top1.update(prec1, concepts.size(0))
            
            pbar.set_postfix({"Loss": f"{train_loss.avg:.4f}", "Top1": f"{train_top1.avg:.2f}"})
            writer.add_scalar("S2/Train/Loss", train_loss.val, epoch * len(train_loader) + i)
        
        # Validate
        label_model.eval()
        val_loss = AverageMeter()
        val_top1 = AverageMeter()
        
        with torch.no_grad():
            for concepts, labels in val_loader:
                concepts = concepts.cuda()
                labels = labels.cuda()
                
                outputs = label_model(concepts)
                loss = criterion(outputs, labels)
                
                prec1, _ = accuracy_topk(outputs, labels, (1, 5))
                val_loss.update(loss.item(), concepts.size(0))
                val_top1.update(prec1, concepts.size(0))
        
        scheduler.step()
        
        print(f"[S2] Epoch {epoch+1}: Train Loss={train_loss.avg:.4f}, Val Top1={val_top1.avg:.2f}%")
        
        writer.add_scalar("S2/Val/Loss", val_loss.avg, epoch)
        writer.add_scalar("S2/Val/Top1", val_top1.avg, epoch)
        
        is_best = val_top1.avg > best_val_acc
        if is_best:
            best_val_acc = val_top1.avg
            best_epoch = epoch
        
        save_checkpoint({
            "epoch": epoch + 1,
            "state_dict": label_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_acc": best_val_acc
        }, is_best, f"{args.prefix}_labels", args.checkpoint_dir)
    
    # Test
    with torch.no_grad():
        test_top1 = AverageMeter()
        for concepts, labels in test_loader:
            concepts = concepts.cuda()
            labels = labels.cuda()
            outputs = label_model(concepts)
            prec1, _ = accuracy_topk(outputs, labels, (1, 5))
            test_top1.update(prec1, concepts.size(0))
    
    writer.close()
    print(f"\n[S2] Training complete.")
    print(f"     Best val epoch: {best_epoch+1}, Best val acc: {best_val_acc:.2f}%")
    print(f"     Test accuracy: {test_top1.avg:.2f}%")


# ============================================================================#
# Main
# ============================================================================#
if __name__ == "__main__":
    if args.stage == 1:
        train_stage1()
    elif args.stage == 2:
        train_stage2()
