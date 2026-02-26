#!/usr/bin/env python3
"""
CBM Concept Purity Evaluation (QCW-compatible)

Computes per-concept ROC-AUC using concept_dataset only.
No main_dataset required.

Example:
python plot_cbm_purity.py \
    --concept_model model.pth \
    --concept_dir /path/to/concept_dataset \
    --concepts "animal,food" \
    --depth 18 \
    --output_dir results/cbm_purity
"""

import argparse
import os
import json
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from torchvision.datasets import ImageFolder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def discover_subconcepts(concept_dir, supercategories, split="train"):
    """
    Discover all subconcepts under given supercategories.

    Example:
        supercategories = ["animal"]
        returns ["animal/bear", "animal/dog", ...]
    """
    concept_root = os.path.join(concept_dir, f"concept_{split}")

    discovered = []

    for super_cat in supercategories:
        super_path = os.path.join(concept_root, super_cat)

        if not os.path.isdir(super_path):
            logger.warning(f"Supercategory '{super_cat}' not found in {concept_root}")
            continue

        subcats = sorted([
            d for d in os.listdir(super_path)
            if os.path.isdir(os.path.join(super_path, d))
        ])

        for subcat in subcats:
            discovered.append(f"{super_cat}/{subcat}")

    if not discovered:
        logger.warning("No subconcepts discovered!")

    return discovered


# ============================================================
# Model
# ============================================================
def get_resnet(depth, num_outputs):
    import torchvision.models as models
    if depth == 18:
        model = models.resnet18(pretrained=False)
    elif depth == 50:
        model = models.resnet50(pretrained=False)
    else:
        raise ValueError("Unsupported depth")
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_outputs)
    return model


class ConceptPredictor(nn.Module):
    def __init__(self, depth, num_concepts):
        super().__init__()
        self.backbone = get_resnet(depth, num_concepts)

    def forward(self, x):
        return self.backbone(x)


# ============================================================
# Build Concept Dataset (QCW-style)
# ============================================================
def build_concept_dataset(concept_dir,
                          supercategories,
                          split="val",
                          auto_discover=True):
    """
    Build concept dataset with optional autodiscovery of subconcepts.
    """

    if auto_discover:
        concepts_list = discover_subconcepts(
            concept_dir,
            supercategories,
            split="train"  # match training structure
        )
    else:
        concepts_list = supercategories

    concept_root = os.path.join(concept_dir, f"concept_{split}")

    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225])
    ])

    dataset = ImageFolder(concept_root, transform=transform)

    N = len(dataset)
    K = len(concepts_list)
    concept_labels = np.zeros((N, K), dtype=np.float32)

    # Map relative path → index
    path_to_index = {
        os.path.relpath(path, concept_root): idx
        for idx, (path, _) in enumerate(dataset.samples)
    }

    for k, concept_name in enumerate(concepts_list):
        concept_path = os.path.join(concept_root, concept_name)

        if not os.path.isdir(concept_path):
            continue

        for root, _, files in os.walk(concept_path):
            for fname in files:
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    full_path = os.path.join(root, fname)
                    rel_path = os.path.relpath(full_path, concept_root)

                    if rel_path in path_to_index:
                        idx = path_to_index[rel_path]
                        concept_labels[idx, k] = 1.0

    return dataset, concept_labels, concepts_list


# ============================================================
# Compute Purity
# ============================================================
def compute_concept_purity(model, loader, concept_labels, device):
    model.eval()
    all_preds = []

    with torch.no_grad():
        for imgs, _ in tqdm(loader, desc="Computing concept purity"):
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)

    preds = np.concatenate(all_preds, axis=0)

    purity_dict = {}
    aucs = []

    for k in range(preds.shape[1]):
        y_true = concept_labels[:, k]
        y_score = preds[:, k]

        if len(np.unique(y_true)) < 2:
            auc = float("nan")
        else:
            auc = roc_auc_score(y_true, y_score)
            aucs.append(auc)

        purity_dict[k] = float(auc)

    mean_purity = float(np.nanmean(aucs)) if aucs else float("nan")

    return purity_dict, mean_purity


# ============================================================
# CLI
# ============================================================
def get_args():
    p = argparse.ArgumentParser("CBM Concept Purity (QCW-compatible)")
    p.add_argument("--concept_model", required=True)
    p.add_argument("--concept_dir", required=True)
    p.add_argument("--concepts", required=True)
    p.add_argument("--depth", type=int, default=18)
    p.add_argument("--split", default="val")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--output_dir", default="results/cbm_purity")
    return p.parse_args()


# ============================================================
# Main
# ============================================================
def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    supercategories = [x.strip() for x in args.concepts.split(",")]
    dataset, concept_labels, concepts_list = build_concept_dataset(
        args.concept_dir,
        supercategories,
        split=args.split,
        auto_discover=True
    )

    num_concepts = len(concepts_list)

    logger.info(f"Loading model with {num_concepts} concepts")

    model = ConceptPredictor(args.depth, num_concepts).to(device)
    ckpt = torch.load(args.concept_model, map_location=device)
    model.load_state_dict(ckpt["state_dict"])

    dataset, concept_labels, concepts_list = build_concept_dataset(
        args.concept_dir,
        supercategories,
        split=args.split,
        auto_discover=True
    )

    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=False, num_workers=4)

    purity_dict, mean_purity = compute_concept_purity(
        model,
        loader,
        concept_labels,
        device
    )

    os.makedirs(args.output_dir, exist_ok=True)

    result = {}
    for i, name in enumerate(concepts_list):
        result[name] = {"best_axis_auc": purity_dict[i]}

    pd.DataFrame(result).T.to_csv(
        os.path.join(args.output_dir, "result.csv")
    )

    print(f"Mean purity: {mean_purity:.4f}")


if __name__ == "__main__":
    main()