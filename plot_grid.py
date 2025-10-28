#!/usr/bin/env python3
from __future__ import annotations
"""
qcw_topk_grids.py

Generate grids where columns = whitened layers (WL) and rows = concepts.
Each cell is the top-1 (rank_1.jpg) image for that (concept, layer).

Usage example:
  python qcw_topk_grids.py --root /path/to/experiments --out_dir ./grids --cell 128
"""
import argparse
import os
import re
import random
from typing import Dict, List, Sequence, Tuple
import matplotlib.pyplot as plt
from PIL import Image

# Pattern for QCW folders (matches QCW only)
QCW_FOLDER_PAT = re.compile(
    r"^RESNET(?P<depth>\d+)_Places365_QCW_COCO_\{(?P<dataset>.+?)\}_layer_(?P<layer>\d+)_checkpoint$"
)

# Target groups (depth, size_label)
TARGET_GROUPS = [
    (18, "small"),  # {animal,food}
    (18, "large"),  # {animal,food,sports,vehicle}
    (50, "large"),
]


# ----------------------------- CLI --------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("QCW per-layer top-k grids (columns=WL, rows=concepts)")
    p.add_argument("--root", required=True, help="Root directory with experiment folders.")
    p.add_argument("--layers", default="", help="Comma-separated WL numbers to include (e.g. 1,5,8).")
    p.add_argument(
        "--concepts",
        default="",
        help="Comma-separated exact concept folder names to include (if empty, concepts present across all layers are used)."
    )
    p.add_argument("--k", type=int, default=5, help="Top-k images available in folders (used only for checking).")
    p.add_argument("--cell", type=int, default=128, help="Thumbnail size (px) for each cell.")
    p.add_argument("--max_rows", type=int, default=0, help="Cap on number of concept rows (0 = no cap).")
    p.add_argument("--shuffle_rows", action="store_true", help="Shuffle row order.")
    p.add_argument("--out_dir", default="layer_topk_grids", help="Output directory for generated figures.")
    return p.parse_args()


# ----------------------------- Utilities --------------------------------- #
def parse_dataset_items(dataset_str: str) -> List[str]:
    """Accept 'animal,food' or 'animal-food' and return list ['animal','food']."""
    s = dataset_str.replace("-", ",")
    items = [t.strip() for t in s.split(",") if t.strip()]
    return items


def dataset_size_label_from_items(items: List[str]) -> str:
    """Return 'small' if <=2 items else 'large'."""
    return "small" if len(items) <= 2 else "large"


def collect_qcw_folders(root: str) -> Dict[Tuple[int, str], List[Tuple[int, str]]]:
    """
    Scan `root` for QCW folders and return mapping:
      (depth, dataset_size) -> list of (layer, folder_name)
    folder_name is the directory name (not full path).
    """
    mapping: Dict[Tuple[int, str], List[Tuple[int, str]]] = {}
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Root not found: {root}")

    for fname in os.listdir(root):
        m = QCW_FOLDER_PAT.match(fname)
        if not m:
            continue
        depth = int(m.group("depth"))
        dataset_str = m.group("dataset")
        layer = int(m.group("layer"))

        items = parse_dataset_items(dataset_str)
        size_label = dataset_size_label_from_items(items)

        key = (depth, size_label)
        mapping.setdefault(key, []).append((layer, fname))

    # Sort layers in each list
    for k in mapping:
        mapping[k] = sorted(mapping[k], key=lambda x: x[0])
    return mapping


# ----------------------------- Image helpers ----------------------------- #
def load_square_no_border(path: str, size: int) -> Image.Image:
    """Load image, center-crop to square, resize. Return placeholder if missing/corrupt.
       No border is added so images touch when tiled."""
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return Image.new("RGB", (size, size), (240, 240, 240))
    w, h = img.size
    m = min(w, h)
    left = (w - m) // 2
    top = (h - m) // 2
    img = img.crop((left, top, left + m, top + m))
    img = img.resize((size, size), Image.Resampling.LANCZOS)
    return img


def clean_label(concept_folder_name: str) -> str:
    """Humanize concept folder names for display."""
    name = concept_folder_name
    free_tag = ""
    if name.endswith("_free") or name.endswith("-free"):
        free_tag = " (free)"
        name = name.rsplit("_free", 1)[0] if name.endswith("_free") else name.rsplit("-free", 1)[0]
    if name.startswith("has_"):
        name = name[4:]
    name = name.replace("::", ": ").replace("_-_", " - ").replace("_", " ")
    name = re.sub(r"\s{2,}", " ", name).strip()
    return f"{name}{free_tag}"


# ----------------------------- Grid builder ------------------------------ #
def list_concepts(topk_dir: str) -> List[str]:
    if not os.path.isdir(topk_dir):
        return []
    return sorted([d for d in os.listdir(topk_dir) if os.path.isdir(os.path.join(topk_dir, d))],
                  key=lambda s: s.lower())


def build_grid_columns_as_layers(layer_names: Sequence[str],
                                 topk_dirs: Sequence[str],
                                 concepts: Sequence[str],
                                 cell: int,
                                 out_path: str,
                                 show_row_labels: bool = True) -> str:
    """
    Build a grid where columns=layers and rows=concepts.
    - layer_names: list of strings shown as column titles (e.g. 'WL1','WL2' or 'ResNet18_WL1')
    - topk_dirs: list of directories, each containing topk_concept_images/<concept>/rank_1.jpg
    - concepts: list of concept folder names (rows)
    - cell: pixel size per image cell
    - out_path: path to save the PNG
    - show_row_labels: whether to draw concept labels left of each row
    """
    n_cols = len(layer_names)
    n_rows = len(concepts)
    if n_cols == 0 or n_rows == 0:
        raise ValueError("No columns or rows for grid building.")

    # DPI & figure size so that images are pixel-accurate
    dpi = 150
    fig_w = (n_cols * cell) / dpi
    label_col_px = int(cell * 1.4) if show_row_labels else 0
    fig_h = (n_rows * cell) / dpi

    fig = plt.figure(figsize=(fig_w + (label_col_px / dpi), fig_h), dpi=dpi)
    # Use absolute axes for each cell to avoid spacing
    # We'll draw image axes in a tight grid; compute normalized width/height per cell
    total_w_px = n_cols * cell + label_col_px
    total_h_px = n_rows * cell

    def norm_rect(x_px, y_px, w_px, h_px):
        # matplotlib 0..1 coords
        return (x_px / total_w_px, 1 - (y_px + h_px) / total_h_px, w_px / total_w_px, h_px / total_h_px)

    # Column headers (centered above each column)
    for c, lname in enumerate(layer_names):
        # x pixel start
        x_px = label_col_px + c * cell
        # small header axis above top images
        hdr_h = int(cell * 0.08)
        rect = norm_rect(x_px, -hdr_h, cell, hdr_h)
        axh = fig.add_axes(rect)
        axh.text(0.5, 0.5, lname, ha="center", va="center", fontsize=9)
        axh.axis("off")

    # Row labels (left column) if requested
    if show_row_labels:
        for r, concept in enumerate(concepts):
            y_px = r * cell
            # place text centered vertically in the row label column
            rect = norm_rect(0, y_px, label_col_px, cell)
            axlab = fig.add_axes(rect)
            axlab.text(0.95, 0.5, clean_label(concept), ha="right", va="center", fontsize=8)
            axlab.axis("off")

    # Draw images
    for r, concept in enumerate(concepts):
        for c, topk_dir in enumerate(topk_dirs):
            x_px = label_col_px + c * cell
            y_px = r * cell
            rect = norm_rect(x_px, y_px, cell, cell)
            ax = fig.add_axes(rect)
            ax.axis("off")
            img_path = os.path.join(topk_dir, concept, "rank_1.jpg")
            im = load_square_no_border(img_path, cell)
            ax.imshow(im)

    # final save with zero padding
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return out_path


# ------------------------------- Main ----------------------------------- #
def main():
    args = parse_args()
    mapping = collect_qcw_folders(args.root)

    # normalize requested layers to int set if provided
    requested_layers_set = set()
    if args.layers.strip():
        for part in [p.strip() for p in args.layers.split(",") if p.strip()]:
            num = re.sub(r"\D", "", part)
            if num:
                requested_layers_set.add(int(num))

    for depth, size_label in TARGET_GROUPS:
        key = (depth, size_label)
        if key not in mapping:
            print(f"[skip] No QCW folders for ResNet-{depth} {size_label}")
            continue

        layers_and_folders = mapping[key]  # list of (layer, foldername)
        # filter by requested layers if user provided
        if requested_layers_set:
            layers_and_folders = [lf for lf in layers_and_folders if lf[0] in requested_layers_set]
            if not layers_and_folders:
                print(f"[skip] No requested WLs found for ResNet-{depth} {size_label}")
                continue

        # Build topk_dir list per layer
        layer_numbers = [lf[0] for lf in layers_and_folders]
        layer_names = [f"WL{ln}" for ln in layer_numbers]
        topk_dirs = [os.path.join(args.root, lf[1], "topk_concept_images") for lf in layers_and_folders]

        # Collect concept sets per layer and compute intersection (concepts present for ALL columns)
        per_layer_concepts = [set(list_concepts(td)) for td in topk_dirs]
        if not per_layer_concepts:
            print(f"[warn] No topk folders for group ResNet-{depth} {size_label}, skipping.")
            continue
        common_concepts = set.intersection(*per_layer_concepts)

        if args.concepts.strip():
            wanted = {s.strip() for s in args.concepts.split(",") if s.strip()}
            common_concepts = common_concepts.intersection(wanted)
            missing = wanted - common_concepts
            if missing:
                print(f"[warn] Concepts requested but not present across all layers for {key}: {sorted(missing)}")

        concepts = sorted(list(common_concepts), key=lambda s: s.lower())
        if not concepts:
            print(f"[skip] No common concepts across layers for ResNet-{depth} {size_label}")
            continue

        if args.max_rows > 0:
            concepts = concepts[: args.max_rows]
        if args.shuffle_rows:
            random.shuffle(concepts)

        # Prepare output path and name
        group_out_dir = os.path.join(args.out_dir, f"resnet{depth}_{size_label}")
        os.makedirs(group_out_dir, exist_ok=True)
        out_png = os.path.join(group_out_dir, f"grid_resnet{depth}_{size_label}_" + "_".join(map(str, layer_numbers)) + ".png")
        print(f"[run] Building grid for ResNet-{depth} {size_label} | layers={layer_numbers} | concepts={len(concepts)}")

        try:
            saved = build_grid_columns_as_layers(
                layer_names=layer_names,
                topk_dirs=topk_dirs,
                concepts=concepts,
                cell=args.cell,
                out_path=out_png,
                show_row_labels=True,
            )
            print(f"[ok] Wrote {saved}")
        except Exception as e:
            print(f"[error] Failed building grid for {key}: {e}")

    print("[done]")


if __name__ == "__main__":
    main()
