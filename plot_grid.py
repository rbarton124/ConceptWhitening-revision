#!/usr/bin/env python3
"""
plot_grid.py - Generate visualization grids of top concept images across whitening layers.
Refactored to match reference scripts: fig_free_showcase.py, fig_concept_across_layers.py, fig8_cub_cross_layer.py

Three visualization modes:

1. CONCEPT_ROWS (default):
   Rows=concepts, Columns=layers, each cell shows top-1 image
   python plot_grid.py --root /path/to/experiments --mode concept_rows --cell 120 --label_fontsize 16
   
2. TOPK_SINGLE:
   Multiple concepts, show top-K images on fixed layer WL8
   python plot_grid.py --root /path/to/experiments --mode topk_single \\
     --concepts "animal::giraffe,animal::dog" --topk 8 --cell 140 --label_fontsize 16
   
3. TOPK_CONCEPTS:
   1-2 concepts, show top-K images per layer in a grid
   python plot_grid.py --root /path/to/experiments --mode topk_concepts \\
     --concepts "animal::giraffe,food::cake" --topk 3 --cell 130 --label_fontsize 16
"""

import argparse
import os
import re
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("QCW per-layer top-k grids with multiple visualization modes")
    p.add_argument("--root", required=True, help="Root directory with experiment folders.")
    p.add_argument("--mode", default="concept_rows", 
                   choices=["concept_rows", "topk_single", "topk_concepts"],
                   help="Visualization mode")
    p.add_argument("--layers", default="", help="Comma-separated WL numbers to include (e.g. 1,5,8).")
    p.add_argument("--concepts", default="", help="Comma-separated concept names.")
    p.add_argument("--k", type=int, default=5, help="Top-k images available in folders.")
    p.add_argument("--topk", type=int, default=3, help="Number of top images to display per concept (for topk modes).")
    p.add_argument("--cell", type=int, default=120, help="Thumbnail size (px) for each cell.")
    p.add_argument("--label_fontsize", type=int, default=16, help="Font size for labels and headers.")
    p.add_argument("--dpi", type=int, default=180, help="DPI for figure output.")
    p.add_argument("--max_rows", type=int, default=0, help="Cap on number of concept rows (0 = no cap).")
    p.add_argument("--shuffle_rows", action="store_true", help="Shuffle row order.")
    p.add_argument("--concept_mapping", default="", help="Path to JSON file with concept name mappings.")
    p.add_argument("--out_dir", default="layer_topk_grids", help="Output directory for generated figures.")
    return p.parse_args()


def parse_dataset_items(dataset_str: str) -> List[str]:
    """Accept 'animal,food' or 'animal-food' and return list ['animal','food']."""
    s = dataset_str.replace("-", ",")
    items = [t.strip() for t in s.split(",") if t.strip()]
    return items


def dataset_size_label_from_items(items: List[str]) -> str:
    """Return 'small' if <=2 items else 'large'."""
    return "small" if len(items) <= 2 else "large"


def collect_qcw_folders(root: str) -> Dict[Tuple[int, str], List[Tuple[int, str]]]:
    """Scan root for QCW folders and return mapping: (depth, dataset_size) -> list of (layer, folder_name)"""
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

    for k in mapping:
        mapping[k] = sorted(mapping[k], key=lambda x: x[0])
    return mapping


def load_square_no_border(path: str, size: int) -> Image.Image:
    """Load image, center-crop to square, resize."""
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


def get_concept_display_name(concept_folder_name: str, concept_mapping: dict | None = None) -> str:
    """Get hierarchical display name for concept."""
    if concept_mapping and concept_folder_name in concept_mapping:
        return concept_mapping[concept_folder_name]
    
    if "/" in concept_folder_name:
        parts = concept_folder_name.split("/")
        return "::".join(clean_label(p) for p in parts)
    
    return clean_label(concept_folder_name)

def format_concept_label_with_hierarchy(display_name: str) -> str:
    """Format concept name with bold hierarchy on first line, value on second.
    E.g., 'animal::giraffe' -> bold{animal:}\ngiraffe
    """
    if "::" in display_name:
        parts = display_name.split("::")
        hl = "::" .join(parts[:-1]).replace(" ", "~")
        val = parts[-1]
        return f"$\\bf{{{hl}:}}$\n{val}"
    else:
        return display_name

def format_concept_label_single_line(display_name: str) -> str:
    """Format concept name on single line without bold.
    E.g., 'animal::giraffe' -> animal: giraffe
    """
    if "::" in display_name:
        parts = display_name.split("::")
        hl = "::" .join(parts[:-1])
        val = parts[-1]
        return f"{hl}: {val}"
    else:
        return display_name

def list_concepts(topk_dir: str) -> List[str]:
    if not os.path.isdir(topk_dir):
        return []
    return sorted([d for d in os.listdir(topk_dir) if os.path.isdir(os.path.join(topk_dir, d))],
                  key=lambda s: s.lower())


def parse_concept_input(concept_str: str) -> Tuple[str, str]:
    """Parse concept input: "supercategory::concept" -> (folder_name, display_name)"""
    display_name = concept_str.strip()
    if "::" in display_name:
        parts = display_name.split("::")
        folder_name = parts[-1].strip()
    else:
        folder_name = display_name
    return folder_name, display_name


def build_concept_mapping_from_input(concept_strings: List[str]) -> Tuple[List[str], dict]:
    """Convert input concepts to folder_names list and display_mapping dict."""
    folder_names = []
    display_mapping = {}
    
    for concept_str in concept_strings:
        folder_name, display_name = parse_concept_input(concept_str)
        folder_names.append(folder_name)
        display_mapping[folder_name] = display_name
    
    return folder_names, display_mapping


# ============================================================================#
# CONCEPT_ROWS Mode (following fig8_cub_cross_layer.py)
# ============================================================================#
def build_concept_rows_grid(layer_names: Sequence[str],
                            topk_dirs: Sequence[str],
                            concepts: Sequence[str],
                            cell: int,
                            out_path: str,
                            label_fontsize: int = 16,
                            concept_mapping: dict | None = None,
                            dpi: int = 200) -> str:
    """Build grid: rows=concepts, cols=layers, each cell=top-1 image."""
    n_cols = len(layer_names)
    n_rows = len(concepts)
    
    if n_cols == 0 or n_rows == 0:
        raise ValueError("No columns or rows for grid building.")
    
    cell_in = cell / 96  # inches per cell at screen dpi
    fig_w = n_cols * cell_in + 0.1
    fig_h = n_rows * (cell_in + 0.06) + 0.4
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), dpi=dpi)
    if n_rows == 1:
        axes = [axes]
    if n_cols == 1:
        axes = [[ax] for ax in axes]
    
    plt.subplots_adjust(left=0.005, right=0.995, top=0.94, bottom=0.01,
                       wspace=0.03, hspace=0.18)
    
    # Fill images
    for r, concept in enumerate(concepts):
        for c, topk_dir in enumerate(topk_dirs):
            ax = axes[r][c]
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            
            img_path = os.path.join(topk_dir, concept, "rank_1.jpg")
            im = load_square_no_border(img_path, cell)
            ax.imshow(im)
    
    # Column headers: WL labels
    for c, lname in enumerate(layer_names):
        axes[0][c].set_title(lname, fontsize=label_fontsize, fontweight="bold", pad=4)
    
    # Row labels: concept names with bold hierarchy
    for r, concept in enumerate(concepts):
        display_name = get_concept_display_name(concept, concept_mapping)
        formatted_label = format_concept_label_with_hierarchy(display_name)
        axes[r][0].set_ylabel(formatted_label, fontsize=label_fontsize,
                             rotation=0, ha="right", va="center", labelpad=8)
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ============================================================================#
# TOPK_SINGLE Mode (following fig_free_showcase.py)
# ============================================================================#
def build_topk_single_grid(concepts: Sequence[str],
                           topk_dir: str,
                           topk: int,
                           cell: int,
                           out_path: str,
                           label_fontsize: int = 16,
                           concept_mapping: dict | None = None,
                           dpi: int = 180) -> str:
    """Build grid: rows=concepts, cols=ranks (right-to-left), all from WL8."""
    n_cols = topk
    n_rows = len(concepts)
    
    cell_in = cell / 96
    fig_w = n_cols * cell_in + 0.15
    row_h = cell_in + 0.08
    fig_h = n_rows * row_h + 0.5
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), dpi=dpi)
    if n_rows == 1:
        axes = [axes]
    if n_cols == 1:
        axes = [[ax] for ax in axes]
    
    plt.subplots_adjust(left=0.005, right=0.995, top=0.88, bottom=0.01,
                       wspace=0.03, hspace=0.25)
    
    # Add title
    fig.suptitle("Animal Subspace (WL8)", fontsize=label_fontsize + 2, fontweight="bold", y=0.96)
    
    # Fill images (left-to-right: rank 1 on left, rank k on right)
    for r, concept in enumerate(concepts):
        for c in range(n_cols):
            rank = c + 1  # Direct order
            ax = axes[r][c]
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            
            img_path = os.path.join(topk_dir, concept, f"rank_{rank}.jpg")
            im = load_square_no_border(img_path, cell)
            ax.imshow(im)
    
    # Column headers: rank labels (left-to-right)
    for c in range(n_cols):
        rank = c + 1
        axes[0][c].set_title(f"rank {rank}", fontsize=label_fontsize, fontweight="bold", pad=4)
    
    # Row labels: concept names with bold hierarchy
    for r, concept in enumerate(concepts):
        display_name = get_concept_display_name(concept, concept_mapping)
        formatted_label = format_concept_label_with_hierarchy(display_name)
        axes[r][0].set_ylabel(formatted_label, fontsize=label_fontsize,
                             rotation=0, ha="right", va="center", labelpad=10)
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ============================================================================#
# TOPK_CONCEPTS Mode (following fig_concept_across_layers.py)
# ============================================================================#
def build_topk_concepts_grid(layer_names: Sequence[str],
                             topk_dirs: Sequence[str],
                             concepts: Sequence[str],
                             topk: int,
                             cell: int,
                             out_path: str,
                             label_fontsize: int = 16,
                             concept_mapping: dict | None = None,
                             dpi: int = 180) -> str:
    """Build grid: rows=layers, cols=topk*concepts with gaps.
    Transposed layout: each row is a layer, each concept is a vertical section showing topk ranks.
    """
    n_layers = len(topk_dirs)
    n_concepts = len(concepts)
    gap = 1 if n_concepts > 1 else 0
    
    # Column structure: topk columns per concept, with gaps between concepts
    total_cols = n_concepts * topk + (n_concepts - 1) * gap
    total_rows = n_layers
    
    cell_in = cell / 96
    fig_w = total_cols * cell_in + 0.2
    fig_h = n_layers * (cell_in + 0.08) + 0.8
    
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    
    # Width ratios: topk columns per concept, gap between concepts
    width_ratios = []
    for ci in range(n_concepts):
        width_ratios.extend([1] * topk)
        if ci < n_concepts - 1:
            width_ratios.append(0.25)
    
    gs = gridspec.GridSpec(total_rows, total_cols, figure=fig,
                          wspace=0.04, hspace=0.18,
                          left=0.005, right=0.995, top=0.90, bottom=0.01,
                          width_ratios=width_ratios)
    
    def col_idx(concept_i, rank_i):
        """Compute grid column for given concept and rank within that concept."""
        return concept_i * (topk + gap) + rank_i
    
    # Create all axes
    axes = {}
    for ri in range(total_rows):
        for ci in range(total_cols):
            ax = fig.add_subplot(gs[ri, ci])
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            axes[(ri, ci)] = ax
    
    # Fill images: for each layer row, show topk ranks for each concept
    for layer_idx, topk_dir in enumerate(topk_dirs):
        for concept_idx, concept in enumerate(concepts):
            for rank_idx in range(topk):
                rank = rank_idx + 1
                c = col_idx(concept_idx, rank_idx)
                ax = axes[(layer_idx, c)]
                
                img_path = os.path.join(topk_dir, concept, f"rank_{rank}.jpg")
                im = load_square_no_border(img_path, cell)
                ax.imshow(im)
    
    # Row labels: layer names (WL1, WL2, etc.)
    for row_idx, lname in enumerate(layer_names):
        # Place label on leftmost axis in this row
        ax = axes[(row_idx, 0)]
        ax.set_ylabel(lname, fontsize=label_fontsize, fontweight="bold",
                     rotation=0, ha="right", va="center", labelpad=8)
    
    # Concept headers: center-aligned across their topk columns at the top
    # Use single-line formatting for topk_concepts titles
    for concept_idx, concept in enumerate(concepts):
        display_name = get_concept_display_name(concept, concept_mapping)
        formatted_label = format_concept_label_single_line(display_name)
        # Place on middle rank column of this concept group for centering
        middle_rank = topk // 2
        c = col_idx(concept_idx, middle_rank)
        axes[(0, c)].set_title(formatted_label, fontsize=label_fontsize, fontweight="bold", pad=4)

    # Add rank headers (rank 1, rank 2, rank 3...) on top row for each concept
    for concept_idx in range(n_concepts):
        for rank_idx in range(topk):
            rank = rank_idx + 1
            c = col_idx(concept_idx, rank_idx)

            axes[(0, c)].set_title(
                f"rank {rank}",
                fontsize=int(label_fontsize * 0.8),
                fontweight="bold",
                pad=12  # <-- increase this for more vertical spacing
            )
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ============================================================================#
# Main Entry Points
# ============================================================================#
def main():
    args = parse_args()
    
    # Parse concepts
    concept_input = [s.strip() for s in args.concepts.split(",") if s.strip()] if args.concepts.strip() else []
    folder_names, input_display_mapping = build_concept_mapping_from_input(concept_input)
    
    # Load explicit concept mapping if provided
    explicit_mapping = {}
    if args.concept_mapping:
        import json
        try:
            with open(args.concept_mapping, 'r') as f:
                explicit_mapping = json.load(f)
        except Exception as e:
            print(f"[warn] Failed to load concept mapping: {e}")
    
    display_mapping = {**input_display_mapping, **explicit_mapping}
    
    # Dispatch to appropriate mode
    if args.mode == "concept_rows":
        main_concept_rows(args, display_mapping)
    elif args.mode == "topk_single":
        main_topk_single(args, folder_names, display_mapping)
    elif args.mode == "topk_concepts":
        main_topk_concepts(args, folder_names, display_mapping)
    
    print("[done]")


def main_concept_rows(args, display_mapping):
    """Concept rows mode: rows=concepts, cols=layers."""
    mapping = collect_qcw_folders(args.root)
    
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

        layers_and_folders = mapping[key]
        if requested_layers_set:
            layers_and_folders = [lf for lf in layers_and_folders if lf[0] in requested_layers_set]
            if not layers_and_folders:
                print(f"[skip] No requested WLs found for ResNet-{depth} {size_label}")
                continue

        layer_numbers = [lf[0] for lf in layers_and_folders]
        layer_names = [f"WL{ln}" for ln in layer_numbers]
        topk_dirs = [os.path.join(args.root, lf[1], "topk_concept_images") for lf in layers_and_folders]

        per_layer_concepts = [set(list_concepts(td)) for td in topk_dirs]
        if not per_layer_concepts:
            print(f"[warn] No topk folders for group ResNet-{depth} {size_label}, skipping.")
            continue
        common_concepts = set.intersection(*per_layer_concepts)

        # If concepts were explicitly specified, use them in that order
        # Otherwise, auto-discover and sort alphabetically
        if display_mapping:
            wanted = list(display_mapping.keys())  # Preserve order from input
            concepts = [c for c in wanted if c in common_concepts]
            missing = set(wanted) - set(common_concepts)
            if missing:
                print(f"[warn] Concepts requested but not present: {sorted(missing)}")
        else:
            concepts = sorted(list(common_concepts), key=lambda s: s.lower())
        
        if not concepts:
            print(f"[skip] No common concepts across layers for ResNet-{depth} {size_label}")
            continue

        if args.max_rows > 0:
            concepts = concepts[: args.max_rows]
        if args.shuffle_rows:
            import random
            random.shuffle(concepts)

        group_out_dir = os.path.join(args.out_dir, f"resnet{depth}_{size_label}")
        os.makedirs(group_out_dir, exist_ok=True)
        out_png = os.path.join(group_out_dir, f"grid_resnet{depth}_{size_label}_" + "_".join(map(str, layer_numbers)) + ".png")
        print(f"[run] Building grid for ResNet-{depth} {size_label} | layers={layer_numbers} | concepts={len(concepts)}")

        try:
            saved = build_concept_rows_grid(
                layer_names=layer_names,
                topk_dirs=topk_dirs,
                concepts=concepts,
                cell=args.cell,
                out_path=out_png,
                label_fontsize=args.label_fontsize,
                concept_mapping=display_mapping,
                dpi=args.dpi,
            )
            print(f"[ok] Wrote {saved}")
        except Exception as e:
            print(f"[error] Failed building grid for {key}: {e}")


def main_topk_single(args, folder_names, display_mapping):
    """Top-K single mode: multiple concepts, top-K images on fixed WL8."""
    if not folder_names:
        print("[error] --concepts required for topk_single mode")
        return
    
    mapping = collect_qcw_folders(args.root)
    target_layer = 8
    
    for depth, size_label in TARGET_GROUPS:
        key = (depth, size_label)
        if key not in mapping:
            continue

        layers_and_folders = mapping[key]
        
        # Find WL8 folder
        wl8_folder = None
        wl8_path = None
        for layer, fname in layers_and_folders:
            if layer == target_layer:
                wl8_folder = fname
                wl8_path = os.path.join(args.root, fname, "topk_concept_images")
                break
        
        if not wl8_folder:
            print(f"[skip] Layer WL{target_layer} not found for ResNet-{depth} {size_label}")
            continue

        # Check if all concepts exist
        all_exist = True
        for concept_folder in folder_names:
            if not os.path.isdir(os.path.join(wl8_path, concept_folder)):
                print(f"[skip] Concept '{concept_folder}' not found in WL{target_layer} for ResNet-{depth} {size_label}")
                all_exist = False
                break
        
        if not all_exist:
            continue

        group_out_dir = os.path.join(args.out_dir, f"resnet{depth}_{size_label}")
        os.makedirs(group_out_dir, exist_ok=True)
        concept_str = "_".join(folder_names)
        out_png = os.path.join(group_out_dir, f"topk_single_wl{target_layer}_{concept_str}.png")
        
        concept_names = [display_mapping.get(cf, cf) for cf in folder_names]
        print(f"[run] Building top-K grid for {concept_names} | WL{target_layer} | ResNet-{depth} {size_label}")

        try:
            saved = build_topk_single_grid(
                concepts=folder_names,
                topk_dir=wl8_path,
                topk=args.topk,
                cell=args.cell,
                out_path=out_png,
                label_fontsize=args.label_fontsize,
                concept_mapping=display_mapping,
                dpi=args.dpi,
            )
            print(f"[ok] Wrote {saved}")
        except Exception as e:
            print(f"[error] Failed: {e}")


def main_topk_concepts(args, folder_names, display_mapping):
    """Top-K concepts mode: 1-2 concepts, top-K images per layer."""
    if not folder_names:
        print("[error] --concepts required for topk_concepts mode")
        return
    
    if len(folder_names) > 2:
        print("[error] topk_concepts mode supports maximum 2 concepts")
        return
    
    mapping = collect_qcw_folders(args.root)
    
    requested_layers_set = set()
    if args.layers.strip():
        for part in [p.strip() for p in args.layers.split(",") if p.strip()]:
            num = re.sub(r"\D", "", part)
            if num:
                requested_layers_set.add(int(num))

    for depth, size_label in TARGET_GROUPS:
        key = (depth, size_label)
        if key not in mapping:
            continue

        layers_and_folders = mapping[key]
        if requested_layers_set:
            layers_and_folders = [lf for lf in layers_and_folders if lf[0] in requested_layers_set]

        if not layers_and_folders:
            continue

        layer_numbers = [lf[0] for lf in layers_and_folders]
        layer_names = [f"WL{ln}" for ln in layer_numbers]
        topk_dirs = [os.path.join(args.root, lf[1], "topk_concept_images") for lf in layers_and_folders]

        # Check if all concepts exist in all layers
        all_exist = True
        for concept_folder in folder_names:
            has_concept = [os.path.isdir(os.path.join(td, concept_folder)) for td in topk_dirs]
            if not all(has_concept):
                print(f"[skip] Concept '{concept_folder}' not found in all layers for ResNet-{depth} {size_label}")
                all_exist = False
                break
        
        if not all_exist:
            continue

        group_out_dir = os.path.join(args.out_dir, f"resnet{depth}_{size_label}")
        os.makedirs(group_out_dir, exist_ok=True)
        concept_str = "_".join(folder_names)
        out_png = os.path.join(group_out_dir, f"topk_concepts_{concept_str}_layers_" + "_".join(map(str, layer_numbers)) + ".png")
        
        concept_names = [display_mapping.get(cf, cf) for cf in folder_names]
        print(f"[run] Building grid for {concept_names} | ResNet-{depth} {size_label} | layers={layer_numbers}")

        try:
            saved = build_topk_concepts_grid(
                layer_names=layer_names,
                topk_dirs=topk_dirs,
                concepts=folder_names,
                topk=args.topk,
                cell=args.cell,
                out_path=out_png,
                label_fontsize=args.label_fontsize,
                concept_mapping=display_mapping,
                dpi=args.dpi,
            )
            print(f"[ok] Wrote {saved}")
        except Exception as e:
            print(f"[error] Failed: {e}")


if __name__ == "__main__":
    main()
