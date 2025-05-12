from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
from types import SimpleNamespace
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from sklearn.metrics import roc_auc_score
from termcolor import cprint
from torch.utils.data import DataLoader
from tqdm import tqdm

# Reduce console noise
logging.getLogger("matplotlib").setLevel(logging.WARNING)
torch.set_printoptions(sci_mode=False)

from MODELS.model_resnet_qcw import build_resnet_qcw, get_last_qcw_layer
from MODELS.ConceptDataset_QCW import ConceptDataset

TOPK: int = 5  # how many axes to show in diagnostics
LOG_FORMAT = "%(levelname)s | %(asctime)s | %(message)s"

def resume_checkpoint(model: nn.Module, checkpoint_path: str) -> nn.Module:
    """Load *any* QCW/BYOL/SimCLR checkpoint, coping with DataParallel prefixes."""
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        logging.warning("[Checkpoint] No checkpoint at %s", checkpoint_path)
        return model

    logging.info("[Checkpoint] Loading weights from %s", checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    raw_sd = ckpt.get("state_dict", ckpt)
    model_sd = model.state_dict()
    clean_sd: Dict[str, torch.Tensor] = {}

    def _rename(k: str) -> str:
        if k.startswith("module."):
            k = k[7:]
        if k.startswith("model."):
            k = k[6:]
        if not k.startswith("backbone.") and f"backbone.{k}" in model_sd:
            return f"backbone.{k}"
        if k.startswith("fc.") and f"backbone.{k}" in model_sd:
            return f"backbone.{k}"
        return k

    for ckpt_k, ckpt_v in raw_sd.items():
        new_k = _rename(ckpt_k)
        if new_k in model_sd and ckpt_v.shape == model_sd[new_k].shape:
            clean_sd[new_k] = ckpt_v

    missing, unexpected = model.load_state_dict(clean_sd, strict=False)
    if missing:
        logging.debug("Missing keys: %s", missing)
    if unexpected:
        logging.debug("Unexpected keys: %s", unexpected)
    return model


def build_model(ckpt: str, depth: int, whitened: List[int], act_mode: str,
                subspaces: dict | None, num_classes: int) -> nn.Module:
    model = build_resnet_qcw(num_classes=num_classes,
                             depth=depth,
                             whitened_layers=whitened,
                             act_mode=act_mode,
                             subspaces=subspaces)
    model = resume_checkpoint(model, ckpt)
    model = nn.DataParallel(model).cuda().eval()
    return model


def build_dataset(root: str, hl_filter: Sequence[str], bboxes: str,
                  crop_mode: str) -> ConceptDataset:
    tfm = T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return ConceptDataset(
        root_dir=os.path.join(root, "concept_val"),
        bboxes_file=bboxes,
        high_level_filter=list(hl_filter),
        transform=tfm,
        crop_mode=crop_mode,
    )


class HookRunner:
    """Context‑manager to grab activations from the final QCW layer."""

    def __init__(self, model: nn.Module, num_concepts: int):
        self.storage = SimpleNamespace(out=None)
        layer = get_last_qcw_layer(model.module if isinstance(model, nn.DataParallel) else model)

        def _hook(_, __, output):
            with torch.no_grad():
                self.storage.out = output.mean(dim=(2, 3))[:, :num_concepts].cpu()

        self.handle = layer.register_forward_hook(_hook)
        self.model = model

    def run_loader(self, loader: DataLoader, device: str = "cuda") -> tuple[np.ndarray, np.ndarray]:
        acts: List[np.ndarray] = []
        labels: List[np.ndarray] = []
        for imgs, sc_lbl, _ in tqdm(loader, desc="Collecting feats"):
            imgs = imgs.to(device)
            self.storage.out = None
            _ = self.model(imgs)
            if self.storage.out is None:
                continue  # Skip if hook didn't fire (can happen in very old PyTorch versions)
            acts.append(self.storage.out.numpy())
            labels.append(sc_lbl.numpy())
        return np.concatenate(acts, 0), np.concatenate(labels, 0)

    def close(self):
        self.handle.remove()


# -----------------------------------------------------------------------------
# Analysis classes
# -----------------------------------------------------------------------------

class PurityAnalyzer:
    """Compute (sub‑)concept purity, diagnostics, and CSV."""

    def __init__(self, model: nn.Module, ds: ConceptDataset, *,
                 batch_size: int = 64, device: str = "cuda", topk: int = TOPK,
                 output_dir: str = "qcw_plots",
                 cfg: argparse.Namespace | None = None):
        self.ds = ds
        self.model = model
        self.bs = batch_size
        self.device = device
        self.topk = topk
        self.output_dir = output_dir
        self.sc_names = ds.get_subconcept_names()
        self.num_concepts = len(self.sc_names)
        self._collect()  # fills self.acts / self.labels

        # Maps for hierarchy
        self.subspace_map = ds.subspace_mapping  # HL → list[int]
        self.sc_to_hl = ds.get_subconcept_to_hl_name_mapping()  # idx → HL
        self.axis_to_hl = {idx: hl for idx, hl in self.sc_to_hl.items()}

        # Pre‑split
        self.labeled, self.free = [], []
        for i, nm in enumerate(self.sc_names):
            (self.free if ds.is_free_subconcept_name(nm) else self.labeled).append(i)
        self.labeled_set = set(self.labeled)
        
        # keep full CLI so we know which extras to compute
        self.cfg = cfg or SimpleNamespace(auc_max=False, energy_ratio=False)

        # ---------------- internal helpers ----------------

    def _collect(self) -> None:
        loader = DataLoader(self.ds, batch_size=self.bs, shuffle=False, pin_memory=True)
        runner = HookRunner(self.model, self.num_concepts)
        self.acts, self.labels = runner.run_loader(loader, self.device)
        runner.close()

    def _roc_auc(self, mask: np.ndarray,
               axis: int | None = None,
               custom: np.ndarray | None = None) -> float:
        """If `custom` is given use that (shape [N]), else self.acts[:, axis]."""
        scores = custom if custom is not None else self.acts[:, axis]
        if mask.sum() in {0, len(mask)}:
            return float("nan")
        return roc_auc_score(mask.astype(int), scores)

    # ---------------- public API ----------------

    def compute(self, subspace_eval: bool = False) -> Dict[str, dict]:
        info: Dict[str, dict] = {}
        rows: List[dict] = []

        # Pre-compute global max scores since they're the same for all concepts
        if self.cfg.auc_max:
            global_max_scores = self.acts.max(1)
        else:
            global_max_scores = None
            
        for idx in range(self.num_concepts):
            name = self.sc_names[idx]
            hl = self.sc_to_hl[idx]
            hl_axes = self.subspace_map[hl]
            mask = self.labels == idx
            
            # Guard against empty positive set
            if not mask.any():
                # Create a dictionary with nan for all numerical values
                info[name] = {
                    "is_free": idx in self.free,
                    "label_auc": float("nan"),
                    "best_axis_auc": float("nan"),
                    "best_axis_idx": -1,
                    "best_axis_is_labeled": None,
                    "unlabeled_axis_auc": None,
                    "unlabeled_axis_idx": None,
                    "auc_max_global": float("nan"),
                    "auc_max_hl": float("nan"),
                    "delta_max": float("nan"),
                    "energy_ratio": float("nan"),
                }
                rows.append({
                    "sc_name": name,
                    "hl_name": hl,
                    "is_free": idx in self.free,
                    "slice_axes": " ".join(map(str, hl_axes)),
                    "best_ax_global": -1,
                    "auc_global": float("nan"),
                    "best_ax_hl": -1,
                    "auc_hl": float("nan"),
                    "mass_global": float("nan"),
                    "mass_hl": float("nan"),
                    "mass_retained_pct": float("nan"),
                    "topk_global": "",
                    "auc_max_global": float("nan"),
                    "auc_max_hl": float("nan"),
                    "delta_max": float("nan"),
                    "energy_ratio": float("nan"),
                })
                continue
                
            sub_acts = self.acts[mask]
            mean_acts = sub_acts.mean(0)

            # ----------------------------------------------------------------
            # NEW - hierarchy-aware metrics that do NOT rely on single axis
            # ----------------------------------------------------------------
            # Compute HL-max scores for this concept's HL (can't be precomputed globally)
            if self.cfg.auc_max:
                hl_max_scores = self.acts[:, hl_axes].max(1)
                auc_max_global = self._roc_auc(mask, custom=global_max_scores)
                auc_max_hl = self._roc_auc(mask, custom=hl_max_scores)
                delta_max = auc_max_global - auc_max_hl
            else:
                auc_max_global = auc_max_hl = delta_max = None

            if self.cfg.energy_ratio:
                # energy for positive samples only
                g_energy = np.linalg.norm(self.acts[mask], axis=1).mean()
                h_energy = np.linalg.norm(self.acts[mask][:, hl_axes], axis=1).mean()
                energy_ratio = h_energy / (g_energy + 1e-9)
            else:
                energy_ratio = None

            # ---------- NEW LOGIC -------------------------------------------------
            # 1) global and HL-local maxima (same as before)
            best_g = int(mean_acts.argmax())
            auc_g = self._roc_auc(mask, best_g)

            hl_means = mean_acts[hl_axes]
            best_hl = hl_axes[int(hl_means.argmax())]
            auc_hl = self._roc_auc(mask, best_hl)

            # 2) choose PRIMARY axis & SECONDARY (diagnostic) axis
            # -----------------------------------------------------
            if idx in self.free:
                # -- Candidate pool = unlabeled axes ---------------
                unlabeled_axes = [a for a in range(self.num_concepts)
                                  if a not in self.labeled_set]
                if not unlabeled_axes:
                    # degenerate case – fall back to best_hl
                    primary_ax = best_hl
                else:
                    ul_means = mean_acts[unlabeled_axes]
                    primary_ax = unlabeled_axes[int(ul_means.argmax())]

                primary_auc = self._roc_auc(mask, primary_ax)

                # SECONDARY axis = best overall *if* it beats primary
                if mean_acts[best_g] > mean_acts[primary_ax] + 1e-6:
                    secondary_ax = best_g
                    secondary_auc = auc_g
                else:
                    secondary_ax = None
                    secondary_auc = None
            else:
                # ----- LABELLED CONCEPT -------------------------------------------
                primary_ax = idx
                primary_auc = self._roc_auc(mask, idx)
                secondary_ax = None
                secondary_auc = None

            # Decide which axis drives the classic single-axis purity plot
            use_axis = primary_ax if not subspace_eval else (
                primary_ax if (primary_ax in hl_axes) else best_hl)
            use_auc = self._roc_auc(mask, use_axis)

            is_lbl = use_axis in self.labeled_set
            unl_auc = unl_idx = None
            # (We keep unl_* keys for backward compatibility but they now mean
            #  "secondary" rather than "unlabeled fallback".)
            if secondary_ax is not None:
                unl_idx, unl_auc = secondary_ax, secondary_auc

            info[name] = {
                "is_free": idx in self.free,
                "label_auc": None if idx in self.free else use_auc,
                "best_axis_auc": use_auc,
                "best_axis_idx": use_axis,
                "best_axis_is_labeled": is_lbl,
                "unlabeled_axis_auc": unl_auc,
                "unlabeled_axis_idx": unl_idx,
                # NEW hierarchy metrics
                "auc_max_global": auc_max_global,
                "auc_max_hl":     auc_max_hl,
                "delta_max":      delta_max,
                "energy_ratio":   energy_ratio,
                # NEW explicit names to avoid confusion
                "primary_axis_idx":    primary_ax,
                "primary_axis_auc":    primary_auc,
                "secondary_axis_idx":  secondary_ax,
                "secondary_axis_auc":  secondary_auc,
            }

            # ---- diagnostics block ----
            rows.append(self._diagnostic_row(name, hl, idx in self.free, hl_axes,
                                             mean_acts, mask, best_g, auc_g, best_hl, auc_hl,
                                             auc_max_global, auc_max_hl, delta_max, energy_ratio))

        self._save_csv(rows)
        return info

    # ---------------- diagnostics / csv ----------------

    def _diagnostic_row(self, scn: str, hl: str, is_free: bool, hl_axes: List[int],
                     mean_acts: np.ndarray, mask: np.ndarray,
                     best_g: int, auc_g: float, best_hl: int, auc_hl: float,
                     auc_max_global=None, auc_max_hl=None, delta_max=None, energy_ratio=None) -> dict:
        sub_acts = self.acts[mask]
        mass_g = float(np.abs(sub_acts).mean())
        mass_h = float(np.abs(sub_acts[:, hl_axes]).mean())
        pct = 100 * mass_h / (mass_g + 1e-9)

        # pretty console block (only for one concept to avoid spam)
        if self.device == "cuda" and len(getattr(self, "_printed", set())) < 1:  # type: ignore[attr-defined]
            self._printed = getattr(self, "_printed", set())  # type: ignore[attr-defined]
            if scn not in self._printed:
                self._printed.add(scn)
                top_ids = mean_acts.argsort()[::-1][: self.topk]
                top_vals = mean_acts[top_ids]
                top_hls = [self.axis_to_hl[a] for a in top_ids]
                off_slice = [a for a in top_ids if a not in hl_axes]
                print("\n[Diagnostic] Sample analysis:")
                print(f"Subconcept: {scn} (HL: {hl})")
                print(f"  Top-{self.topk} axes: {list(top_ids)}")
                print(f"  Top-{self.topk} values: {[f'{v:.3f}' for v in top_vals]}")
                print(f"  Top-{self.topk} HLs: {top_hls}")
                print(f"  Axes in this HL: {hl_axes}")
                print(f"  Axes outside HL: {off_slice}")
                if off_slice:
                    cprint(f"  ⚠️  Found {len(off_slice)} top-{self.topk} axes outside this concept's HL space", "red")
                print(f"Total activation before masking: {mass_g:,.2f}")
                print(f"Total activation after masking:  {mass_h:,.2f}")
                print(f"Percentage retained: {pct:.2f}%")
                if pct > 95:
                    cprint("⚠️  WARNING: Almost 100% of activation retained after masking.", "red")
                
                # Optional readout of new metrics
                if self.cfg.auc_max and delta_max is not None:
                    print(f"Δ_max (leakage gap): {delta_max:.3f} "
                          f"[global {auc_max_global:.3f} vs HL {auc_max_hl:.3f}]")
                if self.cfg.energy_ratio and energy_ratio is not None:
                    print(f"Energy ratio inside slice: {energy_ratio:.3f}")

        return {
            "sc_name": scn,
            "hl_name": hl,
            "is_free": is_free,
            "slice_axes": " ".join(map(str, hl_axes)),
            "best_ax_global": best_g,
            "auc_global": auc_g,
            "best_ax_hl": best_hl,
            "auc_hl": auc_hl,
            "mass_global": mass_g,
            "mass_hl": mass_h,
            "mass_retained_pct": pct,
            "topk_global": " ".join(
                f"{a}:{mean_acts[a]:.3f}" for a in mean_acts.argsort()[::-1][: self.topk]
            ),
            "auc_max_global": auc_max_global,
            "auc_max_hl":     auc_max_hl,
            "delta_max":      delta_max,
            "energy_ratio":   energy_ratio,
        }

    def _save_csv(self, rows: List[dict]) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, "result.csv")
        pd.DataFrame(rows).to_csv(path, index=False)
        logging.info("[CSV] Diagnostics → %s", path)


class TopKImageExtractor:
    """Dump images that maximally activate chosen axes (re‑uses cached activations)."""

    def __init__(self, acts: np.ndarray, labels: np.ndarray, ds: ConceptDataset,
                 out_dir: str, model: nn.Module | None = None):
        self.acts, self.labels, self.ds = acts, labels, ds
        self.out_dir = out_dir
        self.model = model  # only needed if saving transformed

    def dump(self, k: int = 10, save_transformed: bool = False,
             mean: List[float] | None = None, std: List[float] | None = None,
             device: str = "cuda") -> None:
        from torchvision.utils import save_image  # Import once at the method level
        mean = mean or [0.485, 0.456, 0.406]
        std = std or [0.229, 0.224, 0.225]
        sc_names = self.ds.get_subconcept_names()
        sc_count = len(sc_names)
        labeled = [i for i, n in enumerate(sc_names) if not self.ds.is_free_subconcept_name(n)]
        labeled_set = set(labeled)

        # helpers
        def _best_unlabeled_axis(fc_idx: int) -> int | None:
            mask = self.labels == fc_idx
            if not mask.any():
                return None
            means = self.acts[mask].mean(0)
            for ax in means.argsort()[::-1]:
                if ax not in labeled_set:
                    return int(ax)
            return None

        def _unnorm(t: torch.Tensor) -> torch.Tensor:
            m = torch.tensor(mean, dtype=t.dtype, device=t.device)[:, None, None]
            s = torch.tensor(std, dtype=t.dtype, device=t.device)[:, None, None]
            return t.mul(s).add(m).clamp(0, 1)

        os.makedirs(self.out_dir, exist_ok=True)
        if save_transformed:
            # need images – re‑load via DataLoader once
            logging.info("[Top‑K] Re‑loading images for saving …")
            loader = DataLoader(self.ds, batch_size=64, shuffle=False, pin_memory=True)
            imgs_all: List[torch.Tensor] = []
            for imgs, _, _ in tqdm(loader):
                imgs_all.append(imgs)
            imgs_all = torch.cat(imgs_all, 0)

        for idx, scn in enumerate(sc_names):
            axis = idx if idx in labeled_set else _best_unlabeled_axis(idx)
            if axis is None:
                continue
            scores = self.acts[:, axis]
            top_idx = scores.argsort()[::-1][:k]
            sc_dir = os.path.join(self.out_dir, scn.replace("/", "_"))
            os.makedirs(sc_dir, exist_ok=True)
            for rank, data_idx in enumerate(top_idx):
                dst = os.path.join(sc_dir, f"rank_{rank + 1}.jpg")
                if save_transformed:
                    save_image(_unnorm(imgs_all[data_idx].cpu()), dst)
                else:
                    src_path = self.ds.samples[data_idx][0]  # Get image path from samples
                    shutil.copy2(src_path, dst)
        logging.info("[Top‑K] Images dumped → %s", self.out_dir)


# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------

def bar_plot(concept_info: Dict[str, dict], out_path: str) -> None:
    """
    Re-implement the original "dual-bar" purity figure.

    ─ red  = axis is labelled (primary for labeled concepts, secondary when labeled axis leaks)
    ─ blue = axis is unlabelled (primary for free concepts)
    """
    import math
    import matplotlib.patches as mpatches

    # --- sort by primary-AUC descending (same rule as before) -----------------
    ordered = sorted(
        concept_info.items(),
        key=lambda kv: kv[1]["best_axis_auc"]
        if not math.isnan(kv[1]["best_axis_auc"])
        else -999,
        reverse=True,
    )

    plt.figure(figsize=(12, 6))
    MAIN_W, Fallback_W = 0.5, 0.3   # identical to legacy
    x_positions = list(range(len(ordered)))

    def _color(is_lbl: bool) -> str:
        return "red" if is_lbl else "blue"

    # -------------------------------------------------------------------------
    for i, (sc_name, info) in enumerate(ordered):
        primary_auc = info["best_axis_auc"]
        primary_idx = info["best_axis_idx"]
        primary_is_labeled = info["best_axis_is_labeled"]
        primary_color = _color(primary_is_labeled)
        
        # 1. For free concepts with both bars, draw primary and secondary with offset
        if info["is_free"] and info["unlabeled_axis_auc"] is not None:
            secondary_auc = info["unlabeled_axis_auc"]
            secondary_idx = info["unlabeled_axis_idx"]
            
            # Important: For free concepts, always draw unlabeled primary bar on left,
            # labeled secondary bar on right with different width and higher z-order
            # Primary bar (unlabeled, blue)
            if not math.isnan(primary_auc):
                plt.bar(i - 0.05, primary_auc, width=MAIN_W, color="blue", zorder=2)
                plt.text(
                    i - 0.05,
                    primary_auc + 0.01,
                    f"Ax {primary_idx} ({primary_auc:.2f})",
                    ha="center", va="bottom", fontsize=8, zorder=3
                )
                
            # Secondary bar (labeled, red, always on top and narrower)
            if secondary_auc is not None and not math.isnan(secondary_auc):
                plt.bar(i + 0.12, secondary_auc, width=0.25, color="red", zorder=5) 
                plt.text(
                    i + 0.12,
                    secondary_auc + 0.01,
                    f"Ax {secondary_idx} ({secondary_auc:.2f})",
                    ha="center", va="bottom", fontsize=8, zorder=6
                )
        else:
            # Single bar case (labeled concept)
            if not math.isnan(primary_auc):
                plt.bar(i, primary_auc, width=MAIN_W, color=primary_color, zorder=2)
                plt.text(
                    i,
                    primary_auc + 0.01,
                    f"Ax {primary_idx} ({primary_auc:.2f})",
                    ha="center", va="bottom", fontsize=8, zorder=3
                )

    # --- cosmetics -----------------------------------------------------------
    plt.xticks(x_positions, [n for n, _ in ordered], rotation=45, ha="right")
    plt.ylabel("AUC (Concept Purity)")
    plt.title("Concept Purity")
    plt.ylim(0, 1)
    plt.tight_layout()

    plt.legend(
        handles=[
            mpatches.Patch(color="red", label="Labeled Axis"),
            mpatches.Patch(color="blue", label="Unlabeled Axis"),
        ],
        loc="upper right",
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close()
    logging.info("[PNG] Bar-plot saved → %s", out_path)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("QCW concept analyzer (refactored)")
    # model
    p.add_argument("--model_checkpoint", required=True)
    p.add_argument("--depth", type=int, default=18)
    p.add_argument("--whitened_layers", default="5")
    p.add_argument("--act_mode", default="pool_max")
    p.add_argument("--num_classes", type=int, default=200)
    p.add_argument("--subspaces_json", default="")
    p.add_argument("--use_bn_qcw", action="store_true")

    # data
    p.add_argument("--concept_dir", default="")
    p.add_argument("--hl_concepts", default="")
    p.add_argument("--bboxes_file", default="")
    p.add_argument("--image_state", default="crop", choices=["crop", "redact", "none"])

    # run switches
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--run_purity", action="store_true")
    p.add_argument("--subspace_purity", action="store_true")
    p.add_argument("--topk_images", action="store_true")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--save_transformed", action="store_true")
    p.add_argument("--auc_max", action="store_true",
                   help="also compute AUC_max_global / AUC_max_hl / delta_max")
    p.add_argument("--energy_ratio", action="store_true",
                   help="compute EnergyRatio (= slice-energy / global-energy)")

    # out
    p.add_argument("--output_dir", default="qcw_plots")

    return p.parse_args()


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    if args.use_bn_qcw:
        global build_resnet_qcw, get_last_qcw_layer  # type: ignore[global‑statement]
        from MODELS.model_resnet_qcw_bn import (
            build_resnet_qcw as bn_build,
            get_last_qcw_layer as bn_last,
        )

        build_resnet_qcw, get_last_qcw_layer = bn_build, bn_last  # type: ignore[misc‑assignment]

    wl = [int(x) for x in args.whitened_layers.split(",") if x.strip()] if args.whitened_layers else []
    subspaces = json.load(open(args.subspaces_json)) if args.subspaces_json and os.path.isfile(args.subspaces_json) else None

    model = build_model(
        args.model_checkpoint,
        depth=args.depth,
        whitened=wl,
        act_mode=args.act_mode,
        subspaces=subspaces,
        num_classes=args.num_classes,
    )

    if not args.concept_dir or not os.path.isdir(args.concept_dir):
        logging.error("concept_dir missing / not a folder → nothing to do")
        return

    hl_list = [x.strip() for x in args.hl_concepts.split(",") if x.strip()]
    bboxes = args.bboxes_file or os.path.join(args.concept_dir, "bboxes.json")
    ds = build_dataset(args.concept_dir, hl_list, bboxes, args.image_state)

    analyzer = PurityAnalyzer(model, ds, batch_size=args.batch_size, 
                      output_dir=args.output_dir, cfg=args)

    if args.run_purity:
        info = analyzer.compute(subspace_eval=args.subspace_purity)
        bar_plot(info, os.path.join(args.output_dir, "concept_purity.png"))

    if args.topk_images:
        extractor = TopKImageExtractor(analyzer.acts, analyzer.labels, ds,
                                       os.path.join(args.output_dir, "topk_concept_images"), model if args.save_transformed else None)
        extractor.dump(k=args.k, save_transformed=args.save_transformed)

    logging.info("[Done] QCW concept analysis complete.")


if __name__ == "__main__":
    main()
