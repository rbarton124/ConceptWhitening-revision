from __future__ import annotations

# -----------------------------------------------------------------------------#
# Imports & global constants
# -----------------------------------------------------------------------------#
import argparse, io, json, logging, os, shutil
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

# Output subdirectories (kept stable with earlier analyzer)
PUR_SUBDIR  = "purity"
HIER_SUBDIR = "hierarchy"
MAUC_SUBDIR = "masked_auc"
RANK_SUBDIR = "rank_metrics"

TOPK   = 5
LOGFMT = "%(levelname)s | %(asctime)s | %(message)s"

from rank_metrics import run_rank_metrics

# === Legacy-compatible imports: ResNet-only QCW (BN swap) ==================== #
from MODELS.model_resnet_qcw_bn import build_resnet_qcw, get_last_qcw_layer
from MODELS.ConceptDataset_QCW import ConceptDataset

torch.set_printoptions(sci_mode=False)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

# -----------------------------------------------------------------------------#
# Model & data helpers
# -----------------------------------------------------------------------------#
def _load_checkpoint(model: nn.Module, ckpt_path: str) -> nn.Module:
    """
    Load QCW checkpoints from legacy runs. Robust to:
      - DataParallel prefixes ('module.'),
      - 'model.' wrappers,
      - absence of 'backbone.' in older state-dicts,
      - strict=False loading with diagnostics.
    """
    if not ckpt_path or not os.path.isfile(ckpt_path):
        logging.warning("No checkpoint at %s", ckpt_path)
        return model

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw  = ckpt.get("state_dict", ckpt)
    msd  = model.state_dict()
    clean = {}

    # Diagnostics
    shape_mismatched = []
    not_found_in_model = []

    for k, v in raw.items():
        k2 = k.replace("module.", "").replace("model.", "")
        # If model expects 'backbone.' and the ckpt doesn't have it, try adding it
        if not k2.startswith("backbone.") and f"backbone.{k2}" in msd:
            k2 = f"backbone.{k2}"
        # Also catch the common 'fc.' case
        if k2.startswith("fc.") and f"backbone.{k2}" in msd:
            k2 = f"backbone.{k2}"

        if k2 in msd:
            if v.shape == msd[k2].shape:
                clean[k2] = v
            else:
                shape_mismatched.append((k, k2, tuple(v.shape), tuple(msd[k2].shape)))
        else:
            not_found_in_model.append(k)

    model.load_state_dict(clean, strict=False)

    # Report
    logging.info("Loaded %d/%d tensors from %s", len(clean), len(raw), ckpt_path)

    if shape_mismatched:
        logging.info("Modules with shape mismatch (not loaded):")
        for orig_k, model_k, ckpt_shape, model_shape in shape_mismatched[:10]:
            logging.info("  %s -> %s: checkpoint %s vs model %s",
                         orig_k, model_k, ckpt_shape, model_shape)
        if len(shape_mismatched) > 10:
            logging.info("  ...and %d more", len(shape_mismatched) - 10)

    if not_found_in_model:
        logging.info("Checkpoint keys not found in model:")
        for k in not_found_in_model[:10]:
            logging.info("  %s", k)
        if len(not_found_in_model) > 10:
            logging.info("  ...and %d more", len(not_found_in_model) - 10)

    not_in_checkpoint = [k for k in msd.keys() if k not in clean]
    if not_in_checkpoint:
        logging.info("Model keys not found in checkpoint (showing up to 10):")
        for k in not_in_checkpoint[:10]:
            logging.info("  %s", k)

    return model


def build_model(ckpt: str,
                depth: int,
                whitened: List[int],
                act_mode: str,
                subspaces: dict | None,
                num_classes: int,
                cw_lambda: float = 0.05) -> nn.Module:
    """
    Legacy-compatible model builder: ResNet QCW (BN swap) only.
    """
    mdl = build_resnet_qcw(num_classes=num_classes,
                           depth=depth,
                           whitened_layers=whitened,
                           act_mode=act_mode,
                           subspaces=subspaces,
                           use_subspace=True,
                           use_free=False,
                           pretrained_model=None,
                           vanilla_pretrain=False)
    # cw_lambda is part of layer init in training; at analysis time its value
    # is not used, but we keep the parameter for completeness in signatures.
    mdl = _load_checkpoint(mdl, ckpt)
    return nn.DataParallel(mdl).cuda().eval()


def build_dataset(root: str, hl_filter: Sequence[str],
                  bboxes: str, crop_mode: str) -> ConceptDataset:
    tfm = T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225])
    ])
    return ConceptDataset(root_dir=os.path.join(root, "concept_val"),
                          bboxes_file=bboxes,
                          high_level_filter=list(hl_filter),
                          transform=tfm,
                          crop_mode=crop_mode)

# -----------------------------------------------------------------------------#
# Activation collector
# -----------------------------------------------------------------------------#
class _Collector:
    """Grab channel-averaged activations from the last QCW layer (legacy safe)."""
    def __init__(self, model: nn.Module, K: int):
        self.out = None
        self.K = K
        # resolve last QCW layer (handles old/new shapes)
        last = get_last_qcw_layer(model.module if isinstance(model, nn.DataParallel) else model)

        def _hook(_, __, y):
            # y: [B, C, H, W] -> mean spatial -> [B, C]
            # Clip to at most K axes to match number of subconcepts
            C = y.shape[1]
            clip = min(self.K, C)
            val = y.mean((2, 3))[:, :clip].detach().cpu()
            self.out = val

        last.register_forward_hook(_hook)

    def run(self, model, loader, device="cuda"):
        acts, labels = [], []
        for imgs, lbl, _ in tqdm(loader, desc="Collecting Activations"):
            _ = model(imgs.to(device))
            acts.append(self.out.numpy())
            labels.append(lbl.numpy())
        return np.concatenate(acts), np.concatenate(labels)

# -----------------------------------------------------------------------------#
# Purity / hierarchy metrics
# -----------------------------------------------------------------------------#
def _auc(mask: np.ndarray, scores: np.ndarray) -> float:
    if mask.sum() in {0, len(mask)}:
        return float("nan")
    return roc_auc_score(mask.astype(int), scores)


class PurityAnalyzer:
    """Compute classic purity + Δ_max & EnergyRatio + masked-AUC (optional)."""
    def __init__(self, model, ds: ConceptDataset, bs=64, device="cuda",
                 out_dir="qcw_plots", cfg: argparse.Namespace | None = None):
        self.cfg = cfg or argparse.Namespace(
            auc_max=False, energy_ratio=False, masked_auc=False, verbose=False
        )
        self.ds, self.model, self.device, self.out_dir = ds, model, device, out_dir
        self.sc_names = ds.get_subconcept_names()
        self.K = len(self.sc_names)

        # collect activations once
        loader = DataLoader(ds, bs, shuffle=False, pin_memory=True)
        acts, y = _Collector(model, self.K).run(model, loader, device)
        self.acts, self.y = acts, y  # shapes: [N, <=K], [N]

        # hierarchy maps
        self.subspace = ds.subspace_mapping                       # HL → [axis,...]
        self.sc2hl    = ds.get_subconcept_to_hl_name_mapping()    # axis → HL
        self.labeled  = {i for i, n in enumerate(self.sc_names)
                         if not ds.is_free_subconcept_name(n)}

    def compute(self) -> Dict[str, dict]:
        info = {}

        # Optional caches for Δ_max
        global_max = self.acts.max(1) if self.cfg.auc_max else None
        hl_max_cache = {hl: self.acts[:, ax].max(1) if self.cfg.auc_max else None
                        for hl, ax in self.subspace.items()}

        # For masked AUC
        labels_hl = None
        if self.cfg.masked_auc:
            labels_hl = np.array([self.sc2hl.get(int(l), "") for l in self.y], dtype=object)

        for idx, name in enumerate(self.sc_names):
            hl = self.sc2hl.get(idx, None)
            mask = (self.y == idx)
            if not mask.any():
                info[name] = {"best_axis_auc": float("nan")}
                continue

            # Activation means across axes for this subclass
            means = self.acts[mask].mean(0)

            # Decide primary axis
            if idx in self.labeled:
                prim_ax = idx
                prim_auc = _auc(mask, self.acts[:, prim_ax]) if prim_ax < self.acts.shape[1] else float("nan")
                leak_idx = leak_auc = None
                unlabeled_axis_idx = unlabeled_axis_auc = None
            else:
                # choose the best *unlabeled* axis for free concepts
                unlabeled = [a for a in range(self.acts.shape[1]) if a not in self.labeled]
                if len(unlabeled) == 0:
                    prim_ax, prim_auc = None, float("nan")
                else:
                    # among available axes
                    prim_ax = unlabeled[int(np.argmax(means[unlabeled]))]
                    prim_auc = _auc(mask, self.acts[:, prim_ax])

                # leak detection (labeled axis outperforming the unlabeled one)
                leak_idx = leak_auc = None
                if prim_ax is not None:
                    best_g = int(np.argmax(means))
                    if best_g != prim_ax and best_g in self.labeled and best_g < self.acts.shape[1]:
                        leak_idx, leak_auc = best_g, _auc(mask, self.acts[:, best_g])

                unlabeled_axis_idx, unlabeled_axis_auc = prim_ax, prim_auc

            # Masked AUC (restrict negatives to same HL slice)
            if self.cfg.masked_auc and prim_ax is not None and hl is not None:
                scores = np.copy(self.acts[:, prim_ax])
                scores[labels_hl != hl] = 0.0
                hier_auc = _auc(mask, scores)
            else:
                hier_auc = None

            # Δ-max & Energy ratio
            if self.cfg.auc_max and hl in self.subspace:
                auc_gmax  = _auc(mask, global_max)
                auc_hlmax = _auc(mask, hl_max_cache[hl])
                delta_max = auc_hlmax - auc_gmax
            else:
                auc_gmax = auc_hlmax = delta_max = None

            if self.cfg.energy_ratio and hl in self.subspace:
                e_full = np.linalg.norm(self.acts[mask], axis=1).mean()
                e_hl   = np.linalg.norm(self.acts[mask][:, self.subspace[hl]], axis=1).mean()
                e_ratio = e_hl / (e_full + 1e-9)
            else:
                e_ratio = None

            info[name] = {
                "is_free": idx not in self.labeled,
                "best_axis_idx": prim_ax if prim_ax is not None else -1,
                "best_axis_auc": float(prim_auc) if prim_auc is not None else float("nan"),
                "best_axis_is_labeled": (prim_ax in self.labeled) if prim_ax is not None else False,
                "unlabeled_axis_idx": unlabeled_axis_idx if idx not in self.labeled else None,
                "unlabeled_axis_auc": float(unlabeled_axis_auc) if (idx not in self.labeled and unlabeled_axis_auc is not None) else None,
                "leak_axis_idx": leak_idx,
                "leak_axis_auc": float(leak_auc) if leak_auc is not None else None,
                "auc_max_global": float(auc_gmax) if auc_gmax is not None else None,
                "auc_max_hl": float(auc_hlmax) if auc_hlmax is not None else None,
                "delta_max": float(delta_max) if delta_max is not None else None,
                "energy_ratio": float(e_ratio) if e_ratio is not None else None,
                "baseline_auc": float(prim_auc) if prim_auc is not None else None,
                "hier_auc": float(hier_auc) if hier_auc is not None else None,
            }

            if self.cfg.verbose and not getattr(self, "_printed", False) and hl in self.subspace:
                self._printed = True
                kept = np.linalg.norm(self.acts[mask][:, self.subspace[hl]], axis=1).mean()
                full = np.linalg.norm(self.acts[mask], axis=1).mean()
                logging.info("▶ %s  kept %.1f%% energy", name, 100.0 * kept / (full + 1e-9))

        # dump CSV (detailed)
        os.makedirs(self.out_dir, exist_ok=True)
        pd.DataFrame(info).T.to_csv(os.path.join(self.out_dir, "result.csv"))
        return info

# -----------------------------------------------------------------------------#
# Top-k image extractor
# -----------------------------------------------------------------------------#
class TopKExtractor:
    def __init__(self, acts, labels, ds, out_dir, model=None):
        self.acts, self.labels, self.ds, self.out = acts, labels, ds, out_dir
        self.model = model

    def dump(self, k=10, save_transformed=False):
        from torchvision.utils import save_image
        os.makedirs(self.out, exist_ok=True)
        mean = torch.tensor([0.485, 0.456, 0.406])
        std  = torch.tensor([0.229, 0.224, 0.225])

        labeled = {i for i, n in enumerate(self.ds.get_subconcept_names())
                   if not self.ds.is_free_subconcept_name(n)}

        # Optionally re-load transformed tensors once
        imgs_all = None
        if save_transformed:
            loader = DataLoader(self.ds, 64, shuffle=False, pin_memory=True)
            imgs_all = torch.cat([b[0] for b in loader])

        for idx, scn in enumerate(self.ds.get_subconcept_names()):
            # Choose primary axis for plotting
            ax = idx if idx in labeled else self._best_unlabeled(idx, labeled)
            if ax is None or ax >= self.acts.shape[1]:
                continue

            top_idx = self.acts[:, ax].argsort()[::-1][:k]
            dst_dir = os.path.join(self.out, scn.replace("/", "_"))
            os.makedirs(dst_dir, exist_ok=True)

            for rnk, j in enumerate(top_idx):
                dst = os.path.join(dst_dir, f"rank_{rnk+1}.jpg")
                if save_transformed and imgs_all is not None:
                    img = imgs_all[j].clone()
                    img.mul_(std[:, None, None]).add_(mean[:, None, None]).clamp_(0, 1)
                    save_image(img.cpu(), dst)
                else:
                    shutil.copy2(self.ds.samples[j][0], dst)
        logging.info("Top-k images saved to %s", self.out)

    def _best_unlabeled(self, c_idx, labeled):
        mask = self.labels == c_idx
        if not mask.any():
            return None
        means = self.acts[mask].mean(0)
        for ax in means.argsort()[::-1]:
            if int(ax) not in labeled:
                return int(ax)
        return None

# -----------------------------------------------------------------------------#
# Plots (kept same structure; minor guards around NaNs)
# -----------------------------------------------------------------------------#
def bar_plot(info: Dict[str, dict], out_path: str, show_values: bool = True):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import matplotlib.patches as mpatches

    df = pd.DataFrame(info).T
    df = df.sort_values("best_axis_auc", ascending=False).reset_index()
    N  = len(df)

    prim_col = df["best_axis_is_labeled"].map({True: "red", False: "blue"})
    leak_col = np.where(df["leak_axis_idx"].notna(), "red", "blue")
    fig, ax = plt.subplots(figsize=(max(6, N), 6))

    ax.bar(np.arange(N), df["best_axis_auc"], width=.6, color=prim_col, label="Primary axis")

    leak_mask = df["leak_axis_auc"].notna()
    ax.bar(np.where(leak_mask)[0] + .25,
           df.loc[leak_mask, "leak_axis_auc"],
           width=.25, color="red", alpha=.8, label="Labeled leak axis")

    if show_values:
        for i, (y, idx) in enumerate(zip(df["best_axis_auc"], df["best_axis_idx"])):
            if not pd.isna(y) and not pd.isna(idx):
                ax.text(i, y + .01, f"A{int(idx)}:{y:.2f}", ha="center", va="bottom", fontsize=8)
        for i in np.where(leak_mask)[0]:
            y2, idx2 = df.loc[i, ["leak_axis_auc", "leak_axis_idx"]]
            if not pd.isna(y2) and not pd.isna(idx2):
                ax.text(i + .25, y2 + .01, f"A{int(idx2)}:{y2:.2f}", ha="center",
                        va="bottom", fontsize=8, color="red")

    ax.set_xticks(np.arange(N))
    ax.set_xticklabels(df["index"], rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("AUC (concept purity)")
    ax.set_title("Concept Purity")
    ax.legend(handles=[
        mpatches.Patch(color="blue", label="Unlabeled axis"),
        mpatches.Patch(color="red",  label="Labeled axis"),
    ])

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close()
    logging.info("Bar-plot saved to %s", out_path)


def hierarchy_plot(info: Dict[str, dict], out_path: str,
                   show_delta: bool = True, show_energy: bool = True):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    if not (show_delta or show_energy):
        logging.info("No hierarchy metrics requested → skip extra plot.")
        return

    df = pd.DataFrame(info).T
    cols = []
    if show_delta  and "delta_max"   in df.columns: cols.append("delta_max")
    if show_energy and "energy_ratio" in df.columns: cols.append("energy_ratio")
    if not cols:
        logging.warning("Hierarchy metrics absent in `info` → skip plot.")
        return

    df = df.sort_values("best_axis_auc", ascending=False).reset_index()
    n  = len(df)
    fig_rows = len(cols)
    fig, axes = plt.subplots(fig_rows, 1, figsize=(max(6, n), 3*fig_rows), sharex=True)
    if fig_rows == 1:
        axes = [axes]

    colors = ["steelblue" if fr else "orange" for fr in df["is_free"]]

    for ax, metric in zip(axes, cols):
        ax.bar(np.arange(n), df[metric], color=colors)
        ax.set_ylabel("Δ-max" if metric=="delta_max" else "Energy ratio")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", ls=":", alpha=.4)

    axes[-1].set_xticks(np.arange(n))
    axes[-1].set_xticklabels(df["index"], rotation=45, ha="right")
    axes[0].set_title("Hierarchy metrics (slice vs. global)")

    axes[0].legend(handles=[
        plt.Rectangle((0,0),1,1,color="steelblue", label="free concept"),
        plt.Rectangle((0,0),1,1,color="orange",    label="labelled concept"),
    ], loc="upper right")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close()
    logging.info("Hierarchy plot saved to %s", out_path)


def masked_auc_plot(info: Dict[str, dict], out_path: str):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    df = pd.DataFrame(info).T
    if "hier_auc" not in df.columns or df["hier_auc"].isna().all():
        logging.info("masked_auc flag off or no values → skipping hier-AUC plot.")
        return

    df = df.sort_values("baseline_auc", ascending=False).reset_index()
    n  = len(df)
    x  = np.arange(n)
    w  = 0.35

    plt.figure(figsize=(max(6, n), 5))
    plt.bar(x - w/2, df["baseline_auc"], width=w, label="Baseline AUC",  color="grey")
    plt.bar(x + w/2, df["hier_auc"],     width=w, label="Masked AUC",    color="teal")
    plt.axhline(0.5, color="k", ls=":", lw=.8)

    plt.xticks(x, df["index"], rotation=45, ha="right")
    plt.ylabel("ROC-AUC")
    plt.ylim(0, 1)
    plt.title("Effect of HL masking on axis purity")
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close()
    logging.info("Masked-AUC plot saved to %s", out_path)

# -----------------------------------------------------------------------------#
# CLI
# -----------------------------------------------------------------------------#
def get_args():
    p = argparse.ArgumentParser("QCW concept analyzer (legacy ResNet-only)")
    # model
    p.add_argument("--model_checkpoint", required=True)
    p.add_argument("--depth", type=int, default=18,
                   help="18/50 for ResNet.")
    p.add_argument("--whitened_layers", default="5",
                   help="Comma-separated global BN indices to replace (e.g. '5' or '2,5').")
    p.add_argument("--act_mode", default="pool_max")
    p.add_argument("--cw_lambda", type=float, default=0.05,
                   help="λ used during training; kept for completeness.")
    p.add_argument("--num_classes", type=int, default=200)
    p.add_argument("--subspaces_json", default="",
                   help="Optional mapping HL→axes used during training; not required for analysis.")
    # data
    p.add_argument("--concept_dir", required=True)
    p.add_argument("--hl_concepts", default="",
                   help="Comma-separated list of HL concepts to include (e.g. 'wing,beak,general').")
    p.add_argument("--bboxes_file", default="")
    p.add_argument("--image_state", default="crop", choices=["crop","redact","blur","none"])
    # run
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--run_purity", action="store_true")
    p.add_argument("--topk_images", action="store_true")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--save_transformed", action="store_true")
    p.add_argument("--auc_max", action="store_true")      # Δ-max metric
    p.add_argument("--energy_ratio", action="store_true") # energy metric
    p.add_argument("--masked_auc", action="store_true",
                   help="Compute AUC after zero-filling activations outside each HL slice.")
    p.add_argument("--rank_metrics", action="store_true",
                   help="Compute mean-rank / Hit@k hierarchy metric.")
    p.add_argument("--verbose", action="store_true")
    # out
    p.add_argument("--output_dir", default="qcw_plots")
    return p.parse_args()

# -----------------------------------------------------------------------------#
# Main
# -----------------------------------------------------------------------------#
def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO, format=LOGFMT)

    # Parse whitened layers and validate against ResNet depth (global BN indices)
    wl = [int(x) for x in args.whitened_layers.split(",") if x.strip()]
    max_idx = 8 if args.depth == 18 else 16
    bad = [i for i in wl if i < 1 or i > max_idx]
    if bad:
        raise ValueError(f"whitened_layers {bad} out of range 1..{max_idx} for ResNet-{args.depth}.")

    # Subspaces JSON (optional)
    subspaces = None
    if args.subspaces_json and os.path.isfile(args.subspaces_json):
        with open(args.subspaces_json, "r") as f:
            subspaces = json.load(f)

    # Build model (legacy ResNet QCW) and dataset
    model = build_model(args.model_checkpoint, args.depth, wl, args.act_mode,
                        subspaces, args.num_classes, cw_lambda=args.cw_lambda)

    hl_list = [s.strip() for s in args.hl_concepts.split(",") if s.strip()]
    bboxes  = args.bboxes_file or os.path.join(args.concept_dir, "bboxes.json")
    ds      = build_dataset(args.concept_dir, hl_list, bboxes, args.image_state)

    print("\n================ SUB-CONCEPT → AXIS MAP ================")
    for sc_name in ds.get_subconcept_names():
        idx = ds.sc2idx[sc_name]
        print(f"  [{idx:>3}]  {sc_name}")
    print("========================================================\n")

    print("=========== HIGH-LEVEL SUBSPACE LAYOUT ===========")
    for hl, axes in ds.subspace_mapping.items():
        axes_str = ", ".join(map(str, axes))
        print(f"  {hl:<12}:  [{axes_str}]")
    print("==================================================\n")

    analyzer = PurityAnalyzer(model, ds, bs=args.batch_size,
                              out_dir=args.output_dir, cfg=args)

    info = None
    rank_results = None

    if args.run_purity:
        info = analyzer.compute()

        # Purity plots
        purity_dir = os.path.join(args.output_dir, PUR_SUBDIR)
        os.makedirs(purity_dir, exist_ok=True)
        bar_plot(info, os.path.join(purity_dir, "concept_purity.png"))
        pd.DataFrame({k: {
            "best_axis_auc": v.get("best_axis_auc"),
            "best_axis_idx": v.get("best_axis_idx"),
            "best_axis_is_labeled": v.get("best_axis_is_labeled")}
        for k, v in info.items()}).T.to_csv(os.path.join(purity_dir, "purity.csv"))

        # Hierarchy (Δ-max / energy)
        if args.auc_max or args.energy_ratio:
            hier_dir = os.path.join(args.output_dir, HIER_SUBDIR)
            os.makedirs(hier_dir, exist_ok=True)
            hierarchy_plot(info, os.path.join(hier_dir, "hierarchy_metrics.png"),
                           show_delta=args.auc_max, show_energy=args.energy_ratio)
            cols = []
            if args.auc_max: cols.append("delta_max")
            if args.energy_ratio: cols.append("energy_ratio")
            if cols:
                pd.DataFrame(info).T[cols].to_csv(os.path.join(hier_dir, "hierarchy.csv"))

        # Masked AUC
        if args.masked_auc:
            mauc_dir = os.path.join(args.output_dir, MAUC_SUBDIR)
            os.makedirs(mauc_dir, exist_ok=True)
            masked_auc_plot(info, os.path.join(mauc_dir, "masked_vs_baseline_auc.png"))
            pd.DataFrame(info).T[["baseline_auc", "hier_auc"]].to_csv(
                os.path.join(mauc_dir, "masked_auc.csv"))

        # Rank metrics (optional)
        if args.rank_metrics:
            axis_map = {i: d["best_axis_idx"] for i, d in enumerate(info.values())}
            concept_names = list(info.keys())
            # Identify labeled axes for coloring
            labeled_axes = []
            for hl, axes in analyzer.subspace.items():
                if hl != "_free":
                    labeled_axes.extend(axes)

            rank_results = run_rank_metrics(
                analyzer.acts,
                analyzer.y,
                analyzer.subspace,
                analyzer.sc2hl,
                axis_map,
                os.path.join(args.output_dir, RANK_SUBDIR),
                concept_names=concept_names,
                labeled_axes=labeled_axes
            )

    if args.topk_images:
        TopKExtractor(analyzer.acts, analyzer.y, ds,
                      os.path.join(args.output_dir, "topk_concept_images")).dump(
            k=args.k, save_transformed=args.save_transformed
        )

    # ------------------------------------------------------------------#
    # Combined results markdown + CSV (if purity was run)
    # ------------------------------------------------------------------#
    if info is not None:
        info_df = pd.DataFrame(info).T

        # Strip global-AUC helper columns that clutter the summary
        info_df = info_df.drop(columns=[c for c in
                            ("auc_max_global", "auc_max_hl", "baseline_auc")
                            if c in info_df.columns],
                            errors="ignore")

        if rank_results is not None:
            rank_df = pd.DataFrame(rank_results).T
            desired_metrics = ["mean_rank_raw", "mean_rank_mask", "hit@3_raw", "hit@3_mask"]
            keep_cols = [c for c in desired_metrics if c in rank_df.columns]
            rank_df = rank_df[keep_cols]
            # Align names to info_df
            rank_df.index = [list(info.keys())[i] for i in rank_df.index]
            info_df = info_df.join(rank_df, how="left")

        # Normalize numerics and rounding
        info_df = info_df.replace({None: np.nan})
        num_cols = info_df.select_dtypes(include=["number"]).columns
        info_df[num_cols] = info_df[num_cols].astype(float).round(3)

        # Write markdown + CSV
        csv_buffer = io.StringIO()
        info_df.to_csv(csv_buffer, float_format="%.3g")
        csv_content = csv_buffer.getvalue()

        md_path = os.path.join(args.output_dir, "results.md")
        with open(md_path, "w") as f:
            f.write(f"# QCW Model Analysis Results (Legacy)\n\n")
            f.write(f"## Model Information\n\n")
            f.write(f"- **Model:** ResNet-{args.depth}\n")
            f.write(f"- **Checkpoint:** {args.model_checkpoint}\n")
            f.write(f"- **Whitened Layers:** {args.whitened_layers}\n")
            f.write(f"- **CW Lambda:** {args.cw_lambda}\n")
            f.write(f"- **Concept Directory:** {args.concept_dir}\n")
            f.write(f"- **HL Concepts:** {args.hl_concepts}\n")
            f.write(f"- **Image State:** {args.image_state}\n\n")
            f.write(f"## Metrics Data\n\n")
            f.write(f"```csv\n{csv_content}```\n")

        csv_path = os.path.join(args.output_dir, "result.csv")
        info_df.to_csv(csv_path, float_format="%.3g")

        print(f"[Summary] Markdown results → {md_path}")
        print(f"[Summary] CSV results     → {csv_path}")

if __name__ == "__main__":
    main()
