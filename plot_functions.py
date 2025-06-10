from __future__ import annotations

# -----------------------------------------------------------------------------#
# Imports & global constants
# -----------------------------------------------------------------------------#
import argparse, json, logging, math, os, shutil
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np, pandas as pd, torch
import torch.nn as nn
import torchvision.transforms as T
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from rank_metrics import run_rank_metrics  # NEW helper

# Output subdirectories
PUR_SUBDIR = "purity"
HIER_SUBDIR = "hierarchy"
MAUC_SUBDIR = "masked_auc"
RANK_SUBDIR = "rank_metrics"

from MODELS.model_resnet_qcw import build_resnet_qcw, get_last_qcw_layer
from MODELS.ConceptDataset_QCW import ConceptDataset

torch.set_printoptions(sci_mode=False)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

TOPK   = 5
LOGFMT = "%(levelname)s | %(asctime)s | %(message)s"

# -----------------------------------------------------------------------------#
# Model & data helpers
# -----------------------------------------------------------------------------#
def _load_checkpoint(model: nn.Module, ckpt_path: str) -> nn.Module:
    """Load QCW checkpoints, coping with DataParallel prefixes."""
    if not ckpt_path or not os.path.isfile(ckpt_path):
        logging.warning("No checkpoint at %s", ckpt_path);  return model

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw  = ckpt.get("state_dict", ckpt)
    msd  = model.state_dict(); clean = {}
    
    # Track keys that couldn't be loaded
    shape_mismatched = []
    not_found_in_model = []
    
    # Process checkpoint keys
    for k, v in raw.items():
        k2 = k.replace("module.", "").replace("model.", "")
        if not k2.startswith("backbone.") and f"backbone.{k2}" in msd:   k2 = f"backbone.{k2}"
        if k2.startswith("fc.")        and f"backbone.{k2}" in msd:      k2 = f"backbone.{k2}"
        
        if k2 in msd:
            if v.shape == msd[k2].shape:
                clean[k2] = v
            else:
                shape_mismatched.append((k, k2, str(v.shape), str(msd[k2].shape)))
        else:
            not_found_in_model.append(k)
    
    # Track model keys not found in checkpoint
    not_in_checkpoint = [k for k in msd.keys() if k not in clean]
    
    # Load the state dict
    model.load_state_dict(clean, strict=False)
    
    # Log loading statistics
    logging.info("Loaded %d/%d tensors from %s", len(clean), len(raw), ckpt_path)
    
    # Log detailed information about unloaded modules
    if shape_mismatched:
        logging.info("Modules with shape mismatch (not loaded):")
        for orig_k, model_k, ckpt_shape, model_shape in shape_mismatched[:10]:
            logging.info("  %s -> %s: checkpoint shape %s, model shape %s", orig_k, model_k, ckpt_shape, model_shape)
        if len(shape_mismatched) > 10:
            logging.info("  ...and %d more", len(shape_mismatched) - 10)
    
    if not_found_in_model:
        logging.info("Checkpoint keys not found in model:")
        for k in not_found_in_model[:10]:
            logging.info("  %s", k)
        if len(not_found_in_model) > 10:
            logging.info("  ...and %d more", len(not_found_in_model) - 10)
    
    if not_in_checkpoint:
        logging.info("Model keys not found in checkpoint:")
        for k in not_in_checkpoint[:10]:
            logging.info("  %s", k)
        if len(not_in_checkpoint) > 10:
            logging.info("  ...and %d more", len(not_in_checkpoint) - 10)
    
    return model


def build_model(ckpt: str, depth: int, whitened: List[int], act_mode: str,
                subspaces: dict | None, num_classes: int) -> nn.Module:
    mdl = build_resnet_qcw(num_classes=num_classes, depth=depth,
                           whitened_layers=whitened, act_mode=act_mode,
                           subspaces=subspaces)
    mdl = _load_checkpoint(mdl, ckpt)
    return nn.DataParallel(mdl).cuda().eval()


def build_dataset(root: str, hl_filter: Sequence[str],
                  bboxes: str, crop_mode: str) -> ConceptDataset:
    tfm = T.Compose([
        T.RandomResizedCrop(224), T.RandomHorizontalFlip(),
        T.ToTensor(), T.Normalize([0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225])])
    return ConceptDataset(root_dir=os.path.join(root, "concept_val"),
                          bboxes_file=bboxes, high_level_filter=list(hl_filter),
                          transform=tfm, crop_mode=crop_mode)

# -----------------------------------------------------------------------------#
# Activation collector
# -----------------------------------------------------------------------------#
class _Collector:
    """Small helper to grab activations from the last QCW layer."""
    def __init__(self, model: nn.Module, K: int):
        self.out = None
        layer = get_last_qcw_layer(model.module if isinstance(model, nn.DataParallel) else model)
        layer.register_forward_hook(lambda _,__,y: setattr(self, "out", y.mean((2,3))[:,:K].cpu()))

    def run(self, model, loader, device="cuda"):
        acts, labels = [], []
        for imgs, lbl, _ in tqdm(loader, desc="Collecting Activations"):
            _ = model(imgs.to(device));  acts.append(self.out.detach().cpu().numpy()); labels.append(lbl.numpy())
        return np.concatenate(acts), np.concatenate(labels)

# -----------------------------------------------------------------------------#
# Purity / hierarchy metrics
# -----------------------------------------------------------------------------#
def _auc(mask: np.ndarray, scores: np.ndarray) -> float:
    if mask.sum() in {0, len(mask)}:  return float("nan")
    return roc_auc_score(mask.astype(int), scores)


class PurityAnalyzer:
    """Compute classic purity + Δ_max & EnergyRatio."""
    def __init__(self, model, ds: ConceptDataset, bs=64, device="cuda",
                 out_dir="qcw_plots", cfg: argparse.Namespace | None = None):
        self.cfg = cfg or argparse.Namespace(auc_max=False, energy_ratio=False, verbose=False)
        self.ds, self.model, self.device, self.out_dir = ds, model, device, out_dir
        self.sc_names = ds.get_subconcept_names();  self.K = len(self.sc_names)

        # collect activations once
        loader  = DataLoader(ds, bs, shuffle=False, pin_memory=True)
        acts, y = _Collector(model, self.K).run(model, loader, device)
        self.acts, self.y = acts, y

        # hierarchy maps
        self.subspace = ds.subspace_mapping                 # HL → list[axis]
        self.sc2hl    = ds.get_subconcept_to_hl_name_mapping()  # axis → HL
        self.labeled  = {i for i,n in enumerate(self.sc_names) if not ds.is_free_subconcept_name(n)}

    # -------------------------------------------------------------------------#
    def compute(self) -> Dict[str, dict]:
        info = {}
        global_max = self.acts.max(1) if self.cfg.auc_max else None
        hl_max_cache = {hl: self.acts[:, ax].max(1) if self.cfg.auc_max else None
                        for hl, ax in self.subspace.items()}
                        
        # Map each sample to its high-level concept name (for masked-AUC)
        if self.cfg.masked_auc:
            labels_hl = np.array([self.sc2hl.get(int(l), "") for l in self.y], dtype=object)
        

        for idx, name in enumerate(self.sc_names):
            hl, mask = self.sc2hl[idx], (self.y == idx)
            axes_hl  = self.subspace[hl]
            if not mask.any():  # skip empty subclasses
                info[name] = {"best_axis_auc": float("nan")};  continue

            means = self.acts[mask].mean(0)

            # ---------------- primary / secondary axes -----------------------
            if idx in self.labeled:               # labeled concept → axis = itself
                prim_ax, prim_auc = idx, _auc(mask, self.acts[:, idx])
                leak_idx = leak_auc = None
                # For labeled, 'unlabeled_axis_*' is not meaningful
                unlabeled_axis_idx = unlabeled_axis_auc = None
            else:                                 # free concept → best unlabeled axis
                unlabeled = [a for a in range(self.K) if a not in self.labeled]
                prim_ax = max(unlabeled, key=means.__getitem__)
                prim_auc = _auc(mask, self.acts[:, prim_ax])
                # If a labeled axis leaks (outperforms the unlabeled), record it as leak
                best_g = int(means.argmax())
                if best_g != prim_ax and best_g in self.labeled:
                    leak_idx, leak_auc = best_g, _auc(mask, self.acts[:, best_g])
                else:
                    leak_idx = leak_auc = None
                unlabeled_axis_idx, unlabeled_axis_auc = prim_ax, prim_auc
            
            # -------- masked-AUC ("hier-AUC") --------------------------------
            if self.cfg.masked_auc:
                # Start with a copy of the scores from the primary axis
                scores = np.copy(self.acts[:, prim_ax])
                
                # Zero out scores for samples from different high-level concepts
                # (keep scores only for samples belonging to this concept's high-level group)
                scores[labels_hl != hl] = 0.0
                
                # Calculate AUC with the masked scores
                hier_auc = _auc(mask, scores)
            else:
                hier_auc = None

            # ---------------- hierarchy metrics ------------------------------
            if self.cfg.auc_max:
                auc_gmax   = _auc(mask, global_max)
                auc_hlmax  = _auc(mask, hl_max_cache[hl])
                delta_max  = auc_hlmax - auc_gmax  # Positive delta means HL subspace outperforms global
            else:
                auc_gmax = auc_hlmax = delta_max = None

            if self.cfg.energy_ratio:
                e_full = np.linalg.norm(self.acts[mask],     axis=1).mean()
                e_hl   = np.linalg.norm(self.acts[mask][:, axes_hl], axis=1).mean()
                e_ratio = e_hl / (e_full + 1e-9)
            else:
                e_ratio = None

            # store
            # CSV columns: 'unlabeled_axis_*' is always the primary (blue) for free concepts; 'leak_axis_*' is the labeled leak (red) if present.
            info[name] = {
                "is_free": idx not in self.labeled,
                "best_axis_idx": prim_ax,
                "best_axis_auc": prim_auc,
                "best_axis_is_labeled": prim_ax in self.labeled,
                # For labeled concepts, these are None
                "unlabeled_axis_idx": unlabeled_axis_idx if idx not in self.labeled else None,
                "unlabeled_axis_auc": unlabeled_axis_auc if idx not in self.labeled else None,
                # For free concepts, leak axis is present if a labeled axis leaks
                "leak_axis_idx": leak_idx,
                "leak_axis_auc": leak_auc,
                "auc_max_global": auc_gmax, "auc_max_hl": auc_hlmax,
                "delta_max": delta_max,    "energy_ratio": e_ratio,
                # Masked AUC comparison
                "baseline_auc": prim_auc,
                "hier_auc": hier_auc,
            }
            # Dropped CSV columns: best_ax_hl, auc_hl, mass_global, mass_hl, mass_retained_pct, topk_global (see patch guide for rationale)


            # optional terse diagnostics
            if self.cfg.verbose and not getattr(self, "_printed", False):
                self._printed = True
                kept = np.linalg.norm(self.acts[mask][:, axes_hl], axis=1).mean()
                full = np.linalg.norm(self.acts[mask],            axis=1).mean()
                logging.info("▶ %s  kept %.1f%% energy", name, 100*kept/(full+1e-9))

        # dump CSV
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
        mean, std = torch.tensor([0.485,0.456,0.406]), torch.tensor([0.229,0.224,0.225])
        labeled = {i for i,n in enumerate(self.ds.get_subconcept_names())
                   if not self.ds.is_free_subconcept_name(n)}

        # re-load tensors once if we need transformed images
        imgs_all = None
        if save_transformed:
            loader = DataLoader(self.ds, 64, shuffle=False, pin_memory=True)
            imgs_all = torch.cat([b[0] for b in loader])

        for idx, scn in enumerate(self.ds.get_subconcept_names()):
            ax = idx if idx in labeled else self._best_unlabeled(idx, labeled)
            if ax is None: continue
            top_idx = self.acts[:, ax].argsort()[::-1][:k]
            dst_dir = os.path.join(self.out, scn.replace("/", "_"));  os.makedirs(dst_dir, exist_ok=True)
            for rnk, j in enumerate(top_idx):
                dst = os.path.join(dst_dir, f"rank_{rnk+1}.jpg")
                if save_transformed:
                    img = imgs_all[j].clone()
                    img.mul_(std[:,None,None]).add_(mean[:,None,None]).clamp_(0,1)
                    save_image(img.cpu(), dst)
                else:
                    shutil.copy2(self.ds.samples[j][0], dst)
        logging.info("Top-k images saved to %s", self.out)

    def _best_unlabeled(self, c_idx, labeled):
        mask = self.labels == c_idx
        if not mask.any(): return None
        means = self.acts[mask].mean(0)
        for ax in means.argsort()[::-1]:
            if ax not in labeled: return int(ax)
        return None

# -----------------------------------------------------------------------------#
# Plots
# -----------------------------------------------------------------------------#
def bar_plot(info: Dict[str, dict], out_path: str, show_values: bool = True):
    import matplotlib.pyplot as plt, numpy as np, pandas as pd, math, matplotlib.patches as mpatches

    df = pd.DataFrame(info).T
    df = df.sort_values("best_axis_auc", ascending=False).reset_index()
    N  = len(df)

    prim_col = df["best_axis_is_labeled"].map({True: "red", False: "blue"})
    leak_col  = np.where(df["leak_axis_idx"].notna(), "red", "blue") # Leak bar is red if present, else blue
    fig, ax = plt.subplots(figsize=(max(6, N), 6))

    ax.bar(np.arange(N), df["best_axis_auc"], width=.6, color=prim_col, label="Primary axis") # Primary bars

    # Leak bars
    leak_mask = df["leak_axis_auc"].notna()
    ax.bar(np.where(leak_mask)[0] + .25,
           df.loc[leak_mask, "leak_axis_auc"],
           width=.25, color="red", alpha=.8, label="Labeled leak axis")

    # Value labels
    if show_values:
        for i, (y, idx) in enumerate(zip(df["best_axis_auc"], df["best_axis_idx"])):
            ax.text(i, y + .01, f"A{idx}:{y:.2f}", ha="center", va="bottom", fontsize=8)
        for i in np.where(leak_mask)[0]:
            y2, idx2 = df.loc[i, ["leak_axis_auc", "leak_axis_idx"]]
            if not pd.isna(y2) and not pd.isna(idx2):
                ax.text(i + .25, y2 + .01, f"A{int(idx2)}:{y2:.2f}", ha="center",
                        va="bottom", fontsize=8, color="red")

    # Cosmetics
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

# -----------------------------------------------------------------------------#
# Hierarchy plot (Δ-max  &  Energy-ratio)
# -----------------------------------------------------------------------------#
def hierarchy_plot(info: Dict[str, dict], out_path: str,
                   show_delta: bool = True, show_energy: bool = True):
    if not (show_delta or show_energy):
        logging.info("No hierarchy metrics requested → skip extra plot.")
        return

    import pandas as pd, numpy as np, matplotlib.pyplot as plt

    df = pd.DataFrame(info).T
    cols = []
    if show_delta  and "delta_max"   in df.columns: cols.append("delta_max")
    if show_energy and "energy_ratio" in df.columns: cols.append("energy_ratio")
    if not cols:
        logging.warning("Hierarchy metrics absent in `info` → skip plot.")
        return

    # Same ordering as purity-bars for easy cross-reading
    df = df.sort_values("best_axis_auc", ascending=False).reset_index()
    n  = len(df)

    n_sub = len(cols)
    fig, axes = plt.subplots(n_sub, 1, figsize=(max(6, n), 3*n_sub),
                             sharex=True)

    if n_sub == 1:
        axes = [axes]                       # make iterable

    colors = ["steelblue" if fr else "orange"
              for fr in df["is_free"]]     # colour free vs labelled concepts

    for ax, metric in zip(axes, cols):
        ax.bar(np.arange(n), df[metric], color=colors)
        ax.set_ylabel(("Δ-max" if metric=="delta_max" else "Energy ratio"))
        ax.set_ylim(0, 1)
        ax.grid(axis="y", ls=":", alpha=.4)

    axes[-1].set_xticks(np.arange(n))
    axes[-1].set_xticklabels(df["index"], rotation=45, ha="right")
    axes[0].set_title("Hierarchy metrics (slice vs. global)")

    # Legend only once
    axes[0].legend(handles=[
        plt.Rectangle((0,0),1,1,color="steelblue", label="free concept"),
        plt.Rectangle((0,0),1,1,color="orange",    label="labelled concept"),
    ], loc="upper right")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close()
    logging.info("Hierarchy plot saved to %s", out_path)

# -----------------------------------------------------------------------------#
# Masked-AUC plot
# -----------------------------------------------------------------------------#
def masked_auc_plot(info: Dict[str, dict], out_path: str):
    import pandas as pd, numpy as np, matplotlib.pyplot as plt

    df = pd.DataFrame(info).T
    if "hier_auc" not in df.columns or df["hier_auc"].isna().all():
        logging.info("masked_auc flag off → skipping hier-AUC plot."); return

    df = df.sort_values("baseline_auc", ascending=False).reset_index()
    n  = len(df)
    x  = np.arange(n)

    plt.figure(figsize=(max(6, n), 5))
    w = 0.35
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
    plt.savefig(out_path, dpi=120); plt.close()
    logging.info("Masked-AUC plot saved to %s", out_path)

# -----------------------------------------------------------------------------#
# CLI
# -----------------------------------------------------------------------------#
def get_args():
    p = argparse.ArgumentParser("QCW concept analyzer (compact)")
    # model
    p.add_argument("--model_checkpoint", required=True)
    p.add_argument("--depth", type=int, default=18)
    p.add_argument("--whitened_layers", default="5")
    p.add_argument("--act_mode", default="pool_max")
    p.add_argument("--num_classes", type=int, default=200)
    p.add_argument("--subspaces_json", default="")
    p.add_argument("--use_bn_qcw", action="store_true")
    # data
    p.add_argument("--concept_dir", required=True)
    p.add_argument("--hl_concepts", default="")
    p.add_argument("--bboxes_file", default="")
    p.add_argument("--image_state", default="crop", choices=["crop","redact","blur","none"])
    # run
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--run_purity", action="store_true")
    p.add_argument("--topk_images", action="store_true")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--save_transformed", action="store_true")
    p.add_argument("--auc_max", action="store_true")      # Δ_max metric
    p.add_argument("--energy_ratio", action="store_true") # energy metric
    p.add_argument("--masked_auc", action="store_true",
               help="Also compute AUC after zero-filling activations outside each HL slice.")
    p.add_argument("--rank_metrics", action="store_true",
                help="Compute mean-rank / Hit@k hierarchy metric")
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

    if args.use_bn_qcw:
        global build_resnet_qcw, get_last_qcw_layer
        from MODELS.model_resnet_qcw_bn import build_resnet_qcw as B, get_last_qcw_layer as L
        build_resnet_qcw, get_last_qcw_layer = B, L

    wl = [int(x) for x in args.whitened_layers.split(",") if x.strip()]
    subspaces = json.load(open(args.subspaces_json)) if args.subspaces_json and os.path.isfile(args.subspaces_json) else None

    model = build_model(args.model_checkpoint, args.depth, wl, args.act_mode,
                        subspaces, args.num_classes)

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
        # axes is already a list of global axis indices for that HL slice
        axes_str = ", ".join(map(str, axes))
        print(f"  {hl:<12}:  [{axes_str}]")
    print("==================================================\n")

    analyzer = PurityAnalyzer(model, ds, bs=args.batch_size,
                              out_dir=args.output_dir, cfg=args)

    if args.run_purity:
        info = analyzer.compute()
        # Save full results for debugging
        pd.DataFrame(info).T.to_csv(os.path.join(args.output_dir, "result.csv"))
        
        # Purity plots
        purity_dir = os.path.join(args.output_dir, PUR_SUBDIR)
        os.makedirs(purity_dir, exist_ok=True)
        bar_plot(info, os.path.join(purity_dir, "concept_purity.png"))
        pd.DataFrame({k:{"best_axis_auc":v["best_axis_auc"],
                       "best_axis_idx":v["best_axis_idx"],
                       "best_axis_is_labeled":v["best_axis_is_labeled"]}
                   for k,v in info.items()}
                  ).T.to_csv(os.path.join(purity_dir, "purity.csv"))

        # ---- Hierarchy (delta / energy) -----
        if args.auc_max or args.energy_ratio:
            hierarchy_dir = os.path.join(args.output_dir, HIER_SUBDIR)
            os.makedirs(hierarchy_dir, exist_ok=True)
            hierarchy_plot(info, os.path.join(hierarchy_dir, "hierarchy_metrics.png"),
                          show_delta=args.auc_max, show_energy=args.energy_ratio)
            cols = []
            if args.auc_max: cols.append("delta_max")
            if args.energy_ratio: cols.append("energy_ratio")
            if cols:
                pd.DataFrame(info).T[cols].to_csv(os.path.join(hierarchy_dir, "hierarchy.csv"))

        # ---- Masked AUC -----
        if args.masked_auc:
            masked_auc_dir = os.path.join(args.output_dir, MAUC_SUBDIR)
            os.makedirs(masked_auc_dir, exist_ok=True)
            masked_auc_plot(info, os.path.join(masked_auc_dir, "masked_vs_baseline_auc.png"))
            pd.DataFrame(info).T[["baseline_auc", "hier_auc"]].to_csv(
                os.path.join(masked_auc_dir, "masked_auc.csv"))
        
        # ---- Rank Metrics -----
        if args.rank_metrics:
            # Map subconcept index → designated axis
            axis_map = {i: d["best_axis_idx"] for i, d in enumerate(info.values())}
            concept_names = list(info.keys())
            
            # Get labeled axes for color-coding (all non-free axes)
            labeled_axes = []
            for hl, axes in analyzer.subspace.items():
                if hl != "_free":
                    labeled_axes.extend(axes)
            
            run_rank_metrics(analyzer.acts,
                            analyzer.y,
                            analyzer.subspace,
                            analyzer.sc2hl,
                            axis_map,
                            os.path.join(args.output_dir, RANK_SUBDIR),
                            concept_names=concept_names,
                            labeled_axes=labeled_axes)

    if args.topk_images:
        TopKExtractor(analyzer.acts, analyzer.y, ds,
                      os.path.join(args.output_dir, "topk_concept_images"),
                      model if args.save_transformed else None
                     ).dump(k=args.k, save_transformed=args.save_transformed)

if __name__ == "__main__":
    main()
