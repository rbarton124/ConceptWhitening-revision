from __future__ import annotations
import os, logging
from typing import Dict, List, Tuple, Union, Mapping
import numpy as np, pandas as pd
import matplotlib.pyplot as plt


def _colour_map(is_labeled: np.ndarray) -> List[str]:
    """Vector of bar colours matching the purity plot convention."""
    return ["red" if x else "blue" for x in is_labeled]


def _rank_vector(scores: np.ndarray) -> np.ndarray:
    """
    Return rank (1=highest) of each element in `scores` along last dim.
    argsort twice is O(K logK) and fine for K≈200.
    """
    order = np.argsort(-scores, axis=1)               # indices sorted high→low
    ranks = np.empty_like(order)
    # fill ranks: ranks[i, order[i,j]] = j
    rows = np.arange(order.shape[0])[:, None]
    ranks[rows, order] = np.arange(order.shape[1])
    return ranks + 1                                  # 1-based


def run_rank_metrics(
        acts:         np.ndarray,  # (N_images, K_axes)
        labels:       np.ndarray,  # (N_images,) subconcept indices
        slices:       Dict[str, List[int]],  # hl -> list of axes
        sc2hl:        List[str],   # subconcept -> hl mapping
        axis_map:     Union[List[int], Mapping[int, int]],  # subconcept -> designated axis
        out_dir:      str,
        ks:           Tuple[int, ...] = (1,3,5),
        concept_names: List[str] = None,
        labeled_axes: List[int] = None
) -> Dict[str, dict]:  # Results per subconcept

    N, K = acts.shape
    # Use either passed concept_names, or dictionary keys, or numeric indices
    sc_names = concept_names if concept_names else (list(axis_map.keys()) if isinstance(axis_map, dict) else None)

    # Pre-compute ranks once for all axes
    ranks_raw = _rank_vector(acts)
    results = {}
    os.makedirs(out_dir, exist_ok=True)
    
    # Use provided labeled_axes or try to infer them
    if labeled_axes is None:
        labeled_axes = set()
        if isinstance(axis_map, Mapping):
            for c_idx, axis in axis_map.items():
                if c_idx < len(sc2hl):
                    hl = sc2hl[c_idx]
                    if hl in slices and axis in slices[hl]:
                        labeled_axes.add(axis)
    else:
        labeled_axes = set(labeled_axes)
        
    # Reuse this array instead of copying for each concept
    masked_acts = np.zeros_like(acts)
    
    for c_idx in range(K):
        hl = sc2hl[c_idx]
        slice_axes = slices[hl]
        axis = axis_map[c_idx]                        # designated axis for this subconcept

        pos_mask = (labels == c_idx)
        if not pos_mask.any():
            continue

        # Mask out activations outside the slice
        masked_acts.fill(0.0)
        np.copyto(masked_acts, acts)
        masked_acts[:, np.setdiff1d(np.arange(K), slice_axes, assume_unique=True)] = 0.0
        ranks_mask = _rank_vector(masked_acts)

        # ---------- aggregate over positive images ----------
        r_raw  = ranks_raw [pos_mask, axis]
        r_mask = ranks_mask[pos_mask, axis]

        mean_raw  = r_raw .mean()
        # build a single result dict for this concept
        is_lbl = axis in labeled_axes
        res = {
            "mean_rank_raw": r_raw.mean(),
            "mean_rank_mask": r_mask.mean(),
            "is_labeled_axis": is_lbl,
        }

        k_full = len(slice_axes)
        for k in ks + (k_full,):
            res[f"hit@{k}_raw"]  = float((r_raw  <= k).mean())
            res[f"hit@{k}_mask"] = float((r_mask <= k).mean())

        results[c_idx] = res

    # ---------------- plot: mean-rank -----------------------    # prettify
    df = pd.DataFrame(results).T
    
    # Add axis index to concept names (e.g. "bird_eye (#42)")
    if sc_names:
        name_map = {}
        for i, name in enumerate(sc_names):
            if i in df.index:
                axis_idx = axis_map[i] if isinstance(axis_map, Mapping) else axis_map[i] if i < len(axis_map) else None
                if axis_idx is not None:
                    name_map[i] = f"{name} (#{axis_idx})"
                else:
                    name_map[i] = name
        df.rename(index=name_map, inplace=True)

    order = df.sort_values("mean_rank_raw").index
    df = df.loc[order]
    
    n = len(df)
    w = 0.4
    x = np.arange(n)
    
    # Get colors matching purity plot convention
    colours = _colour_map(df["is_labeled_axis"].values)

    # ---------------- plot: mean rank -------------------
    plt.figure(figsize=(max(6, n), 5))
    plt.bar(x-w/2, df["mean_rank_raw"],  width=w, color=colours, label="raw")
    plt.bar(x+w/2, df["mean_rank_mask"], width=w, color=colours, alpha=0.6, label="masked")
    plt.ylabel("Mean rank (↓ better)")
    
    concept_labels = df.index
    
    plt.xticks(x, concept_labels, rotation=45, ha="right")
    plt.ylim(0, df[["mean_rank_raw","mean_rank_mask"]].to_numpy().max()*1.1)
    plt.title("Mean rank of designated axis")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "rank_mean.png"), dpi=120)
    plt.close()

    # ---------------- plot: Hit@k (example k=3) -------------
    if "hit@3_raw" in df.columns:
        plt.figure(figsize=(max(6, n), 5))
        plt.bar(x-w/2, df["hit@3_raw"],  width=w, color=colours, label="raw")
        plt.bar(x+w/2, df["hit@3_mask"], width=w, color=colours, alpha=0.6, label="masked")
        plt.ylabel("Hit@3 (↑ better)")
        plt.ylim(0, 1)
        plt.axhline(1.0, ls=":", color="k")
        
        # Use same concept label handling as in mean rank plot
        plt.xticks(x, concept_labels, rotation=45, ha="right")
        plt.title("Hit@3 of designated axis")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "hit3.png"), dpi=120)
        plt.close()

    # Write CSV with just the metrics columns
    minimal_cols = [c for c in df.columns if c.startswith(("mean_rank", "hit@"))]
    df[minimal_cols + ["is_labeled_axis"]].to_csv(os.path.join(out_dir, "rank_metrics.csv"))
    logging.info("Rank-metrics saved to %s", out_dir)
    return results
