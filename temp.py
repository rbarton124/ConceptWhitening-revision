import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


def collect_metrics(root_dir):
    """
    Collect purity, baseline AUC, masked AUC from purity.csv and masked_auc.csv.
    Also parse model depth (18 vs 50) from folder names.
    """
    # regex matches both RESNET18 and RESNET50
    pattern = re.compile(r"RESNET(\d+)_COCO_(QCW|CW_ORIG)_(.+)_layer_(\d+)_checkpoint")

    results = defaultdict(lambda: defaultdict(list))

    for folder in os.listdir(root_dir):
        match = pattern.match(folder)
        if not match:
            continue

        depth, method, dataset_name, layer_str = match.groups()
        depth = int(depth)
        layer = int(layer_str)

        # Purity
        purity_path = Path(root_dir) / folder / "purity" / "purity.csv"
        mean_purity = None
        if purity_path.exists():
            df = pd.read_csv(purity_path)
            mean_purity = df["best_axis_auc"].mean()

        # Masked AUC
        masked_path = Path(root_dir) / folder / "masked_auc" / "masked_auc.csv"
        mean_baseline_auc, mean_masked_auc = None, None
        if masked_path.exists():
            df_m = pd.read_csv(masked_path)
            mean_baseline_auc = df_m["baseline_auc"].mean()
            mean_masked_auc = df_m["hier_auc"].mean()

        # Dataset size: small if <=2 concepts, else large
        dataset_items = dataset_name.split(",")
        dataset_size = "small" if len(dataset_items) <= 2 else "large"

        results[(dataset_size, depth)][method].append(
            (layer, mean_purity, mean_baseline_auc, mean_masked_auc, folder)
        )

    return results


def plot_purity(results, save_dir):
    """Line plots: QCW vs CW_ORIG purity for each dataset size/depth"""
    for (dataset_size, depth), method_data in results.items():
        plt.figure(figsize=(7, 5))

        for method, vals in method_data.items():
            vals_sorted = sorted(vals, key=lambda x: x[0])
            layers = [v[0] for v in vals_sorted]
            purities = [v[1] for v in vals_sorted]

            if method == "QCW":
                plt.plot(layers, purities, marker="s", color="green", label="QCW Purity")
            elif method == "CW_ORIG":
                plt.plot(layers, purities, marker="o", color="crimson", label="CW_ORIG Purity")

        plt.xlabel("Whitened Layer (WL)")
        plt.ylabel("Mean Purity (AUC)")
        plt.ylim(0.6, 0.9)
        plt.title(f"Concept Purity ({dataset_size.capitalize()} / ResNet-{depth})")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)

        save_path = Path(save_dir) / f"purity_{dataset_size}_resnet{depth}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[Saved] {save_path}")
        plt.close()


def plot_baseline_vs_masked(results, save_dir):
    """Bar plots: Baseline vs Masked AUC (QCW) + Baseline AUC (CW_ORIG)"""
    for (dataset_size, depth), method_data in results.items():
        if "QCW" not in method_data:
            continue

        vals_qcw = sorted(method_data["QCW"], key=lambda x: x[0])
        layers = [v[0] for v in vals_qcw]
        baseline_qcw = [v[2] for v in vals_qcw]
        masked_qcw = [v[3] for v in vals_qcw]

        # CW_ORIG baseline auc
        baseline_cw_orig = []
        if "CW_ORIG" in method_data:
            vals_cw_orig = sorted(method_data["CW_ORIG"], key=lambda x: x[0])
            cw_orig_dict = {v[0]: v[2] for v in vals_cw_orig}
            baseline_cw_orig = [cw_orig_dict.get(layer, None) for layer in layers]

        x = range(len(layers))
        width = 0.25

        plt.figure(figsize=(8, 5))
        # left: Baseline QCW
        plt.bar([i - width for i in x], baseline_qcw, width=width,
                color="dodgerblue", label="Baseline AUC (QCW)")
        # middle: Baseline CW_ORIG
        if baseline_cw_orig:
            plt.bar([i for i in x], baseline_cw_orig, width=width,
                    color="gray", label="Baseline AUC (CW_ORIG)")
        # right: Masked QCW
        plt.bar([i + width for i in x], masked_qcw, width=width,
                color="tomato", label="Masked AUC (QCW)")

        plt.xticks(x, layers)
        plt.xlabel("Whitened Layer (WL)")
        plt.ylabel("AUC")
        plt.ylim(0.2, 0.95)
        plt.title(f"Baseline vs Masked AUC (ResNet-{depth} {dataset_size.capitalize()})")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)

        save_path = Path(save_dir) / f"baseline_vs_masked_{dataset_size}_resnet{depth}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[Saved] {save_path}")
        plt.close()


def summarize_tables(results, root_dir):
    table1 = []
    table2 = []

    pattern = re.compile(r"RESNET(\d+)_COCO_(QCW|CW_ORIG)_(.+)_layer_(\d+)_checkpoint")

    for (dataset_size, depth), method_data in results.items():
        if "QCW" not in method_data:
            continue

        vals_qcw = sorted(method_data["QCW"], key=lambda x: x[0])
        layers = [v[0] for v in vals_qcw]
        purities = [v[1] for v in vals_qcw]
        baseline_aucs = [v[2] for v in vals_qcw]
        masked_aucs = [v[3] for v in vals_qcw]
        folders = [v[4] for v in vals_qcw]

        # find optimal WL (argmax of purity)
        opt_idx = max(range(len(purities)), key=lambda i: purities[i])
        opt_wl = layers[opt_idx]
        opt_purity = purities[opt_idx]
        wl1_purity = purities[0]  # purity at layer 1
        delta_purity = opt_purity - wl1_purity

        dataset_label = f"{dataset_size.capitalize()} / ResNet-{depth}"

        # ---------- Table 1 ----------
        table1.append({
            "Dataset/Model": dataset_label,
            "Optimal WL": opt_wl,
            "Mean Purity (Optimal WL)": opt_purity,
            "Mean Purity (WL-1)": wl1_purity,
            "Δ Purity": delta_purity
        })

        # ---------- Table 2 ----------
        opt_masked_auc = masked_aucs[opt_idx]
        opt_baseline_auc = baseline_aucs[opt_idx]
        delta_max = opt_masked_auc - opt_baseline_auc

        # read energy_ratio from result.csv
        energy_ratio = None
        folder = folders[opt_idx]
        result_path = Path(root_dir) / folder / "result.csv"
        if result_path.exists():
            df_res = pd.read_csv(result_path)
            energy_ratio = df_res["energy_ratio"].mean()

        table2.append({
            "Model/Dataset": f"ResNet-{depth} {dataset_size.capitalize()}",
            "Optimal WL": opt_wl,
            "Masked AUC": opt_masked_auc,
            "Δ-max": delta_max,
            "Energy Ratio": energy_ratio
        })

    return pd.DataFrame(table1), pd.DataFrame(table2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, help="Path to directory containing experiment folders")
    parser.add_argument("--out_dir", type=str, default=".", help="Directory to save plots and tables")
    args = parser.parse_args()

    results = collect_metrics(args.root_dir)
    plot_purity(results, args.out_dir)
    plot_baseline_vs_masked(results, args.out_dir)

    df1, df2 = summarize_tables(results, args.root_dir)
    print("\nTable 1: Purity Summary\n", df1.to_string(index=False))
    print("\nTable 2: Masked AUC Summary\n", df2.to_string(index=False))

    df1.to_csv(Path(args.out_dir) / "purity_summary.csv", index=False)
    df2.to_csv(Path(args.out_dir) / "masked_auc_summary.csv", index=False)
