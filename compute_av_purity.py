import argparse
import os
import csv
from statistics import mean
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate purity.csv files from each layer folder and plot bar chart of average purity vs. layer index")
    parser.add_argument("--analysis_root", default="analysis/larger_layers",
                        help="Root analysis directory that contains WL* sub‑dirs with purity/purity.csv inside")
    parser.add_argument("--output", default="layer_purity.png",
                        help="Path for the output PNG bar chart")
    parser.add_argument("--summary_csv", default=None,
                        help="Optional path to write a CSV containing <layer,avg_purity>")
    return parser.parse_args()


def read_purity_csv(csv_path: str) -> list[float]:
    """Return list of best_axis_auc floats found in *csv_path*."""
    vals: list[float] = []
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return vals
        try:
            auc_idx = header.index("best_axis_auc")
        except ValueError:
            return vals  # column missing
        for row in reader:
            if len(row) <= auc_idx:
                continue
            try:
                vals.append(float(row[auc_idx]))
            except ValueError:
                continue
    return vals


def collect_layer_averages(root: str) -> dict[str, float]:
    """Walk *root* and gather average purity for each WL* folder."""
    results: dict[str, float] = {}
    for entry in sorted(os.listdir(root)):
        layer_dir = os.path.join(root, entry)
        purity_csv = os.path.join(layer_dir, "purity", "purity.csv")
        if not os.path.isfile(purity_csv):
            # silently skip non‑matching folders
            continue
        auc_vals = read_purity_csv(purity_csv)
        if auc_vals:
            results[entry] = mean(auc_vals)
    return results


def make_bar_chart(values: dict[str, float], out_png: str):
    layers = list(values.keys())
    averages = [values[k] for k in layers]

    plt.figure(figsize=(10, 6))
    plt.bar(layers, averages)
    plt.xlabel("Whitened Layer (folder)")
    plt.ylabel("Average Purity (AUC)")
    plt.title("Average Concept Purity vs. Whitened Layer")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    print(f"[INFO] Bar chart written to {out_png}")


def write_summary_csv(values: dict[str, float], out_csv: str):
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "avg_purity_auc"])
        for layer, score in values.items():
            writer.writerow([layer, score])
    print(f"[INFO] Summary CSV written to {out_csv}")


def main():
    args = parse_args()

    if not os.path.isdir(args.analysis_root):
        raise SystemExit(f"[ERROR] analysis_root '{args.analysis_root}' not found")

    layer_scores = collect_layer_averages(args.analysis_root)
    if not layer_scores:
        raise SystemExit("[ERROR] No purity.csv files discovered – check your directory structure")

    make_bar_chart(layer_scores, args.output)

    if args.summary_csv:
        write_summary_csv(layer_scores, args.summary_csv)


if __name__ == "__main__":
    main()