import argparse
import re
import os
from pathlib import Path
import subprocess


def sweep_and_run(checkpoint_root: str, concept_dir: str, output_root: str):
    """
    Loop through all checkpoints in checkpoint_root and run analyzer.py
    for each one, setting arguments dynamically.
    """
    ckpt_pattern = re.compile(
        r"RESNET(?P<depth>\d+)_Places365_(?P<model_type>QCW|CW_ORIG)_COCO_{(?P<dataset>[^}]*)}_layer_(?P<layer>\d+)_checkpoint\.pth$"
    )

    for ckpt in Path(checkpoint_root).glob("*.pth"):
        m = ckpt_pattern.match(ckpt.name)
        if not m:
            continue

        depth = int(m.group("depth"))
        model_type = m.group("model_type")
        dataset = m.group("dataset")             # e.g. "animal,food" or "animal-food"
        whitened_layer = m.group("layer")
        out_dir = os.path.join(output_root, ckpt.stem)

        # Choose bbox file depending on model type
        if model_type == "QCW":
            bbox_file = os.path.join(concept_dir, "bboxes.json")
            cw_lambda = 0.05
        else:
            bbox_file = os.path.join(concept_dir, "bboxes_merged.json")
            cw_lambda = 0

        # Run the analyzer
        cmd = [
            "python", "plot_functions.py",
            "--model_checkpoint", str(ckpt),
            "--depth", depth,
            "--whitened_layers", str(whitened_layer),
            "--act_mode", "pool_max",
            "--cw_lambda", cw_lambda,
            "--num_classes", 365,
            "--concept_dir", concept_dir,
            "--hl_concepts", dataset,
            "--bboxes_file", bbox_file,
            "--image_state", "crop",
            "--batch_size", 64,
            "--run_purity",
            "--topk_images",
            # "--save_transformed",
            "--auc_max",
            "--energy_ratio",
            "--masked_auc",
            "--rank_metrics",
            "--output_dir", out_dir,
        ]
        cmd = list(map(str, cmd))

        print(f"[RUN] {' '.join(cmd)}")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sweep QCW/CW_ORIG checkpoints and run analysis")
    parser.add_argument("--checkpoint_root", type=str, required=True,
                        help="Path to directory containing model checkpoints (RESNET*.pth)")
    parser.add_argument("--concept_dir", type=str, required=True,
                        help="Path to concept dataset root directory")
    parser.add_argument("--output_root", type=str, required=True,
                        help="Path to the output directory")

    args = parser.parse_args()

    sweep_and_run(
        checkpoint_root=args.checkpoint_root,
        concept_dir=args.concept_dir,
        output_root=args.output_root
    )
