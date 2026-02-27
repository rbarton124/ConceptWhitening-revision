# Quantized Concept Whitening (QCW)

An implementation of **Quantized Concept Whitening** for interpretable image recognition, extending the original Concept Whitening method (Chen, Bei, and Rudin, 2020). QCW organizes latent axes into **hierarchical subspaces** — one per high-level concept — and discovers **unlabeled sub-concepts** via winner-takes-all alignment within each subspace.

## Setup

```bash
conda env create -f environment.yml
conda activate QCW
```

## Repository Structure

```
├── train_qcw.py                  Training script (classification + concept alignment)
├── plot_functions.py             Evaluation: purity, AUC, hierarchy metrics, top-K images
├── rank_metrics.py               Mean rank / Hit@k metrics (called by plot_functions)
│
├── MODELS/
│   ├── iterative_normalization.py    ZCA whitening + Cayley rotation + gradient accumulation
│   ├── factory_qcw.py               build_qcw() dispatcher for all architectures
│   ├── ConceptDataset_QCW.py         Concept dataset with crop/redact/blur and free detection
│   └── backbones/
│       ├── base_qcw.py              Abstract base: BN replacement loop, mode/rotation helpers
│       ├── resnet_qcw.py            ResNet-18/50 (indices 1-8 / 1-16)
│       ├── densenet_qcw.py          DenseNet-121/161 (indices 1-5)
│       └── vgg_qcw.py               VGG16-BN (indices 1-13)
│
├── dataset_scripts/
│   ├── prepare_CUB_QCW.py           CUB-200-2011 to QCW format
│   ├── add_free_concepts.py          Add free (unlabeled) concept folders
│   ├── manage_concept_data.py        Interactive concept dataset curation tool
│   ├── cull_runs.py                  Clean up short/incomplete TensorBoard runs
│   ├── download_imagenet.py          Download pretrained ResNet-18 weights
│   └── download_places_365.py        Download Places365 dataset
│
├── environment.yml
├── LICENSE
└── README.md
```

## Method Overview

### Concept Whitening (Background)

Concept Whitening replaces BatchNorm layers with a whitening + rotation module. The whitening step decorrelates features via ZCA (Newton iteration). A rotation matrix Q, updated via the Cayley transform, aligns each latent axis with a labeled concept so that axis *j* fires maximally for images of concept *j*.

### Quantized Concept Whitening (QCW)

QCW extends the original method with three changes:

1. **Subspace partitioning.** Axes are grouped into contiguous subspaces per high-level concept (e.g., axes 0-4 = "beak", 5-10 = "wing"). An axis index directly indicates which body-part subspace it belongs to.

2. **Free (unlabeled) sub-concepts.** Within each subspace, some axes have no label. These are trained via winner-takes-all: each concept image activates whichever free axis responds most strongly, encouraging emergent specialization. Controlled by `--cw_lambda`.

3. **Multi-architecture support.** QCW layers can replace selected BatchNorm layers in ResNet-18/50, DenseNet-121/161, or VGG16-BN via a factory pattern.

### Training Loop

Training alternates between two steps:

- **Classification** (every iteration): Standard cross-entropy forward/backward on full images.
- **Concept alignment** (every `--cw_align_freq` iterations): Forward concept images through the network in eval mode. For labeled subconcepts, accumulate alignment gradients on the designated axis. For free subconcepts, use winner-takes-all within the subspace. Then update Q via Cayley transform with Wolfe line search.

## Data Preparation

### Main Dataset

Standard ImageFolder layout for classification:

```
main_data/
├── train/
│   ├── 001/
│   │   ├── img1.jpg
│   │   └── ...
│   └── 200/
├── val/
└── test/
```

### Concept Dataset

Two-level hierarchy: high-level concepts contain subconcept folders, each with images. A `bboxes.json` maps image paths to `[x1, y1, x2, y2]` bounding-box coordinates.

```
concept_data/
├── bboxes.json
├── concept_train/
│   ├── wing/
│   │   ├── has_wing_color::grey/
│   │   ├── has_wing_color::brown/
│   │   └── wing_free/              # free (unlabeled) subconcept
│   ├── beak/
│   │   └── ...
│   └── general/
└── concept_val/
    └── (same structure)
```

Free subconcepts are detected by the `_free` suffix (or `_free1`, `_free2`, etc.). Use `dataset_scripts/add_free_concepts.py` to generate them automatically.

### CUB-200 Setup

The data preparation script requires a **mappings file** that maps CUB attribute names (e.g., `has_wing_color::grey`) to high-level concepts (e.g., `wing`). A pre-built mappings file for CUB-200 is included at `docs/cub_attribute_mappings.json`.

```bash
# 1. Download and extract CUB_200_2011 to some path

# 2. Generate main classification dataset + concept dataset
python dataset_scripts/prepare_CUB_QCW.py \
    --cub_dir /path/to/CUB_200_2011 \
    --output_main data/CUB/main_data \
    --output_concept data/CUB/concept_data \
    --concepts wing,beak,eye,nape \
    --mode attributes \
    --mappings docs/cub_attribute_mappings.json \
    --draw_or_copy crop -v

# 3. (Optional) Add free concept folders for winner-takes-all training
python dataset_scripts/add_free_concepts.py \
    --concept_dir data/CUB/concept_data \
    --n_free_per_hl 2
```

## Training

```bash
python train_qcw.py \
    --data_dir data/CUB/main_data \
    --concept_dir data/CUB/concept_data \
    --concepts wing,beak,eye,nape \
    --whitened_layers 7 \
    --prefix QCW18_WL7 \
    --model resnet --depth 18 \
    --resume model_checkpoints/res18_CUB.pth \
    --only_load_weights
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | required | Path to main dataset (train/val/test) |
| `--concept_dir` | required | Path to concept dataset |
| `--concepts` | required | Comma-separated high-level concepts |
| `--prefix` | required | Experiment name for logging and checkpoints |
| `--whitened_layers` | `"5"` | BN layer indices to replace (comma-separated) |
| `--model` | `resnet` | Architecture: `resnet`, `densenet`, `vgg16` |
| `--depth` | `18` | ResNet: 18/50, DenseNet: 121/161 |
| `--act_mode` | `pool_max` | Activation reduction: `mean`, `max`, `pos_mean`, `pool_max` |
| `--cw_align_freq` | `40` | Concept alignment every N mini-batches |
| `--cw_lambda` | `0.05` | Weight for free-concept winner-takes-all alignment |
| `--concept_image_mode` | `crop` | Bounding-box handling: `crop`, `redact`, `blur`, `none` |
| `--epochs` | `100` | Training epochs |
| `--batch_size` | `64` | Mini-batch size |
| `--lr` | `5e-4` | Learning rate |
| `--resume` | `""` | Checkpoint to resume from |
| `--only_load_weights` | flag | Load weights only (ignore epoch/optimizer state) |
| `--vanilla_pretrain` | flag | Train without CW (vanilla baseline) |
| `--disable_subspaces` | flag | Disable hierarchical subspace partitioning |

### Whitened Layer Indices

Indices are 1-based global block numbers:

| Architecture | Valid Indices | Mapping |
|-------------|---------------|---------|
| ResNet-18 | 1-8 | 1-2 = layer1, 3-4 = layer2, 5-6 = layer3, 7-8 = layer4 |
| ResNet-50 | 1-16 | [3, 4, 6, 3] blocks across layer1-4 |
| DenseNet-121/161 | 1-5 | norm0, transition1-3, norm5 |
| VGG16-BN | 1-13 | 13 BatchNorm layers in features |

### Multi-Architecture Examples

```bash
# ResNet-50
python train_qcw.py --model resnet --depth 50 --whitened_layers 14 ...

# DenseNet-161
python train_qcw.py --model densenet --depth 161 --whitened_layers 5 ...

# VGG16-BN
python train_qcw.py --model vgg16 --whitened_layers 10 ...
```

### Monitoring

```bash
tensorboard --logdir runs/
```

Logged metrics:
- `Train/Loss`, `Train/Top1`, `Train/Top5` -- classification performance
- `CW/Alignment/SubspaceTop1` -- % of labeled concepts where the top activation within the correct subspace matches the target axis
- `CW/Alignment/GlobalTop1`, `GlobalTop5` -- global axis ranking
- `CW/Alignment/ConceptLoss` -- average alignment score from Cayley update
- `CW/FreeConcept/AxisConsistency` -- % of free-concept images that consistently choose the same axis
- `CW/FreeConcept/AxisPurity` -- exclusivity of chosen axis to that free concept
- `CW/FreeConcept/ActStrengthRatio` -- free concept activation relative to labeled concepts

## Evaluation

```bash
python plot_functions.py \
    --model_checkpoint model_checkpoints/QCW18_WL7_best.pth \
    --concept_dir data/CUB/concept_data \
    --hl_concepts wing,beak,eye,nape \
    --whitened_layers 7 \
    --model resnet --depth 18 \
    --output_dir analysis/QCW18_WL7 \
    --run_purity --topk_images --auc_max --energy_ratio --masked_auc --rank_metrics
```

This computes:
- **Concept purity** (per-axis ROC-AUC)
- **Masked AUC** (purity within hierarchical subspaces)
- **Delta-max** (subspace vs. global max-activation AUC)
- **Energy ratio** (fraction of activation energy within the designated subspace)
- **Rank metrics** (mean rank, Hit@k of the designated axis)
- **Top-K images** per concept axis

Results are saved as CSVs and plots in the output directory.

## Common Issues

1. **No alignment improvement.** Check that your concept dataset has enough images per subconcept (>50 recommended). Try lowering `--cw_align_freq` (more frequent alignment) or increasing `--batches_per_concept`.

2. **Out of memory.** Lower `--batch_size`. Concept alignment processes images one at a time, so OOM usually comes from the classification step.

3. **Invalid whitened_layers.** The script validates indices at startup. For ResNet-18, valid indices are 1-8. The error message will show the valid range.

4. **Multiple whitened layers.** Using `--whitened_layers 2,5,8` replaces three BN layers. Each gets its own rotation matrix and alignment. Empirically, a single well-chosen layer (typically 5-7 for ResNet-18) works best.

5. **Checkpoint shape mismatch.** When loading a vanilla pretrained checkpoint into a QCW model, the replaced BN layer's weights won't match (BN has `[C]` shapes, QCW has `[1,C,1,1]`). This is expected -- those keys are skipped, and the QCW layer initializes fresh.

## Citation

If you use this code, please cite the original Concept Whitening paper:

```bibtex
@article{chen2020concept,
  title={Concept whitening for interpretable image recognition},
  author={Chen, Zhi and Bei, Yijie and Rudin, Cynthia},
  journal={Nature Machine Intelligence},
  volume={2},
  number={12},
  pages={772--782},
  year={2020}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
