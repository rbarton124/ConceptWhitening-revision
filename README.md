# Quantized Concept Whitening (QCW)

An implementation of **Quantized Concept Whitening** for interpretable image recognition, extending the original Concept Whitening method (Chen, Bei, and Rudin, 2020). QCW organizes latent axes into **hierarchical subspaces** — one per high-level concept — and discovers **unlabeled sub-concepts** via winner-takes-all alignment within each subspace.

## Quick Start

```bash
# 1. Create environment
conda env create -f environment.yml
conda activate QCW

# 2. Prepare CUB-200 data (once)
python dataset_scripts/prepare_CUB_QCW.py \
    --cub_dir /path/to/CUB_200_2011 \
    --output_main data/CUB/main_data \
    --output_concept data/CUB/concept_data \
    --concepts wing,beak,eye,nape \
    --mode attributes \
    --mappings data/CUB/mappings.json \
    --draw_or_copy crop -v

# 3. Train
python train_qcw.py \
    --data_dir data/CUB/main_data \
    --concept_dir data/CUB/concept_data \
    --concepts wing,beak,eye,nape \
    --whitened_layers 7 \
    --prefix my_experiment \
    --resume model_checkpoints/res18_CUB.pth \
    --only_load_weights

# 4. Evaluate
python plot_functions.py \
    --checkpoint model_checkpoints/my_experiment_best.pth \
    --concept_dir data/CUB/concept_data \
    --concepts wing,beak,eye,nape \
    --output_dir analysis/my_experiment
```

## How It Works

### Original Concept Whitening

Concept Whitening replaces BatchNorm layers with a whitening + rotation module. The whitening decorrelates features (ZCA via Newton iteration). The rotation matrix Q is updated via the Cayley transform to align each axis with a labeled concept, so that axis *j* fires maximally for images of concept *j*.

### Quantized CW (This Repo)

QCW extends this with three changes:

1. **Subspace partitioning.** Axes are grouped into contiguous subspaces per high-level concept (e.g., axes 0-4 = "beak", axes 5-10 = "wing"). Knowing an axis index tells you which part it belongs to.

2. **Free (unlabeled) sub-concepts.** Within each subspace, some axes have no label. These are trained via winner-takes-all: each image activates whichever free axis responds most strongly, encouraging emergent specialization. Controlled by the `cw_lambda` parameter.

3. **Multi-architecture support.** QCW layers can be inserted into ResNet-18/50, DenseNet-121/161, or VGG16-BN via a factory pattern, replacing selected BatchNorm layers.

### Training Loop

The training alternates between two steps:

- **Classification step** (every iteration): Standard CE forward/backward on full images.
- **Concept alignment step** (every `cw_align_freq` iterations): Forward concept images through the network in eval mode. For labeled subconcepts, accumulate alignment gradients on the designated axis. For free subconcepts, use winner-takes-all within the subspace. Then update Q via Cayley transform (with Wolfe line search).

## Repository Structure

```
├── train_qcw.py                         Main QCW training script
├── plot_functions.py                    Evaluation: purity, AUC, rank metrics, top-K images
├── rank_metrics.py                      MRR, hit@k, rank histograms (used by plot_functions)
│
├── MODELS/
│   ├── iterative_normalization.py       Core: ZCA whitening + Cayley rotation + gradient accumulation
│   ├── factory_qcw.py                   build_qcw(model_type) → ResNet/DenseNet/VGG
│   ├── ConceptDataset_QCW.py            Dataset: crop/redact/blur, labeled vs free detection
│   ├── backbones/
│   │   ├── base_qcw.py                  Abstract base: BN replacement loop, mode/rotation helpers
│   │   ├── resnet_qcw.py               ResNet-18/50 (indices 1-8 / 1-16, replaces block.bn1)
│   │   ├── densenet_qcw.py             DenseNet-121/161 (indices 1-5)
│   │   └── vgg_qcw.py                  VGG16-BN (indices 1-13)
│
├── dataset_scripts/
│   ├── prepare_CUB_QCW.py              CUB-200 → QCW format (main + concept datasets)
│   ├── manage_concept_data.py           Interactive concept dataset curation tool
│   ├── add_free_concepts.py             Add free (unlabeled) concept folders to a dataset
│   ├── cull_runs.py                     Clean up short/incomplete TensorBoard runs
│   ├── download_imagenet.py             Download pretrained ResNet-18
│   └── download_places_365.py           Download Places365 dataset
│
├── legacy/
│   ├── train_basic.py                   Original-CW training script (baseline comparisons)
│   ├── model_resnet.py                  Legacy CW model definitions (ResNet/DenseNet/VGG)
│   └── notes/                           Paper drafts, experiment notes, metrics guides
│
├── environment.yml                      Conda environment (Python 3.11, PyTorch, CUDA 12.1)
├── .gitignore
├── LICENSE                              MIT
└── README.md
```

**Gitignored local directories** (not in the repo, created during use):

```
data/                  Raw datasets + prepared main/concept data
model_checkpoints/     Saved .pth model weights
runs/                  TensorBoard event logs
analysis/              Evaluation outputs from plot_functions.py
figs/                  Generated figures
notes/                 Personal experiment notes
figure_scripts/        Figure generation scripts
experiments/           Shell scripts for running experiments
```

## Data Preparation

### Main Dataset

Standard ImageFolder layout for classification:

```
main_data/
├── train/
│   ├── 001/          # class folders
│   │   ├── img1.jpg
│   │   └── ...
│   └── 200/
├── val/
└── test/
```

### Concept Dataset

Two-level hierarchy: high-level concepts contain subconcept folders, each with images. A `bboxes.json` maps image paths to `[x1, y1, x2, y2]` coordinates for cropping/redaction.

```
concept_data/
├── bboxes.json
├── concept_train/
│   ├── wing/
│   │   ├── has_wing_color::grey/
│   │   │   ├── 0001_img.jpg
│   │   │   └── ...
│   │   ├── has_wing_color::brown/
│   │   └── wing_free/              # free (unlabeled) subconcept
│   ├── beak/
│   │   ├── has_bill_shape::curved/
│   │   └── ...
│   └── general/
└── concept_val/
    └── (same structure)
```

**Free subconcepts** are detected by the `_free` suffix (or `_free1`, `_free2`, etc.). Use `dataset_scripts/add_free_concepts.py` to generate them automatically.

### CUB-200 Setup

```bash
# Download CUB_200_2011 and extract to some path

# Generate both datasets
python dataset_scripts/prepare_CUB_QCW.py \
    --cub_dir /path/to/CUB_200_2011 \
    --output_main data/CUB/main_data \
    --output_concept data/CUB/concept_data \
    --concepts wing,beak,eye,nape,general \
    --mode attributes \
    --mappings data/CUB/mappings.json \
    --draw_or_copy crop \
    -v

# Optionally add free concepts
python dataset_scripts/add_free_concepts.py \
    --concept_dir data/CUB/concept_data/concept_train \
    --n_free_per_hl 2
```

## Training

### Basic Usage

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
| `--concept_dir` | required | Path to concept dataset (concept_train/) |
| `--concepts` | required | Comma-separated high-level concepts |
| `--whitened_layers` | `"5"` | BN layer indices to replace (comma-separated) |
| `--model` | `resnet` | Architecture: `resnet`, `densenet`, `vgg16` |
| `--depth` | `18` | ResNet: 18/50. DenseNet: 121/161. |
| `--act_mode` | `pool_max` | Activation reduction: `mean`, `max`, `pos_mean`, `pool_max` |
| `--cw_align_freq` | `40` | Concept alignment every N mini-batches |
| `--cw_lambda` | `0.05` | Weight for free-concept WTA alignment |
| `--concept_image_mode` | `crop` | How to handle bboxes: `crop`, `redact`, `blur`, `none` |
| `--epochs` | `100` | Training epochs |
| `--batch_size` | `64` | Mini-batch size |
| `--lr` | `5e-4` | Learning rate |
| `--resume` | `""` | Checkpoint to resume from |
| `--only_load_weights` | flag | Load weights only (ignore epoch/optimizer) |
| `--vanilla_pretrain` | flag | Train without CW (vanilla baseline) |
| `--disable_subspaces` | flag | Disable hierarchical subspaces |

### Whitened Layer Indices

Indices are global block numbers (1-indexed):

| Architecture | Blocks per Layer Group | Valid Indices |
|-------------|----------------------|---------------|
| ResNet-18 | [2, 2, 2, 2] | 1-8 |
| ResNet-50 | [3, 4, 6, 3] | 1-16 |
| DenseNet-121/161 | norm0, transition1-3, norm5 | 1-5 |
| VGG16-BN | 13 BN layers in features | 1-13 |

For ResNet-18: indices 1-2 = layer1, 3-4 = layer2, 5-6 = layer3, 7-8 = layer4.

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

Training logs to TensorBoard:

```bash
tensorboard --logdir runs/
```

Logged metrics:
- `Train/Loss`, `Train/Top1`, `Train/Top5` — classification performance
- `CW/Alignment/SubspaceTop1` — % of labeled concepts where the top activation within the correct subspace matches the target axis
- `CW/Alignment/GlobalTop1`, `GlobalTop5` — global axis ranking
- `CW/Alignment/ConceptLoss` — average alignment score from Cayley update
- `CW/FreeConcept/AxisConsistency` — % of free-concept images that consistently choose the same axis
- `CW/FreeConcept/AxisPurity` — exclusivity of chosen axis to that free concept
- `CW/FreeConcept/ActStrengthRatio` — free concept activation relative to labeled concepts

## Evaluation

```bash
python plot_functions.py \
    --checkpoint model_checkpoints/my_experiment_best.pth \
    --concept_dir data/CUB/concept_data \
    --concepts wing,beak,eye,nape \
    --whitened_layers 7 \
    --model resnet --depth 18 \
    --output_dir analysis/my_experiment
```

This computes per-axis purity (AUC), masked AUC, energy ratios, hierarchy AUC, rank metrics (MRR, hit@k), and saves top-K activated images for each concept. Results are saved as CSVs and images in the output directory.

## Common Issues

1. **No alignment improvement.** Check that your concept dataset has enough images per subconcept (>50 recommended). Try lowering `cw_align_freq` (more frequent alignment) or increasing `batches_per_concept`.

2. **Out of memory.** Lower `--batch_size`. Concept alignment processes images one at a time, so OOM usually comes from the classification step.

3. **Invalid whitened_layers.** The script validates indices at startup. For ResNet-18, valid indices are 1-8. The error message will show the valid range.

4. **Multiple whitened layers.** Using `--whitened_layers 2,5,8` replaces three BN layers. Each gets its own rotation matrix and alignment. Empirically, a single well-chosen layer (typically 5-7 for ResNet-18) works best.

5. **Checkpoint shape mismatch.** When loading a vanilla pretrained checkpoint into a QCW model, the replaced BN layer's weights won't match (BN has `[C]` shapes, QCW has `[1,C,1,1]`). This is expected — those keys are skipped, and the QCW layer initializes fresh.

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

MIT License. See `LICENSE` for details.
