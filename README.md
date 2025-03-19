# **Quantized Concept Whitening (QCW) for Interpretable Image Recognition**

**Maintainers**:   
**Contact**:   

## **Table of Contents**

1. [Introduction](#1-introduction)  
2. [Repository Overview](#2-repository-overview)  
3. [Installation & Environment](#3-installation--environment)  
4. [Data Preparation](#4-data-preparation)  
   - [Main Dataset Structure](#41-main-dataset-structure)  
   - [Concept Dataset Structure](#42-concept-dataset-structure)  
   - [Example: CUB-200-2011 Setup](#43-example-cub-200-2011-setup)  
5. [Running the Code](#5-running-the-code)  
   - [Training a QCW Model](#51-training-a-qcw-model)  
   - [Evaluating a QCW Model](#52-evaluating-a-qcw-model)  
   - [Visualizing Concept Activations](#53-visualizing-concept-activations)  
   - [Debugging Common Issues](#54-debugging-common-issues)  
6. [Code Structure](#6-code-structure)  
   - [Top-Level Directories and Files](#61-top-level-directories-and-files)  
   - [Key Scripts](#62-key-scripts)  
     1. [`prepare_cub_qcw.py`](#1-prepare_cub_qcwpy)  
     2. [`train_qcw.py`](#2-train_qcwpy)  
     3. [`model_resnet_qcw.py` & `iterative_normalization.py`](#3-model_resnet_qcwpy--iterative_normalizationpy)  
     4. [`ConceptDataset_QCW.py`](#4-conceptdataset_qcwpy)  
7. [Concept Whitening Details](#7-concept-whitening-details)  
   - [Classic CW vs. Quantized CW](#71-classic-cw-vs-quantized-cw)  
   - [Subspaces, Redaction, and Unlabeled Axes](#72-subspaces-redaction-and-unlabeled-axes)  
8. [Advanced Usage & Customization](#8-advanced-usage--customization)  
   - [Changing Hyperparameters](#81-changing-hyperparameters)  
   - [Multiple BN/CW Layers](#82-multiple-bncw-layers)  
   - [Free Concepts and Unlabeled Axes](#83-free-concepts-and-unlabeled-axes)  
9. [Contributing](#9-contributing)  
10. [License](#10-license)  

---

## **1. Introduction**

This repository provides an implementation of **Quantized Concept Whitening (QCW)** for enhancing **interpretability** in deep neural networks, building upon the original “Concept Whitening” approach introduced by Chen, Bei, and Rudin (Nature Machine Intelligence, 2020). QCW extends classic CW by:

1. Organizing latent axes into **subspaces**, one per high-level concept (e.g., “wing,” “beak,” “eye”).  
2. Optionally redacting or masking irrelevant portions of images so the model only sees the concept-relevant region.  
3. Allowing **free/unlabeled** axes in each subspace, discovered via a winner-takes-all approach.

Over time, this code evolved from a prior “classic CW” repository to a more advanced “Quantized” version.

---

## **2. Repository Overview**

- **`train_qcw.py`**: The main training script for QCW, supporting command-line arguments for architecture, dataset paths, concept data, etc.  
- **`prepare_cub_qcw.py`**: Pre-processing script for converting raw CUB-200-2011 data into the QCW-friendly format (both main dataset and concept dataset).  
- **`MODELS/`**: Contains core code for ResNet-based QCW modules:
  - **`model_resnet_qcw.py`**: Wraps a `torchvision` ResNet with QCW blocks.  
  - **`iterative_normalization.py`**: Implements the IterNorm-based whitening + rotation procedure.  
  - **`ConceptDataset_QCW.py`**: A custom Dataset class that physically crops or redacts images for sub-concept alignment.  

We provide a typical **CUB** example as the default domain, but it can be extended to other classification tasks.

---

## **3. Installation & Environment**

**Recommended environment**: Python 3.11 with PyTorch + CUDA 12.1.  
If using **Conda**, you can create an environment from `environment.yml`:

```bash
conda env create -f environment.yml
conda activate QCW
```

The environment file includes:

- `pytorch-cuda=12.1`, `pytorch`, `torchvision`, `numpy`, `scikit-learn`, `scikit-image`, `matplotlib`, `pillow`, `seaborn`, `tqdm`, `tensorboard`, etc.

We tested on an **NVIDIA A5000** GPU with ~24GB VRAM. For large batch sizes, you may need similarly high-memory GPUs.

---

## **4. Data Preparation**

### **4.1 Main Dataset Structure**

Our code expects a **main dataset** with train/val/test splits in an **ImageFolder**-style layout, e.g.:

```
main_data/
├── train
│   ├── 001
│   │   ├── image1.jpg
│   │   ├── ...
│   └── 002
│       ├── ...
├── val
│   ├── 001
│   │   ├── ...
│   └── 002
│       ├── ...
└── test
    ├── 001
    │   ├── ...
    └── 002
        ├── ...
```

Each subfolder (`001`, `002`, etc.) is a class label containing images. The repository can handle other naming schemes (like species names in CUB), but numeric class labels are typical.

### **4.2 Concept Dataset Structure**

The concept dataset includes images for each **high-level** concept (e.g., `wing`, `beak`). Each high-level concept is subdivided into sub-concepts (like “has_wing_color::grey”). The code also supports bounding boxes for redaction or cropping.

Typical structure:

```
concept_data/
├── bboxes.json
├── concept_train
│   ├── wing
│   │   ├── has_wing_color::grey
│   │   │   ├── imageXYZ.jpg
│   │   │   ├── ...
│   │   ├── has_wing_color::brown
│   │   └── ...
├── concept_val
│   ├── wing
│   │   ├── ...
├── mappings.json  (may also reside at data/CUB, used for advanced subspace logic)
└── ...
```

`bboxes.json` is a dictionary mapping each concept image path to `[x1,y1,x2,y2]` bounding box coordinates, used if you choose **crop** or **redact** modes.

### **4.3 Example: CUB-200-2011 Setup**

1. Download the official **CUB_200_2011** dataset.  
2. Run **`prepare_cub_qcw.py`** to produce:
   - `main_data/` (train/val/test for classification).  
   - `concept_data/` (concept_train, concept_val, bounding box JSON).  
3. Adjust your `--concepts` argument to specify high-level parts (e.g. `wing,beak,nape,general`).

---

## **5. Running the Code**

### **5.1 Training a QCW Model**

Use **`train_qcw.py`** with arguments specifying:

1. **Data directories**:
   - `--data_dir`: Path to your `main_data` folder.  
   - `--concept_dir`: Path to your `concept_data` folder.  
2. **Concept List**: `--concepts "wing,beak,general"`.
3. **Architecture details**:
   - `--depth`: `18` or `50` for ResNet.
   - `--whitened_layers`: Comma-separated global block indices (e.g. `5` or `2,5`).
   - `--act_mode`: one of `mean`, `max`, `pos_mean`, `pool_max`.
4. **Hyperparams**:
   - `--epochs`, `--batch_size`, `--lr`, `--momentum`, `--weight_decay`, etc.
5. **Checkpoint**:
   - `--resume`: Path to checkpoint to resume from.
   - `--only_load_weights`: If set, only load model weights from checkpoint (ignore epoch/optimizer).
6. **Important Arguments**:
   - `--bboxes`: Path to bounding box JSON file. You only need this if the bounding box JSON is not in the `concept_dir`.

**Example**:

```bash
python train_qcw.py \
  --data_dir data/CUB/main_data \
  --concept_dir data/CUB/concept_data \
  --concepts "wing,beak,nape,general" \
  --prefix "RESNET18_QCW" \
  --whitened_layers "5" \
  --depth 18 \
  --act_mode "pool_max" \
  --epochs 100 \
  --batch_size 64 \
  --lr 0.1 \
  --momentum 0.9 \
  --weight_decay 1e-4
```

This will:

- Replace the 5th global residual block’s BN with **IterNormRotation**.  
- Use concept data from `concept_train/wing`, `concept_train/beak`, etc.  
- Periodically do concept alignment (default every 40 mini-batches).

### **5.2 Evaluating a QCW Model**

- Provide a `--resume` checkpoint to the same script and optionally add an evaluation mode.  
- For a quick example, to measure final test accuracy:

```bash
python train_qcw.py \
  --data_dir data/CUB/main_data \
  --concept_dir data/CUB/concept_data \
  --concepts "wing,nape" \
  --whitened_layers "5" \
  --resume checkpoints/RESNET18_QCW_5_wing_nape_checkpoint.pth \
  --prefix "RESNET18_QCW_wing_nape_eval" \
  --epochs 1
```

The script will run a single epoch but effectively skip training, then do a final test at the end. For more thorough or specialized evaluation (like concept top-k images), you may adapt the script or add your own analysis code. We have plans to make checkpoint loading load things like concepts and whitened layers to save you remembering them and make it easier to evaluate models.

### **5.3 Visualizing Concept Activations**

Currently, code to visualize top-k images for a concept axis is not fully integrated.

### **5.4 Debugging Common Issues**
1. **Consecutive Whitening**: If you specify multiple consecutive whitened layers, it is likely that alignment will fail. We also notice little to no improvement with multiple whitened layers. If you are looking for good insights for multiple layers we reccomend training multiple models with different whitened layers.
2. **No BN replaced**: If `--whitened_layers` is out of range (e.g., “12” on a ResNet18 with only 8 blocks), no replacement occurs. Check logs for “Attaching CW AFTER residual in layerX[...] (global idx N)”.  
3. **Concept alignment not improving**: Possibly your concept dataset is too small or the main classification overshadows alignment. Try adjusting `CW_ALIGN_FREQ`, or add more concept images.  
4. **Runtime out of memory**: Lower `--batch_size`, or switch from “crop” to “none” in concept dataset if you’re doing too big bounding boxes.  
5. **No free concepts**: The code has a `--use_free` flag but is not fully implemented. You must adapt `IterNormRotation` to define unlabeled axes.  

---

## **6. Code Structure**

A simplified directory tree:

```
├── data/
│   └── CUB/
│       ├── concept_data/          (bboxes.json, concept_train, concept_val)
│       ├── main_data/             (train, val, test)
│       └── mappings.json
├── dataset_scripts/
│       ├── prepare_cub_qcw.py     (Builds main_data, concept_data from raw CUB)
│       └── ...                    (misc. dataset prep and download scripts)
├── MODELS/
│   ├── ConceptDataset_QCW.py      (Dataset for concept images)
│   ├── iterative_normalization.py (IterNorm-based whitening & rotation)
│   ├── model_resnet_qcw.py        (ResNet with QCW blocks)
│   └── model_resnet.py            (Older code)
├── train_qcw.py                   (Main training loop)
├── environment.yml                (Conda env file)
├── checkpoints/                   (stores model checkpoints)
└── runs/                          (TensorBoard logs)
```

### **6.1 Top-Level Directories and Files**

- **`dataset_scripts/`**: Additional scripts for e.g. COCO or other datasets.  
- **`checkpoints/`**: Where .pth checkpoint files are saved.  
- **`runs/`**: Stores TensorBoard logs; view via `tensorboard --logdir runs/`.  
- **`environment.yml`**: Describes recommended conda environment.  

### **6.2 Key Scripts**

#### 1. `prepare_cub_qcw.py`
- Creates `main_data/` (train, val, test) from raw CUB images, splits them.  
- Creates `concept_data/` for each high-level concept (like “wing”), subdivided by sub-concepts (like “has_wing_color::brown”), plus bounding boxes if needed.  

#### 2. `train_qcw.py`
- The main entry point to train or resume a QCW model.  
- Incorporates concept alignment steps in `align_concepts(...)`.  
- Logs metrics to TensorBoard, including classification losses and concept alignment metrics.

#### 3. `model_resnet_qcw.py` & `iterative_normalization.py`
- **`model_resnet_qcw.py`**: Replaces certain BN blocks with a custom `BottleneckCW` or `BasicBlockCW` that calls `IterNormRotation` after the residual is added.  
- **`iterative_normalization.py`**: Contains `IterNormRotation`, which:
  1. Whitens input features.  
  2. Accumulates gradient-like signals for concept alignment.  
  3. Updates an orthogonal rotation matrix that aligns each axis with the concept.

#### 4. `ConceptDataset_QCW.py`
- A PyTorch `Dataset` that physically **crops** or **redacts** images using bounding boxes from `bboxes.json`.  
- Each subfolder is a sub-concept. The `__getitem__` returns `(image, subconcept_label)`.

---

## **7. Concept Whitening Details**

### **7.1 Classic CW vs. Quantized CW**
- **Classic**: One axis per concept; or each BN replaced with a single multi-axis rotation.  
- **Quantized**: Subdivides axes into subspaces (e.g., “beak subspace” of 5 axes). Offers $\max_{j \in S}$ logic to discover unlabeled axes.  
- The code base is partially in transition, so it has placeholders for free concepts, subspace-based logic, etc.

### **7.2 Subspaces, Redaction, and Unlabeled Axes**
- Setting `--disable_subspaces` lumps all concepts into a single set of axes.  
- Setting `crop_mode` in `ConceptDataset_QCW.py` to “redact” or “crop” zeroes out or physically crops irrelevant image areas.  
- `use_free` is available but not fully integrated. You’ll see references to unlabeled sub-concepts in `IterNormRotation`.

---

## **8. Advanced Usage & Customization**

### **8.1 Changing Hyperparameters**
- In `train_qcw.py`, you can specify `--lr`, `--epochs`, `--batch_size`, `--weight_decay`.  
- Edit `CW_ALIGN_FREQ` inside `train_qcw.py` if you want concept alignment to happen more or less often (default 40).

### **8.2 Multiple BN/CW Layers**
- Provide a comma-separated list to `--whitened_layers` like `--whitened_layers "2,5,8"`. Ensure your network has enough blocks. For ResNet18, only up to 8 blocks exist.

### **8.3 Free Concepts and Unlabeled Axes**
- The code has a `--use_free` flag and references to `use_free` in `model_resnet_qcw.py`, but the actual logic for adding extra “free axes” and updating them with a $\max$ operator is incomplete. To add unlabeled axes, you would need to:
  - Modify how subspaces are built in `build_concept_loaders()` or in `IterNormRotation`.  
  - Implement the $\max_{j \in S}$ strategy for each subspace.  

---

## **9. Contributing**

We welcome contributions that:
- **Implement free/unlabeled concept axes** e.g. implement the full “winner-takes-all” or “max_{j in S}” approach.  
- **Add improved alignment metrics** or new visualizations.  
- **Support more architectures** (e.g., DenseNet, VGG).  

**Pull Requests**: Please create a feature branch and open a PR referencing any relevant issues. Use descriptive commit messages and follow PEP-8 styling where practical.

**Issues**: If you encounter bugs (or perplexing alignment behaviors), open a GitHub issue with logs, config, and environment details so we can reproduce.

---

## **10. License**

This code is released under the MIT License. Please refer to the `LICENSE` file for details. If you use this work in academic research, please cite our Quantized Concept Whitening paper as well as the original Concept Whitening paper.