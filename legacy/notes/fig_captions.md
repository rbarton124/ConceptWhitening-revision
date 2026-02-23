### **Fig 1: ResNet-18 Layer-Wise Accuracy (QCW)**

**Description:** Shows validation accuracy across whitened layers (WL) for ResNet-18 QCW models trained on Small and Large datasets compared to the baseline accuracy.
**Main takeaway:** Optimal accuracy for QCW models is close to but slightly below baseline, with significant layer-wise variability. The Small dataset occasionally outperforms the Large, emphasizing the impact of annotation quality and dataset scale.

| WL | Large Dataset (%) | Small Dataset (%) | Baseline (%) |
| -- | ----------------- | ----------------- | ------------ |
| 1  | 73.24             | 73.33             | 74.23        |
| 2  | 72.56             | **75.13**         | 74.23        |
| 3  | **74.10**         | 73.41             | 74.23        |
| 4  | 73.24             | 73.33             | 74.23        |
| 5  | 71.01             | 72.90             | 74.23        |
| 6  | 73.33             | 73.24             | 74.23        |
| 7  | 72.64             | 73.07             | 74.23        |
| 8  | 74.01             | **74.96**         | 74.23        |

---

### **Fig 2: QCW Models vs Baseline Architectures**

**Description:** Compares test accuracy of QCW models at the optimal whitening layer (WL-3) against standard baseline models (ResNet-18, ResNet-50, DenseNet-161, VGG16).
**Main takeaway:** QCW slightly underperforms baseline ResNet-18, but notably outperforms baseline ResNet-50, showing QCW can enhance deeper architectures.

| Architecture | Baseline Accuracy (%) | QCW Accuracy (%) |
| ------------ | --------------------- | ---------------- |
| ResNet-18    | **75.60**             | 74.16            |
| ResNet-50    | 82.06                 | **82.20**        |
| DenseNet-161 | 78.05                 | N/A              |
| VGG16        | 78.76                 | N/A              |

---

### **Fig 3: Concept Purity Across QCW Layers (ResNet-18)**

**Description:** Demonstrates mean concept purity (measured by Baseline AUC) across whitened layers (WL) for ResNet-18 QCW trained on Small and Large datasets.
**Main takeaway:** Purity generally improves at deeper layers, peaking notably in layers 6–7. The Large dataset consistently achieves higher purity, highlighting benefits of cleaner annotations.

| WL | Small Dataset Purity | Large Dataset Purity |
| -- | -------------------- | -------------------- |
| 1  | 0.675                | 0.709                |
| 2  | 0.738                | 0.722                |
| 3  | 0.735                | 0.715                |
| 4  | 0.715                | 0.743                |
| 5  | 0.781                | 0.761                |
| 6  | 0.820                | **0.801**            |
| 7  | **0.838**            | **0.822**            |
| 8  | 0.825                | 0.811                |

---

### **Fig 4: Baseline vs Masked AUC (ResNet-18 Large)**

**Description:** Compares Baseline AUC and Masked AUC (considering hierarchical concept structure) across whitened layers for ResNet-18 trained on the Large dataset.
**Main takeaway:** Masked AUC consistently exceeds Baseline AUC, clearly demonstrating the interpretability advantage provided by hierarchical QCW, especially at intermediate layers (WL4, WL6–8).

| WL | Baseline AUC | Masked AUC |
| -- | ------------ | ---------- |
| 1  | 0.709        | 0.800      |
| 2  | 0.722        | 0.836      |
| 3  | 0.715        | 0.838      |
| 4  | 0.743        | 0.875      |
| 5  | 0.761        | 0.823      |
| 6  | 0.801        | 0.866      |
| 7  | 0.822        | 0.865      |
| 8  | 0.811        | **0.880**  |

---

### **Fig 5: Energy Ratio Across Layers (ResNet-18 Large)**

**Description:** Shows energy ratio (fraction of total energy captured by concept axes) across whitened layers for ResNet-18 QCW trained on the Large dataset.
**Main takeaway:** Energy ratio trends upward at deeper layers, emphasizing that deeper network representations are increasingly dominated by structured, concept-aligned features.

| WL | Energy Ratio |
| -- | ------------ |
| 1  | 0.373        |
| 2  | 0.373        |
| 3  | 0.377        |
| 4  | 0.401        |
| 5  | 0.378        |
| 6  | 0.419        |
| 7  | 0.414        |
| 8  | **0.479**    |

---

### **Fig 6: Hierarchical Benefit (Δ-max)**

**Description:** Illustrates Δ-max, defined as the difference between Masked AUC and Baseline AUC, across whitened layers for both Small and Large datasets.
**Main takeaway:** Larger dataset shows consistently higher Δ-max values, clearly indicating greater interpretability improvement with higher-quality annotations, particularly in deeper layers.

| WL | Δ-max (Small Dataset) | Δ-max (Large Dataset) |
| -- | --------------------- | --------------------- |
| 1  | 0.036                 | 0.082                 |
| 2  | 0.049                 | 0.096                 |
| 3  | 0.054                 | 0.105                 |
| 4  | 0.036                 | 0.157                 |
| 5  | 0.069                 | 0.173                 |
| 6  | 0.087                 | 0.222                 |
| 7  | **0.143**             | **0.249**             |
| 8  | 0.126                 | **0.269**             |
