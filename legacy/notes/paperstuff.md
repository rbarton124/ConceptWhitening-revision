# Experimental Setup, Accuracy Analysis & Reflections

You're helping to build a detailed scientific markdown report on our Quantized Concept Whitening (QCW) experiments. Before diving into detailed concept metrics, let’s carefully unpack the **experimental setup and accuracy trends**. I'm providing all experimental details, initial observations, hypotheses, and open questions explicitly below. Carefully absorb everything, then perform your own thoughtful analysis.

---

## 🔖 **Experimental Context & Background**

**Datasets Used:**

- **Large Dataset** (9 high-level concepts, 36 subconcepts)
```

back: has\_back\_color::{black, blue, white, yellow}
beak: has\_bill\_length::{longer\_than\_head, shorter\_than\_head}
belly: has\_belly\_color::{black, blue, red, white, yellow}
eye: has\_eye\_color::{black, red, white}
general: has\_size::{medium\_(9\_-*16\_in), very\_small*(3\_-\_5\_in)}
leg: has\_leg\_color::{black, red, yellow}
nape: has\_nape\_color::{blue, green, red, white}
tail: has\_tail\_shape::{fan-shaped, forked, pointed, squared};
has\_upper\_tail\_color::{black, blue, red, white, yellow}
throat: has\_throat\_color::{black, red, white, yellow}

```

- **Small Dataset** (2 high-level concepts, 9 subconcepts)
```

eye: has\_eye\_color::{black, blue, red\_free, white}
nape: has\_nape\_color::{black, blue\_free, green, red, white}

```

**Important notes:**

- Small dataset generally produced poorer results due to fewer samples and noisy labels (many annotation mistakes in original CUB dataset). Large dataset was carefully curated and cleaner.
- Each model was trained using pretrained ResNet-18 or ResNet-50 models from CUB (explicit choice for consistency).
- Training ran ~200 epochs (intentionally long to ensure convergence and thorough exploration).
- **Checkpoint chosen**: Best validation top-1 accuracy (NOT best alignment metric), as the goal was good concept performance without significantly compromising classification accuracy.

---

## 🔍 **Important Observations & Initial Hypotheses (guidance for your reflection)**

- Accuracy varies significantly by layer whitened.
- The model can sometimes become untrainable where QCW struggles to disentangle concepts (multi whitened layers, trying to whiten with no pretraining), leading to noisy latent spaces and unstable batch normalization.
- Hypothesis: QCW performs best at layers where the network’s natural latent representation already somewhat aligns with chosen concepts. Some layers (e.g., WL-8 of ResNet18, last layer) may struggle because they try to disentangle attribute-level concepts while already optimizing for class-level separations—potential conflict.
- Training from scratch with QCW in place consistently underperforms or simply doesnt learn, strongly indicating that QCW relies on a model having already learned general features. Pretraining (e.g., ImageNet or CUB) significantly improves performance.
- Simultaneous multi-layer whitening usually impairs performance or doesnt learn. Possibly an implementation flaw—e.g., gradient conflicts. Occasionally layers (5 & 7 once) trained successfully together, indicating the issue may be layer-specific or subtle implementation-dependent or could be just a challenging thing for the model to have two bn layers changing every alignment step (a staggered approach may help this).

---

## 📈 **Accuracy Results


### **Table A: ResNet-18 QCW Classification Summary** 

| Concept Group | Avg Val Top-1 | Avg Test Top-1 |   Val Range |  Test Range | Best Val WL | Val\@Best | Test\@Best | ΔVal vs Res18\_CUB | ΔTest vs Res18\_CUB |
| ------------: | ------------: | -------------: | ----------: | ----------: | ----------: | --------: | ---------: | -----------------: | ------------------: |
|       **LGR** |         73.02 |          74.05 | 71.01–74.10 | 72.71–74.96 |        WL 3 |     74.10 |      74.16 |              –1.21 |               –1.55 |
|       **SML** |         73.67 |          75.09 | 72.90–75.13 | 74.18–75.71 |        WL 2 |     75.13 |      75.71 |              –0.56 |               –0.51 |

> **Notes**:
> – **Δ** = group avg – ResNet-18\_CUB val/test (74.23/75.60).
> – Best WL by validation: WL 3 for LGR, WL 2 for SML.

---

### **Table B: ResNet-50 QCW Classification Summary (LGR only)**

| Concept Group | Avg Val Top-1 | Avg Test Top-1 |   Val Range |  Test Range | Best Val WL | Val\@Best | Test\@Best | ΔVal vs Res50\_CUB | ΔTest vs Res50\_CUB |
| ------------: | ------------: | -------------: | ----------: | ----------: | ----------: | --------: | ---------: | -----------------: | ------------------: |
|       **LGR** |         81.31 |          81.11 | 78.56–82.25 | 78.28–82.20 |        WL 3 |     82.25 |      82.20 |              –0.61 |               –0.95 |

> **Δ** = avg QCW – ResNet-50\_CUB (81.92/82.06).

---

### **Table C: Baseline CUB-Pretrained Models**

| Model             | Val Top-1 | Test Top-1 |
| :---------------- | --------: | ---------: |
| ResNet-18\_CUB    |     74.23 |      75.60 |
| ResNet-50\_CUB    |     81.92 |      82.06 |
| DenseNet-161\_CUB |     77.44 |      78.05 |
| VGG16\_CUB        |     76.59 |      78.76 |

---

### **Table D: ResNet-18 QCW Per-Layer Breakdown**

|  WL | Group | Val Top-1 | Test Top-1 |
| :-: | :---: | --------: | ---------: |
|  1  |  LGR  |     73.24 |      74.37 |
|  2  |  LGR  |     72.56 |      74.29 |
|  3  |  LGR  |     74.10 |      74.16 |
|  4  |  LGR  |     73.24 |      74.63 |
|  5  |  LGR  |     71.01 |      72.71 |
|  6  |  LGR  |     73.33 |      74.22 |
|  7  |  LGR  |     72.64 |      73.08 |
|  8  |  LGR  |     74.01 |      74.96 |
|  1  |  SML  |     73.33 |      74.87 |
|  2  |  SML  |     75.13 |      75.71 |
|  3  |  SML  |     73.41 |      75.39 |
|  4  |  SML  |     73.33 |      75.63 |
|  5  |  SML  |     72.90 |      74.20 |
|  6  |  SML  |     73.24 |      75.02 |
|  7  |  SML  |     73.07 |      74.18 |
|  8  |  SML  |     74.96 |      75.71 |

---

### **Table E: ResNet-50 QCW Per-Layer Breakdown (LGR)**

|  WL | Val Top-1 | Test Top-1 |
| :-: | --------: | ---------: |
|  2  |     81.22 |      80.73 |
|  3  |     82.25 |      81.81 |
|  6  |     81.99 |      81.53 |
|  7  |     81.30 |      81.46 |
|  10 |     81.82 |      82.20 |
|  11 |     82.08 |      81.14 |
|  14 |     78.56 |      78.28 |
|  15 |     81.22 |      81.70 |


---

## ✏️ **Your Analysis Task**

Provide your response clearly in **two parts**:

### (1) **Thoughtful Analysis & Reflections**
- Carefully analyze provided context & initial hypotheses. 
- Identify patterns, confirm or challenge provided hypotheses, propose improved or alternative explanations, and thoughtfully discuss reasons for accuracy variations between layers, pretraining benefits, and multi-layer whitening difficulties.
- Feel free to explore analytical tangents and new hypotheses as you see fit.

### (2) **Structured "Brain-Dump" Report Draft**
- Clearly summarize the experimental setup, key accuracy trends, and your refined or new hypotheses.
- Markdown format, cohesive but raw—this does not yet need to be publication-quality polished.

Provide both parts in a single response.





# Concept Purity, Hierarchy & Interpretability Analysis

Now let's deeply analyze how QCW performs at a concept-level, using interpretability and hierarchical metrics. Carefully read and internalize all details below.

---

## 🔖 **Metric Interpretation **



### Quick‐Reference: QCW Metric Cheat-Sheet

| **Metric**                     | **Core Idea**                                                | **How It’s Computed**                                                                                                                               | **Read-at-a-glance**                                                                              | **Why It Matters**                                                                               |
| ------------------------------ | ------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| **Baseline AUC** (Purity)      | Does the chosen axis separate “has concept” vs “no concept”? | ROC-AUC on that single axis.                                                                                                                        | **1.0 ≈** perfect; **0.5 ≈** random.                                                              | Higher values mean the axis is a clean detector for the concept.                                 |
| **Δ-max** (or **Slice-Ratio**) | Does hierarchy help?                                         | Compare AUC of best **global** axis vs best axis *inside* the concept’s high-level (HL) slice. Δ = Slice – Global.<br>Slice-Ratio = Slice / Global. | **Δ > 0** or **Ratio > 1** ⇒ hierarchy is useful.                                                 | Shows whether restricting attention to an HL subspace clarifies the concept.                     |
| **Energy-Ratio**               | How much activation energy sits in the HL slice?             | ‖activations inside slice‖ / ‖activations globally‖ (mean over positives).                                                                          | **≈ 1** ⇒ nearly all energy is in-slice.                                                          | High ratio means the model’s internal representation already aligns with the intended hierarchy. |
| **Masked AUC** (Hier-AUC)      | Can the slice alone do the job?                              | Re-compute AUC after zeroing axes outside the HL slice.                                                                                             | **Masked ≥ Baseline** ⇒ hierarchy retains purity.<br>Lower ⇒ model still needs off-slice context. | Tests dependence on “leaky” context outside the slice.                                           |
| **Mean Rank**                  | Relative salience of the designated axis.                    | Average rank (1 = highest) among all axes for positive images. Computed for raw activations and for masked-slice activations.                       | Lower is better; a drop after masking is good.                                                    | Indicates whether the axis becomes more prominent within its slice.                              |
| **Hit\@k**                     | Probability the axis is in the top-k.                        | Fraction of positives where rank ≤ k (raw & masked).                                                                                                | Higher is better; at k = slice size, ideal = 100 %.                                               | Quick binary view of rank improvement.                                                           |

**Relationships & Typical Targets**

* **Purity first.** If Baseline AUC < 0.8 the concept isn’t well captured.
* **Masked AUC & ranks** After purity is decent, hierarchy shouldn’t *hurt* purity, and ranks should improve to claim hierarchy adds value.
* **Hierarchy again** Provide sanity-checks: look for **Δ-max > 0.02** or Slice-Ratio > 1.05 to solidify that hierarchy is valuable.
* **Energy alignment.** Energy-Ratio ≥ 0.9 signals the network’s own features are already partitioned as desired.


---

## 🔍 **Important Observations & Open Questions (guidance)**

- Free concepts consistently achieve nearly perfect purity, far outperforming labeled subconcepts. This remains a puzzle—provide thoughtful hypotheses on why this happens.
- Annotation quality strongly correlated with purity. Small dataset especially noisy; large dataset cleaner.
- Results differ between ResNet18 and ResNet50—reflect thoughtfully on possible reasons.
- Metrics like Δ-max, energy ratio, masked-AUC, mean rank, and hit@k reveal varied hierarchy effectiveness—identify clear trends, note irregularities, and propose nuanced interpretations.

---

## 📈 **Metrics Results**

### QCW Model Analysis Report on Resnet18 Whitened on Small CUB Concept Dataset

#### Per-Layer, Per-Sub-Concept Metrics

##### Whitened Layer 1

| subconcept   |   Purity |   delta_max |   energy_ratio |   hier_AUC |   mean_rank_raw |   mean_rank_mask |   hit@3_raw |   hit@3_mask |
|--------------|----------|-------------|----------------|------------|-----------------|------------------|-------------|--------------|
| False        |    0.597 |       0.034 |          0.587 |      0.757 |            3.93 |             3.23 |       0.554 |        0.751 |
| False        |    0.709 |       0.071 |          0.654 |      0.729 |            3.93 |             3.57 |       0.619 |        0.738 |
| True         |    0.546 |       0.066 |          0.675 |      0.487 |            5.05 |             5.77 |       0.311 |        0.402 |
| False        |    0.559 |       0.04  |          0.628 |      0.667 |            3.12 |             2.61 |       0.632 |        0.809 |
| False        |    0.639 |      -0.04  |          0.73  |      0.886 |            2.99 |             2.06 |       0.668 |        0.934 |
| True         |    0.655 |       0.068 |          0.766 |      0.604 |            4.43 |             4.89 |       0.404 |        0.37  |
| False        |    0.812 |       0.053 |          0.754 |      0.868 |            3    |             2.44 |       0.699 |        0.877 |
| False        |    0.858 |       0.007 |          0.939 |      0.917 |            1.73 |             1.64 |       0.878 |        0.929 |
| False        |    0.701 |       0.028 |          0.773 |      0.836 |            2.58 |             2.17 |       0.746 |        0.865 |

##### Whitened Layer 2

| subconcept   |   Purity |   delta_max |   energy_ratio |   hier_AUC |   mean_rank_raw |   mean_rank_mask |   hit@3_raw |   hit@3_mask |
|--------------|----------|-------------|----------------|------------|-----------------|------------------|-------------|--------------|
| False        |    0.558 |      -0.032 |          0.689 |      0.7   |            4.44 |             3.82 |       0.354 |        0.688 |
| False        |    0.741 |       0.087 |          0.759 |      0.804 |            3.05 |             2.67 |       0.69  |        0.881 |
| True         |    0.906 |       0.035 |          0.724 |      0.871 |            2.88 |             2.7  |       0.713 |        0.779 |
| False        |    0.594 |       0.082 |          0.739 |      0.746 |            1.86 |             1.51 |       0.928 |        0.954 |
| False        |    0.618 |      -0.056 |          0.648 |      0.718 |            4.08 |             3.23 |       0.509 |        0.714 |
| True         |    0.843 |       0.176 |          0.771 |      0.695 |            4.34 |             4.38 |       0.426 |        0.559 |
| False        |    0.846 |       0.145 |          0.729 |      0.911 |            2.67 |             1.95 |       0.712 |        0.904 |
| False        |    0.843 |       0.01  |          0.875 |      0.922 |            1.85 |             1.54 |       0.832 |        0.949 |
| False        |    0.694 |      -0.002 |          0.663 |      0.796 |            3.23 |             2.55 |       0.594 |        0.805 |

##### Whitened Layer 3

| subconcept   |   Purity |   delta_max |   energy_ratio |   hier_AUC |   mean_rank_raw |   mean_rank_mask |   hit@3_raw |   hit@3_mask |
|--------------|----------|-------------|----------------|------------|-----------------|------------------|-------------|--------------|
| False        |    0.52  |      -0.014 |          0.588 |      0.809 |            3.3  |             2.38 |       0.604 |        0.835 |
| False        |    0.749 |       0.159 |          0.604 |      0.791 |            2.81 |             2.48 |       0.667 |        0.833 |
| True         |    0.921 |       0.239 |          0.67  |      0.859 |            4.01 |             3.52 |       0.516 |        0.705 |
| False        |    0.576 |       0.072 |          0.613 |      0.654 |            4.55 |             3.56 |       0.296 |        0.75  |
| False        |    0.608 |      -0.026 |          0.747 |      0.809 |            3.33 |             2.72 |       0.605 |        0.807 |
| True         |    0.895 |       0.034 |          0.76  |      0.76  |            4.21 |             4.01 |       0.422 |        0.593 |
| False        |    0.784 |      -0.005 |          0.761 |      0.934 |            2.67 |             1.93 |       0.699 |        0.904 |
| False        |    0.845 |       0.022 |          0.795 |      0.905 |            1.96 |             1.79 |       0.857 |        0.913 |
| False        |    0.717 |       0.003 |          0.818 |      0.893 |            2.16 |             1.82 |       0.797 |        0.905 |

##### Whitened Layer 4

| subconcept   |   Purity |   delta_max |   energy_ratio |   hier_AUC |   mean_rank_raw |   mean_rank_mask |   hit@3_raw |   hit@3_mask |
|--------------|----------|-------------|----------------|------------|-----------------|------------------|-------------|--------------|
| False        |    0.503 |      -0.038 |          0.533 |      0.468 |            5.86 |             5.25 |       0.049 |        0.468 |
| False        |    0.726 |       0.021 |          0.527 |      0.814 |            3.62 |             2.05 |       0.452 |        0.929 |
| True         |    0.946 |       0.267 |          0.679 |      0.836 |            4.84 |             4.66 |       0.336 |        0.508 |
| False        |    0.552 |       0.063 |          0.544 |      0.712 |            3.8  |             1.96 |       0.441 |        0.928 |
| False        |    0.645 |      -0.004 |          0.828 |      0.918 |            2.06 |             1.76 |       0.862 |        0.962 |
| True         |    0.882 |      -0.006 |          0.777 |      0.641 |            4.87 |             5.08 |       0.407 |        0.448 |
| False        |    0.765 |       0.009 |          0.858 |      0.919 |            2.18 |             1.93 |       0.836 |        0.945 |
| False        |    0.734 |       0.011 |          0.836 |      0.729 |            4.25 |             4.16 |       0.383 |        0.459 |
| False        |    0.682 |       0.004 |          0.848 |      0.915 |            1.65 |             1.47 |       0.923 |        0.983 |

##### Whitened Layer 5

| subconcept   |   Purity |   delta_max |   energy_ratio |   hier_AUC |   mean_rank_raw |   mean_rank_mask |   hit@3_raw |   hit@3_mask |
|--------------|----------|-------------|----------------|------------|-----------------|------------------|-------------|--------------|
| False        |    0.691 |       0.108 |          0.775 |      0.915 |            2.55 |             2    |       0.761 |        0.932 |
| False        |    0.736 |       0.059 |          0.807 |      0.767 |            3.86 |             3.17 |       0.476 |        0.833 |
| True         |    0.931 |       0.044 |          0.697 |      0.78  |            4.52 |             4.33 |       0.418 |        0.557 |
| False        |    0.563 |       0.041 |          0.773 |      0.7   |            2.39 |             1.92 |       0.822 |        0.921 |
| False        |    0.703 |       0.058 |          0.633 |      0.863 |            2.68 |             2    |       0.695 |        0.9   |
| True         |    0.959 |       0.133 |          0.744 |      0.775 |            3.95 |             4    |       0.504 |        0.626 |
| False        |    0.827 |       0.011 |          0.589 |      0.832 |            4.15 |             3.04 |       0.397 |        0.767 |
| False        |    0.895 |       0.081 |          0.701 |      0.935 |            1.91 |             1.63 |       0.827 |        0.944 |
| False        |    0.728 |       0.087 |          0.65  |      0.908 |            2.19 |             1.57 |       0.832 |        0.957 |

##### Whitened Layer 6

| subconcept   |   Purity |   delta_max |   energy_ratio |   hier_AUC |   mean_rank_raw |   mean_rank_mask |   hit@3_raw |   hit@3_mask |
|--------------|----------|-------------|----------------|------------|-----------------|------------------|-------------|--------------|
| False        |    0.692 |       0.137 |          0.662 |      0.887 |            4.02 |             2.34 |       0.378 |        0.915 |
| False        |    0.821 |       0.192 |          0.704 |      0.883 |            1.36 |             1.07 |       1     |        1     |
| True         |    0.968 |       0.134 |          0.691 |      0.852 |            4.27 |             4.13 |       0.492 |        0.656 |
| False        |    0.683 |       0.144 |          0.673 |      0.618 |            5.64 |             5    |       0.086 |        0.539 |
| False        |    0.727 |       0.012 |          0.781 |      0.883 |            2.98 |             2.38 |       0.674 |        0.86  |
| True         |    0.979 |       0.043 |          0.763 |      0.857 |            3.1  |             3.07 |       0.652 |        0.722 |
| False        |    0.883 |       0.047 |          0.819 |      0.952 |            1.6  |             1.34 |       0.945 |        0.986 |
| False        |    0.88  |       0.033 |          0.823 |      0.933 |            1.97 |             1.81 |       0.842 |        0.903 |
| False        |    0.743 |       0.045 |          0.798 |      0.923 |            1.8  |             1.53 |       0.886 |        0.962 |

##### Whitened Layer 7

| subconcept   |   Purity |   delta_max |   energy_ratio |   hier_AUC |   mean_rank_raw |   mean_rank_mask |   hit@3_raw |   hit@3_mask |
|--------------|----------|-------------|----------------|------------|-----------------|------------------|-------------|--------------|
| False        |    0.785 |       0.263 |          0.688 |      0.793 |            3.69 |             3.01 |       0.478 |        0.799 |
| False        |    0.743 |       0.138 |          0.692 |      0.792 |            1.43 |             1.02 |       0.976 |        1     |
| True         |    0.981 |       0.129 |          0.616 |      0.866 |            3.68 |             3.74 |       0.598 |        0.672 |
| False        |    0.713 |       0.141 |          0.688 |      0.656 |            4.42 |             4.31 |       0.342 |        0.618 |
| False        |    0.759 |       0.147 |          0.805 |      0.911 |            2.11 |             1.81 |       0.842 |        0.948 |
| True         |    0.982 |       0.1   |          0.724 |      0.74  |            4.13 |             4.4  |       0.496 |        0.567 |
| False        |    0.891 |       0.116 |          0.82  |      0.948 |            1.38 |             1.25 |       0.986 |        1     |
| False        |    0.892 |       0.109 |          0.816 |      0.894 |            2.26 |             2.21 |       0.755 |        0.837 |
| False        |    0.799 |       0.142 |          0.796 |      0.905 |            2.24 |             1.96 |       0.788 |        0.922 |

##### Whitened Layer 8

| subconcept   |   Purity |   delta_max |   energy_ratio |   hier_AUC |   mean_rank_raw |   mean_rank_mask |   hit@3_raw |   hit@3_mask |
|--------------|----------|-------------|----------------|------------|-----------------|------------------|-------------|--------------|
| False        |    0.777 |       0.214 |          0.726 |      0.711 |            4.24 |             3.61 |       0.428 |        0.713 |
| False        |    0.701 |       0.016 |          0.667 |      0.725 |            4.81 |             3.33 |       0.119 |        0.786 |
| True         |    0.901 |       0.215 |          0.731 |      0.689 |            5.62 |             6.26 |       0.262 |        0.328 |
| False        |    0.761 |       0.064 |          0.803 |      0.814 |            1.89 |             1.27 |       0.849 |        0.974 |
| False        |    0.754 |       0.14  |          0.806 |      0.906 |            2.21 |             2.01 |       0.842 |        0.905 |
| True         |    0.961 |       0.106 |          0.762 |      0.753 |            4.78 |             6.17 |       0.389 |        0.048 |
| False        |    0.857 |       0.115 |          0.869 |      0.923 |            1.31 |             1.26 |       0.973 |        0.986 |
| False        |    0.896 |       0.121 |          0.851 |      0.942 |            1.89 |             1.8  |       0.832 |        0.893 |
| False        |    0.815 |       0.147 |          0.818 |      0.903 |            2.28 |             2.11 |       0.811 |        0.888 |

#### Compact 8-Layer Summary

Below is the averaged performance across all subconcepts, per whitened layer.

|   WL |   Purity |   delta_max |   energy_ratio |   hier_AUC |   mean_rank_raw |   mean_rank_mask |   hit@3_raw |   hit@3_mask |
|------|----------|-------------|----------------|------------|-----------------|------------------|-------------|--------------|
|    1 |    0.675 |       0.036 |          0.723 |      0.75  |           3.418 |            3.153 |       0.612 |        0.742 |
|    2 |    0.738 |       0.049 |          0.733 |      0.796 |           3.156 |            2.706 |       0.64  |        0.804 |
|    3 |    0.735 |       0.054 |          0.706 |      0.824 |           3.222 |            2.69  |       0.607 |        0.805 |
|    4 |    0.715 |       0.036 |          0.714 |      0.772 |           3.681 |            3.147 |       0.521 |        0.737 |
|    5 |    0.781 |       0.069 |          0.708 |      0.831 |           3.133 |            2.629 |       0.637 |        0.826 |
|    6 |    0.82  |       0.087 |          0.746 |      0.865 |           2.971 |            2.519 |       0.662 |        0.838 |
|    7 |    0.838 |       0.143 |          0.738 |      0.834 |           2.816 |            2.634 |       0.696 |        0.818 |
|    8 |    0.825 |       0.126 |          0.781 |      0.818 |           3.226 |            3.091 |       0.612 |        0.725 |


---
### QCW Model Analysis Report on Resnet 18 Whitened on Larger CUB Concept Dataset

#### Per-Layer, Per-Sub-Concept Metrics

##### Whitened Layer 1

| subconcept   |   Purity |   delta_max |   energy_ratio |   hier_AUC |   mean_rank_raw |   mean_rank_mask |   hit@3_raw |   hit@3_mask |
|--------------|----------|-------------|----------------|------------|-----------------|------------------|-------------|--------------|
| False        |    0.63  |       0.074 |          0.314 |      0.888 |            9.98 |             4.93 |       0.173 |        0.898 |
| False        |    0.838 |       0.065 |          0.339 |      0.874 |            8.96 |             5.73 |       0.433 |        0.839 |
| False        |    0.648 |       0.064 |          0.308 |      0.907 |            9.16 |             4.07 |       0.179 |        0.917 |
| False        |    0.789 |       0.049 |          0.319 |      0.929 |            7.8  |             3.57 |       0.438 |        0.906 |
| False        |    0.603 |       0.058 |          0.237 |      0.938 |            7.91 |             1.76 |       0.332 |        0.982 |
| False        |    0.58  |       0.048 |          0.222 |      0.882 |           13.6  |             5.64 |       0.016 |        0.886 |
| False        |    0.687 |       0.06  |          0.472 |      0.767 |           12.1  |             9.38 |       0.35  |        0.711 |
| False        |    0.787 |       0.003 |          0.452 |      0.819 |            9.98 |             7.67 |       0.541 |        0.774 |
| False        |    0.855 |       0.03  |          0.61  |      0.907 |            4.94 |             4.04 |       0.742 |        0.911 |
| False        |    0.708 |       0.033 |          0.429 |      0.919 |            7.24 |             3.85 |       0.457 |        0.887 |
| False        |    0.881 |       0.035 |          0.548 |      0.955 |            4.04 |             2.53 |       0.794 |        0.949 |
| False        |    0.569 |       0.059 |          0.217 |      0.972 |           10.5  |             2.18 |       0.02  |        0.976 |
| False        |    0.591 |       0.053 |          0.219 |      0.688 |           18.1  |            12    |       0.008 |        0.705 |
| False        |    0.59  |       0.043 |          0.23  |      0.653 |           18.1  |            13    |       0.092 |        0.671 |
| False        |    0.811 |       0.368 |          0.26  |      0.29  |           25    |            26.2  |       0.06  |        0.274 |
| False        |    0.762 |       0.355 |          0.273 |      0.478 |           19.9  |            19.1  |       0.143 |        0.473 |
| False        |    0.759 |       0.132 |          0.377 |      0.918 |            7.76 |             4.43 |       0.335 |        0.921 |
| False        |    0.758 |       0.092 |          0.363 |      0.914 |            7.75 |             4.19 |       0.29  |        0.929 |
| False        |    0.757 |       0.138 |          0.382 |      0.936 |            7.62 |             3.21 |       0.304 |        0.957 |
| False        |    0.855 |       0.12  |          0.497 |      0.957 |            3.87 |             2.42 |       0.767 |        0.956 |
| False        |    0.761 |       0.025 |          0.363 |      0.821 |           11    |             7.52 |       0.342 |        0.795 |
| False        |    0.847 |       0.006 |          0.44  |      0.892 |            7.35 |             4.88 |       0.602 |        0.888 |
| False        |    0.68  |       0.094 |          0.366 |      0.756 |           13.2  |             9.85 |       0.277 |        0.754 |
| False        |    0.575 |       0.103 |          0.412 |      0.581 |           20.5  |            17    |       0     |        0.147 |
| False        |    0.64  |       0.118 |          0.419 |      0.62  |           18.2  |            15.5  |       0.146 |        0.354 |
| False        |    0.605 |       0.115 |          0.411 |      0.735 |           18.4  |            11.6  |       0     |        0.147 |
| False        |    0.568 |       0.086 |          0.405 |      0.552 |           21.8  |            18    |       0     |        0.036 |
| False        |    0.642 |       0.139 |          0.42  |      0.863 |            9.89 |             5.5  |       0.198 |        0.742 |
| False        |    0.667 |       0.026 |          0.391 |      0.546 |           19.9  |            18.2  |       0.016 |        0.185 |
| False        |    0.549 |       0.03  |          0.381 |      0.832 |           10.1  |             6.06 |       0.239 |        0.672 |
| False        |    0.669 |       0.142 |          0.419 |      0.923 |            8.64 |             3.11 |       0.133 |        0.851 |
| False        |    0.704 |       0.015 |          0.386 |      0.779 |           14.9  |             9.38 |       0.048 |        0.489 |
| False        |    0.728 |       0.033 |          0.311 |      0.632 |           18.1  |            14.5  |       0.133 |        0.592 |
| False        |    0.861 |       0.014 |          0.429 |      0.87  |            8.04 |             5.48 |       0.585 |        0.873 |
| False        |    0.691 |       0.103 |          0.327 |      0.905 |            9.62 |             4.13 |       0.293 |        0.913 |
| False        |    0.867 |       0.038 |          0.485 |      0.912 |            5.88 |             3.98 |       0.69  |        0.915 |

##### Whitened Layer 2

| subconcept   |   Purity |   delta_max |   energy_ratio |   hier_AUC |   mean_rank_raw |   mean_rank_mask |   hit@3_raw |   hit@3_mask |
|--------------|----------|-------------|----------------|------------|-----------------|------------------|-------------|--------------|
| False        |    0.634 |       0.068 |          0.265 |      0.736 |           17.6  |            10.5  |       0.006 |        0.735 |
| False        |    0.803 |       0.161 |          0.365 |      0.912 |            6.9  |             3.71 |       0.59  |        0.926 |
| False        |    0.627 |       0.091 |          0.263 |      0.748 |           16.1  |             9.87 |       0.041 |        0.751 |
| False        |    0.788 |       0.003 |          0.252 |      0.779 |           14.5  |             9.09 |       0.134 |        0.762 |
| False        |    0.618 |       0.064 |          0.18  |      0.898 |           10.8  |             3.32 |       0.093 |        0.937 |
| False        |    0.599 |       0.067 |          0.17  |      0.744 |           18    |            10.4  |       0.01  |        0.747 |
| False        |    0.69  |       0.054 |          0.41  |      0.755 |           13.7  |             9.98 |       0.232 |        0.721 |
| False        |    0.818 |       0.025 |          0.411 |      0.87  |            8.41 |             5.68 |       0.564 |        0.82  |
| False        |    0.853 |       0.028 |          0.517 |      0.933 |            4.97 |             3    |       0.768 |        0.926 |
| False        |    0.73  |       0.036 |          0.364 |      0.9   |           10.5  |             4.73 |       0.233 |        0.852 |
| False        |    0.887 |       0.04  |          0.575 |      0.949 |            4.31 |             2.8  |       0.76  |        0.939 |
| False        |    0.622 |       0.085 |          0.238 |      0.939 |           13.1  |             3.92 |       0.008 |        0.944 |
| False        |    0.656 |       0.19  |          0.289 |      0.774 |           15.1  |             8.88 |       0.025 |        0.803 |
| False        |    0.619 |       0.204 |          0.28  |      0.866 |            7.14 |             3.66 |       0.395 |        0.928 |
| False        |    0.814 |       0.466 |          0.268 |      0.465 |           21.2  |            19.8  |       0.136 |        0.458 |
| False        |    0.798 |       0.441 |          0.29  |      0.422 |           21.2  |            21.1  |       0.12  |        0.418 |
| False        |    0.781 |       0.083 |          0.474 |      0.886 |            8.27 |             5.66 |       0.442 |        0.889 |
| False        |    0.81  |       0.075 |          0.466 |      0.851 |            9.5  |             6.69 |       0.368 |        0.865 |
| False        |    0.781 |       0.077 |          0.496 |      0.894 |            6.24 |             4.48 |       0.587 |        0.913 |
| False        |    0.842 |       0.071 |          0.438 |      0.83  |            9.63 |             7.14 |       0.544 |        0.815 |
| False        |    0.845 |       0.069 |          0.444 |      0.916 |            7.64 |             4.16 |       0.493 |        0.904 |
| False        |    0.867 |       0.059 |          0.473 |      0.994 |            4.28 |             1.24 |       0.735 |        0.995 |
| False        |    0.671 |       0.072 |          0.398 |      0.756 |           13.3  |             9.7  |       0.359 |        0.747 |
| False        |    0.596 |       0.082 |          0.388 |      0.812 |           15.5  |             8.71 |       0     |        0.261 |
| False        |    0.637 |       0.067 |          0.377 |      0.751 |           16.2  |            11.2  |       0.057 |        0.392 |
| False        |    0.613 |       0.123 |          0.398 |      0.909 |            9.59 |             4.04 |       0.133 |        0.69  |
| False        |    0.57  |       0.095 |          0.395 |      0.769 |           18.1  |            11.1  |       0.002 |        0.133 |
| False        |    0.632 |       0.115 |          0.398 |      0.895 |           10.9  |             5.31 |       0.123 |        0.545 |
| False        |    0.78  |       0.028 |          0.376 |      0.917 |           15.8  |             6.47 |       0     |        0.333 |
| False        |    0.534 |      -0.063 |          0.35  |      0.74  |           16.2  |            10.9  |       0.015 |        0.313 |
| False        |    0.576 |       0.142 |          0.412 |      0.857 |           12.2  |             6.65 |       0.071 |        0.484 |
| False        |    0.7   |       0.082 |          0.377 |      0.893 |            8.36 |             4.32 |       0.266 |        0.769 |
| False        |    0.734 |       0.077 |          0.36  |      0.883 |           10    |             5.39 |       0.296 |        0.837 |
| False        |    0.873 |       0.029 |          0.444 |      0.935 |            5.85 |             3.18 |       0.702 |        0.937 |
| False        |    0.719 |       0.133 |          0.37  |      0.974 |            4.06 |             1.55 |       0.66  |        0.987 |
| False        |    0.862 |       0.024 |          0.441 |      0.935 |            6.42 |             3.41 |       0.632 |        0.92  |

##### Whitened Layer 3

| subconcept   |   Purity |   delta_max |   energy_ratio |   hier_AUC |   mean_rank_raw |   mean_rank_mask |   hit@3_raw |   hit@3_mask |
|--------------|----------|-------------|----------------|------------|-----------------|------------------|-------------|--------------|
| False        |    0.615 |       0.098 |          0.286 |      0.806 |           13.9  |             8.04 |       0.038 |        0.791 |
| False        |    0.806 |       0.147 |          0.332 |      0.855 |           10    |             6.17 |       0.406 |        0.834 |
| False        |    0.651 |       0.121 |          0.288 |      0.928 |            9.53 |             3.21 |       0.151 |        0.946 |
| False        |    0.748 |      -0.001 |          0.26  |      0.824 |           12.3  |             7.36 |       0.215 |        0.795 |
| False        |    0.664 |       0.077 |          0.159 |      0.768 |           16.2  |             8.71 |       0.036 |        0.787 |
| False        |    0.615 |       0.117 |          0.149 |      0.909 |           15.2  |             4.24 |       0.001 |        0.914 |
| False        |    0.715 |       0.025 |          0.418 |      0.771 |           12.8  |             9.3  |       0.284 |        0.714 |
| False        |    0.791 |       0.051 |          0.464 |      0.886 |            8.05 |             5.1  |       0.541 |        0.835 |
| False        |    0.848 |       0.05  |          0.47  |      0.881 |            8.19 |             5.34 |       0.537 |        0.832 |
| False        |    0.745 |       0.034 |          0.379 |      0.917 |            8.55 |             4.09 |       0.284 |        0.888 |
| False        |    0.888 |       0.078 |          0.508 |      0.982 |            2.98 |             1.4  |       0.801 |        0.979 |
| False        |    0.588 |       0.121 |          0.235 |      0.916 |           11.8  |             4.37 |       0.109 |        0.92  |
| False        |    0.644 |       0.147 |          0.259 |      0.854 |           12.8  |             5.22 |       0.074 |        0.902 |
| False        |    0.566 |       0.126 |          0.25  |      0.8   |           13.4  |             6.79 |       0.138 |        0.855 |
| False        |    0.845 |       0.482 |          0.284 |      0.493 |           20.8  |            18.8  |       0.159 |        0.488 |
| False        |    0.845 |       0.521 |          0.298 |      0.541 |           18.8  |            16.9  |       0.204 |        0.539 |
| False        |    0.789 |       0.01  |          0.606 |      0.879 |            7.42 |             6.07 |       0.613 |        0.882 |
| False        |    0.814 |       0.027 |          0.639 |      0.937 |            4.67 |             3.38 |       0.684 |        0.955 |
| False        |    0.824 |       0.024 |          0.66  |      0.936 |            4.45 |             3.08 |       0.772 |        0.957 |
| False        |    0.854 |       0.192 |          0.358 |      0.914 |            8.17 |             4.1  |       0.504 |        0.904 |
| False        |    0.803 |       0.137 |          0.287 |      0.901 |           15.3  |             5.23 |       0.014 |        0.808 |
| False        |    0.807 |       0.102 |          0.328 |      0.916 |            8.17 |             3.89 |       0.434 |        0.923 |
| False        |    0.66  |       0.131 |          0.285 |      0.807 |           13.3  |             7.88 |       0.262 |        0.804 |
| False        |    0.6   |       0.037 |          0.447 |      0.906 |           11.6  |             4.26 |       0.073 |        0.592 |
| False        |    0.663 |       0.125 |          0.488 |      0.772 |           13.6  |            10    |       0.278 |        0.481 |
| False        |    0.534 |       0.064 |          0.455 |      0.853 |           16.7  |             7.76 |       0.001 |        0.219 |
| False        |    0.561 |       0.042 |          0.448 |      0.566 |           22.2  |            17.9  |       0     |        0.1   |
| False        |    0.609 |       0.063 |          0.458 |      0.833 |           13.6  |             7.79 |       0.09  |        0.45  |
| False        |    0.684 |       0.016 |          0.434 |      0.9   |           11.2  |             4.69 |       0.169 |        0.582 |
| False        |    0.52  |      -0.06  |          0.404 |      0.574 |           20    |            16.7  |       0.06  |        0.284 |
| False        |    0.63  |       0.141 |          0.479 |      0.912 |            6.17 |             2.9  |       0.448 |        0.869 |
| False        |    0.676 |      -0.03  |          0.412 |      0.737 |           19.4  |            12.4  |       0.013 |        0.192 |
| False        |    0.737 |       0.143 |          0.311 |      0.841 |           11.5  |             6.79 |       0.281 |        0.79  |
| False        |    0.816 |       0.108 |          0.321 |      0.916 |            9.42 |             4.02 |       0.293 |        0.873 |
| False        |    0.723 |       0.181 |          0.31  |      0.944 |            8.47 |             2.95 |       0.35  |        0.928 |
| False        |    0.869 |       0.145 |          0.386 |      0.975 |            4.93 |             1.62 |       0.62  |        0.978 |

##### Whitened Layer 4

| subconcept   |   Purity |   delta_max |   energy_ratio |   hier_AUC |   mean_rank_raw |   mean_rank_mask |   hit@3_raw |   hit@3_mask |
|--------------|----------|-------------|----------------|------------|-----------------|------------------|-------------|--------------|
| False        |    0.642 |       0.122 |          0.292 |      0.883 |           11.4  |             4.95 |       0.195 |        0.874 |
| False        |    0.792 |       0.092 |          0.294 |      0.889 |           12.7  |             4.95 |       0.203 |        0.853 |
| False        |    0.661 |       0.144 |          0.298 |      0.797 |           15.3  |             8.13 |       0.145 |        0.784 |
| False        |    0.731 |       0.134 |          0.289 |      0.791 |           13.7  |             8.48 |       0.258 |        0.759 |
| False        |    0.696 |       0.143 |          0.217 |      0.918 |           11.3  |             3    |       0.198 |        0.949 |
| False        |    0.587 |       0.136 |          0.193 |      0.912 |           15.8  |             4.38 |       0.039 |        0.916 |
| False        |    0.732 |       0.06  |          0.374 |      0.779 |           14.5  |             9.46 |       0.267 |        0.645 |
| False        |    0.852 |       0.154 |          0.402 |      0.951 |            4.17 |             2.19 |       0.662 |        0.97  |
| False        |    0.856 |       0.136 |          0.475 |      0.964 |            5.39 |             2.04 |       0.7   |        0.921 |
| False        |    0.754 |       0.133 |          0.357 |      0.916 |           10.8  |             4.32 |       0.275 |        0.82  |
| False        |    0.873 |       0.178 |          0.448 |      0.942 |            6.34 |             3.09 |       0.576 |        0.883 |
| False        |    0.679 |       0.123 |          0.347 |      0.98  |            7.14 |             2.26 |       0.317 |        0.984 |
| False        |    0.686 |       0.093 |          0.362 |      0.898 |            8.59 |             3.33 |       0.32  |        0.959 |
| False        |    0.629 |       0.108 |          0.348 |      0.87  |            8.8  |             4.05 |       0.355 |        0.934 |
| False        |    0.846 |       0.55  |          0.29  |      0.504 |           21.2  |            18.5  |       0.147 |        0.496 |
| False        |    0.803 |       0.615 |          0.316 |      0.435 |           21.6  |            20.6  |       0.126 |        0.43  |
| False        |    0.812 |       0.064 |          0.563 |      0.826 |            9.36 |             7.72 |       0.594 |        0.829 |
| False        |    0.85  |       0.069 |          0.584 |      0.863 |            8.83 |             6    |       0.568 |        0.877 |
| False        |    0.855 |       0.08  |          0.64  |      0.885 |            6.13 |             4.79 |       0.75  |        0.902 |
| False        |    0.835 |       0.211 |          0.394 |      0.962 |            5.5  |             2.41 |       0.604 |        0.926 |
| False        |    0.762 |       0.13  |          0.35  |      0.908 |           11.3  |             4.7  |       0.329 |        0.781 |
| False        |    0.821 |       0.134 |          0.368 |      0.969 |            7.06 |             2.16 |       0.469 |        0.964 |
| False        |    0.717 |       0.163 |          0.332 |      0.958 |           10.1  |             3.05 |       0.267 |        0.903 |
| False        |    0.63  |       0.14  |          0.459 |      0.868 |           15    |             6.64 |       0.079 |        0.361 |
| False        |    0.715 |       0.153 |          0.514 |      0.886 |           10.8  |             6.01 |       0.316 |        0.5   |
| False        |    0.651 |       0.17  |          0.477 |      0.918 |           11.2  |             4.53 |       0.109 |        0.559 |
| False        |    0.623 |       0.131 |          0.465 |      0.869 |           17.9  |             7.55 |       0.002 |        0.114 |
| False        |    0.687 |       0.123 |          0.456 |      0.887 |           17.5  |             7.63 |       0.006 |        0.173 |
| False        |    0.737 |       0.133 |          0.456 |      0.937 |            8.93 |             3.42 |       0.376 |        0.683 |
| False        |    0.623 |       0.105 |          0.452 |      0.774 |           13.1  |             9.4  |       0.254 |        0.507 |
| False        |    0.651 |       0.17  |          0.474 |      0.835 |           12.3  |             7.18 |       0.249 |        0.572 |
| False        |    0.697 |       0.166 |          0.481 |      0.826 |           16.5  |             8.83 |       0.061 |        0.358 |
| False        |    0.785 |       0.159 |          0.411 |      0.973 |            3.79 |             1.65 |       0.689 |        0.95  |
| False        |    0.871 |       0.102 |          0.406 |      0.975 |            5.64 |             1.85 |       0.556 |        0.907 |
| False        |    0.734 |       0.168 |          0.407 |      0.97  |            4.62 |             1.82 |       0.618 |        0.967 |
| False        |    0.857 |       0.17  |          0.441 |      0.977 |            4.71 |             1.78 |       0.62  |        0.934 |

##### Whitened Layer 5

| subconcept   |   Purity |   delta_max |   energy_ratio |   hier_AUC |   mean_rank_raw |   mean_rank_mask |   hit@3_raw |   hit@3_mask |
|--------------|----------|-------------|----------------|------------|-----------------|------------------|-------------|--------------|
| False        |    0.708 |       0.198 |          0.239 |      0.811 |           13.5  |             7.81 |       0.067 |        0.787 |
| False        |    0.799 |       0.207 |          0.27  |      0.892 |            7.68 |             4.47 |       0.378 |        0.899 |
| False        |    0.695 |       0.176 |          0.244 |      0.79  |           13.5  |             8.42 |       0.148 |        0.764 |
| False        |    0.774 |       0.139 |          0.233 |      0.812 |           12.1  |             7.72 |       0.205 |        0.787 |
| False        |    0.7   |       0.116 |          0.187 |      0.771 |           12.3  |             8.69 |       0.26  |        0.784 |
| False        |    0.696 |       0.211 |          0.162 |      0.906 |           11.5  |             4.41 |       0.1   |        0.909 |
| False        |    0.756 |       0.061 |          0.409 |      0.844 |            9.44 |             6.66 |       0.45  |        0.773 |
| False        |    0.854 |       0.071 |          0.446 |      0.941 |            4.59 |             3.06 |       0.684 |        0.91  |
| False        |    0.885 |       0.08  |          0.505 |      0.946 |            4.85 |             2.8  |       0.642 |        0.921 |
| False        |    0.761 |       0.081 |          0.416 |      0.907 |            6.4  |             4.18 |       0.583 |        0.881 |
| False        |    0.892 |       0.091 |          0.515 |      0.931 |            4.94 |             3.56 |       0.722 |        0.883 |
| False        |    0.679 |       0.192 |          0.238 |      0.87  |           11.8  |             6.29 |       0.127 |        0.873 |
| False        |    0.705 |       0.204 |          0.257 |      0.871 |            9.16 |             4.46 |       0.32  |        0.918 |
| False        |    0.663 |       0.202 |          0.261 |      0.85  |            7.78 |             4.83 |       0.289 |        0.901 |
| False        |    0.878 |       0.513 |          0.404 |      0.277 |           25.9  |            26.5  |       0.071 |        0.259 |
| False        |    0.892 |       0.601 |          0.439 |      0.358 |           23.4  |            23.3  |       0.137 |        0.346 |
| False        |    0.819 |       0.065 |          0.586 |      0.813 |            8.88 |             8.24 |       0.571 |        0.816 |
| False        |    0.838 |       0.049 |          0.626 |      0.814 |            8.04 |             7.57 |       0.652 |        0.826 |
| False        |    0.847 |       0.09  |          0.669 |      0.843 |            6.57 |             6.48 |       0.685 |        0.859 |
| False        |    0.873 |       0.224 |          0.359 |      0.936 |            4.71 |             3.28 |       0.689 |        0.93  |
| False        |    0.821 |       0.217 |          0.324 |      0.912 |            7.03 |             4.19 |       0.493 |        0.904 |
| False        |    0.875 |       0.198 |          0.33  |      0.93  |            7.37 |             3.79 |       0.52  |        0.888 |
| False        |    0.726 |       0.187 |          0.288 |      0.922 |            7.76 |             4.01 |       0.401 |        0.901 |
| False        |    0.618 |       0.136 |          0.44  |      0.78  |           12.9  |             9.24 |       0.129 |        0.425 |
| False        |    0.724 |       0.179 |          0.489 |      0.849 |            8.02 |             6.46 |       0.475 |        0.608 |
| False        |    0.63  |       0.156 |          0.438 |      0.86  |           12    |             6.68 |       0.077 |        0.435 |
| False        |    0.608 |       0.144 |          0.432 |      0.757 |           12.7  |             9.75 |       0.133 |        0.459 |
| False        |    0.648 |       0.165 |          0.435 |      0.808 |           12.6  |             8.73 |       0.131 |        0.405 |
| False        |    0.79  |       0.134 |          0.397 |      0.78  |           13.3  |            10.2  |       0.095 |        0.339 |
| False        |    0.671 |       0.13  |          0.439 |      0.767 |           11.9  |             9.58 |       0.299 |        0.507 |
| False        |    0.649 |       0.174 |          0.448 |      0.796 |            9.8  |             8.04 |       0.397 |        0.635 |
| False        |    0.724 |       0.146 |          0.423 |      0.565 |           20.4  |            18.5  |       0.009 |        0.162 |
| False        |    0.761 |       0.209 |          0.306 |      0.916 |            6.98 |             3.81 |       0.481 |        0.89  |
| False        |    0.831 |       0.168 |          0.325 |      0.919 |            6.88 |             3.65 |       0.488 |        0.902 |
| False        |    0.734 |       0.12  |          0.279 |      0.931 |            7.14 |             3.14 |       0.408 |        0.939 |
| False        |    0.879 |       0.196 |          0.335 |      0.943 |            6.22 |             3.15 |       0.552 |        0.911 |

##### Whitened Layer 6

| subconcept   |   Purity |   delta_max |   energy_ratio |   hier_AUC |   mean_rank_raw |   mean_rank_mask |   hit@3_raw |   hit@3_mask |
|--------------|----------|-------------|----------------|------------|-----------------|------------------|-------------|--------------|
| False        |    0.711 |       0.252 |          0.291 |      0.832 |           12.9  |             7.15 |       0.158 |        0.8   |
| False        |    0.86  |       0.254 |          0.328 |      0.935 |            5.49 |             2.88 |       0.558 |        0.945 |
| False        |    0.716 |       0.234 |          0.302 |      0.853 |           12.4  |             6.34 |       0.181 |        0.807 |
| False        |    0.818 |       0.213 |          0.29  |      0.879 |            9.19 |             5.29 |       0.357 |        0.856 |
| False        |    0.752 |       0.234 |          0.214 |      0.762 |           13.6  |             9.12 |       0.251 |        0.769 |
| False        |    0.769 |       0.333 |          0.168 |      0.846 |           15.3  |             6.45 |       0.059 |        0.848 |
| False        |    0.785 |       0.102 |          0.415 |      0.829 |           10.8  |             7.71 |       0.377 |        0.674 |
| False        |    0.89  |       0.106 |          0.487 |      0.975 |            3.09 |             1.62 |       0.767 |        0.955 |
| False        |    0.903 |       0.11  |          0.546 |      0.962 |            4.03 |             2.45 |       0.716 |        0.911 |
| False        |    0.815 |       0.147 |          0.453 |      0.952 |            5.71 |             2.83 |       0.543 |        0.909 |
| False        |    0.906 |       0.108 |          0.562 |      0.975 |            2.6  |             1.64 |       0.854 |        0.965 |
| False        |    0.752 |       0.278 |          0.241 |      0.952 |            8.69 |             2.91 |       0.254 |        0.957 |
| False        |    0.802 |       0.322 |          0.272 |      0.871 |           10.6  |             5.15 |       0.295 |        0.902 |
| False        |    0.772 |       0.31  |          0.273 |      0.889 |            9.23 |             3.88 |       0.303 |        0.934 |
| False        |    0.889 |       0.571 |          0.358 |      0.357 |           23.6  |            23.7  |       0.108 |        0.339 |
| False        |    0.912 |       0.607 |          0.397 |      0.458 |           21.2  |            19.9  |       0.131 |        0.448 |
| False        |    0.815 |       0.082 |          0.62  |      0.814 |            8.71 |             8.03 |       0.639 |        0.817 |
| False        |    0.814 |       0.065 |          0.644 |      0.727 |           10.7  |            10.9  |       0.581 |        0.735 |
| False        |    0.844 |       0.089 |          0.678 |      0.833 |            7.18 |             6.6  |       0.685 |        0.848 |
| False        |    0.91  |       0.196 |          0.415 |      0.983 |            4.16 |             1.81 |       0.693 |        0.956 |
| False        |    0.849 |       0.149 |          0.398 |      0.977 |            4.53 |             1.89 |       0.644 |        0.959 |
| False        |    0.87  |       0.12  |          0.391 |      0.967 |            5.75 |             2.43 |       0.561 |        0.913 |
| False        |    0.762 |       0.199 |          0.364 |      0.964 |            5.74 |             2.57 |       0.577 |        0.92  |
| False        |    0.677 |       0.25  |          0.484 |      0.872 |           10.5  |             6.03 |       0.24  |        0.472 |
| False        |    0.782 |       0.249 |          0.529 |      0.885 |            9.44 |             6.21 |       0.354 |        0.513 |
| False        |    0.689 |       0.246 |          0.493 |      0.871 |           10.6  |             6.33 |       0.214 |        0.469 |
| False        |    0.67  |       0.23  |          0.484 |      0.848 |           12.5  |             7.38 |       0.102 |        0.357 |
| False        |    0.752 |       0.253 |          0.487 |      0.916 |            9.69 |             5.1  |       0.177 |        0.474 |
| False        |    0.85  |       0.247 |          0.492 |      0.935 |            6.92 |             3.67 |       0.434 |        0.704 |
| False        |    0.738 |       0.256 |          0.489 |      0.829 |           11.9  |             7.88 |       0.254 |        0.478 |
| False        |    0.701 |       0.255 |          0.496 |      0.879 |            9.28 |             5.76 |       0.359 |        0.563 |
| False        |    0.771 |       0.255 |          0.47  |      0.76  |           15.5  |            11.6  |       0.1   |        0.275 |
| False        |    0.789 |       0.221 |          0.391 |      0.96  |            4.38 |             2.06 |       0.678 |        0.958 |
| False        |    0.878 |       0.178 |          0.404 |      0.944 |            5.37 |             2.93 |       0.58  |        0.946 |
| False        |    0.745 |       0.181 |          0.368 |      0.967 |            4.67 |             1.78 |       0.616 |        0.983 |
| False        |    0.88  |       0.095 |          0.379 |      0.935 |            8.56 |             4.06 |       0.333 |        0.793 |

##### Whitened Layer 7

| subconcept   |   Purity |   delta_max |   energy_ratio |   hier_AUC |   mean_rank_raw |   mean_rank_mask |   hit@3_raw |   hit@3_mask |
|--------------|----------|-------------|----------------|------------|-----------------|------------------|-------------|--------------|
| False        |    0.743 |       0.283 |          0.294 |      0.868 |           10.2  |             5.79 |       0.235 |        0.834 |
| False        |    0.865 |       0.237 |          0.294 |      0.923 |            6.54 |             3.48 |       0.456 |        0.926 |
| False        |    0.725 |       0.259 |          0.3   |      0.833 |           11.7  |             7.03 |       0.237 |        0.767 |
| False        |    0.846 |       0.222 |          0.307 |      0.919 |            6.63 |             3.67 |       0.491 |        0.924 |
| False        |    0.757 |       0.278 |          0.216 |      0.801 |           12.1  |             7.54 |       0.275 |        0.814 |
| False        |    0.811 |       0.404 |          0.17  |      0.852 |           12.7  |             6.29 |       0.155 |        0.853 |
| False        |    0.805 |       0.102 |          0.484 |      0.883 |            6.91 |             5.37 |       0.581 |        0.789 |
| False        |    0.902 |       0.073 |          0.528 |      0.982 |            2.44 |             1.38 |       0.85  |        0.97  |
| False        |    0.915 |       0.093 |          0.557 |      0.944 |            4.74 |             3.19 |       0.7   |        0.847 |
| False        |    0.83  |       0.116 |          0.504 |      0.955 |            3.78 |             2.61 |       0.734 |        0.936 |
| False        |    0.918 |       0.077 |          0.57  |      0.967 |            2.72 |             2.06 |       0.846 |        0.94  |
| False        |    0.819 |       0.321 |          0.277 |      0.927 |            7.95 |             3.79 |       0.404 |        0.931 |
| False        |    0.858 |       0.266 |          0.313 |      0.914 |            6.19 |             3.09 |       0.492 |        0.951 |
| False        |    0.818 |       0.293 |          0.314 |      0.903 |            8.32 |             3.49 |       0.349 |        0.947 |
| False        |    0.902 |       0.585 |          0.407 |      0.343 |           24    |            24.2  |       0.142 |        0.326 |
| False        |    0.917 |       0.613 |          0.433 |      0.398 |           22.4  |            22    |       0.124 |        0.387 |
| False        |    0.811 |       0.071 |          0.621 |      0.862 |            7.47 |             6.47 |       0.65  |        0.865 |
| False        |    0.839 |       0.087 |          0.657 |      0.826 |            7.83 |             7.57 |       0.613 |        0.839 |
| False        |    0.814 |       0.048 |          0.664 |      0.895 |            5    |             4.23 |       0.75  |        0.913 |
| False        |    0.894 |       0.22  |          0.424 |      0.971 |            4.01 |             2.15 |       0.763 |        0.948 |
| False        |    0.877 |       0.181 |          0.418 |      0.978 |            3.66 |             1.85 |       0.685 |        0.973 |
| False        |    0.864 |       0.176 |          0.388 |      0.967 |            5.93 |             2.52 |       0.551 |        0.883 |
| False        |    0.778 |       0.192 |          0.344 |      0.958 |            6.05 |             2.86 |       0.503 |        0.917 |
| False        |    0.716 |       0.314 |          0.446 |      0.843 |           10.8  |             6.89 |       0.226 |        0.519 |
| False        |    0.819 |       0.334 |          0.489 |      0.866 |            9.59 |             6.75 |       0.323 |        0.494 |
| False        |    0.718 |       0.362 |          0.448 |      0.853 |           11    |             6.87 |       0.188 |        0.5   |
| False        |    0.697 |       0.328 |          0.436 |      0.703 |           16.2  |            12.5  |       0.073 |        0.296 |
| False        |    0.784 |       0.343 |          0.443 |      0.903 |            9.05 |             4.9  |       0.296 |        0.61  |
| False        |    0.885 |       0.324 |          0.451 |      0.888 |            9.43 |             5.81 |       0.365 |        0.587 |
| False        |    0.735 |       0.254 |          0.424 |      0.765 |           12.1  |             9.45 |       0.254 |        0.567 |
| False        |    0.75  |       0.351 |          0.454 |      0.842 |           10.5  |             7.14 |       0.273 |        0.505 |
| False        |    0.822 |       0.302 |          0.416 |      0.826 |           12.2  |             8.16 |       0.205 |        0.515 |
| False        |    0.818 |       0.234 |          0.349 |      0.946 |            5.08 |             2.57 |       0.597 |        0.931 |
| False        |    0.875 |       0.252 |          0.376 |      0.954 |            5.56 |             2.37 |       0.551 |        0.941 |
| False        |    0.792 |       0.192 |          0.319 |      0.944 |            7.06 |             2.9  |       0.407 |        0.932 |
| False        |    0.861 |       0.165 |          0.376 |      0.936 |            6.05 |             3.29 |       0.529 |        0.898 |

##### Whitened Layer 8

| subconcept   |   Purity |   delta_max |   energy_ratio |   hier_AUC |   mean_rank_raw |   mean_rank_mask |   hit@3_raw |   hit@3_mask |
|--------------|----------|-------------|----------------|------------|-----------------|------------------|-------------|--------------|
| False        |    0.727 |       0.25  |          0.422 |      0.884 |            9.6  |             5.44 |       0.327 |        0.822 |
| False        |    0.841 |       0.218 |          0.464 |      0.96  |            4.04 |             1.83 |       0.673 |        0.972 |
| False        |    0.743 |       0.246 |          0.44  |      0.91  |            7.73 |             4.22 |       0.399 |        0.863 |
| False        |    0.816 |       0.163 |          0.406 |      0.914 |            7.67 |             4.1  |       0.471 |        0.873 |
| False        |    0.773 |       0.405 |          0.278 |      0.845 |           10.1  |             6.02 |       0.359 |        0.859 |
| False        |    0.817 |       0.364 |          0.261 |      0.857 |           11    |             6.08 |       0.324 |        0.859 |
| False        |    0.835 |       0.203 |          0.514 |      0.908 |            7.79 |             4.67 |       0.457 |        0.752 |
| False        |    0.892 |       0.167 |          0.561 |      0.979 |            2.74 |             1.45 |       0.782 |        0.947 |
| False        |    0.904 |       0.186 |          0.586 |      0.948 |            5.12 |             3.04 |       0.605 |        0.863 |
| False        |    0.832 |       0.204 |          0.544 |      0.96  |            3.98 |             2.31 |       0.709 |        0.94  |
| False        |    0.911 |       0.168 |          0.597 |      0.974 |            2.8  |             1.75 |       0.828 |        0.951 |
| False        |    0.813 |       0.198 |          0.366 |      0.871 |            9.6  |             5.67 |       0.436 |        0.874 |
| False        |    0.836 |       0.189 |          0.466 |      0.858 |            8.52 |             5.47 |       0.533 |        0.877 |
| False        |    0.844 |       0.135 |          0.463 |      0.874 |            8.25 |             4.54 |       0.52  |        0.908 |
| False        |    0.887 |       0.438 |          0.336 |      0.527 |           19.3  |            17.6  |       0.185 |        0.516 |
| False        |    0.887 |       0.432 |          0.298 |      0.486 |           20.6  |            19.1  |       0.122 |        0.475 |
| False        |    0.813 |       0.275 |          0.555 |      0.909 |            6.08 |             4.4  |       0.685 |        0.912 |
| False        |    0.859 |       0.311 |          0.613 |      0.939 |            6.22 |             3.87 |       0.613 |        0.955 |
| False        |    0.836 |       0.282 |          0.613 |      0.948 |            4.45 |             2.76 |       0.772 |        0.967 |
| False        |    0.908 |       0.247 |          0.451 |      0.982 |            3.93 |             1.66 |       0.704 |        0.978 |
| False        |    0.841 |       0.207 |          0.413 |      0.987 |            4.52 |             1.4  |       0.562 |        1     |
| False        |    0.868 |       0.211 |          0.409 |      0.91  |            7.96 |             4.56 |       0.51  |        0.883 |
| False        |    0.764 |       0.2   |          0.367 |      0.861 |           10.2  |             6.37 |       0.363 |        0.823 |
| False        |    0.704 |       0.355 |          0.565 |      0.84  |           11.7  |             7.55 |       0.232 |        0.44  |
| False        |    0.798 |       0.393 |          0.615 |      0.907 |            6.67 |             4.58 |       0.551 |        0.665 |
| False        |    0.735 |       0.409 |          0.582 |      0.917 |            7.28 |             4.16 |       0.387 |        0.64  |
| False        |    0.652 |       0.385 |          0.568 |      0.84  |           12.2  |             7.67 |       0.155 |        0.374 |
| False        |    0.732 |       0.404 |          0.58  |      0.901 |            8.88 |             5.42 |       0.318 |        0.547 |
| False        |    0.887 |       0.378 |          0.582 |      0.944 |            6.52 |             3.69 |       0.503 |        0.704 |
| False        |    0.726 |       0.323 |          0.559 |      0.827 |           12.8  |             8.36 |       0.209 |        0.343 |
| False        |    0.733 |       0.396 |          0.581 |      0.806 |           11.7  |             8.91 |       0.29  |        0.466 |
| False        |    0.8   |       0.372 |          0.56  |      0.844 |           10.9  |             7.64 |       0.336 |        0.533 |
| False        |    0.784 |       0.161 |          0.377 |      0.826 |           11.1  |             7.1  |       0.368 |        0.773 |
| False        |    0.784 |       0.169 |          0.407 |      0.885 |            8.83 |             4.42 |       0.429 |        0.893 |
| False        |    0.778 |       0.143 |          0.381 |      0.91  |            8.07 |             3.9  |       0.48  |        0.9   |
| False        |    0.829 |       0.097 |          0.446 |      0.933 |            7.41 |             3.03 |       0.515 |        0.922 |

#### Compact 8-Layer Summary

Below is the averaged performance across all subconcepts, per whitened layer.

|   WL |   Purity |   delta_max |   energy_ratio |   hier_AUC |   mean_rank_raw |   mean_rank_mask |   hit@3_raw |   hit@3_mask |
|------|----------|-------------|----------------|------------|-----------------|------------------|-------------|--------------|
|    1 |    0.709 |       0.082 |          0.373 |      0.8   |          11.94  |            8.183 |       0.283 |        0.719 |
|    2 |    0.722 |       0.096 |          0.373 |      0.836 |          11.433 |            6.993 |       0.294 |        0.733 |
|    3 |    0.715 |       0.105 |          0.377 |      0.838 |          11.821 |            6.901 |       0.276 |        0.739 |
|    4 |    0.743 |       0.157 |          0.401 |      0.875 |          10.806 |            5.635 |       0.342 |        0.748 |
|    5 |    0.761 |       0.173 |          0.378 |      0.823 |          10.279 |            7.379 |       0.357 |        0.732 |
|    6 |    0.801 |       0.222 |          0.419 |      0.866 |           9.292 |            5.946 |       0.409 |        0.754 |
|    7 |    0.822 |       0.249 |          0.414 |      0.865 |           8.719 |            5.895 |       0.441 |        0.772 |
|    8 |    0.811 |       0.269 |          0.479 |      0.88  |           8.507 |            5.409 |       0.464 |        0.78  |
### QCW Model Analysis Report on Resnet 50 Whitened on Larger CUB Concept Dataset
For this one I will only give you the summary with averages as we don't need the layer minutiae we just want to do a broadstrokes comparison with the res18 large results.

#### Compact 4-Layer Summary

Below is the averaged performance across all subconcepts, per whitened layer.

|   WL |   Purity |   delta_max |   energy_ratio |   hier_AUC |   mean_rank_raw |   mean_rank_mask |   hit@3_raw |   hit@3_mask |
|------|----------|-------------|----------------|------------|-----------------|------------------|-------------|--------------|
|    3 |    0.715 |       0.093 |          0.362 |      0.849 |          11.911 |            6.561 |       0.271 |        0.734 |
|    7 |    0.75  |       0.168 |          0.366 |      0.836 |          11.319 |            7.057 |       0.319 |        0.704 |
|   11 |    0.805 |       0.226 |          0.419 |      0.871 |           9.11  |            5.729 |       0.425 |        0.761 |
|   15 |    0.613 |       0.098 |          0.305 |      0.644 |          14.809 |           13.51  |       0.132 |        0.503 |

## ✏️ **Your Analysis Task**

Provide your response clearly in **two parts**:

### (1) **In-depth Metrics Analysis & Reflections**
- Analyze provided context, data, and your thoughts deeply. 
- Discuss concept purity variations, hypotheses about free-concept purity phenomenon, annotation quality impact, hierarchical metric insights, and differences between architectures.

### (2) **Structured "Brain-Dump" Report Draft**
- Clearly describe QCW's concept-level performance, hierarchy benefit, and interpretability insights.
- Raw markdown prose—unpolished but thorough.

Provide both parts in a single response.





### Task 3: QCW vs Original CW Comparative Reflection

Finally, we'll explicitly analyze how Hierarchical QCW compares directly to original CW. Carefully read context below, then thoughtfully analyze.

---

## 🔖 **Context & Comparison Details**

- Baseline Original CW (small dataset, WL-7, ResNet18) purity metrics provided.
- QCW without free concepts (same setup) allows direct comparison.
- Placeholder for more layers, ResNet50, DenseNet, ImageNet (future TODO).

---

## 📈 **Comparison Results Placeholder (insert later)**

```

\[CW vs QCW RESULTS HERE]

```

---

## ✏️ **Your Analysis Task**

Provide your response clearly in **two parts**:

### (1) **Reflective Comparative Analysis**
- Carefully compare CW vs QCW results. Quantify hierarchical benefits explicitly.
- Identify clearly when and why hierarchy improves purity or interpretability, discuss any accuracy trade-offs, note clearly when hierarchy might not help, and hypothesize why.

### (2) **Structured "Brain-Dump" Comparative Draft**
- Write markdown summarizing CW vs QCW insights, advantages, limitations, and future directions.
- Unpolished, exploratory style encouraged.

Provide both parts in a single response.


Ok we need to throw in CBM results and Classic CW results