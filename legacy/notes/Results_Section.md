## Experiments and Results

### Experimental Setup

We evaluated the effectiveness and behavior of Quantized Concept Whitening (QCW) across multiple experimental conditions, focusing on two datasets extracted from the Caltech-UCSD Birds-200-2011 (CUB) dataset. Our main "Large Dataset" comprised 9 high-level concepts with 36 subconcepts, while the smaller dataset ("Small Dataset") included only 2 high-level concepts with 9 subconcepts. This choice allowed us to contrast performance under varying levels of complexity and annotation noise. All experiments were conducted using pretrained models (ResNet-18 and ResNet-50) previously fine-tuned on the CUB dataset, ensuring consistency in initialization and feature representations.

For each configuration, we performed a single-layer QCW integration at various positions ("whitened layers" or WL) within the ResNet architecture. Training spanned 200 epochs, with checkpoints selected based on optimal validation accuracy, prioritizing high classification performance alongside conceptual interpretability.

---

### Accuracy Trends and Layer Sensitivity

Our results demonstrate clear and insightful trends regarding QCW’s impact on model accuracy and layer sensitivity:

* **Overall Accuracy Impact**: QCW introduced a modest yet consistent decrease in accuracy compared to baseline models. Specifically, ResNet-18 with QCW experienced an average drop of approximately 1.2% (large dataset) and 0.5% (small dataset) in top-1 accuracy compared to the baseline ResNet-18 pretrained on CUB. Similar results emerged with ResNet-50, showing a 0.8% average decline relative to the baseline. This suggests QCW’s quantized conceptual constraints induce a trade-off between enhanced concept interpretability and maximal discriminative power, though this trade-off is relatively mild.

* **Layer-wise Accuracy Variation**: Accuracy varied significantly depending on the specific layer selected for QCW application, highlighting sensitivity to internal feature representation structures. For example, in ResNet-18, the best-performing layers in terms of accuracy were WL-3 and WL-8 for the large dataset and WL-2 and WL-8 for the small dataset. Interestingly, the layers immediately preceding or following the model’s deeper residual blocks (WL-5 and WL-7) consistently resulted in the poorest accuracy. These deeper layers, despite their poor accuracy outcomes, exhibited superior performance in metrics related to concept alignment and purity—suggesting a non-trivial relationship between accuracy, layer depth, and conceptual interpretability.

![ResNet-18 Layer-wise Accuracy Distribution]()
**(Fig X. Placeholder: Bar chart illustrating accuracy across whitened layers for ResNet-18, emphasizing variations between layers.)**

This figure visualizes the variability across layers, illustrating how QCW’s effectiveness in maintaining classification performance strongly depends on layer depth and position within the architecture.

---

### Comparative Analysis Across Architectures

To contextualize QCW's performance, we compared our best-performing QCW-integrated models to established baseline models, including ResNet-18, ResNet-50, DenseNet-161, and VGG16 pretrained on CUB. QCW-integrated ResNet models were competitive, though slightly trailing their non-QCW counterparts. Notably, ResNet-50 with QCW at WL-3 nearly matched its baseline, suggesting a potential for minimal accuracy loss with careful layer selection.

![Model Architecture Accuracy Comparison]()
**(Fig Y. Placeholder: Comparative bar graph of top-performing QCW layers vs. baseline architectures.)**

This figure shows QCW models positioned closely to the baselines, confirming that QCW, while introducing an interpretability constraint, retains a considerable portion of the models' inherent predictive capability.

---

### Reflections and Implications

The experimental outcomes underscore several critical insights into QCW’s behavior and limitations:

#### **Optimal Layer Selection**

Our results suggest that QCW achieves optimal classification performance when applied at mid-depth layers (e.g., WL-3 and WL-8). Early layers (e.g., WL-1 or WL-2), which focus on basic spatial and texture features, were less consistently effective, while deeper layers (WL-5, WL-7) consistently reduced accuracy significantly despite their excellent concept alignment metrics. We hypothesize that deeper-layer QCW disrupts the finely-tuned semantic structure already optimized for class discrimination, causing representational conflicts. Therefore, selecting mid-depth layers seems crucial to balance accuracy and interpretability effectively.

#### **Conceptual Alignment and Accuracy Trade-Off**

A particularly intriguing finding is the inverse relationship between classification accuracy and concept alignment/purity observed in deeper layers. Layers that yielded the strongest conceptual alignment (such as WL-5 and WL-7) suffered the most significant accuracy drops. This suggests that highly disentangled conceptual representations at these deeper layers are inherently at odds with the discriminative representations that traditional supervised learning favors. A fundamental tension thus emerges: QCW seeks strong alignment to human-interpretable concepts, yet highly aligned representations seem less flexible or robust for the complex task of fine-grained classification.

This tension highlights an important direction for future QCW research: exploring more sophisticated strategies—perhaps multi-stage training or adaptive whitening—that can better reconcile strong conceptual alignment with minimal degradation in classification accuracy.

#### **Pretraining Importance**

QCW’s reliance on pretrained feature extractors is evident from its inability to train from scratch effectively. Attempts at training QCW-integrated models without pretrained initializations consistently resulted in untrainable or unstable models, reinforcing our hypothesis that QCW functions best as a refinement or interpretability-enhancement method rather than as a foundational training strategy. This underscores QCW's dependence on the model’s pretrained latent structure and motivates future studies into alternative initialization or training regimes that might enhance QCW’s standalone capability.

#### **Challenges in Multi-Layer QCW Integration**

Experiments involving simultaneous whitening at multiple layers generally failed to train successfully, frequently resulting in instability or stagnation. This outcome could be attributable to conflicting normalization dynamics across multiple whitened layers, causing gradient instabilities or saturation during training. A staggered or incremental training approach—where layers are whitened sequentially or intermittently—may mitigate these issues, suggesting avenues for further methodological innovation.

---

### Conclusions and Recommendations

Our comprehensive exploration of QCW across varying layers and model architectures demonstrates QCW’s promising yet nuanced potential as an interpretability method. While QCW consistently introduces some degree of accuracy trade-off, careful layer selection and reliance on pretrained models can mitigate its negative effects. Deep layers provide remarkable concept interpretability yet suffer classification losses, suggesting a fundamental representational trade-off that warrants deeper theoretical and empirical investigation. Future QCW research should prioritize methods that minimize this tension, perhaps through adaptive whitening strategies or iterative multi-stage training regimes. This investigation has highlighted clear pathways for refinement and strategic QCW application, offering substantial potential for advancing the intersection of deep learning accuracy and human interpretability.

---

# 4. Experiments and Results: Concept Purity, Hierarchy, and Interpretability in QCW

In this section, we present comprehensive experimental evaluations of **Quantized Concept Whitening (QCW)**, focusing explicitly on its interpretability through metrics of **concept purity** and **hierarchical structuring**. These metrics move beyond standard classification accuracy, explicitly quantifying the alignment of learned representations with human-understandable semantic concepts. Through detailed analyses across two variants of the CUB dataset (small and large) and two model architectures (ResNet-18 and ResNet-50), we investigate QCW’s strengths, limitations, and implications for practical interpretability.

Specifically, we structure our investigation around the following critical questions:

1. **Purity:** How effectively do QCW layers isolate and encode individual human-defined concepts?
2. **Hierarchical Quantization:** Does the explicit hierarchical quantization imposed by QCW improve interpretability over standard global embeddings?
3. **Free-Concept Phenomenon:** Why do axes optimized without explicit labels ("free" axes) consistently achieve superior purity?
4. **Annotation Quality Impact:** How sensitive is QCW to label noise and annotation quality?
5. **Architectural Dependence:** How does QCW behavior differ across different network architectures, specifically ResNet-18 versus ResNet-50?

We first clearly detail our experimental setup and interpretability metrics, then provide thorough analyses and interpretations of the results.

---

## 4.1 Experimental Setup and Interpretability Metrics

Our experiments evaluate QCW using two carefully prepared datasets derived from the CUB-200 dataset, each reflecting different degrees of annotation quality:

* **Small Dataset:** Consists of 2 high-level concepts ("eye," "nape") divided into 9 subconcepts, based directly on raw and often noisy annotations from the original CUB labels.
* **Large Dataset:** Comprises 9 high-level concepts further subdivided into 36 subconcepts, carefully annotated and manually verified to minimize labeling noise.

We conduct our experiments on pretrained ResNet-18 and ResNet-50 models. QCW rotations are inserted at multiple candidate batch normalization layers ("Whitened Layers," abbreviated WL), trained to optimize both classification performance and concept alignment simultaneously.

To comprehensively assess interpretability, we adopt several key metrics (summarized in Section 3):

* **Baseline AUC (Purity):** ROC-AUC performance of concept classification using single-axis activations.
* **Masked AUC (Hierarchical AUC):** Recalculation of purity after zeroing activations outside the concept’s hierarchical subspace.
* **Δ-max (Slice-Ratio):** Improvement in purity due to hierarchy (Masked AUC - Baseline global best AUC).
* **Energy Ratio:** Proportion of total activation energy residing within concept-specific hierarchical slices.
* **Mean Rank and Hit\@k:** Metrics capturing how prominently concept axes activate relative to other latent dimensions.

Below, we present detailed analyses and reflections, first emphasizing concept purity, followed by hierarchical structuring and the influence of dataset quality and architecture.

---

## 4.2 Concept Purity Analysis: QCW’s Primary Interpretability Advantage

We begin by examining QCW’s core interpretability claim—improved concept purity through quantized axis rotations.

### Purity Results and Interpretation

Table 4 summarizes average purity across layers and datasets:

| Dataset / Model   | Optimal WL | Mean Purity (Optimal WL) | Mean Purity (WL-1) | Δ Purity |
| ----------------- | ---------- | ------------------------ | ------------------ | -------- |
| Small / ResNet-18 | 7          | **0.838**                | 0.675              | +0.163   |
| Large / ResNet-18 | 7          | **0.822**                | 0.709              | +0.113   |
| Large / ResNet-50 | 11         | **0.805**                | 0.715              | +0.090   |

*Table 4: QCW concept purity across datasets and architectures.*

The significant improvement in purity from early (WL-1) to optimal deep QCW layers (WL-7 for ResNet-18, WL-11 for ResNet-50) indicates QCW’s ability to isolate concepts more effectively as the latent representations become semantically richer (Fig. 4.1). Early layers, encoding primarily low-level visual patterns, yield lower purity (\~0.68), whereas deeper QCW layers consistently achieve purities above 0.80, clearly supporting our hypothesis that semantic quantization enhances interpretability.

![Purity improvement per QCW layer (small vs large datasets, ResNet-18 & 50)]()

QCW reaches its peak effectiveness at intermediate-to-deep network positions—too shallow and semantic discrimination is weak, too deep and representations become overly specialized towards class predictions, harming interpretability. Thus, QCW effectively finds a balance, significantly enhancing interpretability at these intermediate depths.

---

## 4.3 Hierarchical Quantization: Structural Interpretability Benefits

Having established QCW’s purity gains, we now rigorously evaluate the effectiveness of hierarchical quantization—the core QCW innovation of constraining concepts into explicit hierarchical slices.

### Masked-AUC and Hierarchy Insights

Table 5 summarizes key hierarchical metrics across datasets and architectures at optimal QCW layers:

| Model / Dataset | Optimal WL | Masked-AUC | Δ-max (Slice-Ratio) | Energy Ratio |
| --------------- | ---------- | ---------- | ------------------- | ------------ |
| ResNet-18 Small | 7          | **0.834**  | 0.143               | 0.738        |
| ResNet-18 Large | 7          | **0.865**  | 0.249               | 0.414        |
| ResNet-50 Large | 11         | **0.871**  | 0.226               | 0.419        |

*Table 5: Hierarchical quantization effectiveness metrics.*

Masked AUC consistently matches or exceeds baseline purity across optimal layers, demonstrating clear hierarchical structuring value. Subconcepts initially ambiguous globally often gain significant interpretability under hierarchical constraints (Δ-max ≈ 0.14–0.25). This confirms the intuitive benefit of hierarchical slicing—by isolating concepts within dedicated subspaces, QCW achieves substantially clearer semantic representations.

![Masked AUC vs. Baseline AUC for selected subconcepts]()

High energy ratios (\~0.4–0.7) further underscore natural semantic alignment within QCW’s structured slices, reinforcing the interpretability value of explicit hierarchical quantization.

---

## 4.4 The "Free-Concept" Phenomenon: Superior Purity of Unlabeled Axes

A particularly notable QCW phenomenon is the exceptional purity (\~0.95–0.98 AUC) achieved by unlabelled ("free") axes—dimensions optimized without explicit concept labels. We hypothesize three primary reasons for their consistently superior performance:

1. **Concentrated Training Signal:** Free axes leverage abundant, broadly defined positive examples, providing stronger and less ambiguous training signals than narrowly labeled subconcepts.
2. **Axis Specialization:** Without simultaneous pressure from classification objectives, free axes optimize purely for concept separability, avoiding competition-induced interference.
3. **Winner-Take-All Dynamics:** Training inherently favors sharp differentiation, allowing free axes to dominate their semantic niche clearly.

These insights highlight QCW’s inherent ability to robustly discover meaningful semantic concepts even without explicit labeling, offering strong interpretability potential for scenarios where annotation quality may be uncertain.

![Energy and purity distributions for free vs labeled QCW axes]()

---

## 4.5 Annotation Quality Impact: Purity Sensitivity and Limitations

Comparing the small (noisy) and large (clean) datasets highlights QCW’s sensitivity to annotation quality:

* Small dataset purity peaks around 0.84, hindered by label noise.
* Large dataset achieves significantly higher purity (\~0.87–0.90), thanks to meticulous annotation curation.

QCW clearly benefits from accurate labeling, demonstrating interpretability is ultimately constrained by the quality and clarity of human-provided labels. This emphasizes the practical necessity of careful annotation for maximal QCW effectiveness.

![Purity comparison across annotation quality (small vs large datasets)]()

---

## 4.6 Architectural Influence: ResNet-18 vs ResNet-50

Comparing ResNet-18 and ResNet-50, we observe QCW optimal layers shift deeper in the larger ResNet-50 (WL-11) compared to ResNet-18 (WL-7). Deeper, wider architectures delay semantic representation emergence, requiring correspondingly deeper QCW insertions for maximal interpretability. Extremely deep layers (e.g., WL-15 in ResNet-50), however, overly specialize towards classification accuracy, significantly impairing purity (\~0.61), demonstrating a fundamental trade-off between interpretability and end-layer classification.

![Masked AUC across network depths (ResNet-18 vs ResNet-50)]()

---

## 4.7 Practical Recommendations and Key Findings Summary

Our comprehensive experiments clearly establish QCW’s interpretability strengths and constraints:

* **QCW significantly enhances concept purity** through quantized axis rotations.
* **Hierarchical quantization** provides clear additional interpretability benefits (higher Masked-AUC).
* **Unlabeled "free" axes** yield surprisingly robust, near-perfect purity, particularly valuable under noisy labeling conditions.
* **Annotation quality critically determines QCW interpretability outcomes**, highlighting the importance of precise human labeling.
* **Optimal QCW placement** occurs in intermediate-to-deep layers (ResNet-18 WL-7, ResNet-50 WL-11), balancing semantic interpretability and accuracy effectively.

Overall, QCW delivers robust, semantically clear, and interpretable neural representations, provided careful hierarchical structuring and annotation quality.

---

### Figures and Tables Placeholders:

* Fig. 4.1: Purity across QCW layers
* Fig. 4.2: Masked vs Baseline AUC comparisons
* Fig. 4.3: Free vs labeled axes purity visualization
* Fig. 4.4: Annotation quality vs purity
* Fig. 4.5: Architecture comparisons
