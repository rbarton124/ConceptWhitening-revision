## 0 Why measure?

**Concept-Whitening (CW)** gives every labelled concept its own latent axis. **Quantised CW (QCW)** goes a step further: it groups axes into *high-level (HL) slices* such as *wing*, *tail*, *beak* … and forces free (unlabelled) axis to live inside one slice.
We therefore have **two intertwined questions**

1. **Axis⇔concept alignment** Did the training actually place “red-wing” on a single axis?
2. **Slice usefulness** Does restricting axes to slices *help* or merely constrain the model?

---

## 1 Axis selection & terminology

Before *any* metric that involves ROC-AUC, we must agree which axis stands for which subconcept.

| step                         | what we look at                                                                                                                                                                                                                 | why we look at it                                          |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| **1.1 Mean activation scan** | For *each* subconcept we take the mean activation over all its positives and pick the axis with the highest mean. (Labeled concepts are already pinned to their assigned axis, but free concepts may land on any free axis in their slice.) | Cheap heuristic—works surprisingly well.                   |
| **1.2 Verify with AUC**      | Usually the top-mean axis is *not* the best classifier.  After we choose, we then compute AUC to judge quality.                                                                                | ROC-AUC is truly ranking-sensitive; mean activation isn’t. |

---

## 2 Purity metrics – *“One axis <=> one concept?”*

### 2.1 Best-axis ROC-AUC (the old CW score)

* Mask = 1 for images of the subconcept, 0 otherwise
* Score = activation of its designated axis
* **AUC** answers: “How often does the axis fire higher on positives than on negatives?”

```
auc_best = roc_auc_score(mask, axis_scores)
```

> **Reading the plot**
> *0.5* = random, *< 0.5* = inverted sign, *≥ 0.9* = superb axis.

---

### 2.2 Leak bar (exclusive to free concepts)

If the **most-activated** axis for a free concept turns out to be a *labelled* axis, we call that a *leak*.
We draw the free axis in **blue**, the leaking labelled axis in **red**.
Large red bar ⇒ the slice-isolation loss failed to protect a labelled slot.

---

### 2.3 Masked-AUC

*What happens if we zero the score whenever the image’s HL slice ≠ current slice?*

1. Copy the axis-score vector.
2. `score[image_hl != my_hl] = 0`
3. Re-compute ROC-AUC.

Interpretation

| outcome           | meaning                                                              |
| ----------------- | -------------------------------------------------------------------- |
| Masked ≈ Baseline | Axis was already slice-pure.                                         |
| Masked > Baseline | Noise outside the slice hurt discrimination → **good sign for QCW**. |
| Masked < Baseline | Axis exploited off-slice context → maybe slice too strict.           |

---

## 3 Hierarchy metrics – *“Is the slice *helping* at the aggregate level?”*
> (all these metrics kind of do this but these were our first hierarchy metrics that worked so we lumped them in as "hierarchy metrics")

### 3.1 Δ-max

For every positive image *i* we take

* `g_max(i)` = highest activation across **all** axes
* `hl_max(i)` = highest activation restricted to **my slice**

Compute two AUCs and subtract:

$$
\Delta_{\text{max}} = \operatorname{AUC}(hl_{\max})\;-\;\operatorname{AUC}(g_{\max})
$$

* **Positive** Δ → an oracle that only looks *inside* the slice beats one that can peek everywhere.
* **Negative** Δ → valuable clues leak outside the slice.

### 3.2 Energy ratio

Average over positives:

$$
\mathrm{EnergyRatio} =
\frac{\lVert \mathbf{a}_{\text{slice}}\rVert_2}{\lVert \mathbf{a}_{\text{full}}\rVert_2},
\quad 0 < \text{ratio} \le 1
$$

*Closer to 1* → nearly all activation power lives where it should.

---

## 4 Rank metrics – *“How high does the axis rank among all K axes?”*

Both of these metrics are re-computed after slice-masking, so we obtain *(raw, masked)* pairs. The comparison between these two shows us how well our hierarchy is working.

### 4.1 Mean rank

For each positive image, rank axes descending by activation; record the rank of the designated axis.

*Lower is better* (1 = highest possible).

### 4.2 Hit\@k

Binary: is the axis in the top-k? We pre-compute **Hit\@1, 3, 5, k\_full** (where *k\_full* = size of the HL slice; masking forces Hit@ k\_full to 1).

> **Guarantees**
> *Raw Hit@ k\_full ≤ Masked Hit@ k\_full = 1*.
> Gains at **smaller k** (1,3,5) quantify practical benefit.

---

## 5 Putting it all together – decision tree for interpretation

1. **Check best-axis AUC**
   *If < 0.7* → axis mis-aligned ⇒ stop, slice doesn’t matter.
2. **Look at Masked-AUC**
   *Masked > Baseline* → QCW slice removes noise ⇒ success.
   *Masked ≈ Baseline* → already slice-pure ⇒ neutral.
   *Masked < Baseline* → slice hides useful context ⇒ investigate.
3. **Consult Δ-max / Energy ratio**
   *High Δ & high energy* → slice is both necessary and sufficient.
   *Low energy but positive Δ* → information is concentrated in few axes; potential to compress.
4. **Rank metrics** tell usability inside the full classifier:
   *Mean-rank should drop* and *Hit\@k should rise* after masking.
   