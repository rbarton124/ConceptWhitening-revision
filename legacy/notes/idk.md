Ok now we have  everything working I want to provide you with some results and metrics and we are going to put together a markdown document in the style of a scientific paper presenting all our results and experiments etcetera. First I am going to give you a full rundown on all the experimetns we have had so far.

To set this up we usually run on 2 datasets:

**Large dataset — 9 high-level concepts, 36 total subconcepts**

```
back: has_back_color::black, has_back_color::blue, has_back_color::white, has_back_color::yellow
beak: has_bill_length::longer_than_head, has_bill_length::shorter_than_head
belly: has_belly_color::black, has_belly_color::blue, has_belly_color::red, has_belly_color::white, has_belly_color::yellow
eye: has_eye_color::black, has_eye_color::red, has_eye_color::white
general: has_size::medium_(9_-_16_in), has_size::very_small_(3_-_5_in)
leg: has_leg_color::black, has_leg_color::red, has_leg_color::yellow
nape: has_nape_color::blue, has_nape_color::green, has_nape_color::red, has_nape_color::white
tail: has_tail_shape::fan-shaped_tail, has_tail_shape::forked_tail, has_tail_shape::pointed_tail, has_tail_shape::squared_tail, has_upper_tail_color::black, has_upper_tail_color::blue, has_upper_tail_color::red, has_upper_tail_color::white, has_upper_tail_color::yellow
throat: has_throat_color::black, has_throat_color::red, has_throat_color::white, has_throat_color::yellow
```
**Small dataset — 2 high-level concepts, 9 total subconcepts**

```
eye: has_eye_color::black, has_eye_color::blue, has_eye_color::red_free, has_eye_color::white
nape: has_nape_color::black, has_nape_color::blue_free, has_nape_color::green, has_nape_color::red, has_nape_color::white
```


The small dataset is what we did most of our testing on and tested free concepts on, we will get around to doing free concepts for the large dataset later, that is a future experiment as is integrating  the free concept loss into the actual main optimization loss function, and figuring out why free concepts have such near perfect purity when the regular concepts don't, also looking into using redaction and blurring instead of cropping concept images, I have already experimented with this once but the results were shockingly bad which made me think that something else went wrong so I will be looking into that in the future ( add placeholder parts for all of this in the writeup as we dont have data on this yet but will eventually)

The small dataset yields the worst results mostly because it was the first dataset so the concepts we picked didnt have very many training samples and werent very well annotated (the cub dataset has lots and lots of mistakes in terms of parts annotation, looking yourself you will quickly find about half of the attributes are mislabled for some attributes). The larger dataset was more carefully picked. We also found that concept purity was a pretty direct correlation with the quality of the annotation or the number of incorrect samples in that attribute/concept directory.

Overall though results have been good. Each model was trained from a pretrained CUB resnet 18 or 50 that will be clear later. They were trained for about 200 epochs which was total overkill but we decided on it early and stuck with it. Then we performed our metrics testing not on the checkpoints with the best concept alignement metrics (alignment is an in training metric we measure to keep track of how well the models are learning the concepts), but the checkpoints with the best validation top 1 accuracy so the best quality models not the models most optimized for concept whitening (this is important because the goal here is to not affect model quality while still getting good concept data not just to get the best concept purity result).

Now the accuracies change dramatically depending on which layer is whitened as you will see. This is just symptom of some layers posing a harder task to the concept whitening task leading to noiser worse disentanglement and thus a worse batch norm layer that is constantly changing leading to worse accuracy. This is not so much a limitation of concept whitening as it is inevitable, it doesnt work great for every spot in the network it works great for layers that are naturally just already associated with the concepts so at the point in the network where the network on its own has distilled the problem down to something that somewhat resembles our concepts or a complexity level that is on par with our concepts (idk if this makes sense or is even true it is just a badly put, and I am sure there is a better way to put it, hypothesis). In this same vein and likely for a similar reason we have found that we get much much better concept purity and alignemnt results when training from a mdoel checkpoint that is already trained on the task. Trying to train a model from scratch on a task with concept whitening layers already in place and active greatly hinder training. It is at least reccomended to use some sort of starting point like imagenet. Again this is likely because trying to do concept whitening on a network or at a spot in the network (in the different layers having different accuracies case) where the model doesn't really have a strong grasp on the main task and thus obviously doesnt have a strong grasp on the concept and hasnt learned to seperate or is not at a point where it is even possible to seperate the space into different concepts because it is at a point where the concepts haven't really emerged or are simply to tangled up with the classification objective (idk maybe you can come up with some better hypothesis here this might sound a bit wonky) the whitening layer will have trouble doing its job and will likely be changing the space a lot after each whitening step because it is having trouble grasping and creating any structure in the latent space, this then will hinder the main classification task because the normalization layer is not acting predictably and is chagning the latent space in weird ways often. We have also found that training with multiple layers whitening at the same time can often pose too challenging a task for the network and thus hinders the main classification task (best to whiten one layer at a time) (it could also just be a limit of our implementation, one thought is we are accidentally pushing the same gradients to both layers when we have more than one layer active at the same time, no testing has been done to verify this, overall the arhcitecture just wasnt built with multiple layers at the same time in mind so a lot of things could be causing it to go wrong, it may not be a limitation of concept whitening itself, sometimes in certain situations multiple layers are able to learn at the same time I think layer 3 and 7 learned well at the same time once). However, for training the model from a pretrained checkpoint on the task with only one layer at a time on a good layer CW performs well and doesnt hinder the main classifcation task very much as you can see from our results.


So I will need some commentary on the accuracy drop. Noting that accuracy drop is expected in layers where concept whitening struggles to seperate effectively or maybe is seperating concept at an inoportune moment for the network (like the drop we see in purity and accuracy for WL 8 (the last layer for resnet18) this might be happening because the network is trying to seperate stuff into classes and we are trying to seperate the latent space into attributes [idk tho this is just a hypothesis]). 

[accuracy results here]

Now that we know how accuracy is affected we need to go over how we did concept-wise. After training the models we test for concept performance with the concept purity metric. but we don't only want to see concept purity we also want to see how free concepts and hierarchical concepts are doing as this is what QCW improves upon regular CW. So we have a few metrics other than just cocnept purity.


Here's a comprehensive guide on interpreting the metrics used in your **Hierarchical Quantized Concept Whitening (QCW)** analysis:

---

# Comprehensive Guide to QCW Metrics Interpretation

QCW aims to disentangle the latent space of neural networks into clearly defined, human-understandable concepts, arranged in a hierarchical structure. Each metric below measures a different facet of how effectively QCW achieves this interpretability, purity, and hierarchical organization.

---

## 1. **Concept Purity (Baseline AUC)**

### **What is it?**

* **Purity** is measured using **ROC-AUC**, a standard metric from classification tasks, computed independently per concept.
* **ROC-AUC** quantifies how effectively the designated axis distinguishes between positive examples (images that have a concept) and negative examples (images without that concept).
* An AUC of:

  * `0.5` means random guessing (no purity).
  * `1.0` indicates perfect separation of positives from negatives.

### **Why do we measure it?**

* Purity tells us how "cleanly" or accurately each concept is captured by its designated axis.
* Higher purity indicates that the QCW model has clearly identified and isolated each concept onto a distinct axis.

### **How to interpret it:**

* Values close to `1.0`: The axis strongly activates only for relevant images, showing excellent disentanglement.
* Values around `0.5-0.6`: The axis is weakly discriminative—concept clarity is poor.

### **Implications for model quality:**

* High purity values (≥0.9) suggest that the QCW model is successfully learning meaningful concepts.

---

## 2. **Hierarchy Benefit (Δ-max or Slice-Ratio)**

### **What is it?**

* Compares the ROC-AUC of the most active axis in the entire latent space (`global-max`) versus the most active axis only within the concept’s **high-level (HL) subspace** (`slice-max`).
* **Δ-max**: Slice-max AUC minus Global-max AUC.
* **Slice-Ratio**: Slice-max AUC divided by Global-max AUC.

### **Why do we measure it?**

* To verify if the hierarchical structure (restricting attention to high-level subspaces) helps identify concepts more clearly.
* Positive Δ-max or Slice-Ratio > 1 means that restricting to the HL subspace improves the model’s interpretability.

### **How to interpret it:**

* **Δ-max > 0 or Slice-Ratio > 1**: Hierarchy is beneficial, meaning the relevant HL subspace enhances the discriminative power for the concept.
* **Δ-max < 0 or Slice-Ratio < 1**: Hierarchy restriction reduces performance—implying important context exists outside the slice.

### **Implications for model quality:**

* A strongly positive Δ-max or Slice-Ratio significantly >1 demonstrates the value of hierarchical organization in QCW.

---

## 3. **Energy-Ratio**

### **What is it?**

* Measures the fraction of total activation energy (the magnitude of latent activations) that resides inside the concept’s high-level subspace.
* Defined as:

$$
\text{Energy Ratio} = \frac{\text{Mean activation energy in HL slice}}{\text{Mean activation energy globally}}
$$

### **Why do we measure it?**

* Quantifies how focused activation energy is within the relevant high-level subspace.
* Indicates whether the hierarchical structure aligns well with the model's internal activation patterns.

### **How to interpret it:**

* **Energy-Ratio near 1**: Most energy resides inside the relevant subspace, showing clear hierarchical alignment.
* **Energy-Ratio far below 1**: Significant activation energy is leaking outside the intended subspace, showing weaker hierarchical alignment.

### **Implications for model quality:**

* High energy ratio values (\~0.9+) show strong hierarchical alignment, boosting interpretability.

---

## 4. **Masked-AUC (Hierarchical AUC)**

### **What is it?**

* A modified ROC-AUC calculated after setting activations of all axes outside the relevant HL subspace to zero.
* Compares "Baseline AUC" (with all axes active) against this masked version.

### **Why do we measure it?**

* Tests explicitly how reliant a concept’s discriminative power is on axes within its high-level slice.
* Helps understand if out-of-subspace axes provide meaningful contextual clues.

### **How to interpret it:**

* **Masked-AUC ≥ Baseline AUC**: The concept’s clarity improves or stays stable when restricted to its slice, confirming that hierarchy works well.
* **Masked-AUC < Baseline AUC**: Removing context from out-of-subspace axes worsens clarity—indicating important contextual information is encoded outside the HL subspace.

### **Implications for model quality:**

* Consistently higher or equal Masked-AUC suggests effective hierarchical structure.
* Lower Masked-AUC highlights areas where the hierarchical model might benefit from adjustments.

---

## 5. **Rank-Based Metrics (Mean Rank & Hit\@k)**

### **What are they?**

* **Mean Rank**: The average position (rank) of the designated axis activation among all axes for images containing the concept (lower is better).
* **Hit\@k**: The proportion of times the designated axis activation is among the top-k highest activations in images containing the concept (higher is better).

These metrics are calculated twice:

* **Baseline**: Using all axes.
* **Masked**: After zeroing activations outside the concept's HL subspace.

### **Why do we measure them?**

* They assess whether the hierarchy restriction helps the concept’s axis stand out relative to others.
* Captures relative ordering (rather than absolute activation), directly testing hierarchical interpretability.

### **How to interpret it:**

* **Mean Rank (lower is better)**:

  * A reduction after masking means the designated axis improves relative prominence within its slice.
* **Hit\@k (higher is better)**:

  * Increase after masking means the concept’s axis ranks more prominently within the HL slice.
  * For `k` equal to the number of axes in the HL slice, this metric should ideally reach 100%.

### **Implications for model quality:**

* Improving mean rank and hit\@k after masking indicates that hierarchical structure effectively organizes latent space, improving interpretability.
* Failure to reach high hit\@k (especially for k=slice size) identifies axes that do not clearly capture their intended concepts even within their HL subspace.

---

## **Summary Table for Quick Reference**

| Metric Name       | Measures                                | Good result                                       | Interpretation (Ideal Result)                |
| ----------------- | --------------------------------------- | ------------------------------------------------- | -------------------------------------------- |
| Baseline AUC      | Purity per axis (absolute separation)   | Close to 1                                        | Axis is highly concept-specific              |
| Δ-max/Slice-Ratio | Hierarchy improvement (slice vs global) | Δ-max > 0, Slice-Ratio > 1                        | Hierarchy improves discriminative clarity    |
| Energy-Ratio      | Focused activation within HL subspace   | Close to 1                                        | Strong hierarchical alignment                |
| Masked-AUC        | Reliance on HL subspace axes            | ≥ Baseline AUC                                    | Hierarchy retains or improves concept purity |
| Mean Rank         | Axis relative prominence                | Reduced rank after masking                        | Axis improves hierarchical prominence        |
| Hit\@k            | Axis within top-k axes                  | Higher after masking (ideally 100% at slice size) | Axis clearly ranks within HL subspace        |

---

## **Overall Implications & Value**

These metrics together provide a nuanced and multi-faceted assessment of how effectively QCW learns human-interpretable and hierarchically-structured concept representations:

* **Purity (AUC)** shows if each axis individually captures a meaningful concept.
* **Δ-max/Slice-Ratio and Energy-Ratio** explicitly evaluate if hierarchical structuring enhances clarity and interpretability.
* **Masked-AUC** quantifies how concepts rely on their intended subspace.
* **Rank/Hit metrics** measure whether hierarchy clarifies relative prominence among concepts.


Now that you have a good handle on the metrics here are the results so you can interpret them. You will note I am not giving you everything just the metrics and results that I found important.


[results here]


You will note that purity is extraordinarily high for free concepts, this is something I am still investigating and will continue to look into. right now I do not have a good answer (if you have a hypothesis I am open to it). You will notices that the results are different between resnet50 and resnet18, densenet and imagenet results are not done yet but add placeholders for that stuff in the writeup.


We will also need to compare our results with that of normal OG CW and show that hierarchy actually improves purity for hierarchical tasks. I have the purity results from training original CW on the small dataset just for WL_7 resnet18 (We should probably get more results from other layers you can add that in as a placeholder). I will also give you the results of doing hierarchical QCW on the smaller dataset for WL_7 (once again we should probably get other layers for this [we might already have that data I am not sure]) but without the two concepts marked free for a more direct comparison of hierarchy value.

[results here]

 I believe that is everything. is there any other data I likely have that you might want to fully flesh out and finish this experiments and results writeup.