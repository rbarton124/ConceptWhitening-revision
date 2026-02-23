Update plan

Below is a **two‑step plan** followed by the **patched files**.

---

## 🔧 Upgrade plan & design choices

| Goal | What has to change? | Why? |
|------|--------------------|------|
| **1. Add the true  \(1/n_S\)  normalisation**<br>(so every high‑level concept contributes equally) | *iterative_normalization.py* ( `IterNormRotation._accumulate_gradient_subspace`) + one tiny edit in *train_qcw.py* (`align_concepts`) | We need to know the **total number of images** for the currently‑active sub‑space.  The training loop already knows that, so we pass it once per alignment call. |
| **2. Keep the rotation strictly block‑diagonal** | *iterative_normalization.py* – three places:<br>• build a **sub‑space mask** once in `__init__`<br>• mask gradients in `_accumulate_gradient_subspace`<br>• after every Cayley update, zero the cross‑block entries **and re‑orthogonalise each block** in `update_rotation_matrix` | Guarantees that axes from different high‑level concepts never mix, giving true hierarchical disentanglement and stabilising optimisation. |
| **3. No other files need touching** | `model_resnet_qcw_bn.py` already passes the `subspace_map`; the rest of the flow (sampling, λ‑scaling, etc.) remains valid. | — |

Only ~15 LOC change in the training script and <100 LOC inside the layer itself.

---

## 📄 Changed files

### 1. `iterative_normalization.py`
```python
@@
     def __init__(self,
@@
         self.active_subspace=None
+        # --------------------------------------------------
+        #  Build a fast lookup for   axis -> sub‑space id
+        #  and a boolean mask that is 1  *inside*  a block.
+        # --------------------------------------------------
+        self.axis_to_hl = torch.full((num_features,), -1, dtype=torch.long)
+        block_mask = torch.zeros(num_features, num_features, dtype=torch.bool)
+        if self.subspace_map:
+            for block_id, axes in enumerate(self.subspace_map.values()):
+                for a in axes:
+                    self.axis_to_hl[a] = block_id
+                idx = torch.tensor(axes)
+                block_mask[idx.unsqueeze(1), idx.unsqueeze(0)] = True
+        self.register_buffer("block_mask", block_mask)   # shape [C,C]
 
@@  def set_subspace(self, subspace_name:str):
-        self.active_subspace=subspace_name
+        """
+        Optionally pass   n_S   (= #imgs that contain that HL concept).
+        The training loop provides it so we can do the 1/n_S scaling.
+        """
+        if isinstance(subspace_name, tuple):
+            subspace_name, n_total = subspace_name
+            self.current_subspace_size = float(max(n_total, 1))
+        else:
+            self.current_subspace_size = None
+        self.active_subspace = subspace_name
 
@@  def clear_subspace(self):
         self.active_subspace=None
+        self.current_subspace_size = None
 
@@  def update_rotation_matrix(self):
             self.running_rot=R
+            # -------  🔒 enforce block‑diagonal -------------
+            if self.block_mask.any():
+                self.running_rot.mul_(self.block_mask)   # zero cross‑block
+                # Re‑orthogonalise every individual block
+                for axes in self.subspace_map.values():
+                    idx = torch.tensor(axes, device=R.device)
+                    blk = self.running_rot[0].index_select(0, idx).index_select(1, idx)
+                    q,_ = torch.linalg.qr(blk)
+                    self.running_rot[0].index_copy_(0, idx, q)
+                    self.running_rot[0].index_copy_(1, idx, q.t())
             self.counter=torch.ones(size_R[-1],device=G.device)*0.001
 
@@  def _accumulate_gradient_subspace(self, X_hat, subspace_axes):
-        aggregator = torch.zeros(C, C, device=act.device)
+        aggregator = torch.zeros(C, C, device=act.device)
@@
-        aggregator /= float(B)
+        # -----------------------------------------------
+        # scale by 1/n_S  (exact)  if we know it,
+        # else fall back to 1/B  (same as before).
+        # -----------------------------------------------
+        denom = self.current_subspace_size if self.current_subspace_size else float(B)
+        aggregator /= denom
@@
-        for a in subspace_axes:
+        for a in subspace_axes:
             self.sum_G[:, a, :] = (
                 self.momentum * aggregator[a, :]
                 + (1.0 - self.momentum) * self.sum_G[:, a, :]
             )
             self.counter[a] += local_counter[a].item()
+        # Zero gradients that spill outside the block
+        off_block = ~self.block_mask
+        self.sum_G.masked_fill_(off_block, 0.0)
 
@@  def extra_repr(self):
                 f"subspace_map={list(self.subspace_map.keys())}")
```

### 2. `train_qcw.py` – **only the `align_concepts` helper changes**

```python
@@  for hl_name, groups in hl_subconcepts.items():
-            for cw_layer in model.module.cw_layers:
-                cw_layer.clear_subspace()
-                cw_layer.set_subspace(hl_name)
+            # total n_S = all (labeled+free) samples for this HL
+            n_S = sum(len(loader.dataset) for (_, loader) in
+                      (groups["labeled"] + groups["free"]))
+
+            for cw_layer in model.module.cw_layers:
+                cw_layer.clear_subspace()
+                # pass (name, n_S) so the layer can scale by 1/n_S
+                cw_layer.set_subspace((hl_name, n_S))
```

That is the **only modification** needed in the training script.

The model‑building file (`model_resnet_qcw_bn.py`) is untouched – it already forwards `subspace_map` to every QCW layer, so the new block‑mask logic works automatically.

---

## ✅ Why this solves the two issues

* **Balanced objective:**  
  Each sub‑space’s gradient now gets divided by its real dataset size `n_S`, so wing and beak are treated equally even if the training set is imbalanced.

* **True block‑diagonal rotation:**  
  *Gradients* are masked, so no cross‑talk is learned, **and** every Cayley update is followed by a hard projection that zeros the off‑block entries and re‑orthogonalises the within‑block matrices.  
  The running rotation therefore *stays* block‑diagonal for the whole training run.

You can now re‑run training; the logged “axis purity” metric should increase noticeably and adding extra QCW layers later in the network should have the expected positive effect.
