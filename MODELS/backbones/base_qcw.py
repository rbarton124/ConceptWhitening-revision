import logging
import torch
import torch.nn as nn
from MODELS.iterative_normalization import IterNormRotation

class BaseQCW(nn.Module):
    """Common utilities for all QCW backbones."""
    def __init__(self,
                 num_classes,
                 whitened_layers,
                 act_mode,
                 subspaces,
                 **kwargs):
        super().__init__()
        self.whitened_layers = whitened_layers or []
        self.act_mode = act_mode
        self.subspaces = subspaces or {}
        self.cw_layers = []

        # build the vanilla backbone (child class)
        self.backbone, self.bn_dims = self._build_backbone(num_classes=num_classes, **kwargs)

        # replace BN layers
        for mod_ref, c_dim, g_idx in self._bn_iter():
            if g_idx in self.whitened_layers:
                logging.info("Replacing BN layer (global index %d) with QCW layer. "
                             "Expected dimension: %d channels; Activation mode: %s.",
                             g_idx, c_dim, self.act_mode)
                qcw = IterNormRotation(
                    num_features=c_dim,
                    activation_mode=act_mode,
                    cw_lambda=kwargs.get('cw_lambda', 0.1),
                    subspace_map=self.subspaces
                )

                parent, attr = mod_ref
                # --- Minimal fix: handle Sequential / ModuleList integer indexing ---
                if isinstance(attr, int) and isinstance(parent, (nn.Sequential, nn.ModuleList)):
                    parent.__setitem__(attr, qcw)  # e.g., VGG16-BN features[feat_idx] = qcw
                else:
                    setattr(parent, attr, qcw)      # e.g., DenseNet features.norm0 / transition*.norm
                # -------------------------------------------------------------------

                self.cw_layers.append(qcw)

        self._extra_init(**kwargs)

    # ---------- hooks to override ----------
    def _build_backbone(self, **kw):
        raise NotImplementedError
        
    def _bn_iter(self):
        """yield ( (parent_module, attr_name), C, global_idx )"""
        raise NotImplementedError
        
    def _final_pool_and_fc(self, x):
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return self.backbone.fc(x)
        
    def _extra_init(self, **kw):
        pass
    # ---------------------------------------

    # ---------- QCW helpers ----------
    def change_mode(self, mode):
        for cw in self.cw_layers: cw.mode = mode
        
    def update_rotation_matrix(self):
        for cw in self.cw_layers: cw.update_rotation_matrix()
        
    def reset_concept_loss(self):
        for cw in self.cw_layers: cw.reset_concept_loss()
    # ---------------------------------

    # ---------- forward pass ----------
    def forward(self, x):
        raise NotImplementedError   # each child defines its own
