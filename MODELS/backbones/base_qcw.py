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
                print(f"Replacing BN layer (global index {g_idx}) with QCW layer. "
                      f"Expected dimension: {c_dim} channels; "
                      f"Activation mode: {self.act_mode}.")
                qcw = IterNormRotation(num_features=c_dim,
                                       activation_mode=act_mode,
                                       cw_lambda=kwargs.get('cw_lambda', 0.1),
                                       subspace_map=self.subspaces)
                # Use setattr consistently for all module types
                # This works for nn.Module, nn.Sequential, and nn.ModuleList
                parent, attr = mod_ref
                setattr(parent, attr, qcw)
                self.cw_layers.append(qcw)

        self._extra_init(**kwargs)   # optional hook

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
        
    def reset_counters(self):
        for cw in self.cw_layers: cw.reset_counters()
    # ---------------------------------

    # ---------- forward pass ----------
    def forward(self, x):
        raise NotImplementedError   # each child defines its own
