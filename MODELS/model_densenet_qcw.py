# MODELS/model_densenet_qcw.py
import os
import re
import torch
import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict
from MODELS.iterative_normalization import IterNormRotation

class DenseNetQCW(nn.Module):
    """
    DenseNet variant that replaces selected DenseLayer BatchNorms with IterNormRotation (QCW).
    Interface compatible with ResNetQCW used by the training script.
    """

    def __init__(self, num_classes=200, depth=161, whitened_layers=None,
                 act_mode="pool_max", subspaces=None, use_subspace=True,
                 use_free=False, cw_lambda=0, pretrained_model=None, vanilla_pretrain=False):
        super(DenseNetQCW, self).__init__()
        self.use_subspace = use_subspace
        self.use_free = use_free
        self.subspaces = subspaces or {}
        self.whitened_layers = list(whitened_layers or [])

        if depth != 161:
            # you can extend to other DenseNet depths if needed
            raise ValueError("Currently only DenseNet-161 is supported by DenseNetQCW")

        # Build torchvision DenseNet-161 backbone (no pretrained weights by default here)
        self.backbone = models.densenet161(pretrained=False)
        # Replace classifier to match num_classes
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Linear(in_features, num_classes)

        # Will collect all QCW modules here (ordered as encountered)
        self.cw_layers = []

        # Find denseblocks and count total DenseLayer modules to compute a global index
        denseblock_names = ["denseblock1", "denseblock2", "denseblock3", "denseblock4"]

        # Build a flat list of (block_idx, layer_idx, layer_module) with global indexing
        dense_layers = []
        for bi, db_name in enumerate(denseblock_names):
            block = getattr(self.backbone.features, db_name)
            # block is an _DenseBlock, has attribute .dense_layers (or modules with named children)
            # iterate through its children which are DenseLayer instances
            for li, layer in enumerate(block):
                dense_layers.append((bi, li, layer))

        # Replace selected BatchNorms with IterNormRotation.
        # Use global index 1..N for the DenseLayer positions (consistent with ResNet global indices approach).
        for global_idx, (bi, li, layer) in enumerate(dense_layers, start=1):
            if global_idx in self.whitened_layers:
                # For DenseLayer, prefer to replace norm2 (the batchnorm before conv2) which has num_features attribute
                if hasattr(layer, "norm2"):
                    dim = layer.norm2.num_features
                    print(f"[DenseNetQCW] Replacing DenseBlock{bi+1}._DenseLayer[{li}] (global idx {global_idx})"
                          f" norm2 (dim={dim}) -> IterNormRotation (act_mode={act_mode})")
                    qcw = IterNormRotation(num_features=dim, activation_mode=act_mode,
                                           cw_lambda=cw_lambda, subspace_map=self.subspaces)
                    # replace the BatchNorm module with QCW. Keep attribute name 'norm2' so forward still calls it.
                    layer.norm2 = qcw
                    self.cw_layers.append(qcw)
                else:
                    # fallback: try norm1
                    if hasattr(layer, "norm1"):
                        dim = layer.norm1.num_features
                        print(f"[DenseNetQCW] (fallback) Replacing DenseBlock{bi+1}._DenseLayer[{li}]"
                              f" norm1 (dim={dim}) -> IterNormRotation")
                        qcw = IterNormRotation(num_features=dim, activation_mode=act_mode,
                                               cw_lambda=cw_lambda, subspace_map=self.subspaces)
                        layer.norm1 = qcw
                        self.cw_layers.append(qcw)
                    else:
                        print(f"[DenseNetQCW] WARNING: DenseLayer {bi+1}.{li} has no norm1/norm2 to replace; skipping.")

        # Optionally load pretrained weights
        if pretrained_model and os.path.isfile(pretrained_model):
            self.load_model(pretrained_model)

    # Methods expected by training code:
    def change_mode(self, mode):
        for cw in self.cw_layers:
            cw.mode = mode

    def update_rotation_matrix(self):
        for cw in self.cw_layers:
            cw.update_rotation_matrix()

    def reset_counters(self):
        for cw in self.cw_layers:
            cw.reset_counters()

    def load_model(self, pretrain_path):
        print(f"[DenseNetQCW] Loading pretrained weights from {pretrain_path} ...")
        model_dict = self.state_dict()
        pretrain_dict = torch.load(pretrain_path, map_location="cpu")
        if "state_dict" in pretrain_dict:
            pretrain_dict = pretrain_dict["state_dict"]
        new_sd = OrderedDict()
        for key, val in pretrain_dict.items():
            # strip "module." if present from DataParallel checkpoints
            if key.startswith("module."):
                key = key[len("module."):]
            new_sd[key] = val
        # update and load (partial loads allowed)
        model_dict.update({k: v for k, v in new_sd.items() if k in model_dict and model_dict[k].shape == v.shape})
        self.load_state_dict(model_dict, strict=False)
        print("[DenseNetQCW] Pretrained weights loaded (partial allowed).")

    def forward(self, x):
        # Follow torchvision DenseNet forward
        features = self.backbone.features(x)           # conv0..denseblocks..transitions..norm5
        out = nn.functional.relu(features, inplace=True)
        out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.backbone.classifier(out)
        return out


def build_densenet_qcw(num_classes=200, depth=161, whitened_layers=None, act_mode="pool_max",
                       subspaces=None, use_subspace=True, use_free=False,
                       cw_lambda=0, pretrained_model=None, vanilla_pretrain=False):
    """
    Factory to match the ResNet factory interface used by training script.
    """
    model = DenseNetQCW(
        num_classes=num_classes,
        depth=depth,
        whitened_layers=whitened_layers,
        act_mode=act_mode,
        subspaces=subspaces,
        use_subspace=use_subspace,
        use_free=use_free,
        cw_lambda=cw_lambda,
        pretrained_model=pretrained_model,
        vanilla_pretrain=vanilla_pretrain
    )
    return model


# Utility: improved get_last_qcw_layer (works for ResNetQCW or DenseNetQCW)
def get_last_qcw_layer(model):
    # If model has attribute cw_layers (expected), return last if present
    if hasattr(model, "cw_layers") and getattr(model, "cw_layers"):
        return model.cw_layers[-1]

    # search modules in reverse for IterNormRotation
    from MODELS.iterative_normalization import IterNormRotation
    for module in reversed(list(model.modules())):
        if isinstance(module, IterNormRotation):
            return module

    # fallback: None
    return None
