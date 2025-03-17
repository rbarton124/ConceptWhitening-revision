import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from collections import OrderedDict
from MODELS.iterative_normalization import IterNormRotation

NUM_CLASSES = 200

class ResNetQCW(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, depth=18, whitened_layers=None,
                 act_mode="pool_max", subspaces=None, use_subspace=True,
                 use_free=False, pretrained_model=None, vanilla_pretrain=True):
        """
        Build a ResNet with Quantized Concept Whitening layers.
        
        Parameters:
         - num_classes: Output classes.
         - depth: 18 or 50.
         - whitened_layers: List of global block indices (e.g. [5] or [2,5]) to replace.
         - act_mode: Activation mode for QCW.
         - subspaces: A dict mapping high-level concept names to lists of dimension indices.
         - use_subspace: Enable hierarchical subspace partitioning.
         - use_free: Enable free (unlabeled) concept axes.
         - pretrained_model: Path to checkpoint.
         - vanilla_pretrain: Expect standard pretrained weights.
        """
        super(ResNetQCW, self).__init__()
        self.use_subspace = use_subspace
        self.use_free = use_free
        self.subspaces = subspaces  # Dict like {"high1": [dim0, dim1, ...], ...}
        
        # Build backbone
        if depth == 18:
            self.backbone = models.resnet18(pretrained=False)
            bn_dims = [64, 128, 256, 512]
        elif depth == 50:
            self.backbone = models.resnet50(pretrained=False)
            bn_dims = [64, 128, 256, 512]
        else:
            raise ValueError(f"Unsupported depth: {depth}")
        
        self.cw_layers = []
        layer_names = ["layer1", "layer2", "layer3", "layer4"]
        
        # Compute cumulative block counts to convert global index to layer index.
        cum_counts = []
        count = 0
        for i, ln in enumerate(layer_names):
            layer = getattr(self.backbone, ln)
            count += len(layer)
            cum_counts.append(count)
        
        for i, ln in enumerate(layer_names):
            layer = getattr(self.backbone, ln)
            for j, block in enumerate(layer):
                global_idx = sum(len(getattr(self.backbone, ln2)) for ln2 in layer_names[:i]) + j + 1
                if global_idx in whitened_layers:
                    # a print which says at what layer we replaced bn with cw
                    print(f"Replacing BN layer in {ln}[{j}] (global index {global_idx}) "
                          f"with QCW layer. Expected dimension: {bn_dims[i]} channels; "
                          f"Activation mode: {act_mode}.")
                    dim = bn_dims[i]
                    qcw = IterNormRotation(num_features=dim, activation_mode=act_mode, cw_lambda=0.1)
                    if self.use_subspace and self.subspaces is not None:
                        qcw.subspaces = self.subspaces  # QCW layer uses subspace information
                    block.bn1 = qcw
                    self.cw_layers.append(qcw)
        
        # Load pretrained weights if provided, this logic needs to be fleshed out an unified with the train resume logic
        if pretrained_model and os.path.isfile(pretrained_model):
            self.load_model(pretrained_model, vanilla_pretrain)
     
    def change_mode(self, mode):
        for cw in self.cw_layers:
            cw.mode = mode

    def update_rotation_matrix(self):
        for cw in self.cw_layers:
            cw.update_rotation_matrix()

    def reset_counters(self):
        for cw in self.cw_layers:
            cw.reset_counters()

    def load_model(self, pretrain_path, vanilla_pretrain):
        print(f"[ResNetQCW] Loading pretrained weights from {pretrain_path} ...")
        model_dict = self.state_dict()
        pretrain_dict = torch.load(pretrain_path, map_location="cpu")
        if "state_dict" in pretrain_dict:
            pretrain_dict = pretrain_dict["state_dict"]
        new_sd = OrderedDict()
        for key, val in pretrain_dict.items():
            if key.startswith("module."):
                key = key[7:]
            new_sd[key] = val
        model_dict.update(new_sd)
        self.load_state_dict(model_dict)
        print("[ResNetQCW] Pretrained weights loaded.")

    def forward(self, x):
        out = self.backbone.conv1(x)
        out = self.backbone.bn1(out)
        out = self.backbone.relu(out)
        out = self.backbone.maxpool(out)
        for ln in ["layer1", "layer2", "layer3", "layer4"]:
            layer = getattr(self.backbone, ln)
            for block in layer:
                out = block(out)
        out = self.backbone.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.backbone.fc(out)
        return out

def build_resnet_qcw(num_classes=NUM_CLASSES, depth=18, whitened_layers=None, act_mode="pool_max",
                     subspaces=None, use_subspace=True, use_free=False, pretrained_model=None, vanilla_pretrain=True):
    model = ResNetQCW(num_classes=num_classes, depth=depth, whitened_layers=whitened_layers,
                      act_mode=act_mode, subspaces=subspaces, use_subspace=use_subspace,
                      use_free=use_free, pretrained_model=pretrained_model, vanilla_pretrain=vanilla_pretrain)
    return model

def get_last_qcw_layer(model):    
    if hasattr(model, "cw_layers") and len(model.cw_layers) > 0:
        return model.cw_layers[-1]
    
    # if that fails search through all modules
    for module in reversed(list(model.modules())):
        if isinstance(module, IterNormRotation):
            return module
            
    # if all else fails, try the legacy approach
    try:
        return model.backbone.layer4[-1].bn1
    except (AttributeError, IndexError):
        raise ValueError("No IterNormRotation (QCW) layer found in the model")