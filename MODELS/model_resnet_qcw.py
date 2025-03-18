import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from collections import OrderedDict
from MODELS.iterative_normalization import IterNormRotation

NUM_CLASSES = 200

# Create a hybrid approach that preserves original blocks but adds QCW functionality
class ResNetQCW(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, depth=18, whitened_layers=None,
                 act_mode="pool_max", subspaces=None, use_subspace=True,
                 use_free=False, pretrained_model=None, vanilla_pretrain=True):
        """
        Build a ResNet with Quantized Concept Whitening (QCW) layers.
        QCW blocks are attached after the residual addition in each block.
        
        Parameters:
         - num_classes: Number of output classes.
         - depth: ResNet depth (18 or 50).
         - whitened_layers: List of global block indices (e.g., [5] or [2,5]) where QCW is applied.
         - act_mode: Activation mode for QCW.
         - subspaces: A dict mapping high-level concept names to lists of dimension indices.
         - use_subspace: Enable hierarchical subspace partitioning.
         - use_free: Enable free (unlabeled) concept axes.
         - pretrained_model: Path to a checkpoint with pretrained weights.
         - vanilla_pretrain: Expect standard pretrained weights.
        """
        super(ResNetQCW, self).__init__()
        self.use_subspace = use_subspace
        self.use_free = use_free
        self.subspaces = subspaces  # e.g., {"wing": [0,1,2], "beak": [3,4]}
        
        # Build backbone
        if depth == 18:
            self.backbone = models.resnet18(pretrained=False)
            bn_dims = [64, 128, 256, 512]
        elif depth == 50:
            self.backbone = models.resnet50(pretrained=False)
            bn_dims = [64, 128, 256, 512]
        else:
            raise ValueError(f"Unsupported depth: {depth}")
        
        # Keep track of QCW layers and their locations
        self.cw_layers = []
        self.cw_locations = []
        layer_names = ["layer1", "layer2", "layer3", "layer4"]
        
        # Create QCW layers that will be attached to specific locations
        if whitened_layers:
            for i, ln in enumerate(layer_names):
                layer = getattr(self.backbone, ln)
                for j, block in enumerate(layer):
                    global_idx = sum(len(getattr(self.backbone, ln2)) for ln2 in layer_names[:i]) + j + 1
                    if global_idx in whitened_layers:
                        print(f"Creating QCW for {ln}[{j}] (global index {global_idx}) with dimension {bn_dims[i]}, activation mode: {act_mode}.")
                        dim = bn_dims[i]
                        qcw = IterNormRotation(num_features=dim, activation_mode=act_mode, cw_lambda=0.1)
                        if self.use_subspace and self.subspaces is not None:
                            qcw.subspaces = self.subspaces
                        self.cw_layers.append(qcw)
                        self.cw_locations.append((ln, j, global_idx))
                        # Register the QCW layer as a module so PyTorch properly handles it
                        self.add_module(f"qcw_{ln}_{j}", qcw)
        
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
            # Skip QCW layers in the checkpoint if loading vanilla pretrained model
            if vanilla_pretrain and 'qcw' in key:
                continue
            new_sd[key] = val
        
        # Only update parameters present in both dicts to avoid errors
        common_keys = set(model_dict.keys()) & set(new_sd.keys())
        update_dict = {k: new_sd[k] for k in common_keys}
        model_dict.update(update_dict)
        
        self.load_state_dict(model_dict)
        print(f"[ResNetQCW] Loaded {len(update_dict)}/{len(model_dict)} parameters from checkpoint.")

    def forward(self, x, region=None, orig_x_dim=None):
        # Initial layers
        out = self.backbone.conv1(x)
        out = self.backbone.bn1(out)
        out = self.backbone.relu(out)
        out = self.backbone.maxpool(out)
        
        # Process each layer, applying QCW after specific blocks
        layer_names = ["layer1", "layer2", "layer3", "layer4"]
        cw_idx = 0
        
        for ln_idx, ln in enumerate(layer_names):
            layer = getattr(self.backbone, ln)
            for block_idx, block in enumerate(layer):
                # Forward through the original block
                out = block(out)
                
                # Check if this block should have QCW applied after it
                if cw_idx < len(self.cw_locations):
                    target_ln, target_block_idx, _ = self.cw_locations[cw_idx]
                    if ln == target_ln and block_idx == target_block_idx:
                        qcw = self.cw_layers[cw_idx]
                        # Apply QCW layer after residual addition
                        if region is not None and orig_x_dim is not None and hasattr(qcw, 'set_region'):
                            qcw.set_region(region, orig_x_dim)
                        out = qcw(out)
                        cw_idx += 1
        
        # Final layers
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
    for module in reversed(list(model.modules())):
        if isinstance(module, IterNormRotation):
            return module
    try:
        return model.backbone.layer4[-1].bn1
    except (AttributeError, IndexError):
        raise ValueError("No IterNormRotation (QCW) layer found in the model")