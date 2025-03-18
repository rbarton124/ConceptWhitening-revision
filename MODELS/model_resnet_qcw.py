import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from collections import OrderedDict
from MODELS.iterative_normalization import IterNormRotation

NUM_CLASSES = 200

# First, we need to create modified ResNet blocks that support _make_cw_end method
class ModifiedBasicBlock(nn.Module):
    def __init__(self, basic_block):
        super(ModifiedBasicBlock, self).__init__()
        # Copy all attributes from the original block
        self.conv1 = basic_block.conv1
        self.bn1 = basic_block.bn1
        self.relu = basic_block.relu
        self.conv2 = basic_block.conv2
        self.bn2 = basic_block.bn2
        self.downsample = basic_block.downsample
        self.stride = basic_block.stride
        self.cw = None  # Will be set by _make_cw_end
        
    def _make_cw_end(self, cw_layer):
        """Attach a QCW layer after the residual addition"""
        self.cw = cw_layer
        
    def forward(self, x, region=None, orig_x_dim=None):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity  # Residual addition
        
        # Apply QCW layer after residual addition if it exists
        if self.cw is not None:
            if hasattr(self.cw, 'set_region') and region is not None and orig_x_dim is not None:
                self.cw.set_region(region, orig_x_dim)
            out = self.cw(out)
            
        out = self.relu(out)
        return out

class ModifiedBottleneck(nn.Module):
    def __init__(self, bottleneck):
        super(ModifiedBottleneck, self).__init__()
        # Copy all attributes from the original block
        self.conv1 = bottleneck.conv1
        self.bn1 = bottleneck.bn1
        self.conv2 = bottleneck.conv2
        self.bn2 = bottleneck.bn2
        self.conv3 = bottleneck.conv3
        self.bn3 = bottleneck.bn3
        self.relu = bottleneck.relu
        self.downsample = bottleneck.downsample
        self.stride = bottleneck.stride
        self.cw = None  # Will be set by _make_cw_end
        
    def _make_cw_end(self, cw_layer):
        """Attach a QCW layer after the residual addition"""
        self.cw = cw_layer
        
    def forward(self, x, region=None, orig_x_dim=None):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity  # Residual addition
        
        # Apply QCW layer after residual addition if it exists
        if self.cw is not None:
            if hasattr(self.cw, 'set_region') and region is not None and orig_x_dim is not None:
                self.cw.set_region(region, orig_x_dim)
            out = self.cw(out)
            
        out = self.relu(out)
        return out

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
        
        # Replace the standard blocks with our modified versions that support _make_cw_end
        self._replace_blocks_with_modified_versions()
        
        self.cw_layers = []
        layer_names = ["layer1", "layer2", "layer3", "layer4"]
        
        # Compute cumulative block counts to convert global index to layer index.
        cum_counts = []
        count = 0
        for ln in layer_names:
            layer = getattr(self.backbone, ln)
            count += len(layer)
            cum_counts.append(count)
        
        # Iterate through blocks and attach QCW layers (after residual addition)
        for i, ln in enumerate(layer_names):
            layer = getattr(self.backbone, ln)
            for j, block in enumerate(layer):
                global_idx = sum(len(getattr(self.backbone, ln2)) for ln2 in layer_names[:i]) + j + 1
                if whitened_layers and global_idx in whitened_layers:
                    print(f"Attaching QCW block at {ln}[{j}] (global index {global_idx}) with dimension {bn_dims[i]}, activation mode: {act_mode}.")
                    dim = bn_dims[i]
                    qcw = IterNormRotation(num_features=dim, activation_mode=act_mode, cw_lambda=0.1)
                    if self.use_subspace and self.subspaces is not None:
                        qcw.subspaces = self.subspaces
                    # Attach QCW block after the residual addition via the _make_cw_end method
                    block._make_cw_end(qcw)
                    self.cw_layers.append(qcw)
        
        if pretrained_model and os.path.isfile(pretrained_model):
            self.load_model(pretrained_model, vanilla_pretrain)
    
    def _replace_blocks_with_modified_versions(self):
        """Replace standard torchvision blocks with our modified versions that support _make_cw_end"""
        for ln in ["layer1", "layer2", "layer3", "layer4"]:
            layer = getattr(self.backbone, ln)
            for i in range(len(layer)):
                # Check the type of block and replace with the appropriate modified version
                if isinstance(layer[i], models.resnet.BasicBlock):
                    layer[i] = ModifiedBasicBlock(layer[i])
                elif isinstance(layer[i], models.resnet.Bottleneck):
                    layer[i] = ModifiedBottleneck(layer[i])
     
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

    def forward(self, x, region=None, orig_x_dim=None):
        out = self.backbone.conv1(x)
        out = self.backbone.bn1(out)
        out = self.backbone.relu(out)
        out = self.backbone.maxpool(out)
        for ln in ["layer1", "layer2", "layer3", "layer4"]:
            layer = getattr(self.backbone, ln)
            for block in layer:
                # Note: Each block is expected to call its attached QCW layer (if any) after adding the residual
                out = block(out, region=region, orig_x_dim=orig_x_dim)
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