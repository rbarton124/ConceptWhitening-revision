import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from collections import OrderedDict
from MODELS.iterative_normalization import IterNormRotation
import logging
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('resnet_qcw_debug.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

class BottleneckCW(nn.Module):
    expansion = 4
    
    def __init__(self, original_block):
        super(BottleneckCW, self).__init__()
        
        # Copy over all conv/bn layers from the original
        self.conv1 = original_block.conv1
        self.bn1   = original_block.bn1
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = original_block.conv2
        self.bn2   = original_block.bn2
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = original_block.conv3
        self.bn3   = original_block.bn3

        self.downsample = original_block.downsample
        self.outdim = original_block.outdim if hasattr(original_block, 'outdim') else None

        # A final ReLU is used after adding the residual
        self.relu = nn.ReLU(inplace=True)

        # This will be assigned externally if we want this block whitened
        self.cw = None

    def forward(self, x):
        # Standard bottleneck forward
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # Compute residual
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        # Add residual to main path
        out += residual
        # Now we call the cw module if present
        if self.cw is not None:
            out_before = out.clone()
            # self.cw(out)
            out = self.cw(out)
            diff_mean = (out_before - out).abs().mean().item()
            # logging.info(f"BottleneckCW effect - Mean difference: {diff_mean}, shape: {out.shape}, device: {out.device}")


        # Final ReLU after the CW
        out = self.relu(out)
        return out


################################################################################
# 2) The ResNetQCW class that uses the custom BottleneckCW if whitening is needed
################################################################################
class ResNetQCW(nn.Module):
    def __init__(self, num_classes=200, depth=18, whitened_layers=None,
                 act_mode="pool_max", subspaces=None, use_subspace=True,
                 use_free=False, pretrained_model=None, vanilla_pretrain=False):
        """
        Build a ResNet with a CW layer placed AFTER the residual, matching old code.

        - whitened_layers: list of global block indices to whiten
        - We'll wrap those blocks with BottleneckCW (for ResNet-50) or BasicBlockCW (for ResNet-18)
          and attach a CW module at the end of the block.
        """
        super(ResNetQCW, self).__init__()
        self.use_subspace = use_subspace
        self.use_free = use_free
        self.subspaces = subspaces  # Dict like {"high1": [dim0, dim1, ...], ...}

        if whitened_layers is None:
            whitened_layers = []
        
        # Build the standard backbone from torchvision
        if depth == 18:
            self.backbone = models.resnet18(pretrained=False)
            # For BasicBlock-based layers
            block_type = 'basic'
            bn_dims = [64, 128, 256, 512]
        elif depth == 50:
            self.backbone = models.resnet50(pretrained=False)
            block_type = 'bottleneck'
            bn_dims = [64, 128, 256, 512]
        else:
            raise ValueError(f"Unsupported depth: {depth}")
        
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
        
        self.cw_layers: list[IterNormRotation] = []

        layer_names = ["layer1", "layer2", "layer3", "layer4"]

        # Count total blocks to map global indices to (layer, block)
        cum_counts = []
        count = 0
        for ln in layer_names:
            layer_seq = getattr(self.backbone, ln)
            count += len(layer_seq)
            cum_counts.append(count)

        # We'll iterate over each layer and each block, check if global_idx in whitened_layers
        global_counter = 0
        for ln_i, ln in enumerate(layer_names):
            layer_seq = getattr(self.backbone, ln)
            for b_i in range(len(layer_seq)):
                global_counter += 1
                original_block = layer_seq[b_i]

                if global_counter in whitened_layers:
                    print(f"Attaching CW AFTER residual in {ln}[{b_i}] (global idx {global_counter})")
                    
                    # Wrap the original block in a custom block that calls CW after the residual
                    if block_type == 'bottleneck':
                        new_block = BottleneckCW(original_block)
                    else:
                        # If user tries depth=18, define a similar BasicBlockCW or revert to old logic
                        new_block = self._wrap_basicblock(original_block)  # Helper below
                        
                    # Create the CW module
                    dim = bn_dims[ln_i]
                    qcw = IterNormRotation(num_features=dim, activation_mode=act_mode, cw_lambda=0.1, subspace_map=self.subspaces)

                    # Assign it to new_block.cw
                    new_block.cw = qcw
                    self.cw_layers.append(qcw)

                    # Replace the block in the layer with our new wrapped block
                    layer_seq[b_i] = new_block

                # else keep block as-is

        # Load any pretrained weights if given
        if pretrained_model and os.path.isfile(pretrained_model):
            self.load_model(pretrained_model)

    def _wrap_basicblock(self, original_block):
        """
        A helper method to create a BasicBlockCW-like wrapper if depth=18.
        This is analogous to BottleneckCW but for 2 conv layers instead of 3.
        """
        class BasicBlockCW(nn.Module):
            expansion = 1
            def __init__(self, blk):
                super().__init__()
                self.conv1 = blk.conv1
                self.bn1   = blk.bn1
                self.relu1 = nn.ReLU(inplace=True)

                self.conv2 = blk.conv2
                self.bn2   = blk.bn2
                # no third conv for basic block

                self.downsample = blk.downsample
                self.relu = nn.ReLU(inplace=True)
                self.cw = None

            def forward(self, x):
                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu1(out)

                out = self.conv2(out)
                out = self.bn2(out)

                residual = x if self.downsample is None else self.downsample(x)
                out += residual

                if self.cw is not None:
                    out_before = out.clone()
                    # self.cw(out)
                    out = self.cw(out)
                    diff_mean = (out_before - out).abs().mean().item()
                    # logging.info(f"BasicBlockCW effect - Mean difference: {diff_mean}, shape: {out.shape}, device: {out.device}")

                out = self.relu(out)
                return out
                return out

        return BasicBlockCW(original_block)

    def change_mode(self, mode):
        for cw in self.cw_layers:
            cw.mode = mode

    def update_rotation_matrix(self):
        for cw in self.cw_layers:
            cw.update_rotation_matrix()

    def load_model(self, pretrain_path):
        print(f"[ResNetQCW] Loading pretrained weights from {pretrain_path} ...")
        model_dict = self.state_dict()
        pretrain_dict = torch.load(pretrain_path, map_location="cpu")
        if "state_dict" in pretrain_dict:
            pretrain_dict = pretrain_dict["state_dict"]
        new_sd = OrderedDict()
        for key, val in pretrain_dict.items():
            if key.startswith("module."):
                key = key[len("module."):]
            new_sd[key] = val
        model_dict.update(new_sd)
        self.load_state_dict(model_dict, strict=False)
        print("[ResNetQCW] Pretrained weights loaded. (strict=False used)")

    def forward(self, x):
        # Standard ResNet forward
        out = self.backbone.conv1(x)
        out = self.backbone.bn1(out)
        out = self.backbone.relu(out)
        out = self.backbone.maxpool(out)

        for ln in ["layer1", "layer2", "layer3", "layer4"]:
            layer_seq = getattr(self.backbone, ln)
            for block in layer_seq:
                out = block(out)

        out = self.backbone.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.backbone.fc(out)
        return out


def build_resnet_qcw(num_classes=200, depth=18, whitened_layers=None, act_mode="pool_max", subspaces=None, 
                     use_subspace=True, use_free=False, pretrained_model=None, vanilla_pretrain=False):
    """
    Main entry point to construct a ResNet with QCW blocks that attach CW AFTER the residual is added.
    """
    if whitened_layers is None:
        whitened_layers = []
    model = ResNetQCW(
        num_classes=num_classes,
        depth=depth,
        whitened_layers=whitened_layers,
        act_mode=act_mode,
        subspaces=subspaces,
        use_subspace=use_subspace,
        use_free=use_free,
        pretrained_model=pretrained_model,
        vanilla_pretrain=vanilla_pretrain
    )
    return model


def get_last_qcw_layer(model):
    """
    Finds the last IterNormRotation in model.cw_layers or by reversing modules,
    same as before.
    """
    if hasattr(model, "cw_layers") and len(model.cw_layers) > 0:
        return model.cw_layers[-1]

    # fallback: search modules
    for module in reversed(list(model.modules())):
        if isinstance(module, IterNormRotation):
            return module
    raise ValueError("No IterNormRotation (QCW) layer found in the model")

def get_qcw_layer(model, layer_idx):
    """
    Finds the last IterNormRotation in model.cw_layers or by reversing modules,
    same as before.
    """
    if hasattr(model, "cw_layers") and len(model.cw_layers) > 0:
        return model.cw_layers[layer_idx]


    # fallback: search modules
    pot_modules = []
    for module in list(model.modules()):
        if isinstance(module, IterNormRotation):
            pot_modules.append(module)
    if len(pot_modules) > 0:
        return pot_modules[layer_idx]

    raise ValueError("Couldnt find the requested IterNormRotation (QCW) layer in the model")