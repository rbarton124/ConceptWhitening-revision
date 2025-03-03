"""
model_resnet_qcw.py

A revised ResNet that:
 - Uses custom BasicBlockCW/BottleneckCW blocks
 - Avoids nn.Sequential so we can pass (region=None, orig_x_dim=None) into each block
 - Replaces certain BN layers with QCW (IterNormRotation) if whitened_layers is specified
 - Supports bounding-box redaction if `use_redaction=True`
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from MODELS.iternorm_qcw import IterNormRotation as CWLayer
# from MODELS.redact import redact  # optional if you have a redact(...) function

NUM_CLASSES = 200

#######################################
#  Custom BasicBlock / Bottleneck
#######################################
class BasicBlockCW(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x, region=None, orig_x_dim=None):
        identity = x

        out = self.conv1(x)
        if isinstance(self.bn1, CWLayer):
            out = self.bn1(out, X_redact_coords=region, orig_x_dim=orig_x_dim)
        else:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if isinstance(self.bn2, CWLayer):
            out = self.bn2(out, X_redact_coords=region, orig_x_dim=orig_x_dim)
        else:
            out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class BottleneckCW(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(planes * self.expansion)

        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, region=None, orig_x_dim=None):
        identity = x

        out = self.conv1(x)
        if isinstance(self.bn1, CWLayer):
            out = self.bn1(out, X_redact_coords=region, orig_x_dim=orig_x_dim)
        else:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        if isinstance(self.bn2, CWLayer):
            out = self.bn2(out, X_redact_coords=region, orig_x_dim=orig_x_dim)
        else:
            out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        if isinstance(self.bn3, CWLayer):
            out = self.bn3(out, X_redact_coords=region, orig_x_dim=orig_x_dim)
        else:
            out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


#######################################
# ResNet with no nn.Sequential
#######################################

class ResNetQCW(nn.Module):
    """
    A ResNet that:
      - Builds layer1_blocks, layer2_blocks, etc. as lists of BasicBlockCW/BottleneckCW
      - Replaces specified BN layers with QCW
      - Accepts region/orig_x_dim in forward
    """
    def __init__(self, block, layers, num_classes=NUM_CLASSES,
                 whitened_layers=None, act_mode="pool_max",
                 use_redaction=False, use_subspace=False, subspaces=None,
                 use_free=False, cw_lambda=0.1,
                 pretrained_model=None):
        super().__init__()
        self.inplanes = 64
        self.block_type = block
        self.layers_cfg = layers
        self.num_classes = num_classes

        self.use_redaction = use_redaction
        self.use_subspace = use_subspace
        self.subspaces = subspaces
        self.use_free = use_free
        self.cw_lambda = cw_lambda

        self.cw_layers = []  # track replaced BN->CW

        # Stem
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Build layer blocks as ModuleLists (not nn.Sequential!)
        self.layer1_blocks = self._make_layer(block, 64,  self.layers_cfg[0], stride=1)
        self.layer2_blocks = self._make_layer(block, 128, self.layers_cfg[1], stride=2)
        self.layer3_blocks = self._make_layer(block, 256, self.layers_cfg[2], stride=2)
        self.layer4_blocks = self._make_layer(block, 512, self.layers_cfg[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # If the user provided a list of "global block indices" to whiten, do that now
        if whitened_layers is None:
            whitened_layers = []
        self._replace_with_cw(whitened_layers, act_mode)

        # Optionally load weights
        if pretrained_model and isinstance(pretrained_model, str) and len(pretrained_model)>0:
            self.load_weights(pretrained_model)

    def _make_layer(self, block, planes, blocks, stride=1):
        """
        Return a list of blocks (ModuleList) instead of a Sequential.
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layer_blocks = nn.ModuleList()
        layer_blocks.append(block(self.inplanes, planes, stride, downsample=downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layer_blocks.append(block(self.inplanes, planes))

        return layer_blocks

    def _forward_layer(self, blocks, x, region=None, orig_x_dim=None):
        """
        Helper that iterates over each block with (x, region=..., orig_x_dim=...).
        This is how we avoid the "unexpected keyword" error from nn.Sequential.
        """
        for block in blocks:
            x = block(x, region=region, orig_x_dim=orig_x_dim)
        return x

    def _replace_with_cw(self, whitened_layers, act_mode):
        """
        Takes a list of global block indices (like [5,6]) and replaces BN with QCW in those blocks.
        We'll label the blocks in a simple linear sequence:
          1..(layer1_cfg), then continuing for layer2, etc.
        If the index matches, we replace block.bn1, bn2, bn3 if exist.
        """
        global_idx = 1

        for layer_blocks in [self.layer1_blocks, self.layer2_blocks, self.layer3_blocks, self.layer4_blocks]:
            for block in layer_blocks:
                # if global_idx is in whitened_layers => replace BN with QCW
                if global_idx in whitened_layers:
                    self._replace_block_bn(block, act_mode)
                global_idx += 1

    def _replace_block_bn(self, block, act_mode):
        """
        Replaces block.bn1, block.bn2, block.bn3 if they're BN2d with our QCW layer
        """
        # bn1
        if hasattr(block, 'bn1') and isinstance(block.bn1, nn.BatchNorm2d):
            new_cw = CWLayer(num_features=block.bn1.num_features,
                             activation_mode=act_mode,
                             cw_lambda=self.cw_lambda)
            if self.use_subspace and self.subspaces:
                new_cw.subspaces = self.subspaces
            new_cw.use_redaction = self.use_redaction
            block.bn1 = new_cw
            self.cw_layers.append(new_cw)

        # bn2
        if hasattr(block, 'bn2') and isinstance(block.bn2, nn.BatchNorm2d):
            new_cw = CWLayer(num_features=block.bn2.num_features,
                             activation_mode=act_mode,
                             cw_lambda=self.cw_lambda)
            if self.use_subspace and self.subspaces:
                new_cw.subspaces = self.subspaces
            new_cw.use_redaction = self.use_redaction
            block.bn2 = new_cw
            self.cw_layers.append(new_cw)

        # bn3 if it's a Bottleneck
        if hasattr(block, 'bn3') and isinstance(block.bn3, nn.BatchNorm2d):
            new_cw = CWLayer(num_features=block.bn3.num_features,
                             activation_mode=act_mode,
                             cw_lambda=self.cw_lambda)
            if self.use_subspace and self.subspaces:
                new_cw.subspaces = self.subspaces
            new_cw.use_redaction = self.use_redaction
            block.bn3 = new_cw
            self.cw_layers.append(new_cw)

    def load_weights(self, ckpt_path):
        """
        Load standard or custom checkpoint. 
        """
        print(f"[ResNetQCW] Loading checkpoint from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]

        model_sd = self.state_dict()
        new_sd = OrderedDict()
        for k, v in ckpt.items():
            if k.startswith("module."):
                k = k[7:]
            new_sd[k] = v

        model_sd.update(new_sd)
        self.load_state_dict(model_sd, strict=False)
        print("[ResNetQCW] Weights loaded successfully.")

    #####################
    # QCW routines
    #####################
    def change_mode(self, mode):
        for cw in self.cw_layers:
            cw.mode = mode

    def update_rotation_matrix(self):
        for cw in self.cw_layers:
            cw.update_rotation_matrix()

    def reset_counters(self):
        for cw in self.cw_layers:
            cw.reset_counters()

    #####################
    # forward
    #####################
    def forward(self, x, region=None, orig_x_dim=None):
        # stem
        out = self.conv1(x)
        if isinstance(self.bn1, CWLayer):
            out = self.bn1(out, X_redact_coords=region, orig_x_dim=orig_x_dim)
        else:
            out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        # layer1..layer4
        out = self._forward_layer(self.layer1_blocks, out, region=region, orig_x_dim=orig_x_dim)
        out = self._forward_layer(self.layer2_blocks, out, region=region, orig_x_dim=orig_x_dim)
        out = self._forward_layer(self.layer3_blocks, out, region=region, orig_x_dim=orig_x_dim)
        out = self._forward_layer(self.layer4_blocks, out, region=region, orig_x_dim=orig_x_dim)

        out = nn.functional.adaptive_avg_pool2d(out, (1,1))
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


###################################
#  build_resnet_qcw
###################################
def build_resnet_qcw(num_classes=NUM_CLASSES,
                     depth=18,
                     whitened_layers=None,
                     act_mode="pool_max",
                     subspaces=None,
                     use_redaction=False,
                     use_subspace=False,
                     use_free=False,
                     cw_lambda=0.1,
                     pretrained_model=None,
                     vanilla_pretrain=True):
    """
    A convenience function that picks the block/layer config,
    builds ResNetQCW, then does BN->CW as specified by whitened_layers.
    
    `whitened_layers` is a list of global block indices (1-based).
    For example, if layer1 has 2 blocks => indices [1,2].
    layer2 => [3,4,...], etc.
    """
    if depth==18:
        block = BasicBlockCW
        layers = [2,2,2,2]
    elif depth==34:
        block = BasicBlockCW
        layers = [3,4,6,3]
    elif depth==50:
        block = BottleneckCW
        layers = [3,4,6,3]
    elif depth==101:
        block = BottleneckCW
        layers = [3,4,23,3]
    else:
        raise ValueError(f"Unsupported depth {depth}")

    model = ResNetQCW(
        block=block,
        layers=layers,
        num_classes=num_classes,
        whitened_layers=whitened_layers,
        act_mode=act_mode,
        use_redaction=use_redaction,
        use_subspace=use_subspace,
        subspaces=subspaces,
        use_free=use_free,
        cw_lambda=cw_lambda,
        pretrained_model=pretrained_model
    )
    return model


def get_last_qcw_layer(model: nn.Module):
    """
    Grab the last replaced QCW layer from model.cw_layers
    """
    if hasattr(model, "cw_layers") and len(model.cw_layers)>0:
        return model.cw_layers[-1]
    return None


###################################
# Debug
###################################
if __name__=="__main__":
    print("[Debug] Testing build_resnet_qcw with bounding-box logic.")
    dummy_subspaces = {"eye": list(range(5)), "nape": list(range(5,10))}
    net = build_resnet_qcw(num_classes=200,
                           depth=18,
                           whitened_layers=[5],
                           act_mode="pool_max",
                           subspaces=dummy_subspaces,
                           use_redaction=True,
                           use_subspace=True,
                           use_free=False,
                           cw_lambda=0.1,
                           pretrained_model="")  # no model for test
    print(net)

    # Test forward pass
    x = torch.randn(2,3,224,224)
    region = torch.tensor([[50,50,174,174],[56,56,168,168]],dtype=torch.float32)
    out = net(x, region=region, orig_x_dim=224)
    print("Output:", out.shape)
