import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math
from torch.nn import init
from .iterative_normalization import IterNormRotation as cw_layer

def build_resnet_cw(
    num_classes=1000,
    depth=18,
    whitened_layers=None,
    act_mode='pool_max',
    model_file=None
):
    """
    Build a ResNet (18 or 50) that replaces certain BN layers with CW layers.

    :param num_classes: number of output classes
    :param depth: int, typically 18 or 50
    :param whitened_layers: list of int, e.g. [8], specifying which BN -> CW
    :param act_mode: 'mean', 'max', 'pos_mean', or 'pool_max' for the CW axis
    :param model_file: optional path to .pth/.tar that has 'state_dict'
    """
    if whitened_layers is None:
        whitened_layers = [8]

    # Decide if resnet18 or resnet50
    if depth == 50:
        layers_cfg = [3, 4, 6, 3]
        model = ResidualNetTransfer(
            num_classes=num_classes,
            args=None,
            whitened_layers=whitened_layers,
            arch='resnet50',
            layers=layers_cfg,
            model_file=None,      # We'll load model_file manually below
            act_mode=act_mode
        )
    elif depth == 18:
        layers_cfg = [2, 2, 2, 2]
        model = ResidualNetTransfer(
            num_classes=num_classes,
            args=None,
            whitened_layers=whitened_layers,
            arch='resnet18',
            layers=layers_cfg,
            model_file=None,
            act_mode=act_mode
        )
    else:
        raise ValueError(f"Unsupported resnet depth: {depth}")

    # If user has a model_file, load only the state_dict
    if model_file is not None and os.path.isfile(model_file):
        print(f"[build_resnet_cw] Loading weights from {model_file}")
        ckpt = torch.load(model_file, map_location='cpu')
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            # user might have saved raw state_dict
            state_dict = ckpt
        new_sd = {}
        for k, v in state_dict.items():
            nk = k.replace('module.', '')
            nk = nk.replace('bw', 'bn')  # handle old naming if needed
            new_sd[nk] = v
        model.load_state_dict(new_sd, strict=False)

    return model


def build_resnet_bn(
    num_classes=1000,
    depth=18,
    model_file=None
):
    """
    Build a standard ResNet (BN only) with no CW.
    """
    if depth == 50:
        layers_cfg = [3, 4, 6, 3]
        model = ResidualNetBN(
            num_classes=num_classes,
            args=None,
            arch='resnet50',
            layers=layers_cfg,
            model_file=None
        )
    elif depth == 18:
        layers_cfg = [2, 2, 2, 2]
        model = ResidualNetBN(
            num_classes=num_classes,
            args=None,
            arch='resnet18',
            layers=layers_cfg,
            model_file=None
        )
    else:
        raise ValueError(f"Unsupported resnet depth: {depth}")

    if model_file is not None and os.path.isfile(model_file):
        print(f"[build_resnet_bn] Loading weights from {model_file}")
        ckpt = torch.load(model_file, map_location='cpu')
        if 'state_dict' in ckpt:
            sd = ckpt['state_dict']
        else:
            sd = ckpt
        new_sd = {}
        for k, v in sd.items():
            nk = k.replace('module.', '')
            new_sd[nk] = v
        model.load_state_dict(new_sd, strict=False)

    return model


def build_densenet_cw(
    num_classes=1000,
    arch='densenet161',
    whitened_layers=None,
    act_mode='pool_max',
    model_file=None
):
    if whitened_layers is None:
        whitened_layers = [1]

    model = DenseNetTransfer(
        num_classes=num_classes,
        args=None,
        whitened_layers=whitened_layers,
        arch=arch,
        model_file=None,    # load below
        act_mode=act_mode
    )

    if model_file is not None and os.path.isfile(model_file):
        print(f"[build_densenet_cw] Loading {model_file}")
        ckpt = torch.load(model_file, map_location='cpu')
        if 'state_dict' in ckpt:
            sd = ckpt['state_dict']
        else:
            sd = ckpt
        new_sd = {}
        for k, v in sd.items():
            nk = k.replace('module.', '')
            new_sd[nk] = v
        model.load_state_dict(new_sd, strict=False)

    return model


def build_densenet_bn(
    num_classes=1000,
    arch='densenet161',
    model_file=None
):
    model = DenseNetBN(
        num_classes=num_classes,
        args=None,
        arch=arch,
        model_file=None
    )

    if model_file is not None and os.path.isfile(model_file):
        print(f"[build_densenet_bn] Loading {model_file}")
        ckpt = torch.load(model_file, map_location='cpu')
        if 'state_dict' in ckpt:
            sd = ckpt['state_dict']
        else:
            sd = ckpt
        new_sd = {}
        for k,v in sd.items():
            nk = k.replace('module.', '')
            new_sd[nk] = v
        model.load_state_dict(new_sd, strict=False)

    return model


def build_vgg_cw(
    num_classes=1000,
    whitened_layers=None,
    act_mode='pool_max',
    model_file=None
):
    if whitened_layers is None:
        whitened_layers = [1]
    model = VGGBNTransfer(
        num_classes=num_classes,
        args=None,
        whitened_layers=whitened_layers,
        arch='vgg16_bn',
        model_file=None,
        act_mode=act_mode
    )

    if model_file is not None and os.path.isfile(model_file):
        ckpt = torch.load(model_file, map_location='cpu')
        if 'state_dict' in ckpt:
            sd = ckpt['state_dict']
        else:
            sd = ckpt
        # load partial
        new_sd = {}
        for k,v in sd.items():
            nk = k.replace('module.model.', '')
            new_sd[nk] = v
        model.load_state_dict(new_sd, strict=False)

    return model


def build_vgg_bn(
    num_classes=1000,
    arch='vgg16_bn',
    model_file=None
):
    model = VGGBN(
        num_classes=num_classes,
        args=None,
        arch=arch,
        model_file=None
    )

    if model_file is not None and os.path.isfile(model_file):
        ckpt = torch.load(model_file, map_location='cpu')
        if 'state_dict' in ckpt:
            sd = ckpt['state_dict']
        else:
            sd = ckpt
        new_sd = {}
        for k,v in sd.items():
            nk = k.replace('module.model.', '')
            new_sd[nk] = v
        model.load_state_dict(new_sd, strict=False)

    return model


############################################
# ORIGINAL CLASSES, with references to
#   "args.start_epoch" and "args.best_prec1" REMOVED.
# We keep the same param signature but do not rely on `args`.
############################################

class ResidualNetTransfer(nn.Module):
    def __init__(self, num_classes, args, whitened_layers=None,
                 arch='resnet18', layers=[2,2,2,2], model_file=None,
                 act_mode='pool_max'):

        super(ResidualNetTransfer, self).__init__()
        self.layers = layers
        self.model = models.__dict__[arch](num_classes=num_classes)

        # old code tried to set args.start_epoch from checkpoint
        # we remove that. We'll only do a partial model_file load:
        if model_file is not None:
            if not os.path.exists(model_file):
                raise Exception(f"checkpoint {model_file} not found!")
            checkpoint = torch.load(model_file, map_location='cpu')
            # We only load the state_dict
            # no references to "start_epoch" or "best_prec1"
            if 'state_dict' in checkpoint:
                raw_sd = checkpoint['state_dict']
            else:
                raw_sd = checkpoint
            new_sd = {}
            for k,v in raw_sd.items():
                nk = k.replace('module.', '')
                nk = nk.replace('bw', 'bn')  # old naming
                new_sd[nk] = v
            self.model.load_state_dict(new_sd, strict=False)

        self.whitened_layers = whitened_layers if whitened_layers else []
        self.act_mode = act_mode

        # Replace BN with CW
        c_dims = [64, 128, 256, 512]  # layer1->64, layer2->128, ...
        # We'll do the same approach
        offset = 0
        for wh_lyr in self.whitened_layers:
            if wh_lyr <= layers[0]:
                # belongs to layer1
                self.model.layer1[wh_lyr-1].bn1 = cw_layer(64, activation_mode=self.act_mode)
            elif wh_lyr <= layers[0] + layers[1]:
                self.model.layer2[wh_lyr - layers[0] - 1].bn1 = cw_layer(128, activation_mode=self.act_mode)
            elif wh_lyr <= sum(layers[0:3]):
                self.model.layer3[wh_lyr - (layers[0]+layers[1]) - 1].bn1 = cw_layer(256, activation_mode=self.act_mode)
            else:
                self.model.layer4[wh_lyr - sum(layers[0:3]) - 1].bn1 = cw_layer(512, activation_mode=self.act_mode)

    def change_mode(self, mode):
        """
        mode = -1: no gradient update
        0..k-1: concept index
        """
        l = self.layers
        for wh_lyr in self.whitened_layers:
            if wh_lyr <= l[0]:
                self.model.layer1[wh_lyr - 1].bn1.mode = mode
            elif wh_lyr <= l[0] + l[1]:
                self.model.layer2[wh_lyr - l[0] - 1].bn1.mode = mode
            elif wh_lyr <= (l[0]+l[1]+l[2]):
                self.model.layer3[wh_lyr - (l[0]+l[1]) - 1].bn1.mode = mode
            else:
                self.model.layer4[wh_lyr - (l[0]+l[1]+l[2]) - 1].bn1.mode = mode

    def update_rotation_matrix(self):
        l = self.layers
        for wh_lyr in self.whitened_layers:
            if wh_lyr <= l[0]:
                self.model.layer1[wh_lyr-1].bn1.update_rotation_matrix()
            elif wh_lyr <= l[0] + l[1]:
                self.model.layer2[wh_lyr - l[0] -1].bn1.update_rotation_matrix()
            elif wh_lyr <= l[0]+l[1]+l[2]:
                self.model.layer3[wh_lyr - (l[0]+l[1]) - 1].bn1.update_rotation_matrix()
            else:
                self.model.layer4[wh_lyr - (l[0]+l[1]+l[2]) -1].bn1.update_rotation_matrix()

    def forward(self, x):
        return self.model(x)


class DenseNetTransfer(nn.Module):
    def __init__(self, num_classes, args,
                 whitened_layers=None, arch='densenet161',
                 model_file=None, act_mode='pool_max'):
        super(DenseNetTransfer, self).__init__()
        self.model = models.__dict__[arch](num_classes=num_classes)

        if model_file is not None and os.path.exists(model_file):
            checkpoint = torch.load(model_file, map_location='cpu')
            if 'state_dict' in checkpoint:
                raw_sd = checkpoint['state_dict']
            else:
                raw_sd = checkpoint
            new_sd = {}
            for k,v in raw_sd.items():
                nk = k.replace('module.', '')
                new_sd[nk] = v
            self.model.load_state_dict(new_sd, strict=False)

        self.whitened_layers = whitened_layers if whitened_layers else []
        self.act_mode = act_mode

        # BN->CW replacements
        # e.g. if whitened_layer=1 => self.model.features.norm0 = cw_layer(64, act_mode)
        for w_l in self.whitened_layers:
            if w_l == 1:
                self.model.features.norm0 = cw_layer(64, activation_mode=self.act_mode)
            elif w_l == 2:
                self.model.features.transition1.norm = cw_layer(384, activation_mode=self.act_mode)
            elif w_l == 3:
                self.model.features.transition2.norm = cw_layer(768, activation_mode=self.act_mode)
            elif w_l == 4:
                self.model.features.transition3.norm = cw_layer(2112, activation_mode=self.act_mode)
            elif w_l == 5:
                self.model.features.norm5 = cw_layer(2208, activation_mode=self.act_mode)

    def change_mode(self, mode):
        for w_l in self.whitened_layers:
            if w_l == 1:
                self.model.features.norm0.mode = mode
            elif w_l == 2:
                self.model.features.transition1.norm.mode = mode
            elif w_l == 3:
                self.model.features.transition2.norm.mode = mode
            elif w_l == 4:
                self.model.features.transition3.norm.mode = mode
            elif w_l == 5:
                self.model.features.norm5.mode = mode

    def update_rotation_matrix(self):
        for w_l in self.whitened_layers:
            if w_l == 1:
                self.model.features.norm0.update_rotation_matrix()
            elif w_l == 2:
                self.model.features.transition1.norm.update_rotation_matrix()
            elif w_l == 3:
                self.model.features.transition2.norm.update_rotation_matrix()
            elif w_l == 4:
                self.model.features.transition3.norm.update_rotation_matrix()
            elif w_l == 5:
                self.model.features.norm5.update_rotation_matrix()

    def forward(self, x):
        return self.model(x)


class VGGBNTransfer(nn.Module):
    def __init__(self, num_classes, args,
                 whitened_layers=None, arch='vgg16_bn',
                 model_file=None, act_mode='pool_max'):
        super(VGGBNTransfer, self).__init__()
        self.model = models.__dict__[arch](num_classes=num_classes)
        if model_file is not None and os.path.exists(model_file):
            checkpoint = torch.load(model_file, map_location='cpu')
            if 'state_dict' in checkpoint:
                raw_sd = checkpoint['state_dict']
            else:
                raw_sd = checkpoint
            new_sd = {}
            for k,v in raw_sd.items():
                nk = k.replace('module.model.', '')
                new_sd[nk] = v
            self.model.load_state_dict(new_sd, strict=False)

        self.whitened_layers = whitened_layers if whitened_layers else []
        self.layers = [1,4,8,11,15,18,21,25,28,31,35,38,41]
        self.act_mode = act_mode

        for wl in self.whitened_layers:
            idx = wl - 1
            if idx in range(0,2):
                channel = 64
            elif idx in range(2,4):
                channel = 128
            elif idx in range(4,7):
                channel = 256
            else:
                channel = 512
            self.model.features[self.layers[idx]] = cw_layer(channel, activation_mode=self.act_mode)

    def change_mode(self, mode):
        for wl in self.whitened_layers:
            idx = wl - 1
            self.model.features[self.layers[idx]].mode = mode

    def update_rotation_matrix(self):
        for wl in self.whitened_layers:
            idx = wl - 1
            self.model.features[self.layers[idx]].update_rotation_matrix()

    def forward(self, x):
        return self.model(x)


class ResidualNetBN(nn.Module):
    def __init__(self, num_classes, args,
                 arch='resnet18', layers=[2,2,2,2],
                 model_file=None):
        super(ResidualNetBN, self).__init__()
        self.layers = layers
        self.model = models.__dict__[arch](num_classes=num_classes)

        if model_file is not None and os.path.exists(model_file):
            checkpoint = torch.load(model_file, map_location='cpu')
            if 'state_dict' in checkpoint:
                raw_sd = checkpoint['state_dict']
            else:
                raw_sd = checkpoint
            new_sd = {}
            for k,v in raw_sd.items():
                nk = k.replace('module.', '')
                new_sd[nk] = v
            self.model.load_state_dict(new_sd, strict=False)

    def forward(self, x):
        return self.model(x)


class DenseNetBN(nn.Module):
    def __init__(self, num_classes, args,
                 arch='densenet161', model_file=None):
        super(DenseNetBN, self).__init__()
        self.model = models.__dict__[arch](num_classes=num_classes)

        if model_file is not None and os.path.exists(model_file):
            checkpoint = torch.load(model_file, map_location='cpu')
            if 'state_dict' in checkpoint:
                raw_sd = checkpoint['state_dict']
            else:
                raw_sd = checkpoint
            import re
            new_sd = {}
            for k,v in raw_sd.items():
                nk = k.replace('module.', '')
                new_sd[nk] = v
            self.model.load_state_dict(new_sd, strict=False)

    def forward(self, x):
        return self.model(x)


class VGGBN(nn.Module):
    def __init__(self, num_classes, args,
                 arch='vgg16_bn', model_file=None):
        super(VGGBN, self).__init__()
        # user had "self.model = models.__dict__[arch](num_classes=365)"
        # but let's adapt to the user param:
        self.model = models.__dict__[arch](num_classes=num_classes)

        if model_file == 'vgg16_bn_places365.pt':
            # old logic that set start_epoch=0
            state_dict = torch.load(model_file, map_location='cpu')
            d = self.model.state_dict()
            new_sd = {}
            for k in d.keys():
                new_sd[k] = state_dict[k] if k in state_dict else d[k]
            self.model.load_state_dict(new_sd, strict=False)
        elif model_file is not None and os.path.exists(model_file):
            checkpoint = torch.load(model_file, map_location='cpu')
            if 'state_dict' in checkpoint:
                raw_sd = checkpoint['state_dict']
            else:
                raw_sd = checkpoint
            new_sd = {}
            for k,v in raw_sd.items():
                nk = k.replace('module.model.', '')
                new_sd[nk] = v
            self.model.load_state_dict(new_sd, strict=False)

    def forward(self, x):
        return self.model(x)
