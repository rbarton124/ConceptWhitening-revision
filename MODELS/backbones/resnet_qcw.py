import torch
import torch.nn as nn
import torchvision.models as models
from .base_qcw import BaseQCW

class ResNetQCW(BaseQCW):
    def _build_backbone(self, num_classes, depth, **kw):
        if depth == 18:
            _model = models.resnet18(num_classes=num_classes, weights=None)
            _bn_dims = [64, 128, 256, 512]
        elif depth == 50:
            _model = models.resnet50(num_classes=num_classes, weights=None)
            _bn_dims = [64, 128, 256, 512]
        else:
            raise ValueError(f"ResNet depth {depth} not supported. Use 18 or 50.")
            
        return _model, _bn_dims

    def _bn_iter(self):
        """Global-index = 1..8 (R18) or 1..16 (R50)."""
        layers = ["layer1", "layer2", "layer3", "layer4"]
        offset = 0
        for i, ln in enumerate(layers):
            layer = getattr(self.backbone, ln)
            for j, block in enumerate(layer):
                g_idx = offset + j + 1
                yield ((block, "bn1"), self.bn_dims[i], g_idx)
            offset += len(layer)

    def _extra_init(self, pretrained_model=None, **kw):
        # Optional: load pretrained weights if provided
        if pretrained_model and isinstance(pretrained_model, str):
            self.load_model(pretrained_model)
    
    def load_model(self, pretrain_path):
        print(f"[ResNetQCW] Loading pretrained weights from {pretrain_path} ...")
        import os
        import torch
        from collections import OrderedDict
        
        if not os.path.isfile(pretrain_path):
            print(f"[Warning] File not found: {pretrain_path}")
            return
            
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
            for blk in getattr(self.backbone, ln):
                out = blk(out)
        return self._final_pool_and_fc(out)
