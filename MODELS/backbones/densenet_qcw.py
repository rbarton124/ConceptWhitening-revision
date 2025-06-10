import torch
import torch.nn as nn
import torchvision.models as models
from .base_qcw import BaseQCW

class DenseNetQCW(BaseQCW):
    def _build_backbone(self, num_classes, depth, **kw):
        if depth == 121:
            _model = models.densenet121(num_classes=num_classes, weights=None)
        elif depth == 161:
            _model = models.densenet161(num_classes=num_classes, weights=None)
        else:
            raise ValueError(f"DenseNet depth {depth} not supported. Use 121 or 161.")
        
        # CW-replaceable BN locations & dims are fixed:
        self._bn_map = {
            1: ("features.norm0", 64),
            2: ("features.transition1.norm",  256 if depth==121 else 384),
            3: ("features.transition2.norm",  512 if depth==121 else 768),
            4: ("features.transition3.norm", 1024 if depth==121 else 2112),
            5: ("features.norm5",             1024 if depth==121 else 2208)
        }
        return _model, None

    def _bn_iter(self):
        for g_idx, (attr, c) in self._bn_map.items():
            parent, name = attr.rsplit(".", 1)
            mod = self._get_attr(parent)
            # Return in a format compatible with both indexing and attribute access
            yield ((mod, name), c, g_idx)

    def _get_attr(self, dotted):
        obj = self.backbone
        for token in dotted.split("."):
            obj = getattr(obj, token)
        return obj

    def _final_pool_and_fc(self, x):
        return self.backbone.classifier(x)

    def forward(self, x):
        out = self.backbone.features(x)
        out = nn.functional.relu(out, inplace=True)
        out = nn.functional.adaptive_avg_pool2d(out, (1, 1)).flatten(1)
        return self.backbone.classifier(out)
