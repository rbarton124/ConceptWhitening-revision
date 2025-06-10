import torch
import torch.nn as nn
import torchvision.models as models
from .base_qcw import BaseQCW

class VGGQCW(BaseQCW):
    def _build_backbone(self, num_classes, **kw):
        _model = models.vgg16_bn(num_classes=num_classes, weights=None)
        # BN layers positions in features list
        self._bn_pos = [1, 4, 8, 11, 15, 18, 21, 25, 28, 31, 35, 38, 41]  # 13 BNs
        self._bn_dims = [64]*2 + [128]*2 + [256]*3 + [512]*6
        return _model, None

    def _bn_iter(self):
        for idx, feat_idx in enumerate(self._bn_pos):
            g_idx = idx + 1
            mod_ref = (self.backbone.features, feat_idx)
            c = self._bn_dims[idx]
            yield (mod_ref, c, g_idx)

    def _final_pool_and_fc(self, x):
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return self.backbone.classifier(x)

    def forward(self, x):
        out = self.backbone.features(x)
        out = self.backbone.avgpool(out)
        out = torch.flatten(out, 1)
        return self.backbone.classifier(out)
