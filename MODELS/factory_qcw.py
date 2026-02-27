from MODELS.backbones.resnet_qcw import ResNetQCW
from MODELS.backbones.densenet_qcw import DenseNetQCW
from MODELS.backbones.vgg_qcw import VGGQCW
from MODELS.iterative_normalization import IterNormRotation

def build_qcw(model_type="resnet", **kw):
    """
    Factory function to build QCW models of different architectures.
    """
    model_type = model_type.lower()
    if model_type in ("resnet",):
        return ResNetQCW(**kw)
    elif model_type in ("densenet", "densenet161", "densenet121"):
        return DenseNetQCW(**kw)
    elif model_type in ("vgg16", "vgg"):
        return VGGQCW(**kw)
    else:
        raise ValueError(f"Unknown model type '{model_type}'. Choose from: resnet, densenet, vgg16")

def get_last_qcw_layer(model):
    """
    Return the last QCW (IterNormRotation) layer in the model, or None if not present.
    Prefers the explicit registry (model.cw_layers) if available.
    """
    if hasattr(model, "cw_layers") and model.cw_layers:
        return model.cw_layers[-1]
    last = None
    for m in model.modules():
        if isinstance(m, IterNormRotation):
            last = m
    return last