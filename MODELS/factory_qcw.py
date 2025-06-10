from MODELS.backbones.resnet_qcw import ResNetQCW
from MODELS.backbones.densenet_qcw import DenseNetQCW
from MODELS.backbones.vgg_qcw import VGGQCW
from MODELS.iterative_normalization import IterNormRotation

def build_qcw(model_type="resnet", **kw):
    """
    Factory function to build QCW models of different architectures.
    
    Args:
        model_type (str): One of "resnet", "densenet", or "vgg16"
        **kw: Additional arguments passed to the model constructor
    
    Returns:
        A QCW model of the specified architecture
    """
    model_type = model_type.lower()
    if model_type == "resnet":
        return ResNetQCW(**kw)
    elif model_type == "densenet":
        return DenseNetQCW(**kw)
    elif model_type == "vgg16":
        return VGGQCW(**kw)
    else:
        raise ValueError(f"Unknown model type '{model_type}'. Choose from: resnet, densenet, vgg16")

def get_last_qcw_layer(model):
    """
    Find the last QCW layer in the model.
    
    Args:
        model: A QCW model
        
    Returns:
        The last IterNormRotation layer in the model
    """
    if hasattr(model, "cw_layers") and model.cw_layers:
        return model.cw_layers[-1]
    
    # If that fails, search through all modules
    for m in reversed(list(model.modules())):
        if isinstance(m, IterNormRotation):
            return m
    
    raise ValueError("No QCW layer found in the model")
