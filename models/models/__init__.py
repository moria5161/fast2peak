model_registry = {}

def register_model(cls):
    model_registry[cls.__name__] = cls
    return cls

def build_model(model_name, **kwargs):
    if model_name not in model_registry:
        raise ValueError(f"未注册的模型: {model_name}")
    return model_registry[model_name](**kwargs)


from .f2p import *
from .aunet import AUNet
from .resunet import ResUNet
from .smae import SMAE