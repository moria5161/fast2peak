model_registry = {}

def register_model(cls):
    model_registry[cls.__name__] = cls
    return cls

def build_model(model_name, **kwargs):
    if model_name not in model_registry:
        raise ValueError(f"Unregistered model: {model_name}")
    return model_registry[model_name](**kwargs)


from .f2p import *
from .aunet import AUNet
from .resunet import ResUNet
from .smae import SMAE
from .raman_mae import RamanMAE
from .smae2d import S2DMAE
model_registry['2DSMAE'] = S2DMAE