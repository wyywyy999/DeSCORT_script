"""DeSCORT morphology representation and perturbation prediction models."""
from importlib import import_module

__version__ = "0.1.5"
__all__ = ["DeSCORTConfig", "DeSCORTModel", "DeSCORTForMaskedLM",
           "DeSCORTForContrastiveLM", "DeSCORTForPerturbPrediction"]

def __getattr__(name):
    if name in __all__:
        module = ".model.configuration_descort" if name == "DeSCORTConfig" else ".model.modeling_descort"
        value = getattr(import_module(module, __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
