"""Model configuration and task heads (lazy imports)."""
from importlib import import_module

__all__ = ["DeSCORTConfig", "DeSCORTModel", "DeSCORTForMaskedLM",
           "DeSCORTForContrastiveLM", "DeSCORTForPerturbPrediction"]

def __getattr__(name):
    if name in __all__:
        module = ".configuration_descort" if name == "DeSCORTConfig" else ".modeling_descort"
        value = getattr(import_module(module, __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
