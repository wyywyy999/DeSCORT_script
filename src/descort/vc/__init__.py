"""Virtual-cell processing, prediction and evaluation (lazy public API)."""
from importlib import import_module

__all__ = ['CellEvalMixin', 'ExtraMetrics', 'MetricsEvaluator', 'RAW_CONTINUOUS_INPUT', 'DE_BACKENDS', 'LEGACY_METRICS', 'VCC2026_METRICS', 'SUPPORTED_METRICS']

def __getattr__(name):
    if name in __all__:
        value = getattr(import_module(".inference", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
