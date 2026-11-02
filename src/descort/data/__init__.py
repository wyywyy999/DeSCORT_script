"""Data utilities; preprocessing does not eagerly import training dependencies."""

from importlib import import_module

from .dataset import (
    DeSCORTForMaskedLMDataProcessor,
    DeSCORTPerturbPredictionProcessor,
    DeSCORTForContrastiveLMDataProcessor,
    merge_datasets_direct,
)

_LAZY_EXPORTS = {
    'DeSCORTForMaskedLMDataCollator': '.collator',
    'DeSCORTForContrastiveLMDataCollator': '.collator',
    'DeSCORTPerturbForMaskedLMDataCollator': '.collator',
    'PerturbPredictionDataCollator': '.collator',
    'GeneGroupedBatchSampler': '.sampler',
    'DistributedGeneGroupedBatchSampler': '.sampler',
    'GeneGroupedTrainer': '.sampler',
}
__all__ = [
    'DeSCORTForMaskedLMDataProcessor', 'DeSCORTPerturbPredictionProcessor',
    'DeSCORTForContrastiveLMDataProcessor', 'merge_datasets_direct',
    *_LAZY_EXPORTS,
]


def __getattr__(name):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    value = getattr(import_module(_LAZY_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value
