"""

This module implements the metrics used by ``04_descope_metric copy.ipynb`` and
opt-in VCC2026-compatible metrics without requiring ``cell_eval``.  Differential
expression has a dedicated raw-continuous path: SciPy is always available, while
Scanpy and pdex can be selected as optional Wilcoxon p-value backends.

The VCC2026 additions are a morphology adaptation: DeSCORT outputs signed,
MAD-normalized Cell Painting features rather than non-negative RNA counts.  The
expression formulas therefore operate directly in continuous feature space and
DE effect sizes use group-mean differences rather than invalid log fold changes.
No count normalization, CPM filtering, log1p or expm1 transformation is performed
in this path.
"""

from __future__ import annotations

import importlib.util
import logging
import multiprocessing as mp
import os
import threading
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from scipy.stats import (
    false_discovery_control,
    mannwhitneyu,
    pearsonr,
    ranksums,
    ttest_ind,
)
from sklearn.metrics import pairwise_distances
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)
evaluator_logger = logging.getLogger("cell_eval._evaluator")
pdex_logger = logging.getLogger("pdex._single_cell")


LEGACY_METRICS = [
    "pearson_delta",
    "mse",
    "mae",
    "mse_delta",
    "mae_delta",
    "discrimination_score_l1",
    "discrimination_score_l2",
    "discrimination_score_cosine",
    "overlap_at_N",
    "overlap_at_50",
    "overlap_at_100",
]

VCC2026_METRICS = [
    "pds_cosine",
    "expr_mse_unbiased",
    "expr_mse_unbiased_capped",
    "expr_distance_unbiased",
    "expr_real_mass_ratio",
    "expr_mse_unbiased_capped_norm",
    "de_wilcoxon_lfc_nmae",
    "de_wilcoxon_direction_fidelity_yield_raw",
    "de_wilcoxon_direction_reach_raw",
    "de_wilcoxon_sig_jaccard",
]

SUPPORTED_METRICS = LEGACY_METRICS + VCC2026_METRICS

VCC2026_DE_METRICS = {
    "de_wilcoxon_lfc_nmae",
    "de_wilcoxon_direction_fidelity_yield_raw",
    "de_wilcoxon_direction_reach_raw",
    "de_wilcoxon_sig_jaccard",
}
DE_METRICS = {
    "overlap_at_N",
    "overlap_at_50",
    "overlap_at_100",
} | VCC2026_DE_METRICS
METRIC_EXECUTION_ORDER = [
    "overlap_at_N",
    "overlap_at_50",
    "overlap_at_100",
    "pearson_delta",
    "mse",
    "mae",
    "mse_delta",
    "mae_delta",
    "discrimination_score_l1",
    "discrimination_score_l2",
    "discrimination_score_cosine",
    *VCC2026_METRICS,
]

RAW_CONTINUOUS_INPUT = "raw-continuous"
DE_BACKENDS = {"auto", "scipy", "scanpy", "pdex"}


@contextmanager
def _elapsed_progress(description: str, *, enabled: bool = True) -> Iterator[None]:
    """Show elapsed-time activity for a backend without progress callbacks.

    Scanpy executes one blocking ``rank_genes_groups`` call and exposes no
    perturbation-level callback. An indeterminate bar is therefore more honest
    than a fabricated percentage: it confirms that the call is alive and shows
    elapsed wall time until the backend returns.
    """
    if not enabled:
        yield
        return

    stopped = threading.Event()
    progress = tqdm(
        total=None,
        desc=description,
        unit="s",
        bar_format="{desc}: {elapsed}",
    )

    def update_elapsed_time() -> None:
        while not stopped.wait(1.0):
            progress.update(1)

    worker = threading.Thread(target=update_elapsed_time, daemon=True)
    worker.start()
    try:
        yield
    finally:
        stopped.set()
        worker.join(timeout=2.0)
        progress.close()


def _enable_info_logging() -> None:
    """Make INFO messages visible in scripts and Jupyter without duplicates."""
    for current_logger in (logger, evaluator_logger, pdex_logger):
        current_logger.setLevel(logging.INFO)
    root_logger = logging.getLogger()
    if not logger.handlers and not root_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
        logger.addHandler(handler)
        evaluator_logger.addHandler(handler)
        pdex_logger.addHandler(handler)
        for current_logger in (logger, evaluator_logger, pdex_logger):
            current_logger.propagate = False


def _dense(matrix: Any) -> np.ndarray:
    """Return a dense two-dimensional NumPy array."""
    if issparse(matrix):
        matrix = matrix.toarray()
    array = np.asarray(matrix)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D matrix, got shape {array.shape}")
    return array


def _load_anndata(value: Any) -> Any:
    if not isinstance(value, (str, os.PathLike)):
        return value

    import anndata as ad

    logger.info("Reading AnnData from %s", value)
    return ad.read_h5ad(value)


def _mean_rows(matrix: Any, indices: np.ndarray) -> np.ndarray:
    values = matrix[indices]
    if issparse(values):
        return np.asarray(values.mean(axis=0)).ravel()
    return np.asarray(values).mean(axis=0)


@dataclass
class PerturbationAnndataPair:
    """Validated real/predicted AnnData pair and cached pseudobulk profiles."""

    real: Any
    pred: Any
    pert_col: str
    control_pert: str
    perts: np.ndarray = field(init=False)
    genes: np.ndarray = field(init=False)
    pert_mask_real: dict[str, np.ndarray] = field(init=False)
    pert_mask_pred: dict[str, np.ndarray] = field(init=False)
    _bulk_real: dict[str, np.ndarray] = field(init=False, repr=False)
    _bulk_pred: dict[str, np.ndarray] = field(init=False, repr=False)
    _correction_real: dict[str, float] = field(init=False, repr=False)
    _correction_pred: dict[str, float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.real.shape[1] != self.pred.shape[1]:
            raise ValueError(
                "Feature-count mismatch: "
                f"real={self.real.shape[1]}, pred={self.pred.shape[1]}"
            )

        real_genes = self.real.var_names.astype(str).to_numpy()
        pred_genes = self.pred.var_names.astype(str).to_numpy()
        if not np.array_equal(real_genes, pred_genes):
            raise ValueError("Feature names or their order differ between real and pred")
        self.genes = real_genes

        for name, adata in (("real", self.real), ("pred", self.pred)):
            if self.pert_col not in adata.obs.columns:
                raise ValueError(
                    f"Column {self.pert_col!r} is absent from {name}.obs; "
                    f"available columns: {list(adata.obs.columns)}"
                )

        real_labels = self.real.obs[self.pert_col].astype(str).to_numpy()
        pred_labels = self.pred.obs[self.pert_col].astype(str).to_numpy()
        real_perts = np.unique(real_labels)
        pred_perts = np.unique(pred_labels)
        if not np.array_equal(real_perts, pred_perts):
            only_real = np.setdiff1d(real_perts, pred_perts).tolist()
            only_pred = np.setdiff1d(pred_perts, real_perts).tolist()
            raise ValueError(
                "Perturbations differ between real and pred: "
                f"only_real={only_real}, only_pred={only_pred}"
            )
        if str(self.control_pert) not in real_perts:
            raise ValueError(
                f"Control {self.control_pert!r} is absent from {self.pert_col!r}"
            )

        self.control_pert = str(self.control_pert)
        self.perts = real_perts[real_perts != self.control_pert]
        self.pert_mask_real = self._make_masks(real_labels)
        self.pert_mask_pred = self._make_masks(pred_labels)
        self._bulk_real = self._build_bulk(self.real, self.pert_mask_real)
        self._bulk_pred = self._build_bulk(self.pred, self.pert_mask_pred)
        self._correction_real = {}
        self._correction_pred = {}

    @staticmethod
    def _make_masks(labels: np.ndarray) -> dict[str, np.ndarray]:
        return {label: np.flatnonzero(labels == label) for label in np.unique(labels)}

    @staticmethod
    def _build_bulk(
        adata: Any, masks: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        return {label: _mean_rows(adata.X, rows) for label, rows in masks.items()}

    def get_perts(self, include_control: bool = False) -> np.ndarray:
        if include_control:
            return np.append(self.perts, self.control_pert)
        return self.perts

    def bulk(self, which: Literal["real", "pred"], pert: str) -> np.ndarray:
        return (self._bulk_real if which == "real" else self._bulk_pred)[pert]

    def effect(self, which: Literal["real", "pred"], pert: str) -> np.ndarray:
        return self.bulk(which, pert) - self.bulk(which, self.control_pert)

    def sampling_correction(
        self, which: Literal["real", "pred"], pert: str
    ) -> float:
        """Return ``sum(feature sample variance) / n`` for one group mean."""
        cache = self._correction_real if which == "real" else self._correction_pred
        if pert in cache:
            return cache[pert]
        adata = self.real if which == "real" else self.pred
        masks = self.pert_mask_real if which == "real" else self.pert_mask_pred
        rows = masks[pert]
        if len(rows) < 2:
            value = 0.0
        else:
            values = adata.X[rows]
            if issparse(values):
                mean = np.asarray(values.mean(axis=0)).ravel()
                mean_square = np.asarray(values.multiply(values).mean(axis=0)).ravel()
                trace = float(np.maximum(mean_square - mean * mean, 0.0).sum())
                trace *= len(rows) / (len(rows) - 1)
            else:
                trace = float(
                    np.var(np.asarray(values), axis=0, ddof=1, dtype=np.float64).sum()
                )
            value = trace / len(rows)
        cache[pert] = value
        return value

    def iter_bulk_arrays(self) -> Iterator["BulkArrays"]:
        for pert in self.perts:
            yield BulkArrays(
                key=str(pert),
                pert_real=self.bulk("real", pert),
                pert_pred=self.bulk("pred", pert),
                ctrl_real=self.bulk("real", self.control_pert),
                ctrl_pred=self.bulk("pred", self.control_pert),
            )


@dataclass(frozen=True)
class BulkArrays:
    key: str
    pert_real: np.ndarray
    pert_pred: np.ndarray
    ctrl_real: np.ndarray
    ctrl_pred: np.ndarray

    def perturbation_effect(
        self, which: Literal["real", "pred"], abs: bool = False
    ) -> np.ndarray:
        if which == "real":
            value = self.pert_real - self.ctrl_real
        else:
            value = self.pert_pred - self.ctrl_pred
        return np.abs(value) if abs else value


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    return float(pearsonr(x, y).statistic)


def _basic_metric(
    pair: PerturbationAnndataPair,
    metric: Literal["pearson_delta", "mse", "mae", "mse_delta", "mae_delta"],
) -> dict[str, float]:
    output: dict[str, float] = {}
    use_delta = metric.endswith("_delta") or metric == "pearson_delta"
    for arrays in tqdm(
        pair.iter_bulk_arrays(),
        total=len(pair.perts),
        desc="Iterating over perturbations...",
    ):
        if use_delta:
            pred = arrays.perturbation_effect("pred")
            real = arrays.perturbation_effect("real")
        else:
            pred, real = arrays.pert_pred, arrays.pert_real

        if metric == "pearson_delta":
            value = _safe_pearson(pred, real)
        elif metric.startswith("mse"):
            value = float(np.mean(np.square(pred - real)))
        else:
            value = float(np.mean(np.abs(pred - real)))
        output[arrays.key] = value
    return output


def _discrimination_score(
    pair: PerturbationAnndataPair,
    metric: Literal["l1", "l2", "cosine"],
) -> dict[str, float]:
    real_effects = np.vstack(
        [
            pair.effect("real", pert)
            for pert in tqdm(pair.perts, desc="Iterating over perturbations...")
        ]
    )
    pred_effects = np.vstack(
        [
            pair.effect("pred", pert)
            for pert in tqdm(pair.perts, desc="Iterating over perturbations...")
        ]
    )
    sklearn_metric = {"l1": "manhattan", "l2": "euclidean", "cosine": "cosine"}[
        metric
    ]

    output: dict[str, float] = {}
    all_feature_indices = np.arange(real_effects.shape[1])
    for index, pert in enumerate(pair.perts):
        # cell_eval excludes the targeted gene if expression features contain it.
        include = all_feature_indices[pair.genes != str(pert)]
        distances = pairwise_distances(
            real_effects[:, include],
            pred_effects[index, include].reshape(1, -1),
            metric=sklearn_metric,
        ).ravel()
        sorted_indices = np.argsort(distances)
        rank = int(np.flatnonzero(sorted_indices == index)[0])
        output[str(pert)] = 1.0 - rank / len(pair.perts)
    return output


def _vcc2026_pds_cosine(
    pair: PerturbationAnndataPair,
    *,
    show_progress: bool = True,
) -> dict[str, float]:
    """VCC2026 PDS ranking, adapted to signed morphology effects.

    This uses the VCC2026 ``n-1`` denominator, mid-ranks ties, and the real
    control as the prediction baseline.  Target-gene exclusion is intentionally
    omitted because Cell Painting feature columns are not expression genes.
    """
    real_effects = np.vstack([pair.effect("real", str(p)) for p in pair.perts])
    real_control = pair.bulk("real", pair.control_pert)
    pred_effects = np.vstack(
        [pair.bulk("pred", str(p)) - real_control for p in pair.perts]
    )
    with _elapsed_progress(
        f"Computing PDS cosine distances ({len(pair.perts)} perturbations)",
        enabled=show_progress,
    ):
        distances = pairwise_distances(pred_effects, real_effects, metric="cosine")
    n_perts = len(pair.perts)
    output: dict[str, float] = {}
    for index, pert in enumerate(
        tqdm(
            pair.perts,
            desc="Ranking PDS perturbations",
            disable=not show_progress,
        )
    ):
        row = distances[index]
        match = row[index]
        if np.isnan(match):
            n_less = int(np.count_nonzero(~np.isnan(row)))
            n_equal = int(np.count_nonzero(np.isnan(row)))
        else:
            n_less = int(np.count_nonzero(row < match))
            n_equal = int(np.count_nonzero(row == match))
        rank = n_less + (n_equal - 1) / 2.0
        output[str(pert)] = 1.0 if n_perts <= 1 else 1.0 - rank / (n_perts - 1)
    return output


def _vcc2026_expression_metrics(
    pair: PerturbationAnndataPair,
    *,
    show_progress: bool = True,
) -> dict[str, dict[str, float]]:
    """Continuous-feature counterparts of the five VCC2026 expression metrics.

    The unbiased-distance formulas are unchanged.  Means and sampling variances
    are evaluated directly in the signed Cell Painting feature space instead of
    applying the VCC raw-count normalization.
    """
    names = VCC2026_METRICS[1:6]
    output = {name: {} for name in names}
    n_features = pair.real.shape[1]
    if n_features == 0:
        return output

    real_ctrl = pair.bulk("real", pair.control_pert)
    real_ctrl_correction = pair.sampling_correction("real", pair.control_pert)
    control_mass = float(np.abs(real_ctrl).sum())
    for pert_value in tqdm(
        pair.perts,
        desc="Computing expression metrics",
        disable=not show_progress,
    ):
        pert = str(pert_value)
        real_mean = pair.bulk("real", pert)
        pred_mean = pair.bulk("pred", pert)
        real_correction = pair.sampling_correction("real", pert)
        pred_correction = pair.sampling_correction("pred", pert)

        squared_error = float(np.square(pred_mean - real_mean).sum())
        mse_unbiased = (
            squared_error - pred_correction - real_correction
        ) / n_features
        mse_capped = (
            squared_error
            - min(pred_correction, real_correction)
            - real_correction
        ) / n_features
        distance = (
            float(np.square(real_mean - real_ctrl).sum())
            - real_correction
            - real_ctrl_correction
        ) / n_features
        mass_ratio = (
            float("nan")
            if control_mass == 0
            else float(np.abs(real_mean).sum()) / control_mass
        )
        norm = float("nan") if distance == 0 else mse_capped / distance

        output["expr_mse_unbiased"][pert] = mse_unbiased
        output["expr_mse_unbiased_capped"][pert] = mse_capped
        output["expr_distance_unbiased"][pert] = distance
        output["expr_real_mass_ratio"][pert] = mass_ratio
        output["expr_mse_unbiased_capped_norm"][pert] = norm
    return output


def _resolve_de_backend(backend: str, method: str = "wilcoxon") -> str:
    """Resolve an optional Wilcoxon engine without importing heavy packages."""
    backend = str(backend).lower()
    if backend not in DE_BACKENDS:
        raise ValueError(
            f"de_backend must be one of {sorted(DE_BACKENDS)}, got {backend!r}"
        )
    if method != "wilcoxon":
        if backend not in {"auto", "scipy"}:
            raise ValueError(
                f"de_backend={backend!r} only supports de_method='wilcoxon'"
            )
        return "scipy"
    if backend == "auto":
        for candidate in ("pdex", "scanpy"):
            if importlib.util.find_spec(candidate) is not None:
                return candidate
        return "scipy"
    if backend != "scipy" and importlib.util.find_spec(backend) is None:
        raise ImportError(
            f"de_backend={backend!r} was requested but {backend} is not installed; "
            "use de_backend='scanpy', 'scipy', or 'auto'"
        )
    return backend


def _raw_continuous_effects(
    adata: Any,
    pair: PerturbationAnndataPair,
    which: Literal["real", "pred"],
) -> pd.DataFrame:
    """Arithmetic mean differences in the original signed feature space."""
    masks = pair.pert_mask_real if which == "real" else pair.pert_mask_pred
    reference_mean = _mean_rows(adata.X, masks[pair.control_pert]).astype(float)
    rows: list[pd.DataFrame] = []
    for target in pd.unique(adata.obs[pair.pert_col].astype(str)):
        target = str(target)
        if target == pair.control_pert:
            continue
        target_mean = _mean_rows(adata.X, masks[target]).astype(float)
        signed_effect = target_mean - reference_mean
        rows.append(
            pd.DataFrame(
                {
                    "target": target,
                    "reference": pair.control_pert,
                    "feature": pair.genes.astype(str),
                    "target_mean": target_mean,
                    "reference_mean": reference_mean,
                    # Ratios are undefined for signed MAD-normalized features.
                    # Compatibility columns remain NaN so old DE readers still
                    # see the expected schema without consuming invalid log2FC.
                    "percent_change": np.nan,
                    "fold_change": np.nan,
                    "signed_effect": signed_effect,
                    "abs_signed_effect": np.abs(signed_effect),
                }
            )
        )
    if not rows:
        raise ValueError("DE requires at least one non-control perturbation")
    return pd.concat(rows, ignore_index=True)


def _scanpy_raw_continuous_pvalues(
    adata: Any,
    *,
    groupby: str,
    reference: str,
    tie_correct: bool,
    num_threads: int,
    show_progress: bool,
) -> pd.DataFrame:
    """Use Scanpy only as a rank-test engine on untouched signed ``adata.X``."""
    import scanpy as sc

    if num_threads not in {-1, 1}:
        evaluator_logger.info(
            "Scanpy Wilcoxon ignores num_threads=%s; its own execution policy is used",
            num_threads,
        )
    work = adata.copy()
    work.obs[groupby] = pd.Categorical(work.obs[groupby].astype(str))
    targets = [
        str(value)
        for value in pd.unique(work.obs[groupby].astype(str))
        if str(value) != reference
    ]
    # rank_genes_groups also tries to produce an RNA-style logFC table. It is
    # meaningless for signed morphology data and may emit divide/log warnings;
    # only its rank-test p-values and scores are retained below.
    with _elapsed_progress(
        f"Scanpy Wilcoxon ({len(targets)} perturbations x {work.n_vars} features)",
        enabled=show_progress,
    ):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            sc.tl.rank_genes_groups(
                work,
                groupby=groupby,
                groups=targets,
                reference=reference,
                method="wilcoxon",
                tie_correct=tie_correct,
                n_genes=work.n_vars,
                use_raw=False,
                layer=None,
            )
    result = work.uns["rank_genes_groups"]
    frames = []
    target_fields = result["names"].dtype.names or ()
    for target in tqdm(
        target_fields,
        desc="Collecting Scanpy DE results",
        disable=not show_progress,
    ):
        frames.append(
            pd.DataFrame(
                {
                    "target": str(target),
                    "feature": np.asarray(result["names"][target], dtype=str),
                    "p_value": np.asarray(result["pvals"][target], dtype=float),
                    "statistic": np.asarray(result["scores"][target], dtype=float),
                }
            )
        )
    if not frames:
        raise RuntimeError("Scanpy returned no Wilcoxon comparisons")
    return pd.concat(frames, ignore_index=True)


def _pdex_raw_continuous_pvalues(
    adata: Any,
    *,
    groupby: str,
    reference: str,
    num_threads: int,
) -> pd.DataFrame:
    """Use pdex only for MWU p-values; discard its count-space effect columns."""
    from pdex import pdex

    threads = 0 if num_threads == -1 else max(1, int(num_threads))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        frame = pdex(
            adata,
            groupby=groupby,
            mode="ref",
            reference=reference,
            threads=threads,
            geometric_mean=False,
            is_log1p=False,
            epsilon=0.0,
            cpm_filter=None,
        )
    if hasattr(frame, "to_dict"):
        try:
            frame = pd.DataFrame(frame.to_dict(as_series=False))
        except TypeError:
            frame = pd.DataFrame(frame)
    else:
        frame = pd.DataFrame(frame)
    required = ["target", "feature", "p_value", "statistic"]
    missing = set(required).difference(frame.columns)
    if missing:
        raise RuntimeError(f"pdex output is missing columns: {sorted(missing)}")
    return frame.loc[:, required].copy()


def _de_batches(
    target_masks: dict[str, np.ndarray],
    target_names: np.ndarray,
    feature_count: int,
    control_pert: str,
    batch_size: int,
) -> Iterator[list[tuple[np.ndarray, np.ndarray, int, str]]]:
    """Yield pdex-compatible target/feature batches."""
    batch: list[tuple[np.ndarray, np.ndarray, int, str]] = []
    control_mask = target_masks[control_pert]
    for target in target_names:
        for feature_index in range(feature_count):
            batch.append(
                (target_masks[str(target)], control_mask, feature_index, str(target))
            )
            if len(batch) == batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


def _process_de_batch(
    batch: list[tuple[np.ndarray, np.ndarray, int, str]],
    *,
    shared_memory_name: str,
    matrix_shape: tuple[int, int],
    matrix_dtype: np.dtype,
    control_pert: str,
    feature_names: np.ndarray,
    method: str,
    tie_correct: bool,
) -> list[dict[str, Any]]:
    """SciPy rank/t-test fallback on untouched raw-continuous values."""
    shared_memory = SharedMemory(name=shared_memory_name)
    matrix = np.ndarray(matrix_shape, dtype=matrix_dtype, buffer=shared_memory.buf)
    rows: list[dict[str, Any]] = []
    try:
        for target_mask, control_mask, feature_index, target_name in batch:
            if target_name == control_pert:
                continue
            target = matrix[target_mask, feature_index]
            reference = matrix[control_mask, feature_index]
            target_mean_value = float(target.mean())
            reference_mean_value = float(reference.mean())
            signed_effect = target_mean_value - reference_mean_value

            p_value, statistic = 1.0, float("nan")
            try:
                if method == "wilcoxon":
                    if tie_correct:
                        # Scanpy's Wilcoxon path applies tie correction but no
                        # 0.5 continuity correction.  SciPy applies tie
                        # correction in its asymptotic MWU variance; explicitly
                        # disabling continuity makes the p-values numerically
                        # equivalent to Scanpy while allowing this fallback to
                        # run across ``num_threads`` worker processes.
                        test = mannwhitneyu(
                            target,
                            reference,
                            method="asymptotic",
                            use_continuity=False,
                        )
                    else:
                        # ranksums implements the untied normal approximation
                        # used when Scanpy receives tie_correct=False.
                        test = ranksums(target, reference)
                else:
                    test = ttest_ind(target, reference)
                p_value = float(test.pvalue)
                statistic = float(test.statistic)
            except ValueError:
                pass

            rows.append(
                {
                    "target": target_name,
                    "reference": control_pert,
                    "feature": str(feature_names[feature_index]),
                    "target_mean": target_mean_value,
                    "reference_mean": reference_mean_value,
                    "percent_change": float("nan"),
                    "fold_change": float("nan"),
                    "p_value": p_value,
                    "statistic": statistic,
                    # VCC2026-compatible DE metrics use this value for signed
                    # continuous features; fold-change is undefined when values
                    # can be negative.
                    "signed_effect": signed_effect,
                    "abs_signed_effect": abs(signed_effect),
                }
            )
    finally:
        shared_memory.close()
    return rows


def _apply_fdr(
    frame: pd.DataFrame, fdr_scope: Literal["global", "per_pert"]
) -> pd.DataFrame:
    """Add global BH for auditing and select the requested FDR scope."""
    if fdr_scope not in {"global", "per_pert"}:
        raise ValueError("fdr_scope must be 'global' or 'per_pert'")
    result = frame.copy()
    result["p_value"] = result["p_value"].fillna(1.0)
    try:
        result["fdr_global"] = false_discovery_control(
            result["p_value"].to_numpy(), method="bh"
        )
    except ValueError:
        logger.warning("Could not calculate global FDR; using p-values")
        result["fdr_global"] = result["p_value"]

    if fdr_scope == "global":
        result["fdr"] = result["fdr_global"]
        return result

    result["fdr"] = 1.0
    for target_name, indices in result.groupby("target", sort=False).groups.items():
        try:
            result.loc[indices, "fdr"] = false_discovery_control(
                result.loc[indices, "p_value"].to_numpy(), method="bh"
            )
        except ValueError:
            logger.warning(
                "Could not calculate per-perturbation FDR for %s; using p-values",
                target_name,
            )
            result.loc[indices, "fdr"] = result.loc[indices, "p_value"]
    return result


def _compute_de(
    adata: Any,
    pair: PerturbationAnndataPair,
    which: Literal["real", "pred"],
    method: str,
    num_threads: int,
    *,
    backend: str = "auto",
    input_type: str = RAW_CONTINUOUS_INPUT,
    tie_correct: bool = True,
    fdr_scope: Literal["global", "per_pert"] = "per_pert",
    batch_size: int = 100,
    show_progress: bool = True,
) -> pd.DataFrame:
    if method not in {"wilcoxon", "t-test"}:
        raise ValueError("de_method must be 'wilcoxon' or 't-test'")
    if input_type != RAW_CONTINUOUS_INPUT:
        raise ValueError(
            f"src.descort.vc only supports de_input_type={RAW_CONTINUOUS_INPUT!r}; "
            "counts/lognorm processing belongs in the official cell_eval2 pipeline"
        )
    resolved_backend = _resolve_de_backend(backend, method)
    evaluator_logger.info(
        "Raw-continuous DE backend resolved: requested=%s, resolved=%s",
        backend,
        resolved_backend,
    )

    if resolved_backend in {"scanpy", "pdex"}:
        if resolved_backend == "scanpy":
            pvalues = _scanpy_raw_continuous_pvalues(
                adata,
                groupby=pair.pert_col,
                reference=pair.control_pert,
                tie_correct=tie_correct,
                num_threads=num_threads,
                show_progress=show_progress,
            )
        else:
            pvalues = _pdex_raw_continuous_pvalues(
                adata,
                groupby=pair.pert_col,
                reference=pair.control_pert,
                num_threads=num_threads,
            )
        effects = _raw_continuous_effects(adata, pair, which)
        pvalues["target"] = pvalues["target"].astype(str)
        pvalues["feature"] = pvalues["feature"].astype(str)
        if pvalues.duplicated(["target", "feature"]).any():
            raise RuntimeError(
                f"{resolved_backend} returned duplicate (target, feature) rows"
            )
        result = effects.merge(
            pvalues.loc[:, ["target", "feature", "p_value", "statistic"]],
            on=["target", "feature"],
            how="left",
            validate="one_to_one",
        )
        missing = int(result["p_value"].isna().sum())
        if missing:
            raise RuntimeError(
                f"{resolved_backend} omitted {missing} target-feature Wilcoxon results"
            )
        result["effect_type"] = "target_mean_minus_control_mean"
        result["de_backend"] = resolved_backend
        result["de_input_type"] = input_type
        return _apply_fdr(result, fdr_scope)

    matrix = np.ascontiguousarray(_dense(adata.X))
    masks = pair.pert_mask_real if which == "real" else pair.pert_mask_pred
    feature_names = pair.genes
    workers = (os.cpu_count() or 1) if num_threads == -1 else max(1, num_threads)
    target_names = np.asarray(
        [
            str(value)
            for value in pd.unique(adata.obs[pair.pert_col].astype(str))
            if str(value) != pair.control_pert
        ]
    )
    combination_count = len(target_names) * len(feature_names)
    batch_count = (combination_count + batch_size - 1) // batch_size

    pdex_logger.info("Input type: raw-continuous (no normalization/log transform)")
    pdex_logger.info("Precomputing masks for each target gene")
    for _ in tqdm(
        target_names,
        desc="Identifying target masks",
        disable=not show_progress,
    ):
        pass
    pdex_logger.info("Precomputing variable indices for each feature")
    for _ in tqdm(
        feature_names,
        desc="Identifying variable indices",
        disable=not show_progress,
    ):
        pass
    pdex_logger.info("Creating shared memory matrix for parallel computing")
    shared_memory = SharedMemory(create=True, size=matrix.nbytes)
    shared_matrix = np.ndarray(
        matrix.shape, dtype=matrix.dtype, buffer=shared_memory.buf
    )
    shared_matrix[:] = matrix
    del shared_matrix

    batches = _de_batches(
        target_masks=masks,
        target_names=target_names,
        feature_count=len(feature_names),
        control_pert=pair.control_pert,
        batch_size=batch_size,
    )
    task = partial(
        _process_de_batch,
        shared_memory_name=shared_memory.name,
        matrix_shape=matrix.shape,
        matrix_dtype=matrix.dtype,
        control_pert=pair.control_pert,
        feature_names=feature_names,
        method=method,
        tie_correct=tie_correct,
    )

    pdex_logger.info("Creating generator of all combinations: N=%d", combination_count)
    pdex_logger.info("Creating generator of all batches: N=%d", batch_count)
    pdex_logger.info("Initializing parallel processing pool with %d workers", workers)
    pdex_logger.info("Processing batches")
    try:
        with mp.Pool(workers) as pool:
            batch_results = list(
                tqdm(
                    pool.imap(task, batches),
                    total=batch_count,
                    desc="Processing batches",
                    disable=not show_progress,
                )
            )
        pdex_logger.info("Flattening results")
        result = pd.DataFrame([row for batch in batch_results for row in batch])
    finally:
        pdex_logger.info("Closing shared memory pool")
        shared_memory.close()
        shared_memory.unlink()

    result["effect_type"] = "target_mean_minus_control_mean"
    result["de_backend"] = resolved_backend
    result["de_input_type"] = input_type
    # VCC2026/v2 defaults to per-perturbation BH. fdr_global is retained
    # alongside fdr so legacy/debug comparisons remain auditable.
    return _apply_fdr(result, fdr_scope)


def _load_de(value: Any) -> pd.DataFrame:
    if isinstance(value, (str, os.PathLike)):
        return pd.read_csv(value)
    if isinstance(value, pd.DataFrame):
        return value.copy()
    # Accept polars-like objects without importing polars.
    if hasattr(value, "to_pandas"):
        return value.to_pandas()
    return pd.DataFrame(value)


def _prepare_de(
    value: pd.DataFrame,
    *,
    require_signed: bool = False,
    fdr_scope: Literal["global", "per_pert"] | None = None,
) -> pd.DataFrame:
    required = {"target", "feature", "p_value", "fdr"}
    missing = required.difference(value.columns)
    if missing:
        raise ValueError(f"DE data is missing columns: {sorted(missing)}")
    value = value.copy()
    if "fold_change" not in value:
        value["fold_change"] = np.nan
    if "log2_fold_change" not in value:
        if require_signed:
            # Do not manufacture RNA-style ratios in raw-continuous space.
            value["log2_fold_change"] = np.nan
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                value["log2_fold_change"] = np.log2(
                    value["fold_change"].to_numpy()
                )
            value["log2_fold_change"] = value["log2_fold_change"].fillna(0.0)
    if "abs_log2_fold_change" not in value:
        value["abs_log2_fold_change"] = value["log2_fold_change"].abs()
    if require_signed and "signed_effect" not in value:
        if {"target_mean", "reference_mean"}.issubset(value.columns):
            value["signed_effect"] = value["target_mean"] - value["reference_mean"]
            logger.warning(
                "DE table has no signed_effect column; derived it from "
                "target_mean - reference_mean. Recompute DE for exact signed-space values."
            )
        else:
            raise ValueError(
                "DE data needs signed_effect (or target_mean and reference_mean) for "
                "VCC2026-compatible signed-feature metrics"
            )
    if "signed_effect" in value and "abs_signed_effect" not in value:
        value["abs_signed_effect"] = value["signed_effect"].abs()
    value["target"] = value["target"].astype(str)
    value["feature"] = value["feature"].astype(str)
    if fdr_scope is not None:
        value = _apply_fdr(value, fdr_scope)
    return value


def _overlap(
    real_de: pd.DataFrame, pred_de: pd.DataFrame, perts: Iterable[str], k: int | None
) -> dict[str, float]:
    """DE-feature overlap ranked by signed Cell Painting effect magnitude."""
    for side, frame in (("real", real_de), ("pred", pred_de)):
        if "abs_signed_effect" not in frame.columns:
            raise ValueError(
                f"{side} DE data has no abs_signed_effect column; recompute DE with "
                "src.descort.vc before calculating overlap for signed features"
            )
    real_sig = real_de.loc[real_de["fdr"] < 0.05]
    pred_sig = pred_de.loc[pred_de["fdr"] < 0.05]
    output: dict[str, float] = {}
    for pert in perts:
        real_genes = (
            real_sig.loc[real_sig["target"] == str(pert)]
            .sort_values("abs_signed_effect", ascending=False, kind="stable")["feature"]
            .to_numpy()
        )
        pred_genes = (
            pred_sig.loc[pred_sig["target"] == str(pert)]
            .sort_values("abs_signed_effect", ascending=False, kind="stable")["feature"]
            .to_numpy()
        )
        k_eff = len(real_genes) if k is None else min(k, len(real_genes))
        output[str(pert)] = (
            0.0
            if k_eff == 0
            else float(np.intersect1d(real_genes[:k_eff], pred_genes[:k_eff]).size)
            / k_eff
        )
    return output


def _drop_exact_target_feature(frame: pd.DataFrame, pert: str) -> pd.DataFrame:
    """Apply VCC target-feature exclusion when an exact feature match exists."""
    return frame.loc[
        ~(
            (frame["target"].astype(str) == str(pert))
            & (frame["feature"].astype(str) == str(pert))
        )
    ]


def _vcc2026_de_metrics(
    real_de: pd.DataFrame,
    pred_de: pd.DataFrame,
    perts: Iterable[str],
    *,
    fdr_threshold: float = 0.05,
    min_gate_size: int = 10,
    reach_purity_floor: float = 0.9,
    show_progress: bool = True,
) -> dict[str, dict[str, float]]:
    """VCC2026 DE formulas using Wilcoxon signed effects instead of log2FC.

    ``signed_effect`` is the raw continuous-feature mean difference.  This is
    the only valid direction/size quantity for MAD-normalized Cell Painting
    features, for which ratios and log2 fold changes are not defined.
    """
    output = {name: {} for name in VCC2026_DE_METRICS}
    for pert_value in tqdm(
        perts,
        desc="Computing VCC2026 DE metrics",
        disable=not show_progress,
    ):
        pert = str(pert_value)
        real = _drop_exact_target_feature(
            real_de.loc[real_de["target"].astype(str) == pert].copy(), pert
        )
        pred = _drop_exact_target_feature(
            pred_de.loc[pred_de["target"].astype(str) == pert].copy(), pert
        )
        real = real.drop_duplicates(["target", "feature"], keep="first")
        pred = pred.drop_duplicates(["target", "feature"], keep="first")

        real_sig = real.loc[real["fdr"] < fdr_threshold]
        pred_sig = pred.loc[pred["fdr"] < fdr_threshold]
        real_set = set(real_sig["feature"].astype(str))
        pred_set = set(pred_sig["feature"].astype(str))
        union = real_set | pred_set
        output["de_wilcoxon_sig_jaccard"][pert] = (
            1.0 if not union else len(real_set & pred_set) / len(union)
        )

        # Real-significant, finite-effect genes form the submission-independent gate.
        gate = real_sig.loc[np.isfinite(real_sig["signed_effect"])].copy()
        pred_effect = pred.set_index("feature")["signed_effect"]
        gate_pred = gate["feature"].map(pred_effect).fillna(0.0).to_numpy(dtype=float)
        gate_real = gate["signed_effect"].to_numpy(dtype=float)
        denominator = float(np.mean(np.abs(gate_real))) if len(gate) else float("nan")
        if len(gate) >= min_gate_size and np.isfinite(denominator) and denominator != 0:
            output["de_wilcoxon_lfc_nmae"][pert] = float(
                np.mean(np.abs(gate_pred - gate_real)) / denominator
            )
        else:
            output["de_wilcoxon_lfc_nmae"][pert] = float("nan")

        real_effect = real.set_index("feature")["signed_effect"]
        scored = pred_sig.loc[pred_sig["feature"].isin(real_effect.index)].copy()
        scored_real = scored["feature"].map(real_effect).to_numpy(dtype=float)
        scored_pred = scored["signed_effect"].to_numpy(dtype=float)
        adjudicable = np.isfinite(scored_real) & (scored_real != 0)
        scored_real = scored_real[adjudicable]
        scored_pred = scored_pred[adjudicable]
        n_pred = len(scored_real)
        n_conf = int(np.count_nonzero(np.isfinite(gate_real) & (gate_real != 0)))
        k_match = int(
            np.count_nonzero(
                (scored_pred != 0) & (np.sign(scored_pred) == np.sign(scored_real))
            )
        )
        fidelity_denominator = max(n_pred, n_conf)
        output["de_wilcoxon_direction_fidelity_yield_raw"][pert] = (
            0.0 if fidelity_denominator == 0 else k_match / fidelity_denominator
        )

        # Rank the reference-significant (adjudicated) universe using predicted DE.
        ranked = gate[["feature", "signed_effect"]].rename(
            columns={"signed_effect": "signed_effect_real"}
        )
        ranked = ranked.merge(
            pred[["feature", "fdr", "p_value", "signed_effect"]],
            on="feature",
            how="left",
        )
        ranked["fdr"] = ranked["fdr"].fillna(1.0)
        ranked["p_value"] = ranked["p_value"].fillna(1.0)
        ranked["signed_effect"] = ranked["signed_effect"].fillna(0.0)
        ranked["_sig"] = ranked["fdr"] < fdr_threshold
        ranked["_abs_effect"] = ranked["signed_effect"].abs()
        ranked = ranked.sort_values(
            ["_sig", "fdr", "p_value", "_abs_effect", "feature"],
            ascending=[False, True, True, False, True],
            kind="stable",
        )
        if n_conf == 0:
            reach = float("nan")
        else:
            pred_sign = np.sign(ranked["signed_effect"].to_numpy(dtype=float))
            real_sign = np.sign(ranked["signed_effect_real"].to_numpy(dtype=float))
            matches = (pred_sign != 0) & (pred_sign == real_sign)
            purity = np.cumsum(matches) / np.arange(1, len(matches) + 1)
            qualifying = np.flatnonzero(purity >= reach_purity_floor)
            k_star = 0 if len(qualifying) == 0 else int(qualifying[-1] + 1)
            reach = k_star / n_conf
        output["de_wilcoxon_direction_reach_raw"][pert] = reach
    return output


def _describe_like_polars(frame: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the aggregate table written by ``polars.DataFrame.describe``."""
    statistics = [
        "count",
        "null_count",
        "mean",
        "std",
        "min",
        "25%",
        "50%",
        "75%",
        "max",
    ]
    output: dict[str, list[Any]] = {"statistic": statistics}
    for column in frame.columns:
        values = frame[column].to_numpy(dtype=float)
        sorted_values = np.sort(values)

        def nearest_quantile(quantile: float) -> float:
            if not len(sorted_values):
                return float("nan")
            index = int(np.floor(quantile * (len(sorted_values) - 1) + 0.5))
            return float(sorted_values[index])

        output[column] = [
            float(len(values)),
            0.0,
            float(np.mean(values)),
            float(np.std(values, ddof=1)) if len(values) > 1 else float("nan"),
            float(np.nanmin(values)) if len(values) else float("nan"),
            nearest_quantile(0.25),
            nearest_quantile(0.50),
            nearest_quantile(0.75),
            float(np.nanmax(values)) if len(values) else float("nan"),
        ]
    return pd.DataFrame(output)


class MetricsEvaluator:
    """Small evaluator object retained for notebook API compatibility."""

    def __init__(
        self,
        anndata_pair: PerturbationAnndataPair,
        de_real: pd.DataFrame | None,
        de_pred: pd.DataFrame | None,
        outdir: str | os.PathLike[str],
        de_backend: str | None = None,
        de_input_type: str | None = None,
    ) -> None:
        self.anndata_pair = anndata_pair
        self.de_real = de_real
        self.de_pred = de_pred
        self.outdir = str(outdir)
        self.de_backend = de_backend
        self.de_input_type = de_input_type


class ExtraMetrics:
    """Additional metric used in the original DeSCOPE notebook."""

    @staticmethod
    def pearson_delta_on_topk_de(
        data: PerturbationAnndataPair,
        de_real: str | os.PathLike[str] | pd.DataFrame,
        topk: int = 20,
    ) -> tuple[pd.DataFrame, float]:
        if topk <= 0:
            raise ValueError("topk must be positive")
        de_frame = _load_de(de_real)
        required = {"target", "feature", "statistic"}
        missing = required.difference(de_frame.columns)
        if missing:
            raise ValueError(f"DE data is missing columns: {sorted(missing)}")

        feature_to_index = {name: index for index, name in enumerate(data.genes)}
        rows: list[dict[str, Any]] = []
        for pert in de_frame["target"].astype(str).unique():
            pert_de = de_frame.loc[de_frame["target"].astype(str) == pert].copy()
            pert_de["_abs_stat"] = pert_de["statistic"].abs()
            top_features = pert_de.nlargest(topk, "_abs_stat")["feature"].astype(str)
            indices = [
                feature_to_index[name]
                for name in top_features
                if name in feature_to_index
            ]
            if not indices or pert not in data.pert_mask_real:
                continue

            real_delta = data.effect("real", pert)[indices]
            pred_delta = data.effect("pred", pert)[indices]
            correlation = _safe_pearson(real_delta, pred_delta)
            rows.append(
                {
                    "perturbation": pert,
                    "pearson_delta_topk_de": 0.0
                    if np.isnan(correlation)
                    else correlation,
                }
            )

        results = pd.DataFrame(
            rows, columns=["perturbation", "pearson_delta_topk_de"]
        )
        mean_value = (
            float(results["pearson_delta_topk_de"].mean()) if not results.empty else 0.0
        )
        return results, mean_value


class CellEvalMixin:
    """Drop-in legacy evaluator plus signed-feature VCC2026 adaptations.

    ``vcc2026_metrics`` follows the VCC2026 metric names and high-level
    formulas, but it is intentionally evaluated in signed morphology-feature
    space.  It is therefore not numerically interchangeable with the official
    raw-count VCC2026 scorer.  ``de_input_type='raw-continuous'`` is the only
    supported DE input mode.  ``de_backend`` accepts ``'auto'``, ``'scanpy'``,
    ``'pdex'`` or ``'scipy'``; the selected engine supplies Wilcoxon p-values
    only, while this module owns the signed arithmetic-mean effect size.
    """

    all_metrics = SUPPORTED_METRICS.copy()
    legacy_metrics = LEGACY_METRICS.copy()
    vcc2026_metrics = VCC2026_METRICS.copy()

    def __init__(self) -> None:
        _enable_info_logging()
        self.extra_metrics_func = ExtraMetrics()

    def compute_metrics(
        self,
        adata_pred: Any,
        adata_real: Any,
        metrics_to_calculate: list[str] | str | None = None,
        de_pred: Any = None,
        de_real: Any = None,
        control_pert: str = "non-targeting",
        pert_col: str = "target",
        de_method: str = "wilcoxon",
        de_backend: str = "auto",
        de_input_type: str = RAW_CONTINUOUS_INPUT,
        num_threads: int = -1,
        outdir: str | os.PathLike[str] = "./cell-eval-outdir",
        **kwargs: Any,
    ) -> tuple[pd.DataFrame, pd.DataFrame, MetricsEvaluator]:
        if metrics_to_calculate is None:
            # Preserve the pre-VCC default.  New metrics are opt-in through the
            # profile name or an explicit metric list.
            requested = self.legacy_metrics.copy()
        elif metrics_to_calculate == "vcc2026":
            requested = self.vcc2026_metrics.copy()
        elif isinstance(metrics_to_calculate, str):
            requested = [metrics_to_calculate]
        else:
            requested = list(metrics_to_calculate)
            if not requested:
                requested = self.legacy_metrics.copy()
        unknown = sorted(set(requested).difference(self.all_metrics))
        if unknown:
            raise ValueError(
                f"Unsupported metrics: {unknown}; supported metrics: {self.all_metrics}"
            )
        requested_set = set(requested)
        requested = [
            metric for metric in METRIC_EXECUTION_ORDER if metric in requested_set
        ]
        vcc_requested = bool(requested_set.intersection(VCC2026_METRICS))
        if vcc_requested:
            evaluator_logger.warning(
                "VCC2026-compatible metrics are being computed in signed Cell Painting "
                "feature space. These are not official raw-count VCC2026 scores."
            )
        if requested_set.intersection(VCC2026_DE_METRICS) and de_method != "wilcoxon":
            raise ValueError(
                "VCC2026-compatible DE metrics require de_method='wilcoxon'"
            )

        real = _load_anndata(adata_real)
        pred = _load_anndata(adata_pred)
        pair = PerturbationAnndataPair(
            real=real, pred=pred, control_pert=control_pert, pert_col=pert_col
        )
        output_dir = Path(outdir)
        if output_dir.exists():
            evaluator_logger.warning(
                "Output directory %s already exists, potential overwrite occurring",
                output_dir,
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        evaluator_logger.warning("-------------------------------------------------")
        evaluator_logger.warning("Running cell-eval without data format validation.")
        evaluator_logger.warning("-------------------------------------------------")

        real_de_frame: pd.DataFrame | None = None
        pred_de_frame: pd.DataFrame | None = None
        if set(requested).intersection(DE_METRICS):
            pdex_kwargs = kwargs.get("pdex_kwargs") or {}
            legacy_is_log1p = pdex_kwargs.get("is_log1p")
            if legacy_is_log1p not in {None, False}:
                raise ValueError(
                    "raw-continuous Cell Painting DE cannot use is_log1p=True; "
                    "remove it or set it to False"
                )
            de_options = {
                key: pdex_kwargs[key]
                for key in (
                    "tie_correct",
                    "fdr_scope",
                    "batch_size",
                    "show_progress",
                )
                if key in pdex_kwargs
            }
            de_options["backend"] = de_backend
            de_options["input_type"] = de_input_type
            de_options.setdefault("fdr_scope", kwargs.get("fdr_scope", "per_pert"))
            de_options.setdefault("batch_size", int(kwargs.get("batch_size", 100)))
            resolved_de_backend = _resolve_de_backend(de_backend, de_method)
            if de_real is None:
                evaluator_logger.info("Computing DE for real data")
                evaluator_logger.info(
                    "Using the following raw-continuous DE settings: %s",
                    {
                        "reference": control_pert,
                        "groupby_key": pert_col,
                        "backend": resolved_de_backend,
                        "input_type": de_input_type,
                        "num_workers": num_threads,
                        "batch_size": de_options["batch_size"],
                        "metric": de_method,
                        "fdr_scope": de_options["fdr_scope"],
                    },
                )
                real_de_frame = _compute_de(
                    real, pair, "real", de_method, num_threads, **de_options
                )
                evaluator_logger.info("Writing real DE results to: real_de.csv")
                real_de_frame.to_csv(output_dir / "real_de.csv", index=False)
            else:
                evaluator_logger.info("Reading real DE results from %s", de_real)
                real_de_frame = _load_de(de_real)
            if de_pred is None:
                evaluator_logger.info("Computing DE for pred data")
                evaluator_logger.info(
                    "Using the following raw-continuous DE settings: %s",
                    {
                        "reference": control_pert,
                        "groupby_key": pert_col,
                        "backend": resolved_de_backend,
                        "input_type": de_input_type,
                        "num_workers": num_threads,
                        "batch_size": de_options["batch_size"],
                        "metric": de_method,
                        "fdr_scope": de_options["fdr_scope"],
                    },
                )
                pred_de_frame = _compute_de(
                    pred, pair, "pred", de_method, num_threads, **de_options
                )
                evaluator_logger.info("Writing pred DE results to: pred_de.csv")
                pred_de_frame.to_csv(output_dir / "pred_de.csv", index=False)
            else:
                evaluator_logger.info("Reading pred DE results from %s", de_pred)
                pred_de_frame = _load_de(de_pred)
            # Both the Cell Painting overlap metrics and the VCC2026-compatible
            # DE metrics rank/compare raw signed mean effects, never log2FC.
            require_signed = bool(requested_set.intersection(DE_METRICS))
            real_de_frame = _prepare_de(
                real_de_frame,
                require_signed=require_signed,
                fdr_scope=de_options["fdr_scope"],
            )
            pred_de_frame = _prepare_de(
                pred_de_frame,
                require_signed=require_signed,
                fdr_scope=de_options["fdr_scope"],
            )

        metric_values: dict[str, dict[str, float]] = {}
        expression_values: dict[str, dict[str, float]] | None = None
        vcc_de_values: dict[str, dict[str, float]] | None = None
        show_metric_progress = bool(
            (kwargs.get("pdex_kwargs") or {}).get(
                "show_progress", kwargs.get("show_progress", True)
            )
        )
        for metric in requested:
            evaluator_logger.info("Computing metric %s", metric)
            if metric in {"pearson_delta", "mse", "mae", "mse_delta", "mae_delta"}:
                metric_values[metric] = _basic_metric(pair, metric)  # type: ignore[arg-type]
            elif metric.startswith("discrimination_score_"):
                distance = metric.removeprefix("discrimination_score_")
                metric_values[metric] = _discrimination_score(pair, distance)  # type: ignore[arg-type]
            elif metric == "pds_cosine":
                metric_values[metric] = _vcc2026_pds_cosine(
                    pair,
                    show_progress=show_metric_progress,
                )
            elif metric.startswith("expr_"):
                if expression_values is None:
                    expression_values = _vcc2026_expression_metrics(
                        pair,
                        show_progress=show_metric_progress,
                    )
                metric_values[metric] = expression_values[metric]
            elif metric in VCC2026_DE_METRICS:
                assert real_de_frame is not None and pred_de_frame is not None
                if vcc_de_values is None:
                    vcc_de_values = _vcc2026_de_metrics(
                        real_de_frame,
                        pred_de_frame,
                        pair.perts.astype(str),
                        fdr_threshold=float(kwargs.get("fdr_threshold", 0.05)),
                        min_gate_size=int(kwargs.get("min_gate_size", 10)),
                        reach_purity_floor=float(
                            kwargs.get("reach_purity_floor", 0.9)
                        ),
                        show_progress=show_metric_progress,
                    )
                metric_values[metric] = vcc_de_values[metric]
            else:
                assert real_de_frame is not None and pred_de_frame is not None
                k = {"overlap_at_N": None, "overlap_at_50": 50, "overlap_at_100": 100}[
                    metric
                ]
                metric_values[metric] = _overlap(
                    real_de_frame, pred_de_frame, pair.perts.astype(str), k
                )

        results = pd.DataFrame({"perturbation": pair.perts.astype(str)})
        for metric in requested:
            results[metric] = results["perturbation"].map(metric_values[metric])
        aggregate = _describe_like_polars(results.drop(columns="perturbation"))
        if (
            "expr_mse_unbiased_capped_norm" in requested_set
            and expression_values is not None
        ):
            numerator = np.asarray(
                list(expression_values["expr_mse_unbiased_capped"].values()),
                dtype=float,
            )
            denominator = np.asarray(
                list(expression_values["expr_distance_unbiased"].values()),
                dtype=float,
            )
            denominator_sum = float(np.nansum(denominator))
            ratio_of_sums = (
                float("nan")
                if denominator_sum == 0
                else float(np.nansum(numerator)) / denominator_sum
            )
            aggregate.loc[
                aggregate["statistic"] == "mean",
                "expr_mse_unbiased_capped_norm",
            ] = ratio_of_sums

        evaluator_logger.info(
            "Writing perturbation level metrics to %s", output_dir / "results.csv"
        )
        evaluator_logger.info(
            "Writing aggregate metrics to %s", output_dir / "agg_results.csv"
        )
        results.to_csv(output_dir / "results.csv", index=False)
        aggregate.to_csv(output_dir / "agg_results.csv", index=False)
        evaluator = MetricsEvaluator(
            pair,
            real_de_frame,
            pred_de_frame,
            output_dir,
            de_backend=(
                _resolve_de_backend(de_backend, de_method)
                if set(requested).intersection(DE_METRICS)
                else None
            ),
            de_input_type=(
                de_input_type if set(requested).intersection(DE_METRICS) else None
            ),
        )
        return results, aggregate, evaluator
