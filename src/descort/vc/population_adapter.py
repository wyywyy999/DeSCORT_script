"""Train-only regularized mean head and deterministic population calibration.

The mean head learns from equal-weight perturbation means. Quantile calibration
uses training residuals; prediction never consumes held-out perturbed values.
It retains the backbone's within-population ranks (empirical copula), but is a
population operation: all generated wells of each gene must be passed together.
"""
import numpy as np


class KernelMeanHead:
    def fit(self, embeddings, targets, ridge):
        if ridge <= 0:
            raise ValueError('ridge must be positive')
        x, y = np.asarray(embeddings, dtype=np.float64), np.asarray(targets, dtype=np.float64)
        if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or not len(x):
            raise ValueError('Expected equally sized nonempty embedding/target matrices')
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('Nonfinite training data')
        self.center = x.mean(0)
        self.reference = self.encode(x)
        self.intercept = y.mean(0)
        kernel = self.reference @ self.reference.T
        self.coefficients = np.linalg.solve(kernel + ridge * np.eye(len(x)), y - self.intercept)
        self.ridge = ridge
        return self

    def encode(self, x):
        x = np.asarray(x, dtype=np.float64) - self.center
        return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)

    def fit_noise_aware(self, embeddings, targets, sampling_variance, ridge):
        """Stronger feature-specific ridge where train mean variance is noisy.

        Sampling variance is estimated from fitting genes only. A small pooled
        floor and bounded ratios avoid singular/negative signal estimates.
        No validation or test values enter these feature-specific penalties.
        """
        self.fit(embeddings, targets, ridge)
        y = np.asarray(targets, dtype=np.float64)
        noise = np.asarray(sampling_variance, dtype=np.float64)
        if len(y) < 2 or noise.shape != (y.shape[1],) or not np.isfinite(noise).all() or np.any(noise < 0):
            raise ValueError('Expected nonnegative per-feature sampling variances and >= 2 genes')
        signal = y.var(0, ddof=1) - noise
        floor = max(float(noise.mean()) * .01, 1e-12)
        ratio = np.clip(noise / np.maximum(signal, floor), .1, 1000.)
        self.ridge = ridge * ratio
        eigenvalues, vectors = np.linalg.eigh(self.reference @ self.reference.T)
        projected_targets = vectors.T @ (y - self.intercept)
        self.coefficients = vectors @ (projected_targets / (
            np.maximum(eigenvalues, 0.)[:, None] + self.ridge[None, :]))
        return self

    def predict(self, embeddings):
        return self.intercept + self.encode(embeddings) @ self.reference.T @ self.coefficients

    def save(self, path):
        np.savez_compressed(path, center=self.center, reference=self.reference,
                            intercept=self.intercept, coefficients=self.coefficients, ridge=self.ridge)

    @classmethod
    def load(cls, path):
        obj = cls()
        with np.load(path, allow_pickle=False) as state:
            for key in ['center', 'reference', 'intercept', 'coefficients', 'ridge']:
                setattr(obj, key, state[key])
        return obj


def fit_residual_quantiles(values, labels, n_quantiles=129):
    """Equal-gene Wasserstein barycenter of training within-gene residuals."""
    labels = np.asarray(labels)
    levels = np.linspace(0, 1, n_quantiles)
    quantiles = []
    for gene in np.unique(labels):
        x = np.asarray(values[labels == gene], dtype=np.float64)
        if len(x) > 1:
            quantiles.append(np.quantile(x - x.mean(0), levels, axis=0))
    if not quantiles:
        raise ValueError('Distribution fitting requires replicated training perturbations')
    return levels, np.mean(quantiles, axis=0)


def calibrate_population(backbone_prediction, labels, gene_means, levels, residual_quantiles):
    """Preserve within-gene ranks; ties receive equal averaged quantiles.

    Exactly preserves requested gene means after finite-sample centering.
    No row pairing or random noise is introduced. Singleton groups output means.
    """
    from scipy.stats import rankdata
    x = np.asarray(backbone_prediction)
    labels = np.asarray(labels)
    if not np.isfinite(x).all() or x.ndim != 2 or len(labels) != len(x):
        raise ValueError('Invalid generated population')
    if residual_quantiles.shape != (len(levels), x.shape[1]):
        raise ValueError('Residual quantile/feature dimensions differ')
    result = np.empty_like(x, dtype=np.float32)
    for gene in np.unique(labels):
        select = labels == gene
        sample = x[select]
        if len(sample) == 1:
            result[select] = gene_means[gene]
            continue
        ranks = (rankdata(sample, axis=0, method='average') - .5) / len(sample)
        residual = np.empty_like(sample, dtype=np.float64)
        for j in range(x.shape[1]):
            residual[:, j] = np.interp(ranks[:, j], levels, residual_quantiles[:, j])
        residual -= residual.mean(0)
        result[select] = gene_means[gene] + residual
    return result
