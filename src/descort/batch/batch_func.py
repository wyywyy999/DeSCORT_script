"""
Batch correction methods for AnnData objects.

Provides sphering (ZCA/PCA/Cholesky) and wrappers for BBKNN, ComBat,
Harmony, Scanorama, MNN, and DESC.
"""

import os
import time
import numpy as np
import scanpy as sc
from scipy import linalg


# ============================================================================
#  Spherize — data whitening
# ============================================================================

class Spherize:
    """Data whitening (sphering) transformation.

    Transforms data so the covariance matrix approaches identity.
    Typically used as pre-processing before MNN correction.

    Parameters
    ----------
    method : str, one of 'zca', 'pca', 'cholesky'.
    epsilon : float, regularization added to eigenvalues (default 1e-6).
    """

    def __init__(self, method='zca', epsilon=1e-6):
        if method not in ('zca', 'pca', 'cholesky'):
            raise ValueError(
                f"Unknown method '{method}'. "
                f"Choose from: zca, pca, cholesky"
            )
        self.method = method
        self.epsilon = epsilon
        self.W_ = None       # transformation matrix
        self.mean_ = None    # training-set mean

    def fit(self, X):
        """Estimate whitening transform on data X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        """
        self.mean_ = np.mean(X, axis=0)
        Xc = X - self.mean_
        n = Xc.shape[0]
        cov = (Xc.T @ Xc) / n

        eigenvalues, eigenvectors = linalg.eigh(cov)
        eigenvalues = np.maximum(eigenvalues, 0)

        if self.method == 'zca':
            D_inv_sqrt = np.diag(1.0 / np.sqrt(eigenvalues + self.epsilon))
            self.W_ = eigenvectors @ D_inv_sqrt @ eigenvectors.T
        elif self.method == 'pca':
            D_inv_sqrt = np.diag(1.0 / np.sqrt(eigenvalues + self.epsilon))
            self.W_ = D_inv_sqrt @ eigenvectors.T
        elif self.method == 'cholesky':
            cov_reg = cov + self.epsilon * np.eye(cov.shape[0])
            L = linalg.cholesky(linalg.inv(cov_reg), lower=True)
            self.W_ = L
        return self

    def transform(self, X):
        """Apply whitening transform.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        X_sphered : ndarray of shape (n_samples, n_features)
        """
        if self.W_ is None:
            raise RuntimeError("Call .fit() before .transform()")
        return (X - self.mean_) @ self.W_

    def fit_transform(self, X):
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)


# ============================================================================
#  Batch correction functions
# ============================================================================

def Sphering_func(adata, column_norm, save_path, values_norm=None,
                  method='zca', epsilon=1e-6, inplace=True, copy=False):
    """Sphering (whitening) pre-processing for AnnData.

    Parameters
    ----------
    adata : AnnData
    column_norm : str, column in ``adata.obs`` used to select training samples.
    save_path : str, output directory for the result .h5ad file.
    values_norm : list or None, subset of ``column_norm`` values for fitting.
    method : str, 'zca', 'pca', or 'cholesky'.
    epsilon : float, regularization.
    inplace : bool, modify adata in-place (default True).
    copy : bool, return a copy (default False).

    Returns
    -------
    AnnData or None
    """
    if copy:
        adata = adata.copy()
        inplace = True

    # Prepare data
    X = adata.X
    if hasattr(X, 'toarray'):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float64)

    # Determine training indices
    if values_norm is not None:
        train_ix = adata.obs[column_norm].isin(values_norm).values
        if train_ix.sum() == 0:
            raise ValueError(
                f"No samples in adata.obs['{column_norm}'] "
                f"match values_norm={values_norm}."
            )
    else:
        train_ix = np.ones(adata.n_obs, dtype=bool)

    # Fit spherer
    spherer = Spherize(method=method, epsilon=epsilon)
    spherer.fit(X[train_ix])

    # Transform all data
    X_sphered = spherer.transform(X).astype(np.float32)

    # Store spherer parameters
    adata.uns['spherer'] = {
        'method': spherer.method,
        'epsilon': spherer.epsilon,
        'W_': spherer.W_,
        'mean_': spherer.mean_,
    }

    # Downstream analysis
    sc.pp.pca(adata, n_comps=50)
    sc.pp.neighbors(adata, use_rep='X_pca')
    sc.tl.umap(adata)
    sc.tl.leiden(adata, key_added='Metadata_Cluster')

    adata.X = X_sphered
    sphering_save_path = os.path.join(save_path, 'Sphering_adata.h5ad')
    adata.write_h5ad(sphering_save_path)
    if copy:
        return adata


def bbknn_func(original_adata, save_path, batch_key):
    """BBKNN graph-based batch correction.

    Parameters
    ----------
    original_adata : AnnData
    save_path : str, output directory for the result .h5ad file.
    batch_key : str, column in ``adata.obs`` identifying batches.
    """
    from bbknn import bbknn
    adata = original_adata.copy()
    adata.obs.index = adata.obs.index.astype(str)

    # The input may already have a MAD neighbor graph.  Remove it so a
    # missing BBKNN output cannot silently fall back to the old graph.
    adata.uns.pop('neighbors', None)
    for graph_name in ('connectivities', 'distances'):
        if graph_name in adata.obsp:
            del adata.obsp[graph_name]

    neighbors_within_batch = 3 if adata.n_obs < 1e5 else 25

    bbknn(adata, batch_key=batch_key,
          neighbors_within_batch=neighbors_within_batch)

    # BBKNN 1.3.x stores its new graph in uns['neighbors'], whereas Scanpy
    # 1.10 reads obsp first.  The copied input can still contain the old
    # connectivities/distances in obsp, so move the BBKNN graph there before
    # computing Leiden or UMAP.  Also support BBKNN versions that already
    # provide the obsp keys.
    neighbors = adata.uns['neighbors']
    for graph_name in ('connectivities', 'distances'):
        graph_key = neighbors.get(f'{graph_name}_key', graph_name)
        if graph_name in neighbors:
            adata.obsp[graph_key] = neighbors.pop(graph_name)
        if graph_key not in adata.obsp:
            raise ValueError(f'BBKNN did not produce a {graph_name} graph')
        neighbors[f'{graph_name}_key'] = graph_key

    # Do not retain a UMAP/cluster annotation inherited from the input.
    if 'X_umap' in adata.obsm:
        del adata.obsm['X_umap']
    if 'Metadata_Cluster' in adata.obs:
        del adata.obs['Metadata_Cluster']

    sc.tl.leiden(adata, key_added='Metadata_Cluster')
    sc.tl.umap(adata)

    bbknn_save_path = os.path.join(save_path, 'Bbknn_adata.h5ad')
    adata.write_h5ad(bbknn_save_path, compression='gzip')


def Combat_func(adata, batch_key, save_path):
    """ComBat batch correction.

    Parameters
    ----------
    adata : AnnData
    batch_key : str, column in ``adata.obs`` identifying batches.
    save_path : str, output directory for the result .h5ad file.
    """
    adata.X = sc.pp.combat(adata, key=batch_key, inplace=False)

    sc.pp.pca(adata, n_comps=50)
    sc.pp.neighbors(adata, use_rep='X_pca', n_neighbors=25, metric='cosine')
    sc.tl.umap(adata)
    sc.tl.leiden(adata, key_added='Metadata_Cluster')

    combat_save_path = os.path.join(save_path, 'Combat_adata.h5ad')
    adata.write_h5ad(combat_save_path, compression='gzip')


def Harmony_func(adata, batch_key, save_path):
    """Harmony batch correction.

    Parameters
    ----------
    adata : AnnData. Requires ``adata.obsm['X_pca']`` to exist.
    batch_key : str, column in ``adata.obs`` identifying batches.
    save_path : str, output directory for the result .h5ad file.
    """
    import harmonypy as hm
    harmony_out = hm.run_harmony(adata.obsm['X_pca'], adata.obs, batch_key)
    adata.obsm['X_pca_harmony'] = harmony_out.Z_corr

    sc.pp.neighbors(adata, use_rep='X_pca_harmony')
    sc.tl.umap(adata)
    sc.tl.leiden(adata, key_added='Metadata_Cluster')

    harmony_save_path = os.path.join(save_path, 'Harmony_adata.h5ad')
    adata.write_h5ad(harmony_save_path, compression='gzip')


def Scanorama_func(adata, batch_key, save_path):
    """Scanorama batch correction.

    Parameters
    ----------
    adata : AnnData
    batch_key : str, column in ``adata.obs`` identifying batches.
    save_path : str, output directory for the result .h5ad file.
    """
    sc.external.pp.scanorama_integrate(
        adata, key=batch_key, adjusted_basis='X_scanorama',
    )
    sc.pp.neighbors(adata, use_rep='X_scanorama')
    sc.tl.umap(adata)
    sc.tl.leiden(adata, key_added='Metadata_Cluster')

    scanorama_save_path = os.path.join(save_path, 'Scanorama_adata.h5ad')
    adata.write_h5ad(scanorama_save_path, compression='gzip')


def Mnn_func(adata, batch_key, save_path):
    """Mutual Nearest Neighbors (MNN) batch correction.

    Parameters
    ----------
    adata : AnnData
    batch_key : str, column in ``adata.obs`` identifying batches.
    save_path : str, output directory for the result .h5ad file.
    """
    import mnnpy
    batches = adata.obs[batch_key].unique()
    hvgs = adata.var_names.tolist()

    expression_matrices = []
    batch_labels = []

    for batch in batches:
        batch_adata = adata[adata.obs[batch_key] == batch, hvgs].copy()
        if hasattr(batch_adata.X, 'A'):
            X = batch_adata.X.A
        else:
            X = batch_adata.X
        if not np.all(np.isfinite(X)):
            print(f"    ⚠ batch {batch} contains NaN/Inf, cleaning ...")
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        expression_matrices.append(X)
        batch_labels.append(batch)
        print(f"   - batch {batch}: {X.shape[0]} cells, {X.shape[1]} genes")

    corrected_result = mnnpy.mnn_correct(
        *expression_matrices,
        var_index=hvgs,
        var_subset=hvgs,
        batch_categories=batch_labels,
        cos_norm_in=True,
        cos_norm_out=True,
        var_adj=True,
    )
    corrected_matrices = corrected_result[0]

    adata.X = corrected_matrices

    sc.pp.pca(adata, n_comps=50)
    sc.pp.neighbors(adata, use_rep='X_pca', n_neighbors=25, metric='cosine')
    sc.tl.umap(adata)
    sc.tl.leiden(adata, key_added='Metadata_Cluster')

    mnn_save_path = os.path.join(save_path, 'Mnn_adata.h5ad')
    adata.write_h5ad(mnn_save_path)


def Desc_func(adata, save_path):
    """DESC deep-learning-based batch correction.

    Parameters
    ----------
    adata : AnnData
    save_path : str, output directory for the result .h5ad file.
    """
    import desc
    save_dir = f"desc_result_{time.strftime('%Y%m%d_%H%M%S')}"

    desc_adata = desc.train(
        adata,
        dims=[adata.shape[1], 128, 32],
        tol=0.005,
        n_neighbors=64,
        batch_size=1024,
        louvain_resolution=[1.0],
        save_dir=save_dir,
        do_tsne=True,
        learning_rate=300,
        do_umap=True,
        num_Cores_tsne=4,
        save_encoder_weights=True,
        use_GPU=True,
        GPU_id=0,
    )
    desc_save_path = os.path.join(save_path, 'Desc_adata.h5ad')
    desc_adata.write_h5ad(desc_save_path)
