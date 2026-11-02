# Support both an installed package and direct execution from this checkout.
from pathlib import Path as _Path
import sys as _sys
_SOURCE = _Path(__file__).resolve().parents[2] / "src"
if str(_SOURCE) not in _sys.path:
    _sys.path.insert(0, str(_SOURCE))

import os
import sys
import pickle
import logging

from tqdm import tqdm



def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate existing batch-corrected AnnData files")
    parser.add_argument("--data-path", default="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/other_model/DeSCORT/outputs/05_batch_effect/Target_2_Compoud/new_batch_adata")
    parser.add_argument("--save-path", default="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/other_model/DeSCORT/outputs/05_batch_effect/Target_2_Compoud/new_metrices_result")
    parser.add_argument("--method-embeddings-json", type=_Path, help='JSON mapping method prefixes to X/obsm keys; default preserves the Bbknn-only experiment')
    args = parser.parse_args()
    import scanpy as sc
    from descort.batch.calculate_metrics import find_h5ad_files, calculate_batch_bio_metrics, h5ad_2_df, calculate_mAP
    data_path = args.data_path
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    data_path_list = find_h5ad_files(data_path)
    
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
)

    data_mapping_dict = {
        # 'MAD':'X',
        # 'DeSCORT':'X',
        # 'DeSCORTlayernorm50epochs':'X',
        # 'DeSCORTlayernorm':'X',
        # 'DeSCORT10epochs3505':'X',
        # 'DeSCORT10epochs1024':'X',
        # BBKNN only corrects a neighbor graph; its X/X_pca remain the MAD input.
        # Use the UMAP obtained from the corrected BBKNN graph so the existing
        # embedding-based scIB, direct-LISI and mAP code evaluates its output.
        'Bbknn':'X_umap',
        # 'Seuratcca':'X',
        # 'Seuratrpca':'X',
        # 'Mnn':'X',
        # 'Combat':'X',
        # 'Sphering':'X',
        # 'Raw':'X',
        # 'Fast':'X',
        # 'Scanorama':'X_scanorama',
        # 'Harmony':'X_pca_harmony',
        # 'Desc':'X_Embeded_z1.0',
        
    }

    if args.method_embeddings_json:
        import json
        data_mapping_dict = json.loads(args.method_embeddings_json.read_text())

    metric_mapping = {
            'graph_conn': 'graph_connectivity',
            'ASW_label/batch': 'silhouette_batch',
            'NMI_cluster/label': 'leiden_NMI',
            'ARI_cluster/label': 'leiden_ARI',
            'ASW_label': 'silhouette_label',
            'kBET':'kBET',
            'clisi_score_fast':'LISI_label',
            'ilisi_score_fast':'LISI_batch',
        }

    cluster_key = 'Metadata_Cluster'
    label_key = 'Metadata_JCP2022'
    batch_key = 'Metadata_Source'
    # batch_key = 'Metadata_Label'
    Perturbation_Gene_key = 'Metadata_JCP2022'
    negative_control = 'JCP2022_033924'
    sample_method = 'random'

    logger.info("=" * 60)
    logger.info(f"Starting metrics calculation for {len(data_path_list)} batch correction methods")
    logger.info("=" * 60)

    for adata_path in tqdm(data_path_list):
        try:
            method_name = adata_path.parts[-1].split('_')[0]
            method_embs = data_mapping_dict[method_name]
            logger.info(f"Processing method: {method_name} (embedding: {method_embs})")
            adata = sc.read_h5ad(adata_path)
            if method_embs == 'X':
                adata.obsm['X_embed'] = adata.X
            else:
                adata.obsm['X_embed'] = adata.obsm[method_embs]

            scib_dict = calculate_batch_bio_metrics(adata,
                                        batch_key,
                                        label_key,
                                        cluster_key )
            clean_scib_dict ={}
            for key, value in scib_dict.items():
                if key in metric_mapping.keys():
                    clean_scib_dict[metric_mapping[key]] = value

            df = h5ad_2_df(adata = adata,
                        embs = method_embs)

            logger.info(f"Converting h5ad to DataFrame: {df.shape}")

            with_control_mAP_results_df, with_other_mAP_results_df = calculate_mAP(df,
                        Perturbation_Gene_key =Perturbation_Gene_key,
                        negative_control = negative_control,
                        sample_method=sample_method,
                        )

            with_control_mAP_mean = with_control_mAP_results_df['mean_average_precision'].mean()
            with_control_mAP_p_value_mean = with_control_mAP_results_df[with_control_mAP_results_df['corrected_p_value']< 0.05]['mean_average_precision'].mean()
            with_control_hit_gene_number = with_control_mAP_results_df[with_control_mAP_results_df['corrected_p_value']< 0.05].shape[0]

            with_other_mAP_mean = with_other_mAP_results_df['mean_average_precision'].mean()
            with_other_mAP_p_value_mean = with_other_mAP_results_df[with_other_mAP_results_df['corrected_p_value']< 0.05]['mean_average_precision'].mean()
            with_other_hit_gene_number = with_other_mAP_results_df[with_other_mAP_results_df['corrected_p_value']< 0.05].shape[0]

            clean_scib_dict['with_control_mAP_mean'] = with_control_mAP_mean
            clean_scib_dict['with_control_mAP_p_value_mean'] = with_control_mAP_p_value_mean
            clean_scib_dict['with_control_hit_gene_number'] = with_control_hit_gene_number

            clean_scib_dict['with_other_mAP_mean'] = with_other_mAP_mean
            clean_scib_dict['with_other_mAP_p_value_mean'] = with_other_mAP_p_value_mean
            clean_scib_dict['with_other_hit_gene_number'] = with_other_hit_gene_number

            logger.info(f"Method {method_name} — "
                        f"with_control mAP mean: {with_control_mAP_mean:.4f}, "
                        f"hits: {with_control_hit_gene_number} | "
                        f"with_other mAP mean: {with_other_mAP_mean:.4f}, "
                        f"hits: {with_other_hit_gene_number}")

            result_save_path = os.path.join(save_path, method_name+'_scib_metric_result.pkl')

            with open(result_save_path, 'wb') as f:
                pickle.dump(clean_scib_dict, f)

            logger.info(f"Results saved to {result_save_path}")
        except Exception as e:
                logger.error(f"Error processing {adata_path}: {e}", exc_info=True)
                continue


    logger.info("=" * 60)
    logger.info("All batch correction methods processed successfully!")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
