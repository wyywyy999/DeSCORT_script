"""Run the existing seven batch-correction adapters on an AnnData file."""
from pathlib import Path
import sys
import argparse
import os


_SOURCE = Path(__file__).resolve().parents[2] / "src"
if str(_SOURCE) not in sys.path:
    sys.path.insert(0, str(_SOURCE))

import anndata as ad
from descort.batch.batch_func import (
        Sphering_func, bbknn_func, Combat_func, Harmony_func,
        Scanorama_func, Mnn_func, Desc_func,
    )
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/other_model/DeSCORT/outputs/05_batch_effect/Target_2_Compoud/batch_adata/MAD_adata.h5ad")
    parser.add_argument("--save-path", default="/fse/home/wangyongyan/Project/CellPainting/DeSCORT/other_model/DeSCORT/outputs/05_batch_effect/Target_2_Compoud/batch_adata/")
    parser.add_argument("--batch-key", default="Metadata_Source")
    args = parser.parse_args()
    
    for key in ["OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"]:
        os.environ[key] = "64"
    data_path, save_path, batch_key = args.data_path, args.save_path, args.batch_key
    os.makedirs(save_path, exist_ok=True)

    print(f"Reading original data: {data_path}")
    original_data = ad.read_h5ad(data_path)

    # Ensure obs_names are unique
    if not original_data.obs_names.is_unique:
        original_data.obs_names_make_unique()

    print(f"Data shape: {original_data.shape}")
    print(f"Batch categories: {original_data.obs[batch_key].unique().tolist()}")
    print("=" * 60)

    # =============================================
    # 1. Sphering
    # =============================================
    print("\n========== 1/7: Sphering ==========")
    try:
        sp_adata = original_data.copy()
        Sphering_func(sp_adata, column_norm=batch_key, save_path=save_path,
                      method='zca', epsilon=1e-6)
        print("  ✓ Sphering completed")
    except Exception as e:
        print(f"  ✗ Sphering failed: {e}")

    # # =============================================
    # # 2. BBKNN
    # # =============================================
    print("\n========== 2/7: BBKNN ==========")
    try:
        bb_adata = original_data.copy()
        bbknn_func(bb_adata, save_path=save_path, batch_key=batch_key)
        print("  ✓ BBKNN completed")
    except Exception as e:
        print(f"  ✗ BBKNN failed: {e}")

    # =============================================
    # 3. ComBat
    # =============================================
    print("\n========== 3/7: ComBat ==========")
    try:
        cb_adata = original_data.copy()
        Combat_func(cb_adata, batch_key=batch_key, save_path=save_path)
        print("  ✓ ComBat completed")
    except Exception as e:
        print(f"  ✗ ComBat failed: {e}")

    # =============================================
    # 4. Harmony
    # =============================================
    print("\n========== 4/7: Harmony ==========")
    try:
        hm_adata = original_data.copy()
        Harmony_func(hm_adata, batch_key=batch_key, save_path=save_path)
        print("  ✓ Harmony completed")
    except Exception as e:
        print(f"  ✗ Harmony failed: {e}")

    # =============================================
    # 5. Scanorama
    # =============================================
    print("\n========== 5/7: Scanorama ==========")
    try:
        sc_adata = original_data.copy()
        Scanorama_func(sc_adata, batch_key=batch_key, save_path=save_path)
        print("  ✓ Scanorama completed")
    except Exception as e:
        print(f"  ✗ Scanorama failed: {e}")

    # =============================================
    # 6. MNN
    # =============================================
    print("\n========== 6/7: MNN ==========")
    try:
        mn_adata = original_data.copy()
        Mnn_func(mn_adata, batch_key=batch_key, save_path=save_path)
        print("  ✓ MNN completed")
    except Exception as e:
        print(f"  ✗ MNN failed: {e}")

    # =============================================
    # 7. DESC
    # =============================================
    print("\n========== 7/7: DESC ==========")
    try:
        dc_adata = original_data.copy()
        Desc_func(dc_adata, save_path=save_path)
        print("  ✓ DESC completed")
    except Exception as e:
        print(f"  ✗ DESC failed: {e}")

    print("\n" + "=" * 60)
    print("All batch correction methods have finished!")


if __name__ == "__main__":
    main()
