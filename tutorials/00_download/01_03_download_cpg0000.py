import os
import pickle
import pandas as pd
import subprocess
from tqdm import tqdm

def download_data_aws(meta_df, save_path, treatment_type):
    """
    从AWS S3下载Cell Painting数据

    Parameters:
    -----------
    plate_list: list
        板名列表
    save_path: str
        本地保存路径
    treatment_type: str
        处理类型（如compound, crispr等）
    """
    unique_combinations = meta_df[['Metadata_Batch', 'Metadata_Plate']].drop_duplicates()
    for index, row in unique_combinations.iterrows():
        batch = row['Metadata_Batch']
        plate = row['Metadata_Plate']
        print(f"Batch: {batch}, Plate: {plate}")

        save_dir = os.path.join(save_path, treatment_type, batch, plate)
        os.makedirs(save_dir, exist_ok=True)

        # 构建S3路径
        s3_path = f"s3://cellpainting-gallery/cpg0000-jump-pilot/source_4/workspace/profiles/{batch}/{plate}/"

        # 构建aws命令
        cmd = ["aws", "s3", "cp", "--recursive", s3_path, save_dir, "--no-sign-request"]
        print(cmd)

        print(f"\nDownloading {plate}...")
        print(f"  From: {s3_path}")
        print(f"  To: {save_dir}")

        try:
            subprocess.run(cmd, check=True)

        except (subprocess.CalledProcessError, OSError) as e:
            print(f"  ✗ Error downloading {plate}: {str(e)}")


def main():
    """
    Main function: Process Jump Pilot dataset, filter and download data for different cell lines and perturbation types
    """

    # 1. Load metadata
    print("[INFO] Loading metadata...")
    meta_data = pd.read_csv('/fse/DC/CP_data/feature_data/cpg000_jump_pilot/cpg0000-jump-pilot_draft.csv')
    print(f"✓ Metadata loaded successfully: {len(meta_data)} rows, {len(meta_data.columns)} columns")

    # 2. Load ESM2 gene feature dictionary
    print("[INFO] Loading ESM2 gene feature dictionary...")
    with open('/fse/home/wangyongyan/Project/CellPainting/Final_LLLM/common_files/ESM2_pert_features_with_nontargeting.pkl','rb') as f:
        ESM2_gene_dict = pickle.load(f)
    print(f"✓ ESM2 gene feature dictionary loaded: {len(ESM2_gene_dict)} genes")

    # Get all gene names from ESM2 dictionary
    # ESM_gene_list = list(ESM2_gene_dict.keys())
    # print(f"  - Valid gene list: {len(ESM_gene_list)} genes")

    # 3. Filter data by perturbation category
    print("\n[INFO] Filtering data by perturbation category...")
    meta_data.columns = ['Metadata_' + col for col in meta_data.columns]
    # 3.1 Filter CRISPR perturbation data
    crispr_data = meta_data[meta_data['Metadata_Perturbation_Category'] == 'crispr']
    print(f"  - CRISPR perturbation data: {len(crispr_data)} rows")

    # Filter CRISPR data for genes in ESM2 list
    # crispr_data_select_160 = crispr_data[crispr_data['Metadata_Treatment'].isin(ESM_gene_list)]
    # print(f"  - CRISPR data matching ESM2 genes: {len(crispr_data_select_160)} rows")
    # print(f"    (Match rate: {len(crispr_data_select_160)/len(crispr_data)*100:.1f}%)")

    # 3.2 Filter compound perturbation data
    compoud_data = meta_data[meta_data['Metadata_Perturbation_Category'] == 'compound']
    print(f"  - Compound perturbation data: {len(compoud_data)} rows")

    # 3.3 Filter ORF (open reading frame) perturbation data
    orf_data = meta_data[meta_data['Metadata_Perturbation_Category'] == 'orf']
    print(f"  - ORF perturbation data: {len(orf_data)} rows")

    # 4. Further filter by cell line
    print("\n[INFO] Further filtering data by cell line...")

    # 4.1 Filter A549 cell line data
    print("  [A549 cell line]")
    crispr_A549_data = crispr_data[crispr_data['Metadata_Cell_Line'] == 'A549']
    print(f"    - CRISPR data: {len(crispr_A549_data)} rows")

    crispr_U2OS_data = crispr_data[crispr_data['Metadata_Cell_Line'] == 'U2OS']
    print(f"  [U2OS cell line]")
    print(f"    - CRISPR data: {len(crispr_U2OS_data)} rows")

    # 4.2 Separate compound data by cell line
    compoud_A549_data = compoud_data[compoud_data['Metadata_Cell_Line'] == 'A549']
    compoud_U2OS_data = compoud_data[compoud_data['Metadata_Cell_Line'] == 'U2OS']
    print(f"    - Compound data (A549): {len(compoud_A549_data)} rows")
    print(f"    - Compound data (U2OS): {len(compoud_U2OS_data)} rows")

    # 4.3 Separate ORF data by cell line
    orf_A549_data = orf_data[orf_data['Metadata_Cell_Line'] == 'A549']
    orf_U2OS_data = orf_data[orf_data['Metadata_Cell_Line'] == 'U2OS']
    print(f"    - ORF data (A549): {len(orf_A549_data)} rows")
    print(f"    - ORF data (U2OS): {len(orf_U2OS_data)} rows")

    # 5. Create save paths
    print("\n[INFO] Creating save paths...")
    A549_save_path = '/fse/DC/CP_data/feature_data/cpg000_jump_pilot/raw_data_bulk_download/A549'
    U2OS_save_path = '/fse/DC/CP_data/feature_data/cpg000_jump_pilot/raw_data_bulk_download/U2OS'

    # Ensure directories exist
    os.makedirs(A549_save_path, exist_ok=True)
    os.makedirs(U2OS_save_path, exist_ok=True)
    print(f"✓ Save paths created successfully:")
    print(f"  - A549 save path: {A549_save_path}")
    print(f"  - U2OS save path: {U2OS_save_path}")

    # 6. Download data files
    print("\n" + "="*50)
    print("[INFO] Starting to download A549 cell line data...")
    print("="*50)

    # 6.1 Download A549 cell line data
    print("\n[Phase 1] Downloading CRISPR data...")
    # download_data_aws(crispr_A549_data, A549_save_path, 'Crispr')

    print("\n[Phase 1] Downloading CRISPR data...")
    # download_data_aws(crispr_U2OS_data, U2OS_save_path, 'Crispr')

    # print("\n[Phase 2] Downloading compound data...")
    download_data_aws(compoud_A549_data, A549_save_path, 'Compoud')
    # print("\n[Phase 2] Downloading compound data...")
    download_data_aws(compoud_U2OS_data, U2OS_save_path, 'Compoud')

    # print("\n[Phase 3] Downloading ORF data...")
    download_data_aws(orf_A549_data, A549_save_path, 'Orf')
    # print("\n[Phase 3] Downloading ORF data...")
    download_data_aws(orf_U2OS_data, U2OS_save_path, 'Orf')

    print("\n" + "="*50)
    print("[INFO] Starting to download U2OS cell line data...")
    print("="*50)

    # 6.2 Download U2OS cell line data


    print("\n" + "="*50)
    print("[INFO] All data download completed!")
    print(f"Total statistics:")
    print(f"  - A549 cell line: 3 perturbation types, {len(crispr_A549_data)+len(compoud_A549_data)+len(orf_A549_data)} data files")
    print(f"  - U2OS cell line: 3 perturbation types, {len(crispr_U2OS_data)+len(compoud_U2OS_data)+len(orf_U2OS_data)} data files")
    print("="*50)



if __name__ == "__main__":
    main()
