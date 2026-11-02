#!/usr/bin/env python3
"""CPU example: synthetic parquet -> small random checkpoint -> embedding parquet."""
import argparse
from pathlib import Path
import pickle
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/minimal_embedding")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("Choose an empty output directory")
    import numpy as np
    import pandas as pd
    import torch
    from descort import DeSCORTConfig, DeSCORTForContrastiveLM

    torch.manual_seed(42)
    torch.set_num_threads(1)
    plate = output / "input/toy_plate.parquet"
    plate.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({
        "Metadata_Gene": ["control", "GENE_A", "GENE_B"],
        "Metadata_Well": ["A01", "A02", "A03"],
        "Cells_Area": [0.0, 0.5, -0.5],
        "Nuclei_Area": [0.2, 0.7, -0.3],
        "Cytoplasm_Intensity": [0.1, 0.3, -0.1],
    })
    frame.to_parquet(plate, index=False)
    features = list(frame.columns[2:])
    for name, value in {
        "labels": {"control": 0, "GENE_A": 1, "GENE_B": 2},
        "vocab": dict(zip(features, [1, 2, 3])), "features": features,
    }.items():
        with (output / f"{name}.pkl").open("wb") as file:
            pickle.dump(value, file)
    config = DeSCORTConfig(
        hidden_size=16, num_attention_heads=2, n_layers=1,
        intermediate_size=32, moe_intermediate_size=16,
        n_routed_experts=2, num_experts_per_tok=1,
        feature_name_vocab_size=4, max_seq_len=4, bin_num=4,
    )
    DeSCORTForContrastiveLM(config).eval().save_pretrained(output / "model")
    command = [sys.executable, str(ROOT / "tutorials/03_finetune/03_calculate_emb.py"),
               "--source_data_paths", str(plate.parent),
               "--gene2index_pkl", str(output / "labels.pkl"),
               "--feature2index_pkl", str(output / "vocab.pkl"),
               "--need_feature_list_pkl", str(output / "features.pkl"),
               "--pretrained_model_path", str(output / "model"),
               "--save_path", str(output / "embeddings"),
               "--key_gene_column", "Metadata_Gene", "--metadata_parts", "-2", "-2",
               "--max_length", "4", "--device", "cpu", "--batch_size", "2"]
    subprocess.run(command, check=True, cwd=output)
    files = list((output / "embeddings").rglob("*.parquet"))
    if len(files) != 1:
        raise AssertionError(f"Expected one embedding parquet, found {files}")
    result = pd.read_parquet(files[0])
    embeddings = result.filter(regex="^emb_")
    assert embeddings.shape == (3, 16), embeddings.shape
    assert np.isfinite(embeddings.to_numpy()).all()
    pd.testing.assert_frame_equal(result[frame.columns[:2]].reset_index(drop=True), frame[frame.columns[:2]])
    print(f"PASS: {embeddings.shape} finite embeddings; metadata preserved. {files[0]}")


if __name__ == "__main__":
    main()
