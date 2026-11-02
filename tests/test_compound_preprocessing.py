"""Regression checks for shared gene/compound perturbation preprocessing."""

import importlib.util
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from datasets import load_from_disk

SCRIPT = Path(__file__).resolve().parents[1] / "tutorials/06_virtual_cell/unseen_gene_generation/Compound/01_process_data.py"
SPEC = importlib.util.spec_from_file_location("compound_preprocessing", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
from descort.data.dataset import DeSCORTPerturbPredictionProcessor, PERTURBATION_FEATURES
GENE_SCRIPT = SCRIPT.parent.parent / "Crispr/01_process_data.py"
GENE_SPEC = importlib.util.spec_from_file_location("gene_preprocessing", GENE_SCRIPT)
GENE_MODULE = importlib.util.module_from_spec(GENE_SPEC)
GENE_SPEC.loader.exec_module(GENE_MODULE)


class CompoundPreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "input"
        self.source.mkdir()
        self.mapping = {"B": 8, "A": 4}
        self.vocab = {"Cells_X": 12, "Cells_Y": 3}
        for name, value in [("mapping", self.mapping), ("vocab", self.vocab),
                            ("features", ["Cells_X", "Cells_Y"])]:
            with open(self.root / f"{name}.pkl", "wb") as file:
                pickle.dump(value, file)

    def write_plate(self, folder, control="vehicle"):
        directory = self.source / folder
        directory.mkdir()
        path = directory / "plate.parquet"
        pd.DataFrame({
            "Metadata_Smiles": [control, "B", "A", "B", None, "unknown"],
            # Intentionally use a different order than the requested vocabulary.
            "Cells_Y": [90., 1., 2., 3., 4., 5.],
            "Cells_unused": [999.] * 6,
            "Cells_X": [np.nan, 6., 7., np.nan, 8., 9.],
        }).to_parquet(path)
        return path

    def cli(self, destination, workers):
        return ["--source_data_paths", str(self.source),
                "--train_output_dir", str(self.root / "tmp"),
                "--train_save_dir", str(self.root / destination),
                "--gene2index_pkl", str(self.root / "mapping.pkl"),
                "--feature2index_pkl", str(self.root / "vocab.pkl"),
                "--need_feature_list_pkl", str(self.root / "features.pkl"),
                "--control_key_word", "vehicle", "--nproc", str(workers), "--verbosity", "0"]

    def test_schema_order_mapping_and_control_pairing(self):
        for control in ("vehicle", "nontargeting", "CS(=O)C"):
            with self.subTest(control=control):
                path = self.write_plate(str(len(list(self.source.iterdir()))), control)
                processor = DeSCORTPerturbPredictionProcessor(control_key_word=control)
                ds = processor.process_dataset(path, "Metadata_Smiles", self.mapping,
                                               self.vocab, need_feature_list=set(self.vocab))
                self.assertEqual(ds.features, PERTURBATION_FEATURES)
                self.assertEqual(ds.format["type"], "torch")
                rows = ds.with_format(None)[:]
                self.assertEqual(rows["Perturb_Gene"], ["B", "B", "A"])
                self.assertEqual(rows["Perturb_Gene_Index"], [8, 8, 4])
                self.assertEqual(rows["Feature_name_index"], [[3, 12]] * 3)
                self.assertEqual(rows["Perturb_expression"], [[1., 6.], [3., 0.], [2., 7.]])
                self.assertEqual(rows["Control_expression"], [[90., 0.]] * 3)

    def test_serial_parallel_and_filtered_runs(self):
        self.write_plate("first")
        self.write_plate("second")
        for destination, workers in [("serial", 1), ("parallel", 2)]:
            with patch.object(sys, "argv", [str(SCRIPT), *self.cli(destination, workers),
                                            "--train_data_names", ""]):
                MODULE.main()
        serial = load_from_disk(str(self.root / "serial"))
        parallel = load_from_disk(str(self.root / "parallel"))
        self.assertEqual(len(serial), 6)
        self.assertEqual(serial.with_format(None)[:], parallel.with_format(None)[:])
        with patch.object(sys, "argv", [str(SCRIPT), *self.cli("subset", 2),
                                        "--train_data_names", "first"]):
            MODULE.main()
        self.assertEqual(len(load_from_disk(str(self.root / "subset"))), 3)

    def test_missing_control_and_features_fail(self):
        path = self.write_plate("first")
        with self.assertRaisesRegex(ValueError, "no controls"):
            DeSCORTPerturbPredictionProcessor(control_key_word="absent").process_dataset(
                path, "Metadata_Smiles", self.mapping, self.vocab, need_feature_list=set(self.vocab))
        with self.assertRaisesRegex(ValueError, "no features"):
            DeSCORTPerturbPredictionProcessor(control_key_word="vehicle").process_dataset(
                path, "Metadata_Smiles", self.mapping, self.vocab, need_feature_list=set())
        with self.assertRaisesRegex(ValueError, "missing column"):
            DeSCORTPerturbPredictionProcessor(control_key_word="vehicle").process_dataset(
                path, "Metadata_Absent", self.mapping, self.vocab, need_feature_list=set(self.vocab))

    def test_gene_and_compound_entry_points_share_outputs(self):
        self.assertIs(MODULE.DeSCORTPerturbPredictionProcessor, GENE_MODULE.DeSCORTPerturbPredictionProcessor)
        self.write_plate("first")
        self.write_plate("second")
        with patch.object(sys, "argv", [str(SCRIPT), *self.cli("compound", 2)]):
            MODULE.main()
        for name, value in [("esm", self.mapping), ("train_genes", ["B", "A"]), ("test_genes", ["A"])]:
            with open(self.root / f"{name}.pkl", "wb") as file:
                pickle.dump(value, file)
        gene_args = self.cli("gene_train", 2) + [
            "--test_output_dir", str(self.root / "test_tmp"),
            "--test_save_dir", str(self.root / "gene_test"),
            "--esm2_dict_pkl", str(self.root / "esm.pkl"),
            "--train_gene_list_pkl", str(self.root / "train_genes.pkl"),
            "--test_gene_list_pkl", str(self.root / "test_genes.pkl"),
            "--key_gene_column", "Metadata_Smiles",
        ]
        with patch.object(sys, "argv", [str(GENE_SCRIPT), *gene_args]):
            GENE_MODULE.main()
        compound = load_from_disk(str(self.root / "compound")).with_format(None)
        gene = load_from_disk(str(self.root / "gene_train")).with_format(None)
        self.assertEqual(gene[:], compound[:])
        test = load_from_disk(str(self.root / "gene_test")).with_format(None)
        self.assertEqual(test['Perturb_Gene'], ['A', 'A'])

    def test_gene_order_and_sampling_match_legacy_across_read_batches(self):
        path = self.source / "plate.parquet"
        frame = pd.DataFrame({
            "Metadata_Gene": ["control", "B", "A", "B", "control", "A", "B", "unselected"],
            "Cells_Y": [100., 1., 2., 3., 200., 4., 5., 999.],
            "Cells_X": [10., 6., 7., 8., 20., 9., np.nan, 999.],
        })
        frame.to_parquet(path)
        processor = DeSCORTPerturbPredictionProcessor(control_key_word="control", read_batch_size=2)
        for order in (["A", "B"], ["B", "A"], ["control", "A", "B"]):
            with self.subTest(order=order):
                ds = processor.process_dataset(path, "Metadata_Gene", self.mapping, self.vocab,
                                               need_genes_list=order, need_feature_list=set(self.vocab))
                rng = np.random.default_rng(42)
                control = frame.loc[frame.Metadata_Gene.eq("control")]
                expected_labels, expected_values, expected_controls = [], [], []
                for gene in order:
                    if gene == "control":
                        continue
                    target = frame.loc[frame.Metadata_Gene.eq(gene)]
                    sampled = control.sample(n=len(target), replace=len(target) > len(control), random_state=rng)
                    expected_labels.extend([gene] * len(target))
                    expected_values.extend(target[["Cells_Y", "Cells_X"]].fillna(0).values.tolist())
                    expected_controls.extend(sampled[["Cells_Y", "Cells_X"]].values.tolist())
                actual = ds.with_format(None)[:]
                self.assertEqual(actual['Perturb_Gene'], expected_labels)
                self.assertEqual(actual['Perturb_expression'], expected_values)
                self.assertEqual(actual['Control_expression'], expected_controls)
                groups, _, retained_rows = processor.process_single_parquet(
                    path, "Metadata_Gene", order, self.vocab, set(self.vocab))
                self.assertEqual(retained_rows, 7)
                np.testing.assert_array_equal(np.concatenate([g['Control_expression'] for g in groups]), expected_controls)

    def test_unselected_file_is_skipped_but_invalid_file_aborts_merge(self):
        self.write_plate("good")
        # No selected targets: no control is needed and no output shard is made.
        pd.DataFrame({'Metadata_Smiles': ['unselected'], 'Cells_X': [1.]}).to_parquet(self.source / 'empty.parquet')
        processor = DeSCORTPerturbPredictionProcessor(control_key_word='vehicle')
        self.assertIsNone(processor.process_dataset(self.source / 'empty.parquet', 'Metadata_Smiles',
                                                   self.mapping, self.vocab))
        pd.DataFrame({'Metadata_Smiles': ['A'], 'Cells_X': [1.]}).to_parquet(self.source / 'invalid.parquet')
        with patch.object(sys, 'argv', [str(SCRIPT), *self.cli('invalid_output', 2)]):
            with self.assertRaisesRegex(ValueError, 'no controls'):
                MODULE.main()
        self.assertFalse((self.root / 'invalid_output').exists())


if __name__ == "__main__":
    unittest.main()
