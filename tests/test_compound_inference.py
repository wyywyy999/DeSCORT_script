"""Regression checks for saved feature axes, compound means and H5AD output."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import sys
import unittest

import anndata as ad
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SPEC = importlib.util.spec_from_file_location('compound_inference', ROOT / 'tutorials/06_virtual_cell/unseen_gene_generation/Compound/03_inference.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CompoundInferenceTests(unittest.TestCase):
    def setUp(self):
        self.config = SimpleNamespace(population_feature_alignment='intersection',
                                      population_feature_ids=[1, 3], max_seq_len=3)
        self.collator, self.axis = MODULE.build_collator(self.config)
        self.rows = [dict(Feature_name_index=torch.tensor([3, 2, 1]),
                         Perturb_expression=torch.tensor([30., 20., 10.]),
                         Control_expression=torch.tensor([3., 2., 1.]),
                         Perturb_Gene_Index=torch.tensor(1)),
                     dict(Feature_name_index=torch.tensor([1, 3]),
                         Perturb_expression=torch.tensor([100., 300.]),
                         Control_expression=torch.tensor([10., 30.]),
                         Perturb_Gene_Index=torch.tensor(2))]

    def test_requires_saved_intersection_and_training_length(self):
        for cfg in (SimpleNamespace(), SimpleNamespace(population_feature_alignment='pad',
                                                       population_feature_ids=[1], max_seq_len=3)):
            with self.assertRaisesRegex(ValueError, 'Checkpoint must contain'):
                MODULE.build_collator(cfg)
        with self.assertRaisesRegex(ValueError, 'must match checkpoint'):
            MODULE.build_collator(self.config, 2)

    def test_prediction_uses_control_only_and_saved_axis(self):
        class Model:
            def inference(self, **inputs):
                assert 'Perturb_Expr' not in inputs
                return SimpleNamespace(logits=inputs['Control_Expr'][:, :2] + 1)
        ids, truth, predictions, controls = MODULE.collect_test_outputs(
            Model(), [self.collator(self.rows)], torch.device('cpu'), self.axis)
        np.testing.assert_array_equal(ids, [1, 2])
        np.testing.assert_array_equal(truth, [[10., 30.], [100., 300.]])
        np.testing.assert_array_equal(controls, [[1., 3.], [10., 30.]])
        np.testing.assert_array_equal(predictions, controls + 1)

    def test_missing_features_and_wrong_batch_order_rejected(self):
        with self.assertRaisesRegex(ValueError, 'missing required intersection'):
            self.collator([dict(self.rows[0], Feature_name_index=torch.tensor([2, 4, 5]))])
        batch = self.collator(self.rows)
        batch['Feature_Name_Index'][1, :2] = torch.tensor([3, 1])
        with self.assertRaisesRegex(ValueError, 'saved training axis'):
            MODULE.validate_batch(batch, self.axis)

    def test_baseline_equal_weights_compounds_not_rows(self):
        batches = [self.collator([self.rows[0], self.rows[0]]), self.collator([self.rows[1]])]
        mean = MODULE.calculate_train_mean(batches, self.axis)
        np.testing.assert_allclose(mean, [55., 165.])

    def test_h5ad_feature_names_controls_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(output_dir=directory, gene_column='Perturb_Gene',
                control_key_word='CS(=O)C', pretrained_model_path=directory,
                train_dataset_path=directory, test_dataset_path=directory,
                index2morgan_pkl=str(Path(directory) / 'embedding.pkl'),
                condition_dim=2048, smiles2index_pkl=str(Path(directory) / 'smiles.pkl'),
                ground_truth_filename='truth.h5ad', prediction_filename='pred.h5ad',
                mean_baseline_filename='mean.h5ad')
            truth = np.array([[10., 30.], [100., 300.]])
            controls = truth / 10
            MODULE.write_outputs(args, self.axis, {1: 'area', 3: 'intensity'},
                {1: 'CCO', 2: 'CC'}, np.array([1, 2]), truth, controls + 1,
                controls, np.array([55., 165.]), True)
            for filename in ('truth.h5ad', 'pred.h5ad', 'mean.h5ad'):
                data = ad.read_h5ad(Path(directory) / filename)
                self.assertEqual(data.shape, (4, 2))
                self.assertEqual(data.var_names.tolist(), ['area', 'intensity'])
                self.assertEqual(data.var.feature_id.tolist(), [1, 3])
                self.assertEqual(data.obs.Metadata_Smiles.tolist(), ['CCO', 'CC', 'CS(=O)C', 'CS(=O)C'])
                self.assertTrue(data.uns['in_sample'])
                self.assertEqual(data.uns['condition_embedding_dim'], 2048)
                self.assertEqual(data.uns['condition_embedding_path'], str(Path(directory) / 'embedding.pkl'))
                np.testing.assert_array_equal(data.X[2:], controls)


if __name__ == '__main__':
    unittest.main()
