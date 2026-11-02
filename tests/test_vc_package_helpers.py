"""Behavioral checks for the shared virtual-cell tutorial helpers."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import anndata as ad
import numpy as np
import torch
from datasets import Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from descort.data.collator import PerturbPredictionDataCollator
from descort.vc.finetune import infer_feature_order
from descort.vc import inference_compound, inference_crispr
from descort.vc.metric_cli import inspect_input, parse_args, parse_crispr_args
from descort.vc.preprocessing_crispr import filter_genes_by_esm2
from descort.vc.fast_cp_metrics import validate_mwu


class VirtualCellHelperTests(unittest.TestCase):
    def test_gene_filter_preserves_supplied_order_duplicates_and_control(self):
        self.assertEqual(filter_genes_by_esm2(
            ['B', 'A', 'B', 'missing', 'nontargeting'], {'A': 1, 'B': 2}),
            ['B', 'A', 'B'])

    def test_feature_intersection_rejects_duplicate_and_unknown_ids(self):
        data = Dataset.from_dict({'Feature_name_index': [[3, 1], [2, 3], [3, 1]]})
        self.assertEqual(infer_feature_order(data, 8), [3])
        for axis, message in [([1, 1], 'duplicate'), ([1, 9], 'invalid')]:
            with self.assertRaisesRegex(ValueError, message):
                infer_feature_order(Dataset.from_dict({'Feature_name_index': [axis]}), 8)

    def test_compound_prediction_keeps_saved_axis_and_excludes_target_values(self):
        config = SimpleNamespace(population_feature_alignment='intersection',
                                 population_feature_ids=[1, 3], max_seq_len=3)
        collator, axis = inference_compound.build_collator(config)
        rows = [dict(Feature_name_index=torch.tensor([3, 2, 1]),
                     Perturb_expression=torch.tensor([30., 20., 10.]),
                     Control_expression=torch.tensor([3., 2., 1.]),
                     Perturb_Gene_Index=torch.tensor(1))]

        class Model:
            def inference(self, **inputs):
                assert 'Perturb_Expr' not in inputs
                return SimpleNamespace(logits=inputs['Control_Expr'][:, :2]+1)

        _, truth, prediction, control = inference_compound.collect_test_outputs(
            Model(), [collator(rows)], torch.device('cpu'), axis)
        np.testing.assert_array_equal(truth, [[10., 30.]])
        np.testing.assert_array_equal(control, [[1., 3.]])
        np.testing.assert_array_equal(prediction, [[2., 4.]])

    def test_crispr_pm_weights_genes_equally_and_removes_padding(self):
        collator = PerturbPredictionDataCollator(max_length=3, sampling=False, pad_value=100)
        def row(gene, values):
            return dict(Feature_name_index=torch.tensor([1, 3]),
                        Perturb_expression=torch.tensor(values),
                        Control_expression=torch.tensor([0., 0.]),
                        Perturb_Gene_Index=torch.tensor(gene))
        batches = [collator([row(1, [2., 4.]), row(1, [4., 6.])]),
                   collator([row(2, [9., 11.])])]
        np.testing.assert_array_equal(inference_crispr.calculate_train_mean(batches), [6., 8.])

    def test_metric_defaults_distinguish_genes_and_compounds(self):
        argv = ['metrics', '--ground_truth_path', 'truth.h5ad', '--prediction_path',
                'pred.h5ad', '--outdir', 'results', '--cache_dir', 'cache']
        with patch.object(sys, 'argv', argv):
            compound, crispr = parse_args(), parse_crispr_args()
        self.assertEqual((compound.pert_col, compound.control_name), ('Metadata_Smiles', 'CS(=O)C'))
        self.assertEqual((crispr.pert_col, crispr.control_name), ('Perturb_Gene', 'nontargeting'))

    def test_metadata_inspection_retains_exact_axis_and_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'data.h5ad'
            data = ad.AnnData(np.array([[1., 2.], [0., 0.]], dtype=np.float32))
            data.obs['Perturb_Gene'] = ['A', 'nontargeting']
            data.var_names = ['Cells_Y', 'Cells_X']
            data.var['feature_id'] = [3, 1]
            data.write_h5ad(path)
            _, _, labels, names, ids = inspect_input(path, None, 'Perturb_Gene', 'nontargeting')
            self.assertEqual(names.tolist(), ['Cells_Y', 'Cells_X'])
            np.testing.assert_array_equal(ids, [3, 1])
            self.assertEqual(labels.tolist(), ['A', 'nontargeting'])

    def test_fast_mwu_matches_independent_scipy_validation(self):
        result = validate_mwu()
        self.assertGreaterEqual(result['checks'], 20)
        self.assertLessEqual(result['max_p_absolute_error'], 1e-12)


if __name__ == '__main__':
    unittest.main()
