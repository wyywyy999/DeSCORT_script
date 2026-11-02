"""Feature alignment regressions for compound population-loss training."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import torch
from datasets import Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from descort.data.collator import PerturbPredictionDataCollator
from descort.model.perturbation_loss import perturbation_loss

SPEC = importlib.util.spec_from_file_location('compound_finetune', ROOT / 'tutorials/06_virtual_cell/unseen_gene_generation/Compound/02_finetune.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CompoundFeatureAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.examples = [
            {'Feature_name_index': torch.tensor([3, 1]),
             'Perturb_expression': torch.tensor([30., 10.]),
             'Control_expression': torch.tensor([3., 1.]),
             'Perturb_Gene_Index': torch.tensor(5)},
            {'Feature_name_index': torch.tensor([2, 3]),
             'Perturb_expression': torch.tensor([20., 300.]),
             'Control_expression': torch.tensor([2., 30.]),
             'Perturb_Gene_Index': torch.tensor(5)},
        ]

    def test_missing_columns_are_aligned_before_truncation(self):
        collator = PerturbPredictionDataCollator(max_length=3, feature_order=[1, 2, 3])
        batch = collator(self.examples)
        torch.testing.assert_close(batch['Feature_Name_Index'], torch.tensor([[1, 0, 3], [0, 2, 3]]))
        torch.testing.assert_close(batch['Perturb_Expr'], torch.tensor([[10., 100., 30.], [100., 20., 300.]]))
        torch.testing.assert_close(batch['Control_Expr'], torch.tensor([[1., 100., 3.], [100., 2., 30.]]))
        self.assertEqual(batch['Attention_mask'].bool().tolist(), [[True, False, True], [False, True, True]])
        short = PerturbPredictionDataCollator(max_length=2, feature_order=[1, 2, 3])(self.examples)
        torch.testing.assert_close(short['Perturb_Expr'], batch['Perturb_Expr'][:, :2])
        self.assertEqual(self.examples[0]['Feature_name_index'].tolist(), [3, 1])

    def test_masked_missing_features_do_not_enter_population_loss(self):
        batch = PerturbPredictionDataCollator(max_length=3, feature_order=[1, 2, 3])(self.examples)
        prediction = batch['Perturb_Expr'].clone()
        prediction[~batch['Attention_mask'].bool()] = 1e6
        prediction.requires_grad_()
        config = SimpleNamespace(mean_weight=1., direction_weight=0., marginal_quantile_weight=0.)
        loss, _ = perturbation_loss(prediction, batch['Perturb_Expr'], batch['Control_Expr'],
                                   batch['Perturb_Gene_Index'], batch['Attention_mask'], config,
                                   batch['Feature_Name_Index'])
        self.assertEqual(loss.item(), 0.)
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_infer_axis_and_reject_duplicate_features(self):
        dataset = Dataset.from_dict({'Feature_name_index': [[3, 1], [2, 3], [3, 1]]})
        self.assertEqual(MODULE.infer_feature_order(dataset, 8), [3])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            MODULE.infer_feature_order(Dataset.from_dict({'Feature_name_index': [[1, 1]]}), 8)

    def test_alignment_does_not_hide_unknown_feature_ids(self):
        collator = PerturbPredictionDataCollator(max_length=3, feature_order=[1, 2])
        with self.assertRaisesRegex(ValueError, 'outside feature_order'):
            collator(self.examples)

    def test_intersection_selects_matching_values_and_pads_only_at_end(self):
        batch = PerturbPredictionDataCollator(
            max_length=2, feature_order=[3], feature_alignment='intersection')(self.examples)
        torch.testing.assert_close(batch['Feature_Name_Index'], torch.tensor([[3, 0], [3, 0]]))
        torch.testing.assert_close(batch['Perturb_Expr'], torch.tensor([[30., 100.], [300., 100.]]))
        torch.testing.assert_close(batch['Control_Expr'], torch.tensor([[3., 100.], [30., 100.]]))
        self.assertEqual(batch['Attention_mask'].bool().tolist(), [[True, False], [True, False]])

    def test_intersection_reorders_before_truncating(self):
        example = dict(self.examples[0], Feature_name_index=torch.tensor([3, 1, 2]),
                       Perturb_expression=torch.tensor([30., 10., 20.]),
                       Control_expression=torch.tensor([3., 1., 2.]))
        batch = PerturbPredictionDataCollator(
            max_length=1, feature_order=[1, 3], feature_alignment='intersection')([example])
        self.assertEqual(batch['Feature_Name_Index'].tolist(), [[1]])
        self.assertEqual(batch['Perturb_Expr'].tolist(), [[10.]])
        self.assertEqual(batch['Control_Expr'].tolist(), [[1.]])

    def test_intersection_rejects_missing_required_features(self):
        collator = PerturbPredictionDataCollator(
            max_length=3, feature_order=[1, 3], feature_alignment='intersection')
        with self.assertRaisesRegex(ValueError, 'missing required intersection'):
            collator(self.examples)

    def test_empty_intersection_is_rejected(self):
        for rows in ([[1], [2]], []):
            with self.assertRaisesRegex(ValueError, 'empty intersection'):
                MODULE.infer_feature_order(Dataset.from_dict({'Feature_name_index': rows}), 8)


if __name__ == '__main__':
    unittest.main()
