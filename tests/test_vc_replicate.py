"""Checks for the PDF's independent-control and full-reference-gate rules."""
from pathlib import Path
import sys
import unittest

import anndata as ad
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from descort.vc.inference import PerturbationAnndataPair
from descort.vc.replicate import (
    full_gate_replicate_lfc, independent_control_pds, split_half_indices,
)


class ReplicateProtocolTests(unittest.TestCase):
    def test_odd_groups_are_disjoint_complete_and_reproducible(self):
        labels = np.array(['A']*7+['B']*8+['ctrl']*11)
        left, right = split_half_indices(labels, 42)
        self.assertFalse(set(left) & set(right))
        np.testing.assert_array_equal(np.sort(np.r_[left, right]), np.arange(len(labels)))
        np.testing.assert_array_equal(left, split_half_indices(labels, 42)[0])
        for label in set(labels):
            self.assertLessEqual(abs(np.sum(labels[left] == label)-np.sum(labels[right] == label)), 1)
        with self.assertRaisesRegex(ValueError, 'at least two'):
            split_half_indices(['singleton', 'ctrl', 'ctrl'], 42)

    def test_pds_uses_each_halfs_control_mean(self):
        obs = pd.DataFrame({'condition': ['A', 'B', 'C', 'ctrl']},
                           index=['a', 'b', 'c', 'control'])
        values = np.vstack([np.eye(3)*3, np.zeros(3)]).astype('float32')
        real = ad.AnnData(values, obs=obs.copy())
        pred = ad.AnnData(values+500, obs=obs.copy())
        pair = PerturbationAnndataPair(real=real, pred=pred, pert_col='condition', control_pert='ctrl')
        self.assertEqual(set(independent_control_pds(pair).values()), {1.})

    def test_lfc_uses_full_gate_even_if_halves_are_not_significant(self):
        full = pd.DataFrame({'target': ['A']*10, 'feature': [f'f{i}' for i in range(10)],
                             'fdr': [.01]*10, 'signed_effect': [2.]*10})
        left, right = full.copy(), full.copy()
        left['signed_effect'], right['signed_effect'] = 1., 2.
        left['fdr'], right['fdr'] = 1., 1.
        scores, gates = full_gate_replicate_lfc(full, left, right, ['A'])
        self.assertEqual(scores['A'], .5)
        self.assertTrue(gates.scored.iloc[0])
        scores, gates = full_gate_replicate_lfc(full.iloc[:9], left, right, ['A'])
        self.assertTrue(np.isnan(scores['A']))
        self.assertFalse(gates.scored.iloc[0])


if __name__ == '__main__':
    unittest.main()
