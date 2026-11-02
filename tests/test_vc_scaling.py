"""Fixed-anchor scaling, PDF clipping and model-independent scores."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from descort.vc.scaling import SCORE_METRICS, baseline_to_replicate_scores


class BaselineReplicateScalingTests(unittest.TestCase):
    def setUp(self):
        self.baseline = pd.Series([.5, 1., .5, 0., 0., 1.], index=SCORE_METRICS)
        self.replicate = pd.Series([.9, .1, .8, .9, .5, .4], index=SCORE_METRICS)

    def test_baseline_zero_replicate_one_and_six_metric_mean(self):
        data = pd.DataFrame([self.baseline, self.replicate], index=['PM', 'replicate'])
        data['mse'] = [1e9, -1e9]
        scores = baseline_to_replicate_scores(data, self.baseline, self.replicate)
        np.testing.assert_allclose(scores.loc['PM'], 0)
        np.testing.assert_allclose(scores.loc['replicate'], 1)
        self.assertEqual(scores.shape[1], 7)

    def test_only_pdf_defined_clamps_are_applied(self):
        data = pd.DataFrame([[0., 3., 0., 0., 0., 10.],
                             [1., -10., 1., 1., 1., 0.]], columns=SCORE_METRICS)
        scores = baseline_to_replicate_scores(data, self.baseline, self.replicate)
        self.assertLess(scores.iloc[0].pds_cosine, 0)
        self.assertEqual(scores.iloc[0].expr_mse_unbiased_capped_norm, 0)
        self.assertEqual(scores.iloc[0].de_wilcoxon_lfc_nmae, -6)
        self.assertEqual(scores.iloc[1].expr_mse_unbiased_capped_norm, 1)
        self.assertGreater(scores.iloc[1].pds_cosine, 1)
        self.assertGreater(scores.iloc[1].de_wilcoxon_lfc_nmae, 1)

    def test_other_models_do_not_change_a_score(self):
        data = pd.DataFrame([self.baseline, self.replicate])
        both = baseline_to_replicate_scores(data, self.baseline, self.replicate)
        one = baseline_to_replicate_scores(data.iloc[[0]], self.baseline, self.replicate)
        pd.testing.assert_series_equal(both.iloc[0], one.iloc[0])
        with self.assertRaisesRegex(ValueError, 'must differ'):
            baseline_to_replicate_scores(data, self.baseline, self.baseline)


if __name__ == '__main__':
    unittest.main()
