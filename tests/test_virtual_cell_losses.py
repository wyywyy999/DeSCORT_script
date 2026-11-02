import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import numpy as np
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from descort.model.perturbation_loss import perturbation_loss
from descort.model.configuration_descort import DeSCORTConfig
from descort.model.modeling_descort import MultiHeadAttention, DeSCORTForPerturbPrediction
from descort.data.sampler import (GeneGroupedBatchSampler,
    DistributedGeneGroupedBatchSampler, BalancedMultiGeneBatchSampler)
from descort.data.collator import PerturbPredictionDataCollator
from descort.vc.population_adapter import KernelMeanHead, fit_residual_quantiles, calibrate_population


class VirtualCellLossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(21)
        self.p = torch.randn(12, 7, requires_grad=True)
        self.t, self.c = torch.randn(12, 7), torch.randn(12, 7)
        self.g = torch.arange(3).repeat_interleave(4)
        self.cfg = SimpleNamespace(mean_weight=1., direction_weight=1.,
            marginal_quantile_weight=.1, centered_quantile_weight=.3,
            variance_weight=.1, sliced_wasserstein_weight=.1,
            sliced_wasserstein_projection_seed=42, sliced_wasserstein_num_projections=8,
            gene_contrastive_weight=.1, gene_contrastive_temperature=.2)

    def loss(self, p=None, t=None, c=None, g=None, mask=None):
        return perturbation_loss(self.p if p is None else p,
            self.t if t is None else t, self.c if c is None else c,
            self.g if g is None else g, mask, self.cfg)[0]

    def test_independent_permutation_within_gene(self):
        perm = torch.tensor([2, 0, 3, 1, 7, 4, 5, 6, 11, 9, 8, 10])
        torch.testing.assert_close(self.loss(), self.loss(t=self.t[perm]))

    def test_padding_and_finite_gradient(self):
        mask = torch.ones_like(self.p, dtype=torch.bool)
        mask[::2, -1] = False
        p = self.p.detach().clone().requires_grad_()
        a = self.loss(p=p, mask=mask)
        b = self.loss(p=p.masked_fill(~mask, 1e5), t=self.t.masked_fill(~mask, -1e5),
                      c=self.c.masked_fill(~mask, 1e5), mask=mask)
        torch.testing.assert_close(a, b)
        a.backward()
        self.assertTrue(torch.isfinite(p.grad).all())
        self.assertEqual(float(p.grad[~mask].abs().sum()), 0.)

    def test_grouped_mean_does_not_cancel_opposite_errors(self):
        cfg = SimpleNamespace(mean_weight=1.)
        p = torch.tensor([[2.], [2.], [-2.], [-2.]])
        loss, _ = perturbation_loss(p, torch.zeros_like(p), torch.zeros_like(p),
                                   torch.tensor([0, 0, 1, 1]), None, cfg)
        self.assertAlmostEqual(float(loss), 4.)

    def test_legacy_single_gene(self):
        self.cfg.centered_quantile_weight = self.cfg.variance_weight = 0.
        self.cfg.sliced_wasserstein_weight = self.cfg.gene_contrastive_weight = 0.
        pm, tm, cm = self.p.mean(0), self.t.mean(0), self.c.mean(0)
        expected = F.mse_loss(pm, tm) + 1 - F.cosine_similarity((pm-cm)[None], (tm-cm)[None]).mean()
        expected += .1 * F.smooth_l1_loss(self.p.sort(0).values, self.t.sort(0).values)
        torch.testing.assert_close(self.loss(g=torch.zeros_like(self.g)), expected)

    def test_null_direction_and_empty_mask(self):
        zero = torch.zeros_like(self.p)
        self.assertEqual(float(self.loss(p=zero, t=zero, c=zero)), 0.)
        self.assertEqual(float(self.loss(mask=torch.zeros_like(self.p, dtype=torch.bool))), 0.)

    def test_sdpa_matches_manual(self):
        cfg = DeSCORTConfig(hidden_size=16, num_attention_heads=4)
        layer = MultiHeadAttention(cfg).eval()
        x = torch.randn(2, 5, 16)
        mask = torch.zeros(2, 1, 1, 5)
        mask[..., -1] = -1e4
        manual = layer(x, mask)[0]
        cfg.use_sdpa = True
        torch.testing.assert_close(layer(x, mask)[0], manual, atol=2e-6, rtol=2e-5)

    def test_samplers_length_and_equal_exposure(self):
        labels = ['a'] * 3 + ['b'] * 5
        for sampler in [GeneGroupedBatchSampler(labels, 4, drop_last=True),
                        DistributedGeneGroupedBatchSampler(labels, 4, drop_last=True)]:
            batches = list(sampler)
            self.assertEqual(len(batches), len(sampler))
            self.assertTrue(all(len(b) == 4 for b in batches))
        s = BalancedMultiGeneBatchSampler(labels, 2, 3, rounds=2)
        self.assertEqual(list(s), list(s))
        observed = [labels[i] for batch in s for i in batch]
        self.assertEqual(observed.count('a'), 6)
        self.assertEqual(observed.count('b'), 6)

    def test_feature_sampling_alignment_and_no_maximum(self):
        rows = [dict(Perturb_expression=torch.arange(20).float(),
                     Control_expression=torch.arange(20).float() + 1000,
                     Feature_name_index=torch.arange(1, 21),
                     Perturb_Gene_Index=torch.tensor(i)) for i in range(3)]
        for maximum in [None, 8, 25]:
            batch = PerturbPredictionDataCollator(max_length=maximum, sampling=True)(rows)
            m = batch['Attention_mask'].bool()
            torch.testing.assert_close(batch['Control_Expr'][m] - batch['Perturb_Expr'][m],
                                       torch.full_like(batch['Control_Expr'][m], 1000.))
            self.assertTrue(torch.equal(batch['Feature_Name_Index'][0], batch['Feature_Name_Index'][1]))

    def test_mean_head_serialization(self):
        rng = np.random.default_rng(42)
        x, y = rng.normal(size=(9, 12)), rng.normal(size=(9, 5))
        head = KernelMeanHead().fit(x, y, 1.)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'head.npz'
            head.save(path)
            np.testing.assert_allclose(head.predict(x), KernelMeanHead.load(path).predict(x))

    def test_noise_aware_head_shrinks_noisy_features(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=(12, 7))
        signal = x[:, 0]
        y = np.stack([signal, signal], axis=1)
        head = KernelMeanHead().fit_noise_aware(x, y, np.array([.001, 100.]), .1)
        prediction = head.predict(x)
        self.assertGreater(prediction[:, 0].var(), prediction[:, 1].var() * 10)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'noise.npz'
            head.save(path)
            np.testing.assert_allclose(prediction, KernelMeanHead.load(path).predict(x))

    def test_population_calibration_preserves_mean_and_permutation(self):
        rng = np.random.default_rng(42)
        train = rng.normal(size=(40, 5))
        labels = np.repeat(['a', 'b'], 20)
        levels, quantiles = fit_residual_quantiles(train, labels)
        prediction = rng.normal(size=(40, 5))
        mu = {'a': np.arange(5), 'b': -np.arange(5)}
        out = calibrate_population(prediction, labels, mu, levels, quantiles)
        for gene in mu:
            np.testing.assert_allclose(out[labels == gene].mean(0), mu[gene], atol=1e-6)
        order = rng.permutation(40)
        shuffled = calibrate_population(prediction[order], labels[order], mu, levels, quantiles)
        np.testing.assert_allclose(shuffled, out[order], atol=1e-6)

    def test_zero_weight_terms_are_inactive(self):
        # Strong distribution differences must not change a mean-only objective.
        cfg = SimpleNamespace(mean_weight=1.)
        loss, parts = perturbation_loss(self.p, self.t, self.c, self.g, None, cfg)
        self.assertAlmostEqual(float(loss), parts['mean_mse'])

    def test_complete_model_backward_and_roundtrip(self):
        cfg = DeSCORTConfig(hidden_size=8, num_attention_heads=2, n_layers=1,
            intermediate_size=16, n_routed_experts=None, feature_name_vocab_size=10,
            max_seq_len=5, use_sdpa=True, hidden_dropout=0., centered_quantile_weight=.1,
            gene_contrastive_weight=.1)
        esm = {i: torch.randn(5120) for i in range(2)}
        model = DeSCORTForPerturbPrediction(cfg, esm)
        batch = dict(Perturb_Gene_Index=torch.tensor([0, 0, 1, 1]),
                     Feature_Name_Index=torch.tensor([[1, 2, 3, 4, 0]]).repeat(4, 1),
                     Control_Expr=torch.randn(4, 5), Perturb_Expr=torch.randn(4, 5),
                     Attention_mask=torch.tensor([[1, 1, 1, 1, 0]]).repeat(4, 1))
        loss = model(**batch).loss
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        model.eval()
        before = model.inference(**batch).logits
        with tempfile.TemporaryDirectory() as d:
            model.save_pretrained(d)
            restored = DeSCORTForPerturbPrediction.from_pretrained(d, INDEX2ESM2_Dict=esm).eval()
            torch.testing.assert_close(restored.inference(**batch).logits, before)
        with self.assertRaisesRegex(KeyError, 'Missing ESM2'):
            model.DeSCORT.get_embeddings(torch.tensor([0, 999]))

    def test_misaligned_features_rejected(self):
        ids = torch.arange(1, 8).repeat(12, 1)
        ids[0, 0] = 99
        with self.assertRaisesRegex(ValueError, 'aligned features'):
            perturbation_loss(self.p, self.t, self.c, self.g, None, self.cfg, ids)


if __name__ == '__main__':
    unittest.main()
