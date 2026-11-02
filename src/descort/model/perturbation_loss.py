"""Gene-balanced losses for unpaired control/perturbation populations.

Never supervise an arbitrary control/target row pairing. Statistics and sorted
quantiles are calculated within each gene, in float32, excluding padding.
"""
import torch
import torch.nn.functional as F


def perturbation_loss(prediction, target, control, genes, mask, config,
                      feature_ids=None):
    pred, target, control = prediction.float(), target.float(), control.float()
    mask = torch.ones_like(pred, dtype=torch.bool) if mask is None else mask.bool()
    zero = pred.masked_fill(~mask, 0).sum() * 0.0
    components = {key: [] for key in (
        'mean_mse', 'direction', 'marginal_quantile', 'centered_quantile',
        'sliced_wasserstein', 'variance')}
    effects_p, effects_t, feature_masks = [], [], []
    if genes is None:
        raise ValueError('Perturb_Gene_Index is required for gene-wise loss')
    genes = genes.reshape(-1)
    if feature_ids is not None:
        # Column-wise population statistics require aligned feature identities.
        hi = feature_ids.masked_fill(~mask, -1).amax(0)
        lo = feature_ids.masked_fill(~mask, torch.iinfo(feature_ids.dtype).max).amin(0)
        if ((hi != lo) & mask.any(0)).any():
            raise ValueError('Population loss requires aligned features; use sampling=False')
    for gene in torch.unique(genes):
        rows = genes == gene
        m = mask[rows]
        count = m.sum(0)
        valid = count > 0
        if not valid.any():
            continue
        p, t, c = pred[rows], target[rows], control[rows]
        count_safe = count.clamp_min(1)
        pm = p.masked_fill(~m, 0).sum(0) / count_safe
        tm = t.masked_fill(~m, 0).sum(0) / count_safe
        cm = c.masked_fill(~m, 0).sum(0) / count_safe
        components['mean_mse'].append(F.mse_loss(pm[valid], tm[valid]))
        pd, td = pm - cm, tm - cm
        # A null effect has no defined direction and must not contribute 1.
        if td[valid].norm() > 1e-8:
            direction = 1 - F.cosine_similarity(pd[valid][None], td[valid][None], eps=1e-8).mean()
        else:
            direction = zero
        components['direction'].append(direction)
        effects_p.append(pd)
        effects_t.append(td)
        feature_masks.append(valid)
        if getattr(config, 'marginal_quantile_weight', 0) or getattr(config, 'centered_quantile_weight', 0):
            # Invalid entries sort to the end independently in each feature.
            ps = p.masked_fill(~m, float('inf')).sort(dim=0).values
            ts = t.masked_fill(~m, float('inf')).sort(dim=0).values
            qm = torch.arange(len(p), device=p.device)[:, None] < count[None]
            if getattr(config, 'marginal_quantile_weight', 0):
                components['marginal_quantile'].append(F.smooth_l1_loss(ps[qm], ts[qm]))
            if getattr(config, 'centered_quantile_weight', 0):
                multi = qm & (count > 1)[None]
                if multi.any():
                    components['centered_quantile'].append(
                        F.smooth_l1_loss((ps - pm)[multi], (ts - tm)[multi]))
        if getattr(config, 'variance_weight', 0):
            multi = count > 1
            if multi.any():
                pv = (p - pm).masked_fill(~m, 0).square().sum(0) / (count - 1).clamp_min(1)
                tv = (t - tm).masked_fill(~m, 0).square().sum(0) / (count - 1).clamp_min(1)
                components['variance'].append(F.smooth_l1_loss(
                    (pv[multi] + 1e-6).sqrt(), (tv[multi] + 1e-6).sqrt()))
        if getattr(config, 'sliced_wasserstein_weight', 0) and len(p) > 1:
            common = m.all(0)
            if common.any():
                gen = torch.Generator(device='cpu').manual_seed(config.sliced_wasserstein_projection_seed)
                projection = torch.randn(int(common.sum()), config.sliced_wasserstein_num_projections,
                                         generator=gen).to(p.device)
                projection = F.normalize(projection, dim=0)
                pp = ((p - pm)[:, common] @ projection).sort(dim=0).values
                tt = ((t - tm)[:, common] @ projection).sort(dim=0).values
                components['sliced_wasserstein'].append(F.smooth_l1_loss(pp, tt))
    values = {key: torch.stack(items).mean() if items else zero for key, items in components.items()}
    values['gene_contrastive'] = zero
    if getattr(config, 'gene_contrastive_weight', 0) and len(effects_p) > 1:
        common = torch.stack(feature_masks).all(0)
        if common.any():
            p = torch.stack(effects_p)[:, common]
            t = torch.stack(effects_t)[:, common]
            # Remove the common response, so matching cannot be satisfied by
            # predicting the same morphology for every perturbation.
            p, t = p - p.mean(0), t - t.mean(0)
            if t.norm() > 1e-8:
                temperature = config.gene_contrastive_temperature
                if temperature <= 0:
                    raise ValueError('gene_contrastive_temperature must be positive')
                scores = F.normalize(p, dim=1) @ F.normalize(t, dim=1).T / temperature
                labels = torch.arange(len(p), device=p.device)
                values['gene_contrastive'] = (F.cross_entropy(scores, labels) + F.cross_entropy(scores.T, labels)) / 2
    weights = {'mean_mse': 'mean_weight', 'direction': 'direction_weight',
               'marginal_quantile': 'marginal_quantile_weight',
               'centered_quantile': 'centered_quantile_weight',
               'sliced_wasserstein': 'sliced_wasserstein_weight',
               'variance': 'variance_weight', 'gene_contrastive': 'gene_contrastive_weight'}
    loss = sum(getattr(config, name, 0) * values[key] for key, name in weights.items())
    return loss, {key: float(value.detach()) for key, value in values.items()}
