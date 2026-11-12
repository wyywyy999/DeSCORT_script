"""Three gene-balanced losses for unpaired perturbation populations.

Never supervise an arbitrary control/target row pairing. Statistics and sorted
quantiles are calculated within each gene, in float32, excluding padding.
"""
import torch
import torch.nn.functional as F


def perturbation_loss(prediction, target, control, genes, mask, config,
                      feature_ids=None):
    pred, target, control = prediction.float(), target.float(), control.float()
    mask = torch.ones_like(pred, dtype=torch.bool) if mask is None else mask.bool()
    components = {key: [] for key in ('mean_mse', 'direction', 'marginal_quantile')}
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
            direction = pred.new_zeros(())
        components['direction'].append(direction)
        # Invalid entries sort to the end independently in each feature.
        ps = p.masked_fill(~m, float('inf')).sort(dim=0).values
        ts = t.masked_fill(~m, float('inf')).sort(dim=0).values
        qm = torch.arange(len(p), device=p.device)[:, None] < count[None]
        components['marginal_quantile'].append(F.smooth_l1_loss(ps[qm], ts[qm]))
    if not components['mean_mse']:
        raise ValueError('Perturbation loss requires at least one valid feature')
    values = {key: torch.stack(items).mean() for key, items in components.items()}
    weights = {'mean_mse': 'mean_weight', 'direction': 'direction_weight',
               'marginal_quantile': 'marginal_quantile_weight'}
    loss = sum(getattr(config, name, 0) * values[key] for key, name in weights.items())
    return loss, {key: float(value.detach()) for key, value in values.items()}
