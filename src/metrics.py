"""Performance measures.

Per replicate (comparing an estimate P_hat with the true P on a block S):
    relfro_all  = ||P_hat - P||_F / ||P||_F
    relfro_avg  = mean over rows i with sum_j P_ij >= min_exp_deg of
                  ||P_hat_i - P_i||_2 / ||P_i||_2
    relfro_cent = ||P_hat - P||_F / ||P - pbar||_F             (stored, not plotted)
    rmse        = ||P_hat - P||_F / sqrt(|S|)                   (stored, not plotted)
    kl          = mean KL(Bern(P_ij) || Bern(P_hat_ij))
    relkl       = kl / mean KL(Bern(P_ij) || Bern(pbar)),  pbar = mean of P over S
    frac_rows   = fraction of rows kept by relfro_avg
    relfro_theta (plug-in only) = ||Theta_hat - Theta||_F / ||Theta||_F

Across replicates (simulation-level summaries):
    mean, SD, Monte Carlo SE = SD / sqrt(R), and a normal 95% interval for the mean.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

METRIC_KEYS = ['relfro_all', 'relfro_avg', 'relfro_cent', 'rmse', 'kl', 'relkl', 'frac_rows']
ESTIMATORS = ['pi', 'mc']
ESTIMATOR_NAMES = {'pi': 'plug-in', 'mc': 'MC posterior predictive'}


def bern_kl(logp, log1mp, logq, log1mq):
    """Elementwise KL(Bern(p) || Bern(q)) from log-probabilities."""
    p = torch.exp(logp)
    return p * (logp - logq) + (1.0 - p) * (log1mp - log1mq)


def prob_metrics(theta, logq, log1mq, mask=None, min_exp_deg=1.0):
    """Metrics for one block.

    theta        : true logits (r, c), float64
    logq, log1mq : log P_hat and log(1 - P_hat), (r, c), float64
    mask         : optional bool (r, c); False entries are excluded
    """
    if mask is None:
        mask = torch.ones_like(theta, dtype=torch.bool)
    w = mask.to(theta.dtype)
    m = w.sum()

    logp, log1mp = F.logsigmoid(theta), F.logsigmoid(-theta)
    p, q = torch.exp(logp), torch.exp(logq)
    sq = (q - p).pow(2) * w

    relfro_all = torch.sqrt(sq.sum()) / torch.sqrt((p.pow(2) * w).sum())
    rmse = torch.sqrt(sq.sum() / m)
    pbar_p = (p * w).sum() / m
    relfro_cent = torch.sqrt(sq.sum()) / torch.sqrt(((p - pbar_p).pow(2) * w).sum())

    row_num = torch.sqrt(sq.sum(1))
    row_den = torch.sqrt((p.pow(2) * w).sum(1))
    valid = (p * w).sum(1) >= min_exp_deg
    relfro_avg = (row_num[valid] / row_den[valid]).mean() if valid.any() else torch.tensor(float('nan'))

    kl = (bern_kl(logp, log1mp, logq, log1mq) * w).sum() / m
    pbar = ((p * w).sum() / m).clamp(1e-300, 1 - 1e-16)
    kl0 = (bern_kl(logp, log1mp, torch.log(pbar), torch.log1p(-pbar)) * w).sum() / m

    return {'relfro_all': relfro_all.item(), 'relfro_avg': relfro_avg.item(),
            'relfro_cent': relfro_cent.item(), 'rmse': rmse.item(),
            'kl': kl.item(), 'relkl': (kl / kl0).item(),
            'frac_rows': valid.double().mean().item()}


def evaluate_blocks(theta_true, estimates, theta_hat_plugin, blocks, min_exp_deg=1.0):
    """Evaluate every estimator on every block.

    theta_true       : (n, n) float64 true logits, nodes in training order
    estimates        : {'pi': (logq, log1mq), 'mc': (logq, log1mq)}; a missing
                       estimator is recorded as nan
    theta_hat_plugin : (n, n) float64 plug-in logits (for relfro_theta)
    blocks           : {name: (row_idx, col_idx, offdiag)} with long tensors
    Returns a flat dict with keys '{est}_{metric}_{block}'.
    """
    out = {}
    for bname, (rows, cols, offdiag) in blocks.items():
        th = theta_true[rows][:, cols]
        mask = ~torch.eye(len(rows), dtype=torch.bool, device=th.device) if offdiag else None
        for ename in ESTIMATORS:
            if ename in estimates:
                lq, l1q = estimates[ename]
                res = prob_metrics(th, lq[rows][:, cols], l1q[rows][:, cols], mask, min_exp_deg)
            else:
                res = {k: float('nan') for k in METRIC_KEYS}
            for k, v in res.items():
                out[f'{ename}_{k}_{bname}'] = v
        th_hat = theta_hat_plugin[rows][:, cols]
        w = 1.0 if mask is None else mask.double()
        out[f'pi_relfro_theta_{bname}'] = (torch.sqrt(((th_hat - th).pow(2) * w).sum())
                                           / torch.sqrt((th.pow(2) * w).sum())).item()
    return out


# --------------------------------------------------------------------------- #
# Simulation-level summaries                                                  #
# --------------------------------------------------------------------------- #
def summarize(values):
    """Mean, SD, Monte Carlo SE and normal 95% CI of the mean over replicates.

    nan replicates are dropped and counted (n_nan); nothing is imputed.
    """
    arr = np.asarray(values, dtype=float)
    n_nan = int(np.isnan(arr).sum())
    arr = arr[~np.isnan(arr)]
    r = len(arr)
    if r == 0:
        return dict(mean=np.nan, sd=np.nan, mcse=np.nan, ci_lo=np.nan, ci_hi=np.nan,
                    n_valid=0, n_nan=n_nan)
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1)) if r > 1 else np.nan
    mcse = sd / math.sqrt(r) if r > 1 else np.nan
    return dict(mean=mean, sd=sd, mcse=mcse, ci_lo=mean - 1.96 * mcse, ci_hi=mean + 1.96 * mcse,
                n_valid=r, n_nan=n_nan)


BLOCKS = ['onlyX_onlyA', 'onlyX_both', 'onlyX_A', 'A_A']


def parse_metric_key(key):
    """'mc_relfro_all_onlyX_A' -> ('mc', 'relfro_all', 'onlyX_A'); None if not a metric key."""
    est = key.split('_', 1)[0]
    if est not in ESTIMATORS:
        return None
    for block in sorted(BLOCKS, key=len, reverse=True):
        suffix = '_' + block
        if key.endswith(suffix):
            metric = key[len(est) + 1:-len(suffix)]
            if metric:
                return est, metric, block
    return None
