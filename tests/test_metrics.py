"""Performance-measure correctness on cases with known answers."""

import math

import numpy as np
import torch
import torch.nn.functional as F

from src import metrics


def _theta(seed=0, shape=(30, 40)):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=g, dtype=torch.float64) * 3 - 2


def test_perfect_prediction_gives_zero_error():
    th = _theta()
    r = metrics.prob_metrics(th, F.logsigmoid(th), F.logsigmoid(-th), min_exp_deg=0.0)
    for k in ('relfro_all', 'relfro_avg', 'relfro_cent', 'rmse', 'kl', 'relkl'):
        assert abs(r[k]) < 1e-12, k


def test_log_space_kl_matches_direct_formula_and_softplus_identity():
    th, thh = _theta(0), _theta(1)
    p, q = torch.sigmoid(th), torch.sigmoid(thh)
    direct = (p * torch.log(p / q) + (1 - p) * torch.log((1 - p) / (1 - q))).mean()
    logspace = metrics.bern_kl(F.logsigmoid(th), F.logsigmoid(-th),
                               F.logsigmoid(thh), F.logsigmoid(-thh)).mean()
    identity = (F.softplus(thh) - F.softplus(th) + p * (th - thh)).mean()
    assert torch.allclose(direct, logspace, rtol=1e-10)
    assert torch.allclose(identity, logspace, rtol=1e-10)


def test_kl_is_finite_for_extremely_small_probabilities():
    th = torch.full((5, 5), -60.0, dtype=torch.float64)       # p ~ 1e-26
    thh = torch.full((5, 5), -40.0, dtype=torch.float64)
    r = metrics.prob_metrics(th, F.logsigmoid(thh), F.logsigmoid(-thh), min_exp_deg=0.0)
    assert math.isfinite(r['kl']) and r['kl'] >= 0


def test_constant_density_predictor_has_relkl_one():
    th = _theta()
    pbar = torch.sigmoid(th).mean()
    lq = torch.full_like(th, math.log(pbar)); l1q = torch.full_like(th, math.log1p(-pbar))
    r = metrics.prob_metrics(th, lq, l1q, min_exp_deg=0.0)
    assert abs(r['relkl'] - 1.0) < 1e-10
    assert abs(r['relfro_cent'] - 1.0) < 1e-10


def test_all_zero_predictor_has_relfro_one():
    th = _theta()
    lq = torch.full_like(th, -1e4); l1q = torch.zeros_like(th)     # P_hat = 0
    r = metrics.prob_metrics(th, lq, l1q, min_exp_deg=0.0)
    assert abs(r['relfro_all'] - 1.0) < 1e-10


def test_mask_excludes_diagonal_and_rows_filter_is_reported():
    th = _theta(shape=(20, 20))
    thh = th.clone(); thh.fill_diagonal_(50.0)                     # wrong only on the diagonal
    mask = ~torch.eye(20, dtype=torch.bool)
    r = metrics.prob_metrics(th, F.logsigmoid(thh), F.logsigmoid(-thh), mask, min_exp_deg=0.0)
    assert r['relfro_all'] < 1e-12
    r2 = metrics.prob_metrics(th, F.logsigmoid(th), F.logsigmoid(-th), min_exp_deg=1e9)
    assert r2['frac_rows'] == 0.0 and math.isnan(r2['relfro_avg'])


def test_summarize_counts_nan_and_computes_mcse():
    s = metrics.summarize([1.0, 2.0, 3.0, float('nan')])
    assert s['n_valid'] == 3 and s['n_nan'] == 1
    assert abs(s['mean'] - 2.0) < 1e-12
    assert abs(s['mcse'] - 1.0 / math.sqrt(3)) < 1e-12


def test_parse_metric_key():
    assert metrics.parse_metric_key('mc_relfro_all_onlyX_A') == ('mc', 'relfro_all', 'onlyX_A')
    assert metrics.parse_metric_key('pi_relkl_onlyX_onlyA') == ('pi', 'relkl', 'onlyX_onlyA')
    assert metrics.parse_metric_key('pi_relfro_theta_A_A') == ('pi', 'relfro_theta', 'A_A')
    assert metrics.parse_metric_key('logvar_mean_onlyX') is None
