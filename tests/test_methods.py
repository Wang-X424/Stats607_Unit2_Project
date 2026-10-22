"""Method correctness: estimators behave as theory says in simple cases."""

import numpy as np
import torch

from src import dgps, methods


class _Decoder:
    PA_dens = torch.tensor(-1.0)


def test_mc_equals_plugin_when_posterior_variance_vanishes():
    torch.manual_seed(0)
    mu = torch.randn(25, 4)
    l1, _ = methods.mc_log_probs(_Decoder, mu, torch.full_like(mu, -40.0), n_samples=5)
    pi, _ = methods.plugin_log_probs(_Decoder, mu)
    assert torch.allclose(l1.exp(), pi.exp(), atol=1e-6)


def test_mc_shrinks_towards_one_half_when_variance_is_large():
    """Jensen: E[sigmoid(x)] is below sigmoid(E x) for likely edges and above for unlikely ones."""
    torch.manual_seed(1)
    mu = torch.randn(40, 4)
    mc = methods.mc_log_probs(_Decoder, mu, torch.full_like(mu, 1.0), n_samples=400)[0].exp()
    pi = methods.plugin_log_probs(_Decoder, mu)[0].exp()
    hi, lo = pi > 0.9, pi < 0.1
    assert mc[hi].mean() < pi[hi].mean()
    assert mc[lo].mean() > pi[lo].mean()


def test_prepare_inputs_produces_valid_normalised_adjacency():
    d = dgps.generate_data(80, 8, 15, -6, 1.0, seed=2)        # sparse: isolated nodes likely
    oA, b, oX = dgps.split_nodes(80, (30, 40, 30), seed=3)
    inp = methods.prepare_inputs(d['A'], d['X'], oA, b, oX, self_weight=1.0)
    AH = inp['AHat']
    assert np.all(np.isfinite(AH))
    assert np.allclose(AH, AH.T)
    assert inp['X_fill'].shape == (80, 15)
    assert np.isfinite(inp['logit_edge_prob'])
    # onlyX nodes' network is never passed to the method
    assert inp['A_obs'].shape == (len(oA) + len(b),) * 2


def test_vgae_fit_reduces_error_against_initialisation():
    """A short fit on an easy, fairly dense problem beats the model at initialisation."""
    from src.config import get_config
    from src.metrics import prob_metrics
    cfg = get_config('smoke'); cfg['n_iter'] = 150
    d = dgps.generate_data(150, 8, 60, -2, 1.0, seed=5)
    oA, b, oX = dgps.split_nodes(150, (30, 40, 30), seed=6)
    inp = methods.prepare_inputs(d['A'], d['X'], oA, b, oX, 1.0)
    th = torch.tensor(dgps.true_logits(d['Z'][inp['num_train']], -2))
    rows, cols = torch.as_tensor(inp['id_A']), torch.as_tensor(inp['id_A'])
    errs = {}

    def cb(it, loss, model, zm, zlv):
        lq, l1q = methods.plugin_log_probs(model, zm)
        errs[it] = prob_metrics(th[rows][:, cols], lq[rows][:, cols], l1q[rows][:, cols],
                                ~torch.eye(len(rows), dtype=torch.bool))['relkl']

    torch.manual_seed(0)
    methods.fit_vgae(inp, cfg, torch.device('cpu'), [0, cfg['n_iter'] - 1], cb)
    assert errs[cfg['n_iter'] - 1] < errs[0]
