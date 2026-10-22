"""Design matrix, output validation, reproducibility and resume behaviour."""

import math
import os
import pickle

import pytest
import torch

from src import simulation
from src.config import get_config, log_iters

CPU = torch.device('cpu')


def tiny_cfg():
    cfg = get_config('smoke')
    cfg.update(n_iter=60, n_reps=2, mc_samples=5)
    return cfg


def assert_identical(a, b):
    """Exact equality of nested results, treating nan == nan (plain == does not)."""
    if isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b)
        for x, y in zip(a, b):
            assert_identical(x, y)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            assert_identical(a[k], b[k])
    elif isinstance(a, float) and math.isnan(a):
        assert isinstance(b, float) and math.isnan(b)
    else:
        assert a == b, (a, b)


SCEN = {'label': 'N60_X30_Aneg2', 'n': 60, 'p': 30, 'alpha': -2}


def test_full_design_has_61_unique_scenarios():
    scen = simulation.build_scenarios(get_config('full'))
    labels = [s['label'] for s in scen]
    assert len(labels) == len(set(labels)) == 61
    # alpha = -6: the full 5 x 5 (n, p) grid; every other alpha: 5 + 5 - 1 = 9 scenarios
    assert sum(s['alpha'] == -6 for s in scen) == 25
    for a in (-8, -4, -2, 0):
        assert sum(s['alpha'] == a for s in scen) == 9
    assert simulation.make_label(700, 700, -6) == 'N700_X700_Aneg6'


def test_outputs_are_finite_and_in_valid_ranges():
    res = simulation.run_one_rep(tiny_cfg(), SCEN, rep=0, device=CPU)
    assert [r['iter'] for r in res] == list(log_iters(tiny_cfg()))
    last = res[-1]
    for est in ('pi', 'mc'):
        for m in ('relfro_all', 'relfro_cent', 'kl', 'relkl', 'rmse'):
            v = last[f'{est}_{m}_onlyX_A']
            assert math.isfinite(v) and v >= 0, (est, m, v)
        assert 0.0 <= last[f'{est}_frac_rows_onlyX_A'] <= 1.0
    assert math.isnan(res[0]['mc_relkl_onlyX_A'])          # MC only at the final iteration
    assert 0.0 < last['edge_density'] < 1.0


def test_same_seed_gives_identical_results():
    a = simulation.run_one_rep(tiny_cfg(), SCEN, rep=1, device=CPU)
    b = simulation.run_one_rep(tiny_cfg(), SCEN, rep=1, device=CPU)
    assert_identical(a, b)


def test_resume_after_interruption_matches_uninterrupted_run(tmp_path):
    cfg = tiny_cfg()
    full_dir, resumed_dir = tmp_path / 'full', tmp_path / 'resumed'
    full_dir.mkdir(); resumed_dir.mkdir()
    simulation.run_scenario(cfg, SCEN, str(full_dir), CPU)

    # simulate a crash after replicate 1: only the partial checkpoint exists
    first = simulation.run_one_rep(cfg, SCEN, rep=0, device=CPU)
    simulation.atomic_pickle([first], str(resumed_dir / f"sens_{SCEN['label']}.partial.pkl"))
    simulation.run_scenario(cfg, SCEN, str(resumed_dir), CPU)

    load = lambda d: pickle.load(open(d / f"sens_{SCEN['label']}.pkl", 'rb'))
    assert_identical(load(full_dir), load(resumed_dir))
    assert not os.path.exists(resumed_dir / f"sens_{SCEN['label']}.partial.pkl")


def test_config_mismatch_is_refused(tmp_path):
    cfg = tiny_cfg()
    simulation.check_config(cfg, str(tmp_path))
    other = dict(cfg, lr=5e-3)
    with pytest.raises(RuntimeError):
        simulation.check_config(other, str(tmp_path))
