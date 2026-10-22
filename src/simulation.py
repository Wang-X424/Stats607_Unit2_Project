"""Main simulation orchestration.

    python -m src.simulation --config full --props 30 40 30 [--device cuda]

For every scenario (n, p, alpha) in the design matrix and every replicate r:
generate data (seeded), split nodes, fit VGAE-X, and evaluate the plug-in and MC
estimators of the edge-probability matrix against the truth every
``log_every`` iterations.  Raw per-iteration results are written to
results/raw/<run_name>/sens_<label>.pkl (list over replicates of lists over
logged iterations of metric dicts) plus a tab-separated .txt copy.

Caching / resuming: a finished scenario is skipped if its .pkl exists; an
unfinished scenario resumes from the last completed replicate
(sens_<label>.partial.pkl).  All writes are atomic.  results/raw/<run_name>/
config.json records the configuration; a mismatch raises instead of silently
mixing results from different settings.
"""

import argparse
import json
import logging
import os
import pickle
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

from . import dgps, methods, metrics
from .config import SCIENTIFIC_KEYS, get_config, log_iters, run_name

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log = logging.getLogger('simulation')


# --------------------------------------------------------------------------- #
# Design matrix                                                               #
# --------------------------------------------------------------------------- #
def make_label(n, p, alpha):
    return f'N{n}_X{p}_A{str(alpha).replace(".", "").replace("-", "neg")}'


def build_scenarios(cfg):
    """Three one-factor-fixed slices of the (n, p, alpha) grid, deduplicated.

      slice 'n x p'     : alpha = fixed_alpha
      slice 'p x alpha' : n     = fixed_n
      slice 'n x alpha' : p     = fixed_p
    """
    seen = {}

    def add(n, p, a):
        lbl = make_label(n, p, a)
        if lbl not in seen:
            seen[lbl] = {'label': lbl, 'n': n, 'p': p, 'alpha': a}

    for n in cfg['n_levels']:
        for p in cfg['p_levels']:
            add(n, p, cfg['fixed_alpha'])
    for p in cfg['p_levels']:
        for a in cfg['alpha_levels']:
            add(cfg['fixed_n'], p, a)
    for n in cfg['n_levels']:
        for a in cfg['alpha_levels']:
            add(n, cfg['fixed_p'], a)
    return list(seen.values())


def seeds_for(cfg, rep):
    s = cfg['seed']
    return {'data': s + rep, 'split': s + rep + 100, 'torch': s + rep + 10000}


# --------------------------------------------------------------------------- #
# One replicate                                                               #
# --------------------------------------------------------------------------- #
def run_one_rep(cfg, scen, rep, device):
    """Simulate, fit and evaluate one replicate. Returns a list of per-iteration dicts."""
    sd = seeds_for(cfg, rep)
    data = dgps.generate_data(scen['n'], cfg['z_dim_true'], scen['p'], scen['alpha'],
                              cfg['z_var'], seed=sd['data'], x_hidden_dim=cfg['x_hidden_dim'])
    idx_oA, idx_b, idx_oX = dgps.split_nodes(scen['n'], cfg['props'], seed=sd['split'])
    inp = methods.prepare_inputs(data['A'], data['X'], idx_oA, idx_b, idx_oX, cfg['self_weight'])

    # truth in training order (the estimand); never passed to the method
    theta_true = torch.tensor(dgps.true_logits(data['Z'][inp['num_train']], scen['alpha']),
                              dtype=torch.float64, device=device)
    tl = lambda a: torch.as_tensor(a, dtype=torch.long, device=device)
    blocks = {
        'onlyX_A': (tl(inp['id_onlyX']), tl(inp['id_A']), False),
        'onlyX_onlyA': (tl(inp['id_onlyX']), tl(inp['id_onlyA']), False),
        'onlyX_both': (tl(inp['id_onlyX']), tl(inp['id_both']), False),
        'A_A': (tl(inp['id_A']), tl(inp['id_A']), True),
    }
    n_A = len(inp['id_A'])
    iters = log_iters(cfg)
    last_it = int(iters[-1])
    results = []

    def callback(it, loss, model, zm, zlv):
        est = {'pi': methods.plugin_log_probs(model, zm)}
        if cfg['mc_every_log'] or it == last_it:
            est['mc'] = methods.mc_log_probs(model, zm, zlv, cfg['mc_samples'])
        rc = {'iter': it, 'loss': loss,
              'logvar_mean_onlyX': zlv[n_A:].mean().item(),
              'logvar_mean_A': zlv[:n_A].mean().item()}
        rc.update(metrics.evaluate_blocks(theta_true, est, methods.plugin_logits(model, zm),
                                          blocks, cfg['min_exp_deg']))
        results.append(rc)

    torch.manual_seed(sd['torch'])
    methods.fit_vgae(inp, cfg, device, iters, callback)
    density = data['A'].sum() / (scen['n'] * (scen['n'] - 1))
    results[-1]['edge_density'] = float(density)
    return results


# --------------------------------------------------------------------------- #
# I/O helpers                                                                 #
# --------------------------------------------------------------------------- #
def atomic_pickle(obj, path):
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_txt(res, path):
    cols = [k for k in res[0][-1].keys() if k not in ('iter', 'loss')]
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        f.write('\t'.join(['rep', 'iter', 'loss'] + cols) + '\n')
        for r, rep in enumerate(res):
            for rc in rep:
                f.write('\t'.join(str(v) for v in [r, rc['iter'], rc['loss']]
                                  + [rc.get(c, float('nan')) for c in cols]) + '\n')
    os.replace(tmp, path)


def raw_dir(cfg):
    return os.path.join(ROOT, 'results', 'raw', run_name(cfg))


def check_config(cfg, out_dir):
    """Write config.json on first use; refuse to reuse a directory made with other settings."""
    path = os.path.join(out_dir, 'config.json')
    current = {k: (list(cfg[k]) if isinstance(cfg[k], tuple) else cfg[k]) for k in SCIENTIFIC_KEYS}
    if os.path.exists(path):
        with open(path) as f:
            stored = json.load(f)
        diff = {k: (stored.get(k), current[k]) for k in SCIENTIFIC_KEYS if stored.get(k) != current[k]}
        if diff:
            raise RuntimeError(f'{path} was produced with different settings {diff}. '
                               'Use a different config/props or remove the directory.')
    else:
        with open(path, 'w') as f:
            json.dump({**current, 'provenance': 'generated by src/simulation.py'}, f, indent=2)


def run_scenario(cfg, scen, out_dir, device):
    final_path = os.path.join(out_dir, f"sens_{scen['label']}.pkl")
    partial_path = os.path.join(out_dir, f"sens_{scen['label']}.partial.pkl")
    if os.path.exists(final_path):
        log.info('[skip] %s (cached)', scen['label'])
        return
    reps = []
    if os.path.exists(partial_path):
        with open(partial_path, 'rb') as f:
            reps = pickle.load(f)
        log.info('[resume] %s from replicate %d', scen['label'], len(reps) + 1)
    for rep in tqdm(range(len(reps), cfg['n_reps']), desc=scen['label'], leave=False):
        t0 = time.time()
        res = run_one_rep(cfg, scen, rep, device)
        reps.append(res)
        atomic_pickle(reps, partial_path)
        f = res[-1]
        log.info('%s rep %d/%d (%.1fs) density=%.4f | onlyX_A relfro_all mc=%.4f pi=%.4f | '
                 'relkl mc=%.4f pi=%.4f', scen['label'], rep + 1, cfg['n_reps'], time.time() - t0,
                 f['edge_density'], f['mc_relfro_all_onlyX_A'], f['pi_relfro_all_onlyX_A'],
                 f['mc_relkl_onlyX_A'], f['pi_relkl_onlyX_A'])
    atomic_pickle(reps, final_path)
    write_txt(reps, final_path.replace('.pkl', '.txt'))
    os.remove(partial_path)
    log.info('[saved] %s', scen['label'])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default='full')
    ap.add_argument('--props', nargs=3, type=int, default=None, metavar=('ONLYA', 'BOTH', 'ONLYX'))
    ap.add_argument('--device', default='auto', help="'auto', 'cpu' or 'cuda'")
    args = ap.parse_args(argv)

    cfg = get_config(args.config, args.props)
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu')
                          if args.device == 'auto' else args.device)
    out_dir = raw_dir(cfg)
    os.makedirs(out_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(os.path.join(out_dir, 'simulation.log'))])
    check_config(cfg, out_dir)
    scenarios = build_scenarios(cfg)
    log.info('run %s | device %s | %d scenarios x %d replicates', run_name(cfg), device,
             len(scenarios), cfg['n_reps'])
    for scen in tqdm(scenarios, desc='scenarios'):
        run_scenario(cfg, scen, out_dir, device)
    log.info('all scenarios complete')


if __name__ == '__main__':
    main()
