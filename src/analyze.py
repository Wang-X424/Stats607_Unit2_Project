"""Turn raw simulation output into tidy tables.

    python -m src.analyze --config full --props 30 40 30

Reads results/raw/<run_name>/sens_*.pkl and writes to results/summary/<run_name>/:
    final_long.csv     one row per (scenario, replicate, estimator, metric, block),
                       value at the FINAL iteration (no fallback to earlier iterations)
    summary.csv        mean, SD, Monte Carlo SE, 95% CI, n_valid, n_nan per cell
    paired_diff.csv    MC minus plug-in, paired within replicate (onlyX x A block)
    trajectories.csv   plug-in metrics and posterior log-variance over iterations
    convergence.csv    late-training change and iteration of the minimum, per scenario
    summary_onlyX_A.md readable table of the main metrics
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd

from . import metrics
from .config import get_config, log_iters, run_name
from .simulation import ROOT, build_scenarios

MAIN_BLOCK = 'onlyX_A'
MAIN_METRICS = ['relfro_all', 'relfro_avg', 'relkl']
TRAJ_COLS = ['loss', 'logvar_mean_onlyX', 'logvar_mean_A'] + \
            [f'pi_{m}_{MAIN_BLOCK}' for m in MAIN_METRICS] + \
            [f'pi_{m}_A_A' for m in MAIN_METRICS]          # network-observed pairs, for contrast


def load_raw(cfg, allow_missing=False):
    rdir = os.path.join(ROOT, 'results', 'raw', run_name(cfg))
    out, missing = {}, []
    for scen in build_scenarios(cfg):
        path = os.path.join(rdir, f"sens_{scen['label']}.pkl")
        if not os.path.exists(path):
            missing.append(scen['label'])
            continue
        with open(path, 'rb') as f:
            out[scen['label']] = (scen, pickle.load(f))
    if missing and not allow_missing:
        raise FileNotFoundError(f'{len(missing)} scenario(s) missing in {rdir}: {missing[:5]} ... '
                                'run `make simulate` first (or pass --allow-missing).')
    return out, missing


def final_long(raw, cfg):
    last_it = int(log_iters(cfg)[-1])
    rows = []
    for label, (scen, reps) in raw.items():
        if len(reps) != cfg['n_reps']:
            raise ValueError(f'{label}: {len(reps)} replicates, expected {cfg["n_reps"]}')
        for r, rep in enumerate(reps):
            last = rep[-1]
            if last['iter'] != last_it:
                raise ValueError(f'{label} rep {r}: last logged iteration {last["iter"]} != {last_it}')
            base = dict(label=label, n=scen['n'], p=scen['p'], alpha=scen['alpha'], rep=r)
            for key, val in last.items():
                parsed = metrics.parse_metric_key(key)
                if parsed is None:
                    if key in ('logvar_mean_onlyX', 'logvar_mean_A', 'edge_density'):
                        rows.append({**base, 'estimator': 'none', 'metric': key,
                                     'block': 'none', 'value': val})
                    continue
                est, met, block = parsed
                rows.append({**base, 'estimator': est, 'metric': met, 'block': block, 'value': val})
    return pd.DataFrame(rows)


def summarize_long(df):
    keys = ['label', 'n', 'p', 'alpha', 'estimator', 'metric', 'block']
    recs = []
    for k, g in df.groupby(keys, sort=False):
        recs.append({**dict(zip(keys, k)), **metrics.summarize(g['value'].values)})
    return pd.DataFrame(recs)


def paired_diff(df):
    sub = df[(df['block'] == MAIN_BLOCK) & df['metric'].isin(MAIN_METRICS)]
    wide = sub.pivot_table(index=['label', 'n', 'p', 'alpha', 'rep', 'metric'],
                           columns='estimator', values='value').reset_index()
    wide['diff'] = wide['mc'] - wide['pi']
    recs = []
    for k, g in wide.groupby(['label', 'n', 'p', 'alpha', 'metric'], sort=False):
        s = metrics.summarize(g['diff'].values)
        recs.append({**dict(zip(['label', 'n', 'p', 'alpha', 'metric'], k)), **s,
                     'frac_mc_better': float(np.mean(g['diff'] < 0))})
    return pd.DataFrame(recs)


def trajectories(raw):
    rows = []
    for label, (scen, reps) in raw.items():
        for r, rep in enumerate(reps):
            for rc in rep:
                rows.append({'label': label, 'n': scen['n'], 'p': scen['p'], 'alpha': scen['alpha'],
                             'rep': r, 'iter': rc['iter'], **{c: rc.get(c, np.nan) for c in TRAJ_COLS}})
    return pd.DataFrame(rows)


def convergence(traj, cfg):
    """Per scenario and plug-in metric: change over the last quarter of training
    (relative to the replicate SD at the end) and where the mean curve is minimal."""
    iters = log_iters(cfg)
    last_it = int(iters[-1])
    ref_it = int(iters[np.argmin(np.abs(iters - 0.75 * cfg['n_iter']))])
    recs = []
    for label, g in traj.groupby('label', sort=False):
        for m in MAIN_METRICS:
            col = f'pi_{m}_{MAIN_BLOCK}'
            piv = g.pivot(index='rep', columns='iter', values=col)
            if piv[last_it].isna().all():
                continue
            change = (piv[last_it] - piv[ref_it])
            sd_end = piv[last_it].std(ddof=1)
            mean_curve = piv.mean(axis=0)
            recs.append(dict(label=label, n=g['n'].iat[0], p=g['p'].iat[0], alpha=g['alpha'].iat[0],
                             metric=m, ref_iter=ref_it, mean_change_last_quarter=change.mean(),
                             sd_at_end=sd_end,
                             change_over_sd=change.mean() / sd_end if sd_end > 0 else np.nan,
                             iter_of_min_mean=int(mean_curve.idxmin()),
                             value_at_min_mean=float(mean_curve.min()),
                             value_at_end_mean=float(mean_curve[last_it])))
    return pd.DataFrame(recs)


def markdown_table(summ, pdiff):
    sub = summ[(summ['block'] == MAIN_BLOCK) & summ['metric'].isin(MAIN_METRICS)]
    lines = ['| n | p | alpha | estimator | ' + ' | '.join(MAIN_METRICS) + ' |',
             '|---|---|---|---|' + '---|' * len(MAIN_METRICS)]
    for (label, n, p, a, est), g in sub.groupby(['label', 'n', 'p', 'alpha', 'estimator'], sort=False):
        cells = []
        for m in MAIN_METRICS:
            row = g[g['metric'] == m].iloc[0]
            cells.append(f"{row['mean']:.3f} ± {row['mcse']:.3f}")
        lines.append(f'| {n} | {p} | {a} | {est} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines) + '\n\nEntries: mean ± Monte Carlo SE over replicates, final iteration.\n'


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='full')
    ap.add_argument('--props', nargs=3, type=int, default=None)
    ap.add_argument('--allow-missing', action='store_true')
    args = ap.parse_args(argv)
    cfg = get_config(args.config, args.props)

    raw, missing = load_raw(cfg, args.allow_missing)
    out_dir = os.path.join(ROOT, 'results', 'summary', run_name(cfg))
    os.makedirs(out_dir, exist_ok=True)

    df = final_long(raw, cfg)
    summ = summarize_long(df)
    pdiff = paired_diff(df)
    traj = trajectories(raw)
    conv = convergence(traj, cfg)

    df.to_csv(os.path.join(out_dir, 'final_long.csv'), index=False)
    summ.to_csv(os.path.join(out_dir, 'summary.csv'), index=False)
    pdiff.to_csv(os.path.join(out_dir, 'paired_diff.csv'), index=False)
    traj.to_csv(os.path.join(out_dir, 'trajectories.csv'), index=False)
    conv.to_csv(os.path.join(out_dir, 'convergence.csv'), index=False)
    with open(os.path.join(out_dir, 'summary_onlyX_A.md'), 'w') as f:
        f.write(markdown_table(summ, pdiff))

    n_nan = int(df[df['estimator'] != 'none']['value'].isna().sum())
    print(f'analyzed {len(raw)} scenarios ({len(missing)} missing); {n_nan} nan metric values '
          f'at the final iteration; tables in {out_dir}')


if __name__ == '__main__':
    main()
