"""Figures.

    python -m src.figures --config full --props 30 40 30

Reads results/summary/<run_name>/ and writes to results/figures/<run_name>/:
    fig1_main_mc.{pdf,png}       publication figure, MC estimator (mean and 95% CI)
    fig1_main_pi.{pdf,png}       same for the plug-in estimator
    fig2_convergence.{pdf,png}   diagnostic: training trajectories in stress scenarios
    fig3_mc_vs_plugin.{pdf,png}  diagnostic: paired per-replicate comparison of estimators
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from .analyze import MAIN_BLOCK, MAIN_METRICS
from .config import get_config, run_name
from .metrics import ESTIMATOR_NAMES
from .simulation import ROOT, make_label

METRIC_LABELS = {'relfro_all': 'relative Frobenius error\n(whole block)',
                 'relfro_avg': 'relative Frobenius error\n(node average)',
                 'relkl': 'relative KL\n(1 = constant-density predictor)'}
METRIC_SHORT = {'relfro_all': 'relFro (all)', 'relfro_avg': 'relFro (avg)', 'relkl': 'relKL'}
# ordinal one-hue ramps (light -> dark); alpha in blue, p in orange; markers as secondary encoding
BLUE = ['#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#104281']
ORANGE = ['#f2a07c', '#eb6834', '#c4511f', '#963d16', '#6b2b0f']
MARKERS = ['o', '^', 's', 'D', 'v']
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#d9d8d4'
LOG_TICKS = [0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10, 20, 30, 50]

plt.rcParams.update({
    'font.size': 9, 'axes.titlesize': 9.5, 'axes.labelsize': 9, 'legend.fontsize': 8,
    'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 110,
})


def ramp(colors, k):
    idx = np.round(np.linspace(0, len(colors) - 1, k)).astype(int) if k > 1 else [len(colors) // 2]
    return [colors[i] for i in idx]


def save(fig, out_dir, name):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(out_dir, f'{name}.{ext}'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('saved', os.path.join(out_dir, name + '.{pdf,png}'))


def style_axis(ax):
    ax.grid(axis='y', color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


# --------------------------------------------------------------------------- #
# Figure 1: main results                                                      #
# --------------------------------------------------------------------------- #
def panels(cfg):
    """(x levels, x label, line levels, line legend fn, label fn, colours, title) per column."""
    return [
        (cfg['n_levels'], 'number of nodes $n$', cfg['p_levels'], lambda v: f'$p$ = {v}',
         lambda x, l: make_label(x, l, cfg['fixed_alpha']), ramp(ORANGE, len(cfg['p_levels'])),
         f"$\\alpha$ = {cfg['fixed_alpha']} fixed"),
        (cfg['p_levels'], 'feature dimension $p$', cfg['alpha_levels'], lambda v: f'$\\alpha$ = {v}',
         lambda x, l: make_label(cfg['fixed_n'], x, l), ramp(BLUE, len(cfg['alpha_levels'])),
         f"$n$ = {cfg['fixed_n']} fixed"),
        (cfg['n_levels'], 'number of nodes $n$', cfg['alpha_levels'], lambda v: f'$\\alpha$ = {v}',
         lambda x, l: make_label(x, cfg['fixed_p'], l), ramp(BLUE, len(cfg['alpha_levels'])),
         f"$p$ = {cfg['fixed_p']} fixed"),
    ]


def fig_main(summ, cfg, est, out_dir):
    s = summ[(summ['block'] == MAIN_BLOCK) & (summ['estimator'] == est)].set_index(['label', 'metric'])
    cols = panels(cfg)
    fig, axes = plt.subplots(len(MAIN_METRICS), 3, figsize=(10.5, 2.75 * len(MAIN_METRICS)), squeeze=False)

    for r, m in enumerate(MAIN_METRICS):
        lo_all, hi_all = [], []
        for c, (xl, xlab, ll, lfmt, lbl, colors, title) in enumerate(cols):
            ax = axes[r, c]
            for k, lv in enumerate(ll):
                rows = [s.loc[(lbl(x, lv), m)] if (lbl(x, lv), m) in s.index else None for x in xl]
                mean = np.array([np.nan if q is None else q['mean'] for q in rows])
                lo = np.array([np.nan if q is None else q['ci_lo'] for q in rows])
                hi = np.array([np.nan if q is None else q['ci_hi'] for q in rows])
                ax.fill_between(xl, lo, hi, color=colors[k], alpha=0.22, linewidth=0)
                ax.plot(xl, mean, color=colors[k], marker=MARKERS[k % len(MARKERS)], ms=4.5,
                        lw=1.8, label=lfmt(lv), mec='white', mew=0.6)
                lo_all += list(lo[~np.isnan(lo)]); hi_all += list(hi[~np.isnan(hi)])
            ax.set_xticks(xl)
            ax.set_xlim(min(xl) - 0.06 * (max(xl) - min(xl)), max(xl) + 0.06 * (max(xl) - min(xl)))
            style_axis(ax)
            if r == len(MAIN_METRICS) - 1:
                ax.set_xlabel(xlab)
            if r == 0:
                ax.set_title(title, color=MUTED, style='italic', pad=30)
                ax.legend(frameon=False, ncol=3, handlelength=1.6, columnspacing=0.9, fontsize=7.5,
                          loc='lower center', bbox_to_anchor=(0.5, 1.0))
            if c == 0:
                ax.set_ylabel(METRIC_LABELS[m])
        if lo_all:
            lo, hi = min(v for v in lo_all if v > 0), max(hi_all)
            for c in range(3):
                ax = axes[r, c]
                ax.set_yscale('log')                      # ratios: log scale, 1 in the middle
                ax.set_ylim(min(lo, 1.0) / 1.15, max(hi, 1.0) * 1.15)
                ax.yaxis.set_major_locator(mticker.FixedLocator(LOG_TICKS))
                ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:g}'))
                ax.yaxis.set_minor_locator(mticker.NullLocator())
                ax.axhline(1.0, color=MUTED, lw=0.8, ls='--', zorder=0)
                if c > 0:
                    ax.tick_params(labelleft=False)

    fig.suptitle(f'Recovery of edge probabilities for cold-start nodes (onlyX $\\times$ A), '
                 f'{ESTIMATOR_NAMES[est]} estimator\n'
                 f'mean and 95% CI over R = {cfg["n_reps"]} replicates at the final iteration; '
                 'lower is better; log scale; dashed line = 1 (reference predictor)', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, out_dir, f'fig1_main_{est}')


# --------------------------------------------------------------------------- #
# Figure 2: convergence diagnostic                                            #
# --------------------------------------------------------------------------- #
def stress_scenarios(cfg):
    fa, fn, fp = cfg['fixed_alpha'], cfg['fixed_n'], cfg['fixed_p']
    cand = [('centre', fn, fp, fa),
            ('largest $n$, $p$', max(cfg['n_levels']), max(cfg['p_levels']), fa),
            ('smallest $n$, largest $p$', min(cfg['n_levels']), max(cfg['p_levels']), fa),
            ('sparsest', fn, fp, min(cfg['alpha_levels'])),
            ('densest', fn, fp, max(cfg['alpha_levels']))]
    out, seen = [], set()
    for name, n, p, a in cand:
        lbl = make_label(n, p, a)
        if lbl not in seen:
            seen.add(lbl)
            out.append((name, n, p, a, lbl))
    return out


def fig_convergence(traj, cfg, out_dir):
    rows = [(f'pi_relfro_all_{MAIN_BLOCK}', 'relFro (all), plug-in', 'pi_relfro_all_A_A'),
            (f'pi_relkl_{MAIN_BLOCK}', 'relKL, plug-in', 'pi_relkl_A_A'),
            ('logvar_mean_onlyX', 'mean posterior log-variance\n(onlyX nodes)', None)]
    sc = stress_scenarios(cfg)
    fig, axes = plt.subplots(len(rows), len(sc), figsize=(2.3 * len(sc) + 0.8, 2.2 * len(rows)),
                             squeeze=False, sharex=True)
    # iterations where the cosine-annealed learning rate is below 10% of its start
    its = np.arange(cfg['n_iter'])
    lr = cfg['lr_min'] + (cfg['lr'] - cfg['lr_min']) * (1 + np.cos(np.pi * its / cfg['n_iter'])) / 2
    low_from = int(its[np.argmax(lr < 0.1 * cfg['lr'])])
    for c, (name, n, p, a, lbl) in enumerate(sc):
        g = traj[(traj['label'] == lbl) & (traj['iter'] > 0)]
        for r, (col, ylab, contrast) in enumerate(rows):
            ax = axes[r, c]
            ax.axvspan(low_from, cfg['n_iter'], color='#f0efec', zorder=0, lw=0)
            if g.empty:
                continue
            piv = g.pivot(index='iter', columns='rep', values=col)
            ax.plot(piv.index, piv.values, color='#86b6ef', lw=0.6, alpha=0.6)
            ax.plot(piv.index, piv.mean(axis=1), color='#104281', lw=1.8, label='onlyX × A (cold start)')
            if contrast is not None and contrast in g:
                pc = g.pivot(index='iter', columns='rep', values=contrast)
                ax.plot(pc.index, pc.mean(axis=1), color='#eb6834', lw=1.8, ls='--',
                        label='A × A (network observed)')
            style_axis(ax)
            if r == 0:
                ax.set_title(f'{name}\n$n$={n}, $p$={p}, $\\alpha$={a}', fontsize=8.5)
            if c == 0:
                ax.set_ylabel(ylab)
            if r == len(rows) - 1:
                ax.set_xlabel('iteration')
    fig.suptitle('Diagnostic: training trajectories, plug-in estimator. Thin blue: replicates (onlyX × A); '
                 'thick blue: their mean; dashed orange: mean for A × A.\n'
                 f'Shaded: learning rate < 10% of initial (iteration ≥ {low_from}), where flattening '
                 'is forced by the schedule and is not by itself evidence of convergence', fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.925), ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    save(fig, out_dir, 'fig2_convergence')


# --------------------------------------------------------------------------- #
# Figure 3: MC vs plug-in                                                     #
# --------------------------------------------------------------------------- #
def fig_mc_vs_plugin(final, cfg, out_dir):
    sub = final[(final['block'] == MAIN_BLOCK) & final['metric'].isin(MAIN_METRICS)]
    wide = sub.pivot_table(index=['label', 'alpha', 'rep', 'metric'], columns='estimator',
                           values='value').reset_index()
    alphas = sorted(cfg['alpha_levels'])
    colors = ramp(BLUE, len(alphas))
    fig, axes = plt.subplots(1, len(MAIN_METRICS), figsize=(3.5 * len(MAIN_METRICS), 3.5))
    for ax, m in zip(axes, MAIN_METRICS):
        w = wide[wide['metric'] == m].dropna(subset=['pi', 'mc'])
        for k, a in enumerate(alphas):
            wa = w[w['alpha'] == a]
            ax.scatter(wa['pi'], wa['mc'], s=10, color=colors[k], marker=MARKERS[k % len(MARKERS)],
                       alpha=0.65, linewidths=0, label=f'$\\alpha$ = {a}')
        if len(w):
            lo = min(w['pi'].min(), w['mc'].min()) / 1.15; hi = max(w['pi'].max(), w['mc'].max()) * 1.15
            ax.set_xscale('log'); ax.set_yscale('log')
            ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.9, ls='--')
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            for axis in (ax.xaxis, ax.yaxis):
                axis.set_major_locator(mticker.FixedLocator(LOG_TICKS))
                axis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:g}'))
                axis.set_minor_locator(mticker.NullLocator())
            frac = np.mean(w['mc'] < w['pi'])
            ax.text(0.97, 0.03, f'MC lower in {100 * frac:.0f}% of\nscenario-replicates',
                    transform=ax.transAxes, va='bottom', ha='right', fontsize=8, color=INK)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel(f'{METRIC_SHORT[m]}, plug-in'); ax.set_ylabel(f'{METRIC_SHORT[m]}, MC')
        style_axis(ax)
    axes[-1].legend(frameon=False, loc='upper left', bbox_to_anchor=(1.02, 1.0), markerscale=1.6)
    fig.suptitle('Diagnostic: MC vs plug-in on the same replicate (onlyX $\\times$ A, final iteration, '
                 'log-log); points below the dashed line favour MC', fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, out_dir, 'fig3_mc_vs_plugin')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='full')
    ap.add_argument('--props', nargs=3, type=int, default=None)
    args = ap.parse_args(argv)
    cfg = get_config(args.config, args.props)
    sdir = os.path.join(ROOT, 'results', 'summary', run_name(cfg))
    if not os.path.exists(os.path.join(sdir, 'summary.csv')):
        raise FileNotFoundError(f'{sdir}/summary.csv not found; run `make analyze` first')
    out_dir = os.path.join(ROOT, 'results', 'figures', run_name(cfg))
    os.makedirs(out_dir, exist_ok=True)

    summ = pd.read_csv(os.path.join(sdir, 'summary.csv'))
    final = pd.read_csv(os.path.join(sdir, 'final_long.csv'))
    traj = pd.read_csv(os.path.join(sdir, 'trajectories.csv'))
    for est in ('mc', 'pi'):
        fig_main(summ, cfg, est, out_dir)
    fig_convergence(traj, cfg, out_dir)
    fig_mc_vs_plugin(final, cfg, out_dir)


if __name__ == '__main__':
    main()
