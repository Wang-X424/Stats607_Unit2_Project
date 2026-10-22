"""Simulation configurations.

Every tunable value of the study lives in this file. ``CONFIGS['full']`` is the
configuration used for the reported results; ``CONFIGS['smoke']`` is a tiny
version for checking that the pipeline runs end to end (minutes on a CPU).
"""

import copy

FULL = dict(
    # ---- random seeds --------------------------------------------------------
    seed=42,                 # replicate r uses data seed  seed + r,
                             #                 split seed seed + r + 100,
                             #                 torch seed seed + r + 10000
    # ---- data-generating mechanism -------------------------------------------
    z_dim_true=8,            # latent dimension d
    z_var=1.0,               # Var(Z_ik); Z_i ~ N(0, z_var * I_d)
    x_hidden_dim=32,         # hidden width of the MLP that generates X from Z
    props=(30, 40, 30),      # % of nodes that are onlyA / both / onlyX
    # ---- simulation grid -----------------------------------------------------
    n_levels=[100, 400, 700, 1100, 1500],      # number of nodes n
    p_levels=[100, 400, 700, 1100, 1500],      # feature dimension p
    alpha_levels=[-8, -6, -4, -2, 0],          # edge intercept alpha
    fixed_alpha=-6,
    fixed_n=700,
    fixed_p=700,
    n_reps=20,
    # ---- model (VGAE-X) and training ------------------------------------------
    z_dim_fit=8,
    h_dim_fit=128,
    lr=1e-3,
    lr_min=1e-5,             # cosine annealing floor
    l2=1e-2,
    self_weight=1.0,         # GCN self-loop weight (times node degree)
    n_iter=1200,
    log_every=50,            # evaluate at iterations 0, 49, 99, ..., n_iter-1
    # ---- evaluation ----------------------------------------------------------
    mc_samples=50,           # posterior draws for the MC estimator
    mc_every_log=False,      # MC estimator only at the final iteration
    min_exp_deg=1.0,         # rows with sum_j P_ij < this are skipped in relfro_avg
)

SMOKE = copy.deepcopy(FULL)
SMOKE.update(
    n_levels=[60, 120],
    p_levels=[40, 80],
    alpha_levels=[-6, -2],
    fixed_alpha=-6,
    fixed_n=120,
    fixed_p=80,
    n_reps=2,
    n_iter=100,
    mc_samples=10,
)

CONFIGS = {'full': FULL, 'smoke': SMOKE}

# Keys that define the science of a run.  A cached raw-results directory whose
# config.json disagrees on any of these is refused (see simulation.check_config).
SCIENTIFIC_KEYS = [
    'seed', 'z_dim_true', 'z_var', 'x_hidden_dim', 'props',
    'n_levels', 'p_levels', 'alpha_levels', 'fixed_alpha', 'fixed_n', 'fixed_p',
    'n_reps', 'z_dim_fit', 'h_dim_fit', 'lr', 'lr_min', 'l2', 'self_weight',
    'n_iter', 'log_every', 'mc_samples', 'mc_every_log', 'min_exp_deg',
]


def get_config(name, props=None):
    if name not in CONFIGS:
        raise ValueError(f'unknown config {name!r}; choose from {list(CONFIGS)}')
    cfg = copy.deepcopy(CONFIGS[name])
    cfg['name'] = name
    if props is not None:
        props = tuple(int(v) for v in props)
        if len(props) != 3 or sum(props) != 100:
            raise ValueError(f'props must be three integers summing to 100, got {props}')
        cfg['props'] = props
    cfg['props'] = tuple(cfg['props'])
    return cfg


def run_name(cfg):
    """Directory name shared by results/raw, results/summary and results/figures."""
    a, b, c = cfg['props']
    return f"{cfg['name']}_oA{a}_b{b}_oX{c}_zvar{cfg['z_var']:g}"


def log_iters(cfg):
    import numpy as np
    return np.insert(np.arange(cfg['log_every'] - 1, cfg['n_iter'], cfg['log_every']), 0, 0)
