"""Import raw results produced by the earlier single-file script
(simulation_sensitivity_fix1_prob.py) on the cluster.

    python -m src.import_legacy PATH/TO/sensitivity_results_prob30_zvar1 --props 30 40 30

The legacy script used the same DGP, split, model, training and evaluation code
and the same data seeds (verified bit-for-bit against src/ when torch is seeded
identically), but it did not seed torch.  The pickles have the same format, so
they are copied unchanged into results/raw/<run>/ together with a config.json
that records this provenance.
"""

import argparse
import json
import os
import pickle
import shutil

from .config import SCIENTIFIC_KEYS, get_config
from .simulation import build_scenarios, raw_dir


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('src_dir')
    ap.add_argument('--config', default='full')
    ap.add_argument('--props', nargs=3, type=int, default=None)
    args = ap.parse_args(argv)
    cfg = get_config(args.config, args.props)
    out = raw_dir(cfg)
    os.makedirs(out, exist_ok=True)

    found, missing = 0, []
    for scen in build_scenarios(cfg):
        src = os.path.join(args.src_dir, f"sens_{scen['label']}.pkl")
        if not os.path.exists(src):
            missing.append(scen['label'])
            continue
        with open(src, 'rb') as f:
            reps = pickle.load(f)                      # validates the file is readable
        if len(reps) != cfg['n_reps']:
            raise ValueError(f'{src}: {len(reps)} replicates, expected {cfg["n_reps"]}')
        shutil.copy2(src, out)
        txt = src.replace('.pkl', '.txt')
        if os.path.exists(txt):
            shutil.copy2(txt, out)
        found += 1

    current = {k: (list(cfg[k]) if isinstance(cfg[k], tuple) else cfg[k]) for k in SCIENTIFIC_KEYS}
    current['provenance'] = ('imported from the legacy single-file script run on the Great Lakes '
                             'cluster; same data seeds, torch not seeded')
    with open(os.path.join(out, 'config.json'), 'w') as f:
        json.dump(current, f, indent=2)
    print(f'imported {found} scenarios into {out}; missing: {missing}')


if __name__ == '__main__':
    main()
