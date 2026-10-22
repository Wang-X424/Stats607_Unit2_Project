# Simulation study: recovering edge probabilities for cold-start nodes with VGAE-X

VGAE-X is a variational graph autoencoder for link prediction in partially
observed graphs. It predicts the edges of *cold-start* nodes, which have node
features but no observed edges. This study simulates graphs from a latent-space
model with known edge probabilities $P_{ij} = \sigma(\alpha + Z_i^\top Z_j)$ and asks:

1. How accurately does VGAE-X recover $P$ for cold-start nodes as the number of
   nodes $n$, the feature dimension $p$ and the network density (intercept $\alpha$)
   vary?
2. Is the Monte Carlo posterior-predictive estimator of $P$ better than the
   plug-in estimator $\sigma(\mu_i^\top\mu_j + b)$?

The full design (61 scenarios × 20 replicates), estimands and performance measures
are in [ADEMP.md](ADEMP.md). The critical discussion is in [ANALYSIS.md](ANALYSIS.md).

## Key findings

VGAE-X recovered the cold-start edge probabilities better than trivial reference
predictors only in dense networks ($\alpha \ge -2$, density ≈ 26% or more). In sparser
networks, its error was larger than that of predicting no edges. The cold-start error was
lowest early in training and then grew while the fit to the observed network kept
improving, and it increased with the feature dimension. The MC posterior-predictive
estimator beat the plug-in estimator in about 75% of replicates, but only by a small
margin. Details are in [ANALYSIS.md](ANALYSIS.md).

## Setup

Python ≥ 3.10.

```bash
python -m venv .venv && source .venv/bin/activate    
pip install -r requirements.txt
```

A CUDA GPU is optional. It is only needed to re-run the full simulation in a
reasonable time.

## Running

```bash
make all       # simulate -> analyze -> figures (full design, node split 30/40/30)
make test      # pytest suite (≈ 5 s)
make smoke     # the whole pipeline on a tiny design (≈ 30 s on a CPU)
make clean     # remove summaries, figures and smoke outputs (keeps full raw results)
make clean-all # also delete all raw simulation results
```

Individual steps: `make simulate`, `make analyze`, `make figures`. Options:
`CONFIG=smoke|full`, `PROPS="30 40 30"` (onlyA/both/onlyX percentages),
`DEVICE=auto|cpu|cuda`, `PY=python3`.

The raw results of the full run are committed in `results/raw/full_oA30_b40_oX30_zvar1/`.
`make all` therefore finds every scenario cached, skips the simulation, and only
rebuilds the tables and figures (under a minute). To re-run the simulation from
scratch, use `make clean-all && make all`.

### Estimated runtime

| what | hardware | time |
|---|---|---|
| `make all` with cached raw results | laptop CPU | < 1 min |
| `make smoke` | laptop CPU | ~ 30 s |
| `make test` | laptop CPU | ~ 5 s |
| full simulation (1220 fits × 1200 iterations) | one NVIDIA V100 | ~ 2 h |

The full simulation was run on the University of Michigan Great Lakes cluster
with the earlier single-file version of this code and then imported with
`python -m src.import_legacy`. For people who wanna reproduce our work, they
should redo the full simulation on their own device.

## Reproducibility

- **Seeds.** Replicate $r$ uses data seed $42 + r$, split seed $42 + r + 100$, and torch
  seed $42 + r + 10000$ (initialisation, reparameterisation and MC draws). The seed
  depends on the replicate, not on the scenario, so replicate $r$ of every scenario
  with the same $n$ shares the same $Z$ draw (common random numbers). This reduces the
  Monte Carlo variance of comparisons between conditions.
- **Configuration.** All settings are in `src/config.py`. `results/raw/<run>/config.json`
  records them, and a cached run with different settings is refused.
- **Re-running.** `make clean-all && make all` regenerates everything.

## Layout

```
src/config.py      all parameters (full and smoke configurations)
src/dgps.py        data-generating mechanism and node split
src/methods.py     VGAE-X model, training, plug-in and MC estimators
src/metrics.py     performance measures and Monte Carlo summaries
src/simulation.py  scenario grid, seeding, caching/resume, raw output
src/analyze.py     raw output -> tidy CSV tables (final iteration, MCSE, paired differences, convergence)
src/figures.py     publication and diagnostic figures
src/import_legacy.py  imports raw results produced on the cluster by the earlier single-file script
tests/             DGP, metric, method, reproducibility and resume tests
results/raw/       per-scenario raw results (.pkl, .txt), config.json, simulation.log
results/summary/   tables produced by `make analyze`
results/figures/   figures produced by `make figures`
```

Outputs in `results/figures/<run>/`:

- `fig1_main_mc`: main result for the MC estimator, faceted by the three design slices.
  Shows the mean and 95% CI.
- `fig1_main_pi`: the same figure for the plug-in estimator.
- `fig2_convergence`: diagnostic. Training trajectories in five stress scenarios.
- `fig3_mc_vs_plugin`: diagnostic. Paired per-replicate comparison of the two estimators.
