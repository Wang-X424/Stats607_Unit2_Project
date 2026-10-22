# Simulation pipeline.  Variables can be overridden, e.g.
#   make all CONFIG=smoke              # tiny end-to-end check (~30 s on a CPU)
#   make simulate PROPS="40 30 30"     # another node-type split
#   make simulate DEVICE=cuda

PY     ?= python
CONFIG ?= full
PROPS  ?= 30 40 30
DEVICE ?= auto
ARGS    = --config $(CONFIG) --props $(PROPS)

.PHONY: all simulate analyze figures test smoke clean clean-all

all: simulate analyze figures

## run all scenarios x replicates; finished scenarios are loaded from results/raw (cached)
simulate:
	$(PY) -m src.simulation $(ARGS) --device $(DEVICE)

## raw per-iteration results -> tidy tables in results/summary/<run>/
analyze:
	$(PY) -m src.analyze $(ARGS)

## tables -> figures in results/figures/<run>/
figures:
	$(PY) -m src.figures $(ARGS)

test:
	$(PY) -m pytest -q

smoke:
	$(MAKE) all CONFIG=smoke

## remove derived outputs (summaries, figures, smoke-run raw results, caches);
## keeps the expensive full-run raw results in results/raw/full_*
clean:
	rm -rf results/summary/*/ results/figures/*/ results/raw/smoke_*/
	rm -rf .pytest_cache src/__pycache__ tests/__pycache__

## also remove ALL raw simulation results (a full rerun takes hours on a GPU)
clean-all: clean
	rm -rf results/raw/*/
