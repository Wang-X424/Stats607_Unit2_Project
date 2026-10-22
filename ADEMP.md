# ADEMP: recovery of edge probabilities for cold-start nodes with VGAE-X

This is an original simulation study (Option B). It comes from my research on
link prediction in *partially observed* graphs. In these graphs some nodes have
an observed network but no features, some have both, and some ("cold-start"
nodes) have features but no observed edges at all. VGAE-X is a variational graph
autoencoder that predicts the missing links of the cold-start nodes from their
features. Earlier evaluations used AUC/AP against a sampled adjacency matrix. The
attainable value of AUC/AP depends on the edge density itself, so those metrics
cannot separate "the model got worse" from "the problem got harder". In a
simulation the true edge-probability matrix is known, so this study evaluates
the estimated probabilities directly.

## A — Aims

1. Quantify how accurately VGAE-X recovers the edge-probability matrix between
   cold-start nodes (onlyX) and network-observed nodes (onlyA ∪ both) as a function of
   the number of nodes $n$, the feature dimension $p$, and the edge intercept $\alpha$,
   which controls the network density.
2. Compare two estimators of the edge probabilities computed from the *same*
   fitted variational posterior: the plug-in estimator, which the method's paper
   currently uses, and the Monte Carlo posterior-predictive estimator.


## D — Data-generating mechanism

For each replicate:

| step | model |
|---|---|
| latent positions | $Z_i \overset{iid}{\sim} N(0, \sigma_Z^2 I_d)$, $d = 8$, $\sigma_Z^2 = 1$ |
| network | $A_{ij} \mid Z \sim \text{Bernoulli}\big(\sigma(\alpha + Z_i^\top Z_j)\big)$ independently for $i<j$; $A$ symmetric, zero diagonal |
| features | $X_{ik} \mid Z \sim \text{Bernoulli}\big(\sigma([W_2\,\text{ReLU}(W_1 Z_i + b_1) + b_2]_k)\big)$, hidden width 32; $W_1, W_2$ entries $\sim N(0,1)$, $b_1, b_2$ entries $\sim N(0, 0.1^2)$ (drawn once per replicate) |
| node split | random partition into onlyA / both / onlyX = 30% / 40% / 30% |
| observed data | $A$ among onlyA ∪ both; $X$ for both ∪ onlyX. The edges of onlyX nodes are hidden. |

**Factors.** $n \in \{100, 400, 700, 1100, 1500\}$,
$p \in \{100, 400, 700, 1100, 1500\}$, $\alpha \in \{-8, -6, -4, -2, 0\}$.
The intercept $\alpha$ sets the density. These are Monte Carlo values of
$E[\sigma(\alpha + Z_i^\top Z_j)]$, with the expected degree computed at $n = 700$:

| $\alpha$ | −8 | −6 | −4 | −2 | 0 |
|---|---|---|---|---|---|
| edge density | 1.1% | 3.8% | 11.1% | 26.5% | 50.0% |
| expected degree ($n=700$) | 8 | 26 | 77 | 186 | 350 |

The full factorial design would have 125 cells. Instead I use three
one-factor-fixed slices through the centre point ($n = 700$, $p = 700$, $\alpha = -6$):

- slice 1: $n \times p$ at $\alpha = -6$ (25 cells)
- slice 2: $p \times \alpha$ at $n = 700$ (25 cells)
- slice 3: $n \times \alpha$ at $p = 700$ (25 cells)

After removing duplicates there are **61 unique scenarios**, each with
**R = 20 replicates** (1220 model fits).


## E — Estimands

The estimand of a replicate is the true edge-probability matrix restricted to a block
of node pairs, $P_{ij} = \sigma(\alpha + Z_i^\top Z_j)$. It is a function of that
replicate's realised $Z$, so this is a *recovery* target that changes from replicate
to replicate, not a fixed parameter.

- **Primary block:** onlyX × A, i.e. cold-start nodes against all network-observed
  nodes (onlyA ∪ both). This is the link-prediction task the method exists for.
- **Secondary blocks (stored, not plotted):** onlyX × onlyA, onlyX × both, and
  A × A (off-diagonal pairs).

$Z$ is identified only up to an orthogonal rotation, but $P$ (and $\Theta = \alpha + ZZ^\top$)
is invariant to it, so no alignment is needed.

## M — Methods

**Model fit (VGAE-X), identical in every scenario:**

- Encoder: 2-layer GCN on the observed adjacency for onlyA ∪ both nodes. The adjacency
  is normalised symmetrically, with self-loop weight equal to the node's degree. The
  features of onlyA nodes are imputed as the average over their "both" neighbours. An
  MLP that shares the GCN weights encodes onlyX nodes.
- Output: $q(z_i) = N(\mu_i, \text{diag}(\sigma_i^2))$.
- Decoder: $\sigma(z_i^\top z_j + b)$ for edges plus an MLP for features.
- Prior: $N(0, \tau^2 I)$ with $\tau^2 = \sigma_Z^2$, matched to the DGP.
- Latent dimension 8 (equal to the truth). Hidden width 128.
- Objective: ELBO with one reparameterised draw per iteration (M = 1), plus L2
  penalty $10^{-2}$ on the weights.
- Optimiser: Adam, lr $10^{-3}$, cosine-annealed to $10^{-5}$ over 1200 full-batch
  iterations.

**Estimators of $P$ (the methods compared):**

| name | formula |
|---|---|
| plug-in (`pi`) | $\hat P_{ij} = \sigma(\mu_i^\top \mu_j + \hat b)$ |
| MC posterior predictive (`mc`) | $\hat P_{ij} = \frac1S \sum_{s=1}^S \sigma(z_i^{(s)\top} z_j^{(s)} + \hat b)$, $z^{(s)} \sim q$, $S = 50$ |

**Reference predictors** are built into the metrics: the all-zero predictor has
relative Frobenius error 1, and the constant-density predictor $\bar p$ has relative
KL 1. The MC estimator is computed only at the final iteration. The plug-in estimator
is computed every 50 iterations, which gives the convergence diagnostics.

## P — Performance measures

Each replicate is evaluated on a block $S = R \times C$ with $m = |S|$ pairs, at the
**final iteration** (iteration 1200). The final iteration is used rather than the best,
because picking the best iteration would select on the true $P$.

| measure | definition | why |
|---|---|---|
| relFro (all) | $\lVert \hat P - P\rVert_F / \lVert P\rVert_F$ | accuracy of the probability values; dominated by high-probability pairs, which are the likely edges |
| relFro (avg) | $\frac{1}{\lvert R^*\rvert}\sum_{i \in R^*} \lVert \hat P_{i\cdot} - P_{i\cdot}\rVert_2 / \lVert P_{i\cdot}\rVert_2$, where $R^* = \{i : \sum_j P_{ij} \ge 1\}$ | gives every cold-start node equal weight; rows with expected degree < 1 have a near-zero denominator and are skipped (the fraction kept is stored as `frac_rows`) |
| relKL | $\frac{\frac1m\sum \text{KL}(\text{Bern}(P_{ij})\,\Vert\,\text{Bern}(\hat P_{ij}))}{\frac1m\sum \text{KL}(\text{Bern}(P_{ij})\,\Vert\,\text{Bern}(\bar p))}$, with $\bar p = \frac1m \sum P_{ij}$ | calibration over the whole probability range, on the same likelihood scale as training; 1 = no better than knowing the density |

KL is computed in log space with float64, so it stays finite for $P_{ij} \approx 10^{-10}$.
The following are also stored but not plotted: RMSE, the centred relative Frobenius error
$\lVert \hat P - P\rVert_F / \lVert P - \bar p\rVert_F$, raw KL, and the logit-scale
error of the plug-in estimator.

**Across replicates:** for each cell I report the mean, SD, Monte Carlo SE
($\text{SD}/\sqrt{R}$), and a normal 95% CI for the mean. Estimators are also compared
*within* replicate through the paired difference MC − plug-in (mean, MCSE, and the
share of replicates where MC is lower). Replicates whose final value is `nan` are
dropped and counted; nothing is imputed.

