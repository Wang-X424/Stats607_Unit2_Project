# Critical analysis (Option B: original simulation)

## Results in brief

The simulation covered 61 scenarios with 20 replicates each. Its central finding was
negative: VGAE-X recovered edge probabilities for cold-start nodes poorly unless the
network was dense.

**Density.** For the MC estimator, the relative Frobenius error on onlyX × A was below 1
(better than predicting no edges) in all 18 scenarios with $\alpha \in \{-2, 0\}$ and in
none with $\alpha \le -4$. Relative KL was below 1 (better than predicting the known
density) in only 6 of 61 scenarios, all with $\alpha \in \{-2, 0\}$. At the centre
($n = p = 700$), relFro fell from 3.22 at $\alpha = -8$ to 0.30 at $\alpha = 0$. Over
the same range, relKL fell from 4.20 to 0.84 (Figure 1).

**Sample size.** Larger $n$ helped, most of all at small $n$. At $p = 700$,
$\alpha = -6$, relKL fell from 11.8 at $n = 100$ to 2.15 at $n = 1500$.

**Feature dimension.** Contrary to H1, more features *hurt*. At $n = 700$, $\alpha = -6$,
relFro rose from 1.35 at $p = 100$ to 2.53 at $p = 1500$, and relKL rose from 1.25 to 4.30.

**Network-observed nodes.** The same fits recovered the A × A block well: plug-in relKL
was below 1 in 53 of 61 scenarios. So the failure was specific to the cold-start path.

**Training dynamics.** Figure 2 explains the pattern. The A × A error decreased
throughout training, while the onlyX × A error was lowest at iteration ≤ 99 in 60 of 61
scenarios (for relKL) and then rose to a plateau. The median final relKL was 2.2 times
its minimum along the trajectory.

**Estimators.** MC was lower than plug-in in about 75% of replicate-scenario pairs. Its
95% CI favoured MC in 45 cells and plug-in in 15. The plug-in cells were all sparse
($\alpha \le -6$) with $p \le 700$. The differences were small (about 0.01 in
relFro) except at $n = 100$. The onlyX posterior log-variances settled between −6 and −2, a
posterior SD of about 0.05–0.33 against a prior SD of 1. The posterior was therefore too
concentrated for posterior averaging to correct the error.

## Why this design

The DGP is the latent space model that the decoder assumes. This was deliberate: with a
well-specified model, errors come from estimation (encoder, variational approximation,
optimisation) and not from misspecification. With hindsight, this choice made the
negative result more informative. The cold-start failure cannot be blamed on model error.

The levels of $n$ (100–1500) cover the range where full-batch training on dense
$n \times n$ matrices is cheap. The levels of $p$ (100–1500) are typical of bag-of-words
features. The levels of $\alpha$ give densities from 1% to 50%. I ran three slices
through a common centre instead of the 125-cell factorial; this halved the cost and
answered the marginal questions. As a result, interactions are only visible along the
slices.

With R = 20 replicates, the Monte Carlo SE of a cell mean was about 1% of the mean
(median SD/mean = 0.04). Every pattern above is many MCSEs wide. Shared seeds
(common random numbers across scenarios with the same $n$) made comparisons across $p$
and $\alpha$ even sharper.

## Fairness

The two estimators come from the same fitted posterior in every replicate. The
comparison is therefore paired, and neither estimator had a tuning advantage. All
hyperparameters were given and identical across scenarios. Replicates were scored at the final
iteration, because selecting the best iteration would use the true $P$. All metrics are
relative to fixed reference predictors.

The probability-scale metrics are not neutral across density. A logit error
$\delta\theta$ costs about $p(1-p)\,\delta\theta$ in probability and
$\tfrac12 p(1-p)\,\delta\theta^2$ in KL, so the metrics penalise dense graphs *more*.
Dense graphs still came out best, so the density conclusion is conservative. The size
of the density effect, however, is not a clean model-quality comparison.

## Limitations and scenarios not included

**Optimistic setting.** The model was correctly specified: the latent dimension, the link
function and the prior variance all matched the truth. Real graphs add degree
heterogeneity, which an inner-product decoder cannot represent. Missingness was
completely at random, with one 30/40/30 split. In practice, cold-start nodes are often
newer and lower-degree.

**Density grid.** The grid stops at $\alpha = 0$, so it cannot distinguish a peak at
density 0.5 from monotone improvement with density.

**Mechanism.** The explanation for the effect of $p$ and for the degradation during
training is a hypothesis that was not tested here. The loss uses `reduction='sum'`, so
the feature-reconstruction term grows with $n \cdot p$ while the edge term grows with
$n_A^2$. A larger $p$ therefore shifts the ELBO towards explaining $X$. In addition, the
shared encoder weights are fitted to graph-smoothed inputs ($\hat A X$), while onlyX nodes
pass raw $X$ through the same weights. As training specialises those weights, the
cold-start path drifts.


## What the results mean for practice and theory

For practice, the ELBO is not a proxy for cold-start accuracy. Optimising it improved
the observed network and degraded the cold-start predictions. A validation criterion
computed on held-out nodes treated as cold start is needed, for example for early
stopping. Without one, reported cold-start probabilities in sparse graphs can be worse
than a constant.

The choice between estimators matters much less than the choice of stopping point. I
would report MC for coherence with the probabilistic model and note that the difference
from plug-in is small.

An earlier AUC-based version of this study is not included in this repository. Its
evaluation differed, so its numbers are not directly comparable. In that version, the
same model reached reasonably good AUC for cold-start nodes. This does not contradict the
results above. AUC measures only how pairs are ranked, while the probability metrics also
measure whether the predicted values are right. Taken together, the two suggest that
VGAE-X ranks candidate links for cold-start nodes sensibly but assigns them poorly
calibrated probabilities. If that reading holds, a post-hoc recalibration fitted on held-out
nodes could repair much of the damage without changing the model. There is a
methodological lesson as well: an AUC-only evaluation would have hidden this failure,
because AUC ignores calibration and its attainable value changes with density.

## Next steps

1. Test the mechanism by reweighting the two likelihood terms (mean reduction), by
   aligning the encoder paths, and by early stopping on a held-out cold-start criterion.
2. Add an $\alpha$-invariant logit-scale metric and a symmetric $\alpha$ grid to settle
   the density question.
3. Compute AUC/AP and their oracle values (using the true $P$) inside this same design,
   to test directly whether ranking is preserved while calibration fails.
4. Add misspecified conditions: wrong latent dimension and degree heterogeneity.

## Implementation challenges

The hardest part was the metric design, not the code. Three points took the longest:
seeing that AUC is confounded with density; seeing that probability-scale errors carry
their own $p(1-p)$ weighting; and separating the Monte Carlo expectation inside the ELBO
(one draw per step suffices for a stochastic gradient) from the expectation needed for
prediction (one draw would only add noise).

Numerically, true probabilities near $10^{-9}$ required computing KL from log-sigmoids
in float64.


## How confident am I?

I am confident in the qualitative patterns. They exceed Monte Carlo error by a wide
margin and are consistent across all three slices and across both estimators.

The thing limit that confidence:

1. **The reporting rule.** Because cold-start error is lowest early in training, the
   final-iteration values describe the converged model, not the best achievable one. A
   different stopping rule would change the levels.

