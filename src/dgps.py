"""Data-generating mechanism: latent space network with node features.

    Z_i ~ N(0, z_var * I_d)
    A_ij | Z ~ Bernoulli(sigmoid(alpha + Z_i^T Z_j)),   i < j,  A symmetric, zero diagonal
    X_ik | Z ~ Bernoulli(sigmoid([W2 ReLU(W1 Z_i + b1) + b2]_k))

The node set is then split into onlyA (network observed, features missing),
both (network and features observed) and onlyX (features observed, network
missing = cold-start nodes).
"""

import math
import numpy as np


def generate_data(n, z_dim, x_dim, alpha, z_var, seed, x_hidden_dim=32):
    """Return dict with Z (n, d) float32, A (n, n) int32, X (n, p) int32."""
    rng = np.random.RandomState(seed)
    Z = rng.randn(n, z_dim).astype(np.float32) * math.sqrt(z_var)

    logits_A = alpha + Z @ Z.T
    with np.errstate(over='ignore'):
        prob_A = 1.0 / (1.0 + np.exp(-logits_A))
    U = rng.rand(n, n)
    A_upper = (U < prob_A).astype(np.int32)
    A = np.triu(A_upper, k=1)
    A = A + A.T

    rng2 = np.random.RandomState(seed + 1000)
    W1 = rng2.randn(z_dim, x_hidden_dim).astype(np.float32)
    b1 = rng2.randn(x_hidden_dim).astype(np.float32) * 0.1
    W2 = rng2.randn(x_hidden_dim, x_dim).astype(np.float32)
    b2 = rng2.randn(x_dim).astype(np.float32) * 0.1
    H = np.maximum(0, Z @ W1 + b1)
    logits_X = H @ W2 + b2
    with np.errstate(over='ignore'):
        prob_X = 1.0 / (1.0 + np.exp(-logits_X))
    X = (rng2.rand(n, x_dim) < prob_X).astype(np.int32)

    return {'Z': Z, 'A': A, 'X': X}


def true_logits(Z, alpha):
    """Theta = alpha + Z Z^T in float64 (the estimand lives on sigmoid(Theta))."""
    Z64 = np.asarray(Z, dtype=np.float64)
    return alpha + Z64 @ Z64.T


def split_nodes(n, props, seed):
    """Random partition into (onlyA, both, onlyX) index arrays; props are percentages."""
    prop_onlyA, prop_both, prop_onlyX = props
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_oA = int(np.round(n * prop_onlyA / 100))
    n_b = int(np.round(n * prop_both / 100))
    n_oX = min(int(np.round(n * prop_onlyX / 100)), n - n_oA - n_b)
    return perm[:n_oA], perm[n_oA:n_oA + n_b], perm[n_oA + n_b:n_oA + n_b + n_oX]


def expected_density(z_dim, z_var, alpha, n_mc=200_000, seed=0):
    """Monte Carlo value of E[sigmoid(alpha + Z_i^T Z_j)] for independent Z_i, Z_j."""
    rng = np.random.RandomState(seed)
    Zi = rng.randn(n_mc, z_dim) * math.sqrt(z_var)
    Zj = rng.randn(n_mc, z_dim) * math.sqrt(z_var)
    return float(np.mean(1.0 / (1.0 + np.exp(-(alpha + np.sum(Zi * Zj, axis=1))))))
