"""DGP verification: generated data have the properties the model assumes."""

import numpy as np

from src import dgps


def test_adjacency_is_symmetric_binary_without_self_loops():
    d = dgps.generate_data(n=150, z_dim=8, x_dim=30, alpha=-2, z_var=1.0, seed=1)
    A = d['A']
    assert A.shape == (150, 150)
    assert np.array_equal(A, A.T)
    assert np.all(np.diag(A) == 0)
    assert set(np.unique(A)) <= {0, 1}
    assert set(np.unique(d['X'])) <= {0, 1}
    assert d['X'].shape == (150, 30)


def test_edge_density_matches_model_expectation():
    """Empirical density over many seeds is close to E[sigmoid(alpha + Z_i'Z_j)]."""
    for alpha in (-6, -2, 0):
        dens = [dgps.generate_data(300, 8, 10, alpha, 1.0, seed=s)['A'].sum() / (300 * 299)
                for s in range(5)]
        expected = dgps.expected_density(8, 1.0, alpha)
        assert abs(np.mean(dens) - expected) < 0.15 * expected + 0.003, (alpha, np.mean(dens), expected)


def test_latent_variance_follows_z_var():
    for z_var in (1.0, 4.0):
        Z = dgps.generate_data(2000, 8, 5, -2, z_var, seed=3)['Z']
        assert abs(Z.var() - z_var) < 0.05 * z_var


def test_same_seed_same_data_different_seed_different_data():
    a = dgps.generate_data(100, 8, 20, -4, 1.0, seed=7)
    b = dgps.generate_data(100, 8, 20, -4, 1.0, seed=7)
    c = dgps.generate_data(100, 8, 20, -4, 1.0, seed=8)
    for k in ('Z', 'A', 'X'):
        assert np.array_equal(a[k], b[k])
    assert not np.array_equal(a['Z'], c['Z'])


def test_split_is_a_disjoint_partition_with_requested_sizes():
    oA, b, oX = dgps.split_nodes(200, (30, 40, 30), seed=0)
    assert (len(oA), len(b), len(oX)) == (60, 80, 60)
    allidx = np.concatenate([oA, b, oX])
    assert len(np.unique(allidx)) == 200
