"""The method under study (VGAE-X) and the two edge-probability estimators.

VGAE-X is a variational graph autoencoder for partially observed graphs.  Nodes
with an observed network (onlyA, both) are encoded with a two-layer GCN; onlyX
nodes (features only) are encoded with an MLP that shares the GCN weights.  The
decoder is P(A_ij = 1 | z) = sigmoid(z_i^T z_j + b) plus an MLP for X.

Estimators of the edge-probability matrix compared in the study
    plug-in : P_hat_ij = sigmoid(mu_i^T mu_j + b)
    MC      : P_hat_ij = E_q[sigmoid(z_i^T z_j + b)]  (posterior predictive),
              approximated with S posterior draws.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Input preparation                                                           #
# --------------------------------------------------------------------------- #
def prepare_inputs(A, X, idx_oA, idx_b, idx_oX, self_weight):
    """Reorder nodes as [onlyA, both, onlyX] and build observed inputs.

    Returns a dict of numpy arrays; only observed parts of A and X are used.
    """
    num_train = np.concatenate([idx_oA, idx_b, idx_oX])
    n = len(num_train)
    id_onlyA = np.arange(len(idx_oA))
    id_both = np.arange(len(idx_oA), len(idx_oA) + len(idx_b))
    id_onlyX = np.arange(len(idx_oA) + len(idx_b), n)
    id_A = np.concatenate([id_onlyA, id_both])
    id_X = np.concatenate([id_both, id_onlyX])

    A_all = A[np.ix_(num_train, num_train)]
    X_obs = X[num_train][id_X]
    A_obs = A_all[np.ix_(id_A, id_A)]
    x_dim = X.shape[1]

    edge_prob = (A_obs.sum() / 2.0) / (A_obs.shape[0] * (A_obs.shape[0] - 1) / 2.0 + 1e-10)
    logit_edge_prob = float(np.log(edge_prob / (1.0 - edge_prob + 1e-10) + 1e-10))

    # features of onlyA nodes: average of observed neighbours' features
    # (column 0 = global feature mean, used when a node has no 'both' neighbour)
    X_obs_mean = np.mean(X_obs, axis=0)
    X_more = np.concatenate((X_obs_mean.reshape(1, x_dim), X_obs[:len(id_both), :]))
    A_avg = A_obs[np.ix_(id_onlyA, id_both)]
    D = np.sum(A_avg, axis=1)
    no_link = 1 * (D == 0)
    A_avg = np.concatenate((no_link.reshape(len(id_onlyA), 1), A_avg), axis=1)
    D = 1.0 / np.sum(A_avg, axis=1)
    X_onlyA = np.matmul(np.diag(D), A_avg) @ X_more
    X_fill = np.concatenate((X_onlyA, X_obs))

    # symmetric-normalised adjacency with degree-proportional self loops
    node_degree = np.sum(A_obs, axis=1)
    node_degree = np.where(node_degree == 0, 1, node_degree)
    A_tilde = A_obs + np.diag(self_weight * node_degree)
    d_inv_sqrt = 1.0 / np.power(np.sum(A_tilde, axis=1), 0.5)
    AHat = np.matmul(np.matmul(np.diag(d_inv_sqrt), A_tilde), np.diag(d_inv_sqrt))

    col = np.tile(np.arange(len(id_A)), (len(id_A), 1))
    mask_upper = np.transpose(col) < col

    return dict(num_train=num_train, id_onlyA=id_onlyA, id_both=id_both,
                id_onlyX=id_onlyX, id_A=id_A, id_X=id_X,
                X_obs=X_obs, A_obs=A_obs, X_fill=X_fill, AHat=AHat,
                mask_upper=mask_upper, logit_edge_prob=logit_edge_prob, x_dim=x_dim)


# --------------------------------------------------------------------------- #
# Model                                                                       #
# --------------------------------------------------------------------------- #
class VAE(torch.nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, logit_edge_prob):
        super().__init__()
        xi = lambda i, o: torch.randn(i, o) * (1.0 / (i / 2.0) ** 0.5)
        self.PX_W1 = torch.nn.Parameter(xi(z_dim, h_dim))
        self.PX_b1 = torch.nn.Parameter(torch.zeros(h_dim))
        self.PX_W2 = torch.nn.Parameter(xi(h_dim, x_dim))
        self.PX_b2 = torch.nn.Parameter(torch.zeros(x_dim))
        self.PA_dens = torch.nn.Parameter(torch.tensor(logit_edge_prob, dtype=torch.float32))
        self.Q_W1 = torch.nn.Parameter(xi(x_dim, h_dim))
        self.Q_b1 = torch.nn.Parameter(torch.zeros(h_dim))
        self.Q_W2_mu = torch.nn.Parameter(xi(h_dim, z_dim))
        self.Q_b2_mu = torch.nn.Parameter(torch.zeros(z_dim))
        self.Q_W2_sigma = torch.nn.Parameter(xi(h_dim, z_dim))
        self.Q_b2_sigma = torch.nn.Parameter(torch.zeros(z_dim))

    def PX(self, z):
        logits = F.relu(z @ self.PX_W1 + self.PX_b1) @ self.PX_W2 + self.PX_b2
        return torch.sigmoid(logits), logits

    def PA(self, z):
        logits = z @ z.t() + self.PA_dens
        return torch.sigmoid(logits), logits

    def Q(self, X_fill, AHat, n_A, n_onlyX):
        X_A, X_oX = X_fill[:n_A], X_fill[n_A:n_A + n_onlyX]
        h_oX = F.relu(X_oX @ self.Q_W1 + self.Q_b1)
        z_oX_mu = h_oX @ self.Q_W2_mu + self.Q_b2_mu
        z_oX_lv = h_oX @ self.Q_W2_sigma + self.Q_b2_sigma
        h_A = F.relu((AHat @ X_A) @ self.Q_W1 + self.Q_b1)
        z_A_mu = (AHat @ h_A) @ self.Q_W2_mu + self.Q_b2_mu
        z_A_lv = (AHat @ h_A) @ self.Q_W2_sigma + self.Q_b2_sigma
        return torch.cat([z_A_mu, z_oX_mu], 0), torch.cat([z_A_lv, z_oX_lv], 0)

    @staticmethod
    def sample_z(mu, lv):
        return mu + torch.exp(lv / 2.0) * torch.randn_like(mu)   # lv = log sigma^2

    def l2_reg(self):
        return (self.PX_W1.pow(2).sum() + self.PX_W2.pow(2).sum() +
                self.Q_W1.pow(2).sum() + self.Q_W2_mu.pow(2).sum() +
                self.Q_W2_sigma.pow(2).sum()) / 2


def fit_vgae(inp, cfg, device, log_iters, callback=None):
    """Maximise the ELBO (single reparameterised draw per iteration, M = 1).

    ``callback(it, loss, model, z_mu, z_logvar)`` is called in eval/no-grad mode
    at every iteration in ``log_iters``.  Returns the fitted model.
    """
    tau2 = cfg['z_var']                      # prior N(0, tau2 I) matched to the DGP
    n_A, n_oX, n_oA = len(inp['id_A']), len(inp['id_onlyX']), len(inp['id_onlyA'])
    n_X = len(inp['id_X'])

    t = lambda a, dt=torch.float32: torch.tensor(a, dtype=dt, device=device)
    t_Xo, t_Ao = t(inp['X_obs']), t(inp['A_obs'])
    t_Xf, t_AH = t(inp['X_fill']), t(inp['AHat'])
    t_mask = t(inp['mask_upper'], torch.bool)

    model = VAE(inp['x_dim'], cfg['h_dim_fit'], cfg['z_dim_fit'], inp['logit_edge_prob']).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg['n_iter'], eta_min=cfg['lr_min'])
    log_set = set(int(i) for i in log_iters)

    for it in range(cfg['n_iter']):
        model.train(); opt.zero_grad()
        z_mu, z_lv = model.Q(t_Xf, t_AH, n_A, n_oX)
        z_s = model.sample_z(z_mu, z_lv)
        _, lgX = model.PX(z_s)
        _, lgA = model.PA(z_s)
        lgX_obs = lgX[n_oA:n_oA + n_X]
        lgA_obs = lgA[:n_A, :n_A]
        loss = (F.binary_cross_entropy_with_logits(lgA_obs[t_mask], t_Ao[t_mask], reduction='sum')
                + F.binary_cross_entropy_with_logits(lgX_obs, t_Xo, reduction='sum')
                + 0.5 * torch.sum((torch.exp(z_lv) + z_mu.pow(2)) / tau2 - 1.0 - z_lv + np.log(tau2))
                + model.l2_reg() * cfg['l2'])
        loss.backward()
        opt.step()
        sched.step()

        if callback is not None and it in log_set:
            model.eval()
            with torch.no_grad():
                zm, zlv = model.Q(t_Xf, t_AH, n_A, n_oX)
                callback(it, loss.item(), model, zm, zlv)
    return model


# --------------------------------------------------------------------------- #
# Estimators of the edge-probability matrix                                   #
# --------------------------------------------------------------------------- #
def plugin_logits(model, z_mu):
    """Theta_hat = mu mu^T + b (float64).  Equals E_q[theta] for off-diagonal pairs."""
    mu = z_mu.double()
    return mu @ mu.t() + model.PA_dens.double()


def plugin_log_probs(model, z_mu):
    th = plugin_logits(model, z_mu)
    return F.logsigmoid(th), F.logsigmoid(-th)


def mc_log_probs(model, z_mu, z_lv, n_samples):
    """log E_q[sigmoid(theta)] and log E_q[1 - sigmoid(theta)] via a running logsumexp."""
    mu, lv = z_mu.double(), z_lv.double()
    b = model.PA_dens.double()
    acc1 = acc0 = None
    for _ in range(n_samples):
        z = mu + torch.exp(lv / 2.0) * torch.randn_like(mu)
        th = z @ z.t() + b
        l1, l0 = F.logsigmoid(th), F.logsigmoid(-th)
        acc1 = l1 if acc1 is None else torch.logaddexp(acc1, l1)
        acc0 = l0 if acc0 is None else torch.logaddexp(acc0, l0)
    return acc1 - math.log(n_samples), acc0 - math.log(n_samples)
