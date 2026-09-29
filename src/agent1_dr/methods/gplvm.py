"""Bayesian GPLVM with variational latent uncertainty and inducing GP variables.

Reference: GPyTorch GPLVM SVI tutorial; Titsias & Lawrence (2010).
The loss includes KL(q(X)||p(X)) through VariationalLatentVariable.
One epoch is ceil(n / batch_size) minibatch steps, so every observation receives the same
expected number of updates at any cohort size (75 epochs = 300 steps at n = 1000).
"""

import math

import gpytorch as gp
import numpy as np
import torch
from gpytorch.models.gplvm import BayesianGPLVM, VariationalLatentVariable
from sklearn.decomposition import PCA

from .base import DRMethod


class Model(BayesianGPLVM):
    def __init__(self, initial, data_dim, inducing):
        n, q = initial.shape
        batch = torch.Size([data_dim])
        qu = gp.variational.CholeskyVariationalDistribution(inducing, batch_shape=batch)
        z = initial[torch.randperm(n)[:inducing]].unsqueeze(0).repeat(data_dim, 1, 1)
        strategy = gp.variational.VariationalStrategy(self, z, qu, learn_inducing_locations=True)
        prior = gp.priors.NormalPrior(torch.zeros_like(initial), torch.ones_like(initial))
        latent = VariationalLatentVariable(n, data_dim, q, initial, prior)
        super().__init__(latent, strategy)
        self.mean_module = gp.means.ZeroMean(batch_shape=batch)
        self.covar_module = gp.kernels.ScaleKernel(
            gp.kernels.RBFKernel(ard_num_dims=q, batch_shape=batch), batch_shape=batch
        )

    def forward(self, X):
        return gp.distributions.MultivariateNormal(self.mean_module(X), self.covar_module(X))


class Method(DRMethod):
    method_id = "gplvm"
    stochastic = True

    def default_params(self, X):
        return {"epochs": 75, "inducing_points": 32, "learning_rate": 0.03, "batch_size": 256}

    def tuning_grid(self, X):
        return [
            self.default_params(X),
            dict(self.default_params(X), inducing_points=48),
            dict(self.default_params(X), learning_rate=0.01),
        ]

    def fit_transform(self, X, seed, params):
        torch.manual_seed(seed)
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        # One global scalar preserves all feature distances up to a constant.
        Y = torch.tensor((X - X.mean(axis=0)) / max(float(X.std()), 1e-8), dtype=torch.float32)
        initial = PCA(n_components=2, svd_solver="full").fit_transform(Y.numpy())
        initial /= max(float(initial.std()), 1e-8)
        initial = torch.tensor(initial, dtype=torch.float32)
        model = Model(initial.clone(), X.shape[1], params["inducing_points"])
        likelihood = gp.likelihoods.GaussianLikelihood(batch_shape=torch.Size([X.shape[1]]))
        elbo = gp.mlls.VariationalELBO(likelihood, model, num_data=len(X))
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(likelihood.parameters()), lr=params["learning_rate"]
        )
        losses = []
        model.train()
        likelihood.train()
        steps = params["epochs"] * math.ceil(len(X) / params["batch_size"])
        for _ in range(steps):
            idx = torch.randperm(len(X))[: params["batch_size"]]
            optimizer.zero_grad()
            loss = -elbo(model(model.sample_latent_variable()[idx]), Y[idx].T).sum()
            if not torch.isfinite(loss):
                raise FloatingPointError("GPLVM ELBO became non-finite")
            losses.append(float(loss.detach()))
            loss.backward()
            optimizer.step()
        embedding = model.X.q_mu.detach().numpy()
        tail = max(len(losses) // 10, 1)
        last, previous = np.mean(losses[-tail:]), np.mean(losses[-2 * tail : -tail] or losses[-tail:])
        return embedding, {
            "loss_history": losses,
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "latent_displacement": float(np.linalg.norm(embedding - initial.numpy())),
            "optimization_steps": len(losses),
            "epochs": params["epochs"],
            # Relative change of the mean minibatch loss between the last two tenths of training.
            "loss_plateau_relative_change": float(abs(last - previous) / max(abs(previous), 1e-12)),
            "device": "cpu",
            "convergence": "fixed-budget optimization; convergence not guaranteed",
        }
