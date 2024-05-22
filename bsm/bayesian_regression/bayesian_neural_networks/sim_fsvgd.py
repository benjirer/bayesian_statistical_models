import time
from typing import Dict, Callable, Optional, Union
import os

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import random, vmap
from jaxtyping import PyTree
import tensorflow_probability.substrates.jax.distributions as tfd

from bsm.bayesian_regression.bayesian_neural_networks.deterministic_ensembles import (
    DeterministicEnsemble,
)
from bsm.bayesian_regression.bayesian_neural_networks.probabilistic_ensembles import (
    ProbabilisticEnsemble,
)
from bsm.utils.normalization import DataStats, Data
from bsm.sims import FunctionSimulator
from bsm.sims import SinusoidsSim
import wandb


def prepare_stein_kernel(h=0.2**2):
    def k(x, y):
        return jnp.exp(-jnp.sum((x - y) ** 2) / (2 * h))

    v_k = vmap(k, in_axes=(0, None), out_axes=0)
    m_k = vmap(v_k, in_axes=(None, 0), out_axes=1)

    def kernel(fs):
        kernel_matrix = m_k(fs, fs)
        return kernel_matrix

    k_x = jax.grad(k, argnums=0)

    v_k_der = vmap(k_x, in_axes=(0, None), out_axes=0)
    m_k_der = vmap(v_k_der, in_axes=(None, 0), out_axes=1)

    def kernel_derivative(fs):
        return m_k_der(fs, fs)

    return kernel, kernel_derivative


class DeterministicSimFSVGDEnsemble(DeterministicEnsemble):
    def __init__(
        self,
        prior_h: float = 1.0,
        function_simulator: FunctionSimulator = SinusoidsSim(),
        num_f_samples: int = 64,
        likelihood_std: Union[float, jnp.ndarray] = 0.2,
        likelihood_exponent: float = 1.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.prior_h = prior_h
        self.function_simulator = function_simulator
        self.num_f_samples = num_f_samples
        self.likelihood_std = (
            jnp.array([likelihood_std] * self.output_dim)
            if isinstance(likelihood_std, float) and self.output_dim > 1
            else likelihood_std
        )
        self.likelihood_exponent = likelihood_exponent
        self.stein_kernel, self.stein_kernel_derivative = prepare_stein_kernel()

    def fsim_samples(
        self, x: jnp.ndarray, key: jax.random.PRNGKey, data_stats: DataStats
    ) -> jnp.ndarray:
        x_unnormalized = vmap(
            lambda xi: self.normalizer.denormalize(xi, data_stats.inputs)
        )(x)
        f_prior = self.function_simulator.sample_function_vals(
            x=x_unnormalized, num_samples=self.num_f_samples, rng_key=key
        )
        f_prior_normalized = vmap(
            lambda fi: vmap(self.normalizer.normalize, in_axes=(0, None))(
                fi, data_stats.outputs
            )
        )(f_prior)
        return f_prior_normalized

    def prior_log_prob_gp_approx(
        self,
        predictions: jnp.ndarray,
        x: jnp.ndarray,
        key: jax.random.PRNGKey,
        data_stats: DataStats,
        eps: float = 1e-4,
    ) -> jnp.ndarray:
        f_samples = self.fsim_samples(x, key, data_stats)
        f_mean = jnp.mean(f_samples, axis=0)
        f_cov = jnp.cov(
            f_samples.reshape(-1, f_samples.shape[-1]), rowvar=False
        ) + eps * jnp.eye(f_samples.shape[-1])
        prior_gp_approx = tfd.MultivariateNormalFullCovariance(
            loc=f_mean, covariance_matrix=f_cov
        )
        prior_logprob = prior_gp_approx.log_prob(predictions)
        return prior_logprob

    def log_likelihood(
        self,
        predictions: jnp.ndarray,
        likelihood_std: jnp.ndarray,
        y_batch: jnp.ndarray,
    ) -> jnp.ndarray:
        log_prob = tfd.MultivariateNormalDiag(
            loc=predictions, scale_diag=likelihood_std
        ).log_prob(y_batch)
        return jnp.mean(log_prob, axis=-1).sum(axis=0)

    def neg_log_posterior(
        self,
        predictions: jnp.ndarray,
        x: jnp.ndarray,
        y_batch: jnp.ndarray,
        num_train_points: Union[float, int],
        key: jax.random.PRNGKey,
        data_stats: DataStats,
    ):
        nll = (
            -num_train_points
            * self.likelihood_exponent
            * self.log_likelihood(predictions, self.likelihood_std, y_batch)
        )
        prior_logprob = self.prior_log_prob_gp_approx(predictions, x, key, data_stats)
        neg_log_post = nll - prior_logprob
        return jnp.mean(neg_log_post)

    def loss(
        self,
        vmapped_params: PyTree,
        inputs: chex.Array,
        outputs: chex.Array,
        data_stats: DataStats,
    ) -> [jax.Array, Dict]:

        # setup vmap: apply to params
        apply_ensemble_one = vmap(
            self._apply_train, in_axes=(0, None, None), out_axes=0
        )

        # setup vmap: apply to inputs
        apply_ensemble = vmap(
            apply_ensemble_one, in_axes=(None, 0, None), out_axes=1, axis_name="batch"
        )

        # apply vmaps to get predictions (mean and stds)
        predicted_mean, predicted_stds = apply_ensemble(
            vmapped_params, inputs, data_stats
        )

        # normalize target outputs
        target_outputs_norm = vmap(self.normalizer.normalize, in_axes=(0, None))(
            outputs, data_stats.outputs
        )

        def neg_log_likelihood(predictions, output):
            return (
                self.neg_log_posterior(
                    predictions,
                    inputs,
                    output,
                    len(inputs),
                    random.PRNGKey(3),
                    data_stats,
                ),
                predictions,
            )

        # calculate nll and nll gradient of posterior
        (negative_log_posterior, predictions), grad_post = jax.value_and_grad(
            neg_log_likelihood, argnums=0, has_aux=True
        )(predicted_mean, target_outputs_norm)

        # calculate mse
        mse = jnp.mean((predictions - target_outputs_norm[None, ...]) ** 2)

        # calculate kernel, kernel derivative and kernal gradient
        k = self.stein_kernel(predicted_mean)
        k_x = self.stein_kernel_derivative(predicted_mean)
        grad_k = jnp.mean(k_x, axis=0)

        # calculate surrogate loss (analog to FSVGD-Paper)
        surrogate_loss = jnp.sum(
            predicted_mean
            * jax.lax.stop_gradient(jnp.einsum("ij,jkm", k, grad_post) - grad_k)
        )
        return surrogate_loss, mse


class ProbabilisticSimFSVGDEnsemble(ProbabilisticEnsemble):
    def __init__(
        self,
        prior_h: float = 1.0,
        function_simulator: FunctionSimulator = SinusoidsSim(),
        num_f_samples: int = 64,
        likelihood_std: Union[float, jnp.ndarray] = 0.2,
        likelihood_exponent: float = 1.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.prior_h = prior_h
        self.function_simulator = function_simulator
        self.num_f_samples = num_f_samples
        self.likelihood_std = (
            jnp.array([likelihood_std] * self.output_dim)
            if isinstance(likelihood_std, float) and self.output_dim > 1
            else likelihood_std
        )
        self.likelihood_exponent = likelihood_exponent
        self.stein_kernel, self.stein_kernel_derivative = prepare_stein_kernel()

    def fsim_samples(
        self, x: jnp.ndarray, key: jax.random.PRNGKey, data_stats: DataStats
    ) -> jnp.ndarray:

        d_l, d_u = 0, 10
        x = jnp.linspace(d_l, d_u, 10).reshape(-1, 1)

        x_unnormalized = vmap(
            lambda xi: self.normalizer.denormalize(xi, data_stats.inputs)
        )(x)

        f_prior = self.function_simulator.sample_function_vals(
            x=x_unnormalized, num_samples=self.num_f_samples, rng_key=key
        )
        f_prior_normalized = vmap(
            lambda fi: vmap(self.normalizer.normalize, in_axes=(0, None))(
                fi, data_stats.outputs
            )
        )(f_prior)
        return f_prior_normalized

    def prior_log_prob_gp_approx(
        self,
        predictions: jnp.ndarray,
        x: jnp.ndarray,
        key: jax.random.PRNGKey,
        data_stats: DataStats,
        eps: float = 1e-4,
    ) -> jnp.ndarray:
        # dont pass x
        f_samples = self.fsim_samples(x, key, data_stats)
        f_mean = jnp.mean(f_samples, axis=0)
        f_cov = jnp.cov(
            f_samples.reshape(-1, f_samples.shape[-1]), rowvar=False
        ) + eps * jnp.eye(f_samples.shape[-1])
        prior_gp_approx = tfd.MultivariateNormalFullCovariance(
            loc=f_mean, covariance_matrix=f_cov
        )
        prior_logprob = prior_gp_approx.log_prob(predictions)
        return prior_logprob

    def log_likelihood(
        self,
        predictions: jnp.ndarray,
        likelihood_std: jnp.ndarray,
        y_batch: jnp.ndarray,
    ) -> jnp.ndarray:
        log_prob = tfd.MultivariateNormalDiag(
            loc=predictions, scale_diag=likelihood_std
        ).log_prob(y_batch)
        return jnp.mean(log_prob, axis=-1).sum(axis=0)

    def neg_log_posterior(
        self,
        predictions: jnp.ndarray,
        x: jnp.ndarray,
        y_batch: jnp.ndarray,
        num_train_points: Union[float, int],
        key: jax.random.PRNGKey,
        data_stats: DataStats,
    ):
        nll = (
            -num_train_points
            * self.likelihood_exponent
            * self.log_likelihood(predictions, self.likelihood_std, y_batch)
        )
        prior_logprob = self.prior_log_prob_gp_approx(predictions, x, key, data_stats)
        neg_log_post = nll - prior_logprob
        return jnp.mean(neg_log_post)

    def loss(
        self,
        vmapped_params: PyTree,
        inputs: chex.Array,
        outputs: chex.Array,
        data_stats: DataStats,
    ) -> [jax.Array, Dict]:

        # adapt "apply_train" to output model
        def apply_fn(params: PyTree, x: jax.Array) -> jax.Array:
            x = self.normalizer.normalize(x, data_stats.inputs)
            out = self.model.apply({"params": params}, x)
            return out

        # setup vmap: apply to params
        apply_ensemble_one = vmap(apply_fn, in_axes=(0, None), out_axes=0)

        # setup vmap: apply to inputs
        apply_ensemble = vmap(
            apply_ensemble_one, in_axes=(None, 0), out_axes=1, axis_name="batch"
        )

        # apply vmaps to get predictions (model)
        f_raw = apply_ensemble(vmapped_params, inputs)

        # normalize target outputs
        target_outputs_norm = vmap(self.normalizer.normalize, in_axes=(0, None))(
            outputs, data_stats.outputs
        )

        def neg_log_likelihood(predictions, output):
            mu, log_sigma = jnp.split(predictions, 2, axis=-1)
            sig = nn.softplus(log_sigma)
            # pass sigma (use GP prior on log sigma)
            # sig = nn.softplus(sig)
            # sig = jnp.clip(sig, self.sig_min, self.sig_max)
            return (
                self.neg_log_posterior(
                    mu, inputs, output, len(inputs), random.PRNGKey(3), data_stats
                ),
                mu,
            )

        # calculate nll and nll gradient of posterior
        (negative_log_posterior, mu), grad_post = jax.value_and_grad(
            neg_log_likelihood, argnums=0, has_aux=True
        )(f_raw, target_outputs_norm)

        # calculate mse
        mse = jnp.mean((mu - target_outputs_norm[None, ...]) ** 2)

        # calculate kernel, kernel derivative and kernal gradient
        k = self.stein_kernel(f_raw)
        k_x = self.stein_kernel_derivative(f_raw)
        grad_k = jnp.mean(k_x, axis=0)

        # calculate surrogate loss (analog to FSVGD-Paper)
        surrogate_loss = jnp.sum(
            f_raw * jax.lax.stop_gradient(jnp.einsum("ij,jkm", k, grad_post) - grad_k)
        )
        return surrogate_loss, mse


if __name__ == "__main__":
    key = random.PRNGKey(0)
    logging_wandb = False
    input_dim = 1
    output_dim = 2

    noise_level = 0.1
    d_l, d_u = 0, 10
    xs = jnp.linspace(d_l, d_u, 10).reshape(-1, 1)
    ys = jnp.concatenate([jnp.sin(xs), jnp.cos(xs)], axis=1)
    ys = ys * (1 + noise_level * random.normal(key=random.PRNGKey(0), shape=ys.shape))
    data_std = noise_level * jnp.ones(shape=(output_dim,))

    data = Data(inputs=xs, outputs=ys)

    sim = SinusoidsSim(output_size=output_dim)

    num_particles = 10
    model = ProbabilisticSimFSVGDEnsemble(
        input_dim=input_dim,
        output_dim=output_dim,
        features=[64, 64, 64],
        num_particles=num_particles,
        num_f_samples=256,
        eval_frequency=500,
        output_stds=data_std,
        logging_wandb=logging_wandb,
        function_simulator=sim,
        likelihood_std=0.05,
    )
    model_state = model.init(model.key)
    start_time = time.time()
    print("Starting with training")
    if logging_wandb:
        wandb.init(
            project="Pendulum",
            group="test group",
        )

    model_state = model.fit_model(
        data=data, num_training_steps=1000, model_state=model_state
    )
    print(f"Training time: {time.time() - start_time:.2f} seconds")

    test_xs = jnp.linspace(-3, 13, 1000).reshape(-1, 1)
    test_ys = jnp.concatenate([jnp.sin(test_xs), jnp.cos(test_xs)], axis=1)

    test_ys_noisy = test_ys * (
        1 + noise_level * random.normal(key=random.PRNGKey(0), shape=test_ys.shape)
    )

    test_stds = noise_level * jnp.ones(shape=test_ys.shape)

    f_dist, y_dist = vmap(model.posterior, in_axes=(0, None))(test_xs, model_state)

    pred_mean = f_dist.mean()
    eps_std = f_dist.stddev()
    al_std = jnp.mean(y_dist.aleatoric_stds, axis=1)
    total_std = jnp.sqrt(jnp.square(eps_std) + jnp.square(al_std))

    output_dir = "plts_sim_fsvgd_prob_new"
    os.makedirs(output_dir, exist_ok=True)

    for j in range(output_dim):
        plt.figure()
        plt.scatter(xs.reshape(-1), ys[:, j], label="Data", color="red")
        for i in range(num_particles):
            plt.plot(
                test_xs,
                f_dist.particle_means[:, i, j],
                label="NN prediction",
                color="black",
                alpha=0.3,
            )
        plt.plot(test_xs, f_dist.mean()[..., j], label="Mean", color="blue")
        plt.fill_between(
            test_xs.reshape(-1),
            (pred_mean[..., j] - 2 * total_std[..., j]).reshape(-1),
            (pred_mean[..., j] + 2 * total_std[..., j]).reshape(-1),
            label=r"$2\sigma$",
            alpha=0.3,
            color="blue",
        )
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.plot(test_xs.reshape(-1), test_ys[:, j], label="True", color="green")
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

        plt.savefig(os.path.join(output_dir, f"plot_{j}.png"))
        plt.close()

    for j in range(output_dim):
        plt.figure()
        for i in range(num_particles):
            plt.plot(
                test_xs,
                f_dist.particle_means[:, i, j],
                label="NN prediction",
                color="black",
                alpha=0.3,
            )
        plt.plot(test_xs, f_dist.mean()[..., j], label="Mean", color="blue")
        plt.fill_between(
            test_xs.reshape(-1),
            (pred_mean[..., j] - 2 * total_std[..., j]).reshape(-1),
            (pred_mean[..., j] + 2 * total_std[..., j]).reshape(-1),
            label=r"$2\sigma$",
            alpha=0.3,
            color="blue",
        )
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.plot(test_xs.reshape(-1), test_ys[:, j], label="True", color="green")
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

        plt.savefig(os.path.join(output_dir, f"plot_{j+output_dim}.png"))
        plt.close()
