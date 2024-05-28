import time
from typing import Dict, Callable, Optional, Union
import os

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from jax import random, vmap
from jax.scipy.stats import norm
from jaxtyping import PyTree
import tensorflow_probability.substrates.jax.distributions as tfd
from tensorflow_probability.substrates import jax as tfp

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
        num_measurement_points: int = 16,
        num_f_samples: int = 64,
        likelihood_exponent: float = 1.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.prior_h = prior_h
        self.function_simulator = function_simulator
        self.num_measurement_points = num_measurement_points
        self.num_f_samples = num_f_samples
        self.likelihood_exponent = likelihood_exponent
        self.stein_kernel, self.stein_kernel_derivative = prepare_stein_kernel()
        self.domain = self.function_simulator.domain

    def fsim_samples(
        self, data_stacked: jnp.ndarray, key: jax.random.PRNGKey, data_stats: DataStats
    ) -> jnp.ndarray:
        random_key, subkey = random.split(key)

        x_unnormalized = vmap(
            lambda xi: self.normalizer.denormalize(xi, data_stats.inputs)
        )(data_stacked)

        f_prior = self.function_simulator.sample_function_vals(
            x=x_unnormalized, num_samples=self.num_f_samples, rng_key=subkey
        )

        f_prior_normalized = self.normalizer._normalize_y(f_prior, data_stats.outputs)
        return f_prior_normalized

    def prior_log_prob_gp_approx(
        self,
        predictions: jnp.ndarray,
        data_stacked: jnp.ndarray,
        key: jax.random.PRNGKey,
        data_stats: DataStats,
        eps: float = 1e-4,
    ) -> jnp.ndarray:

        # calculate prior
        f_samples = self.fsim_samples(data_stacked, key, data_stats)
        f_mean = jnp.mean(f_samples, axis=0).T
        f_cov = jnp.swapaxes(
            tfp.stats.covariance(f_samples, sample_axis=0, event_axis=1), 0, -1
        )
        prior_gp_approx = tfd.MultivariateNormalFullCovariance(
            loc=f_mean, covariance_matrix=f_cov + eps * jnp.eye(data_stacked.shape[0])
        )
        prior_logprob = jnp.sum(
            prior_gp_approx.log_prob(predictions.swapaxes(-1, -2)), axis=(-2, -1)
        )

        return prior_logprob

    # def prior_log_prob(self, mu, data_stacked, key, data_stats):
    #     # predictions shape: (len(data_stacked), 2*output_dim)
    #     prior_logprob = self.prior_log_prob_gp_approx(mu, data_stacked, key, data_stats)
    #     return prior_logprob

    def _neg_log_posterior(
        self,
        mu: jnp.ndarray,
        sigma: jnp.ndarray,
        data_stacked: jnp.ndarray,
        input_till_idx: int,
        outputs: jnp.ndarray,
        num_train_points: Union[float, int],
        key: jax.random.PRNGKey,
        data_stats: DataStats,
    ):

        # calculate nll for input data only
        # split predictions
        mu_inputs = mu[:, :input_till_idx, :]
        sigma_inputs = nn.softplus(sigma[:, :input_till_idx, :])

        nll = jax.vmap(jax.vmap(self._nll), in_axes=(0, 0, None))(
            mu_inputs, sigma_inputs, outputs
        )

        # # calculate prior lof_prob for each particle
        # # predictions shape: (num_particles, len(data_stacked), 2*output_dim)
        # prior_logprob = vmap(
        #     self.prior_log_prob, in_axes=(0, None, None, None)
        # )(mu, data_stacked, key, data_stats)

        prior_logprob = self.prior_log_prob_gp_approx(mu, data_stacked, key, data_stats)

        return nll.mean() - prior_logprob

    def loss(
        self,
        vmapped_params: PyTree,
        inputs: chex.Array,
        outputs: chex.Array,
        data_stats: DataStats,
    ) -> [jax.Array, Dict]:

        # data points: input and measurement points
        input_till_idx = inputs.shape[0]
        measurement_points = self.domain.sample_uniformly(
            key=random.PRNGKey(0), sample_shape=self.num_measurement_points
        )
        # measurement_points = jnp.linspace(-5, 15, self.num_measurement_points).reshape(
        #     -1, 1
        # )
        data_stacked = jnp.concatenate([inputs, measurement_points], axis=0)

        # setup vmap
        apply_ensemble_one = vmap(
            self._apply_train, in_axes=(0, None, None), out_axes=0
        )
        apply_ensemble = vmap(
            apply_ensemble_one, in_axes=(None, 0, None), out_axes=1, axis_name="batch"
        )

        # apply to get predictions
        mu, sigma = apply_ensemble(vmapped_params, data_stacked, data_stats)

        # split predictions to seperate input and measurement points
        mu_inputs = mu[:, :input_till_idx, :]

        # normalize target outputs
        target_outputs_norm = vmap(self.normalizer.normalize, in_axes=(0, None))(
            outputs, data_stats.outputs
        )

        # calculate nll and nll gradient of posterior
        negative_log_posterior, grad_post = jax.value_and_grad(self._neg_log_posterior)(
            mu,
            sigma,
            data_stacked,
            input_till_idx,
            target_outputs_norm,
            len(inputs),
            random.PRNGKey(3),
            data_stats,
        )

        # calculate mse
        mse = jnp.mean((mu_inputs - target_outputs_norm[None, ...]) ** 2)

        # calculate kernel, kernel derivative and kernal gradient
        k = self.stein_kernel(mu)
        k_x = self.stein_kernel_derivative(mu)
        grad_k = jnp.mean(k_x, axis=0)

        # calculate surrogate loss (analog to FSVGD-Paper)
        surrogate_loss = jnp.sum(
            mu * jax.lax.stop_gradient(jnp.einsum("ij,jkm", k, grad_post) - grad_k)
        )
        return surrogate_loss, mse


class ProbabilisticSimFSVGDEnsemble(ProbabilisticEnsemble):
    def __init__(
        self,
        prior_h: float = 1.0,
        function_simulator: FunctionSimulator = SinusoidsSim(),
        num_measurement_points: int = 16,
        num_f_samples: int = 64,
        likelihood_exponent: float = 1.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.prior_h = prior_h
        self.function_simulator = function_simulator
        self.num_measurement_points = num_measurement_points
        self.num_f_samples = num_f_samples
        self.likelihood_exponent = likelihood_exponent
        self.stein_kernel, self.stein_kernel_derivative = prepare_stein_kernel()
        self.domain = self.function_simulator.domain

    def fsim_samples(
        self, data_stacked: jnp.ndarray, key: jax.random.PRNGKey, data_stats: DataStats
    ) -> jnp.ndarray:
        random_key, subkey = random.split(key)

        x_unnormalized = vmap(
            lambda xi: self.normalizer.denormalize(xi, data_stats.inputs)
        )(data_stacked)

        f_prior = self.function_simulator.sample_function_vals(
            x=x_unnormalized, num_samples=self.num_f_samples, rng_key=subkey
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
        data_stacked: jnp.ndarray,
        key: jax.random.PRNGKey,
        data_stats: DataStats,
        eps: float = 1e-4,
    ) -> jnp.ndarray:

        # calculate prior
        f_samples = self.fsim_samples(data_stacked, key, data_stats)
        f_mean = jnp.mean(f_samples, axis=0).T
        f_cov = jnp.swapaxes(
            tfp.stats.covariance(f_samples, sample_axis=0, event_axis=1), 0, -1
        )
        prior_gp_approx = tfd.MultivariateNormalFullCovariance(
            loc=f_mean, covariance_matrix=f_cov + eps * jnp.eye(data_stacked.shape[0])
        )
        prior_logprob = jnp.sum(prior_gp_approx.log_prob(predictions.swapaxes(-1, -2)))

        return prior_logprob

    def _neg_log_posterior(
        self,
        predictions: jnp.ndarray,
        data_stacked: jnp.ndarray,
        input_till_idx: int,
        outputs: jnp.ndarray,
        num_train_points: Union[float, int],
        key: jax.random.PRNGKey,
        data_stats: DataStats,
    ):
        # calculate _neg_log_posterior for each particle
        # predictions shape: (num_particles, len(data_stacked), 2*output_dim)

        def _neg_log_posterior_single(predictions, data_stacked, key, data_stats):
            # predictions shape: (len(data_stacked), 2*output_dim)

            # split predictions
            mu, log_sigma = jnp.split(predictions, 2, axis=-1)
            sigma = nn.softplus(log_sigma)

            # split pred_inputs
            pred_inputs = predictions[:input_till_idx, :]
            mu_inputs, log_sigma_inputs = jnp.split(pred_inputs, 2, axis=-1)
            sigma_inputs = nn.softplus(log_sigma_inputs)

            # calculate nll
            nll = (
                -num_train_points
                * self.likelihood_exponent
                * norm.logpdf(outputs, loc=mu_inputs, scale=sigma_inputs)
            )

            prior_logprob = self.prior_log_prob_gp_approx(
                mu, data_stacked, key, data_stats
            )

            neg_log_post = nll.mean() - prior_logprob
            return neg_log_post

        neg_log_post_final = vmap(
            _neg_log_posterior_single, in_axes=(0, None, None, None)
        )(predictions, data_stacked, key, data_stats)

        return jnp.mean(neg_log_post_final)

    def loss(
        self,
        vmapped_params: PyTree,
        inputs: chex.Array,
        outputs: chex.Array,
        data_stats: DataStats,
    ) -> [jax.Array, Dict]:

        # data points: input and measurement points
        input_till_idx = inputs.shape[0]
        measurement_points = self.domain.sample_uniformly(
            key=random.PRNGKey(0), sample_shape=self.num_measurement_points
        )
        data_stacked = jnp.concatenate([inputs, measurement_points], axis=0)

        # adapt "apply_train" to output model
        def apply_fn(params: PyTree, x: jax.Array) -> jax.Array:
            x = self.normalizer.normalize(x, data_stats.inputs)
            out = self.model.apply({"params": params}, x)
            return out

        # setup vmap
        apply_ensemble_one = vmap(apply_fn, in_axes=(0, None), out_axes=0)
        apply_ensemble = vmap(
            apply_ensemble_one, in_axes=(None, 0), out_axes=1, axis_name="batch"
        )

        # apply to get predictions
        f_raw = apply_ensemble(vmapped_params, data_stacked)

        # split predictions to seperate input and measurement points
        # f_raw_inputs = apply_ensemble_one(vmapped_params, inputs)
        f_raw_inputs = f_raw[:, :input_till_idx, :]
        mu_inputs, log_sigma_inputs = jnp.split(f_raw_inputs, 2, axis=-1)

        # normalize target outputs
        target_outputs_norm = vmap(self.normalizer.normalize, in_axes=(0, None))(
            outputs, data_stats.outputs
        )

        def neg_log_likelihood(predictions, output):
            mu, log_sigma = jnp.split(predictions, 2, axis=-1)
            return (
                self._neg_log_posterior(
                    predictions,
                    data_stacked,
                    input_till_idx,
                    output,
                    len(inputs),
                    random.PRNGKey(3),
                    data_stats,
                ),
                mu,
            )

        # calculate nll and nll gradient of posterior
        (negative_log_posterior, mu), grad_post = jax.value_and_grad(
            neg_log_likelihood, argnums=0, has_aux=True
        )(f_raw, target_outputs_norm)

        # calculate mse
        mse = jnp.mean((mu_inputs - target_outputs_norm[None, ...]) ** 2)

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
    test_simulator = True
    input_dim = 1
    output_dim = 2

    noise_level = 0.1
    d_l, d_u = 0, 10
    xs = jnp.linspace(d_l, d_u, 20).reshape(-1, 1)
    ys = jnp.concatenate([jnp.sin(xs), jnp.cos(xs)], axis=1)
    ys = ys * (1 + noise_level * random.normal(key=random.PRNGKey(0), shape=ys.shape))
    data_std = noise_level * jnp.ones(shape=(output_dim,))

    data = Data(inputs=xs, outputs=ys)

    sim = SinusoidsSim(input_size=input_dim, output_size=output_dim)

    # if test_simulator:
    #     num_samples = 10
    #     # x_test = sim.domain.sample_uniformly(key=key, sample_shape=(100,))
    #     x_test = jnp.linspace(-3, 13, 1000).reshape(-1, 1)

    #     y_test = sim.sample_function_vals(x_test, num_samples=num_samples, rng_key=key)

    #     fig, axs = plt.subplots(1, output_dim, figsize=(12, 6))

    #     for i in range(num_samples):
    #         for j in range(output_dim):
    #             axs[j].plot(x_test, y_test[i, :, j], label=f"Sample {i+1}")

    #     for ax in axs:
    #         ax.legend()
    #         ax.set_xlabel("x")
    #         ax.set_ylabel("f(x)")

    #     plt.savefig("sampled_functions.png")

    num_particles = 10
    model = DeterministicSimFSVGDEnsemble(
        input_dim=input_dim,
        output_dim=output_dim,
        features=[64, 64, 64],
        num_particles=num_particles,
        num_f_samples=256,
        eval_frequency=500,
        output_stds=data_std,
        logging_wandb=logging_wandb,
        function_simulator=sim,
        # likelihood_std=0.05,
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
