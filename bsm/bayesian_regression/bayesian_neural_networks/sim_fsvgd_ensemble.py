import time
from typing import Dict, Callable, Optional, Union
import os
from functools import partial
import chex
import optax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from jax import random, vmap
from jax.scipy.stats import norm
from jaxtyping import PyTree
from collections import OrderedDict
from distrax import MultivariateNormalFullCovariance

from bsm.bayesian_regression.bayesian_neural_networks.sim_ensemble import (
    SimPriorDeterministicEnsemble,
    SimPriorProbabilisticEnsemble,
)
from bsm.bayesian_regression.bayesian_neural_networks.fsvgd_ensemble import (
    prepare_stein_kernel,
)

from bsm.bayesian_regression.bayesian_neural_networks.bnn import BNNState, NO_EVAL_VALUE
from bsm.utils.normalization import DataStats, Data
from bsm.sims import FunctionSimulator
from bsm.sims import SinusoidsSim, GaussianProcessSim
import wandb


class SimFSVGDDeterministicEnsemble(SimPriorDeterministicEnsemble):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.stein_kernel, self.stein_kernel_derivative = prepare_stein_kernel()

    def loss(
        self,
        vmapped_params: PyTree,
        inputs: chex.Array,
        outputs: chex.Array,
        data_stats: DataStats,
        key: jax.random.PRNGKey,
        num_data_points: int,
    ) -> [jax.Array, Dict]:

        # add measurement points from domain
        measurement_points_key, key = jax.random.split(key, 2)
        measurement_points = self.domain.sample_uniformly(
            key=measurement_points_key, sample_shape=(self.num_measurement_points,)
        )
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

        # normalize target outputs
        target_outputs_norm = vmap(self.normalizer.normalize, in_axes=(0, None))(
            outputs, data_stats.outputs
        )

        # vmap over the ensemble axis
        def neg_log_posterior(
            mu, sigma, data, target_outputs_norm, num_data_points, key, data_stats
        ):
            return (
                jax.vmap(
                    self._neg_log_posterior,
                    in_axes=(0, 0, None, None, None, None, None),
                )(
                    mu,
                    sigma,
                    data,
                    target_outputs_norm,
                    num_data_points,
                    key,
                    data_stats,
                )
            ).mean(0)

        negative_log_posterior, grad_post = jax.value_and_grad(neg_log_posterior)(
            mu,
            sigma,
            data_stacked,
            target_outputs_norm,
            num_data_points,
            key,
            data_stats,
        )

        # split predictions to seperate input and measurement points
        mu_inputs = mu[:, : -self.num_measurement_points, :]

        # calculate mse
        mse = jnp.mean((mu_inputs - target_outputs_norm[None, ...]) ** 2)

        # fsvgd loss
        k = self.stein_kernel(mu)
        k_x = self.stein_kernel_derivative(mu)
        grad_k = jnp.mean(k_x, axis=0)

        surrogate_loss = jnp.sum(
            mu * jax.lax.stop_gradient(jnp.einsum("ij,jkm", k, grad_post) - grad_k)
        )

        return surrogate_loss, mse


class SimFSVGDProbabilisticEnsemble(SimPriorProbabilisticEnsemble):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.stein_kernel, self.stein_kernel_derivative = prepare_stein_kernel()

    def prior_log_prob_gp_approx(
        self,
        prediction: jnp.ndarray,
        data_stacked: jnp.ndarray,
        key: jax.random.PRNGKey,
        data_stats: DataStats,
        simulator: FunctionSimulator,
        eps: float = 1e-4,
    ) -> jnp.ndarray:

        # calculate prior
        f_samples = self.fsim_samples(data_stacked, key, data_stats, simulator)

        # calculate log prob per dim
        log_prob_per_dim = jax.vmap(self.get_score_per_dim, in_axes=(2, 1), out_axes=0)(
            f_samples, prediction
        )

        return log_prob_per_dim.sum(0)

    def _neg_log_posterior(
        self,
        mu: jnp.ndarray,
        sigma: jnp.ndarray,
        data_stacked: jnp.ndarray,
        outputs: jnp.ndarray,
        num_data_points: int,
        key: jax.random.PRNGKey,
        data_stats: DataStats,
    ):
        mu_inputs = mu[: -self.num_measurement_points, :]
        sigma_inputs = sigma[: -self.num_measurement_points, :]

        nll = jax.vmap(self._nll)(mu_inputs, sigma_inputs, outputs)

        prior_logprob_mu = self.prior_log_prob_gp_approx(
            mu, data_stacked, key, data_stats, simulator=self.function_simulator
        )

        prior_logprob_sigma = self.prior_log_prob_gp_approx(
            sigma,
            data_stacked,
            key,
            data_stats,
            simulator=GaussianProcessSim(
                input_size=self.input_dim, output_size=self.output_dim
            ),
        )

        return (
            self.likelihood_exponent * nll.mean()
            - (1.0 / num_data_points) * (prior_logprob_mu).mean()
        )

    def loss(
        self,
        vmapped_params: PyTree,
        inputs: chex.Array,
        outputs: chex.Array,
        data_stats: DataStats,
        key: jax.random.PRNGKey,
        num_data_points: int,
    ) -> [jax.Array, Dict]:

        # add measurement points from domain
        measurement_points_key, key = jax.random.split(key, 2)
        measurement_points = self.domain.sample_uniformly(
            key=measurement_points_key, sample_shape=(self.num_measurement_points,)
        )
        data_stacked = jnp.concatenate([inputs, measurement_points], axis=0)

        def apply_fn(
            params: PyTree, x: jnp.ndarray, data_stats: DataStats
        ) -> jnp.ndarray:
            x = self.normalizer.normalize(x, data_stats.inputs)
            out = self.model.apply({"params": params}, x)
            return out

        # setup vmap
        apply_ensemble_one = vmap(apply_fn, in_axes=(0, None, None), out_axes=0)
        apply_ensemble = vmap(
            apply_ensemble_one, in_axes=(None, 0, None), out_axes=1, axis_name="batch"
        )

        # apply to get predictions
        predictions = apply_ensemble(vmapped_params, data_stacked, data_stats)

        # normalize target outputs
        target_outputs_norm = vmap(self.normalizer.normalize, in_axes=(0, None))(
            outputs, data_stats.outputs
        )

        # vmap over the ensemble axis
        def neg_log_posterior(
            predictions, data, target_outputs_norm, num_data_points, key, data_stats
        ):
            mu, log_sigma = jnp.split(predictions, 2, axis=-1)
            sigma = nn.softplus(log_sigma)

            return (
                jax.vmap(
                    self._neg_log_posterior,
                    in_axes=(0, 0, None, None, None, None, None),
                )(
                    mu,
                    sigma,
                    data,
                    target_outputs_norm,
                    num_data_points,
                    key,
                    data_stats,
                )
            ).mean(0)

        negative_log_posterior, grad_post = jax.value_and_grad(neg_log_posterior)(
            predictions,
            data_stacked,
            target_outputs_norm,
            num_data_points,
            key,
            data_stats,
        )

        # split predictions to seperate input and measurement points
        mu, sigma = jnp.split(predictions, 2, axis=-1)
        mu_inputs = mu[:, : -self.num_measurement_points, :]

        # calculate mse
        mse = jnp.mean((mu_inputs - target_outputs_norm[None, ...]) ** 2)

        # fsvgd loss
        k = self.stein_kernel(predictions)
        k_x = self.stein_kernel_derivative(predictions)
        grad_k = jnp.mean(k_x, axis=0)

        surrogate_loss = jnp.sum(
            predictions
            * jax.lax.stop_gradient(jnp.einsum("ij,jkm", k, grad_post) - grad_k)
        )

        return surrogate_loss, mse


if __name__ == "__main__":
    key = random.PRNGKey(0)
    logging_wandb = False
    output_dir = "plts_sim_fsvgd_prob_new"
    input_dim = 1
    output_dim = 1
    sim = SinusoidsSim(input_size=input_dim, output_size=output_dim)
    num_train_points = 2
    num_test_points = 1000
    num_training_steps = 150_000

    noise_level = 0.001
    d_l, d_u = -5, 5
    xs = jnp.linspace(d_l, d_u, num_train_points + num_test_points).reshape(-1, 1)
    sample_key, key = jax.random.split(key, 2)
    ys = jnp.squeeze(
        sim.sample_function_vals(xs, num_samples=1, rng_key=sample_key), axis=0
    )
    test_xs = xs
    test_ys = ys
    ys = ys + noise_level * random.normal(key=random.PRNGKey(0), shape=ys.shape)
    data_std = noise_level * jnp.ones(shape=(output_dim,))

    # split data
    data_sample_key, key = jax.random.split(key, 2)
    indx = jax.random.randint(
        key=data_sample_key,
        shape=(num_train_points,),
        minval=0,
        maxval=num_train_points + num_test_points,
    )
    xs, ys = xs[indx], ys[indx]
    data = Data(inputs=xs, outputs=ys)

    num_particles = 10
    model = SimFSVGDProbabilisticEnsemble(
        input_dim=input_dim,
        output_dim=output_dim,
        features=[64, 64, 64],
        num_particles=num_particles,
        num_f_samples=1024,
        eval_frequency=500,
        output_stds=data_std,
        logging_wandb=logging_wandb,
        function_simulator=sim,
        batch_size=8,
        num_measurement_points=32,
        calibration=False,
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
        data=data, num_training_steps=num_training_steps, model_state=model_state
    )
    print(f"Training time: {time.time() - start_time:.2f} seconds")

    f_dist, y_dist = vmap(model.posterior, in_axes=(0, None))(test_xs, model_state)
    pred_mean = f_dist.mean()
    eps_std = f_dist.stddev()
    al_std = jnp.mean(y_dist.aleatoric_stds, axis=1)
    total_std = jnp.sqrt(jnp.square(eps_std) + jnp.square(al_std))

    # plot
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
        plt.title(
            f"num_train_points: {num_train_points}, num_training_steps: {num_training_steps}"
        )

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
        plt.title(
            f"num_train_points: {num_train_points}, num_training_steps: {num_training_steps}"
        )

        plt.savefig(os.path.join(output_dir, f"plot_{j+output_dim}.png"))
        plt.close()
