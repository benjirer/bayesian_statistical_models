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

from bsm.bayesian_regression.bayesian_neural_networks.deterministic_ensembles import (
    DeterministicEnsemble,
)
from bsm.bayesian_regression.bayesian_neural_networks.probabilistic_ensembles import (
    ProbabilisticEnsemble,
)
from bsm.bayesian_regression.bayesian_neural_networks.bnn import BNNState, NO_EVAL_VALUE
from bsm.utils.normalization import DataStats, Data
from bsm.sims import FunctionSimulator
from bsm.sims import SinusoidsSim
import wandb



class SimPriorDeterministicEnsemble(DeterministicEnsemble):
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
        self.domain = self.function_simulator.domain

    def eval_loss(self,
                vmapped_params: chex.Array,
                inputs: chex.Array,
                outputs: chex.Array,
                data_stats: DataStats) -> chex.Array:
        apply_ensemble_one = vmap(self._apply_train, in_axes=(0, None, None), out_axes=0)
        apply_ensemble = vmap(apply_ensemble_one, in_axes=(None, 0, None), out_axes=1, axis_name='batch')
        predicted_outputs, predicted_stds = apply_ensemble(vmapped_params, inputs, data_stats)

        target_outputs_norm = vmap(self.normalizer.normalize, in_axes=(0, None))(outputs, data_stats.outputs)
        negative_log_likelihood = super()._neg_log_posterior(predicted_outputs, predicted_stds, target_outputs_norm)
        mse = jnp.mean((predicted_outputs - target_outputs_norm[None, ...]) ** 2)
        return negative_log_likelihood, mse

    def evaluate_model(self,
                       vmapped_params: PyTree,
                       eval_data: Data,
                       data_stats: DataStats) -> OrderedDict:
        eval_nll, eval_mse = self.eval_loss(vmapped_params, eval_data.inputs,
                                       eval_data.outputs, data_stats)
        eval_stats = OrderedDict(eval_nll=eval_nll, eval_mse=eval_mse)
        return eval_stats

    def fsim_samples(
        self, data_stacked: jnp.ndarray, key: jax.random.PRNGKey, data_stats: DataStats
    ) -> jnp.ndarray:

        f_prior = self.function_simulator.sample_function_vals(
            x=data_stacked, num_samples=self.num_f_samples, rng_key=key
        )
        # vmap over measurement points and then over samples
        f_prior_normalized = jax.vmap(jax.vmap(lambda y: self.normalizer._normalize_y(y,
                                                                                 data_stats.outputs)))(f_prior)
        return f_prior_normalized

    def get_score_per_dim(self, fsamples, predictions, eps=1e-4):
        # fsamples.ndim = 2, predictions.ndim = 1
        f_mean = jnp.mean(fsamples, axis=0)
        f_cov = jnp.cov(fsamples, rowvar=False, bias=True)
        prior_gp_approx = MultivariateNormalFullCovariance(
            loc=f_mean, covariance_matrix=f_cov + eps * jnp.eye(f_cov.shape[0])
        )
        # TODO: Normalization wrt the number of points in the prior helps in training.
        #  -> Should check if this is generally necessary
        return prior_gp_approx.log_prob(predictions) / (self.num_measurement_points + self.batch_size)

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
        # calculate log prob per dim
        log_prob_per_dim = jax.vmap(
            self.get_score_per_dim, in_axes=(2, 1), out_axes=0
        )(f_samples, predictions)
        log_prob = log_prob_per_dim.sum(0)
        return log_prob

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

        mu_inputs = mu[:-self.num_measurement_points, :]
        sigma_inputs = nn.softplus(sigma[:-self.num_measurement_points, :])

        nll = jax.vmap(self._nll)(mu_inputs, sigma_inputs, outputs)

        prior_logprob = self.prior_log_prob_gp_approx(mu, data_stacked, key, data_stats)

        return self.likelihood_exponent * nll.mean() -  (1.0 / num_data_points) * prior_logprob.mean()

    def loss(
        self,
        vmapped_params: PyTree,
        inputs: chex.Array,
        outputs: chex.Array,
        data_stats: DataStats,
        key: jax.random.PRNGKey,
        num_data_points: int,
    ) -> [jax.Array, Dict]:
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
        neg_log_posterior = jax.vmap(self._neg_log_posterior, in_axes=(0, 0, None, None, None, None, None))
        negative_log_posterior = neg_log_posterior(
            mu,
            sigma,
            data_stacked,
            target_outputs_norm,
            num_data_points,
            key,
            data_stats,
        )

        # split predictions to seperate input and measurement points
        mu_inputs = mu[:, :-self.num_measurement_points, :]

        # calculate mse
        mse = jnp.mean((mu_inputs - target_outputs_norm[None, ...]) ** 2)
        return negative_log_posterior.mean(0), mse

    @partial(jax.jit, static_argnums=0)
    def step_jit(self,
                 opt_state: optax.OptState,
                 vmapped_params: chex.PRNGKey,
                 inputs: chex.Array,
                 outputs: chex.Array,
                 data_stats: DataStats,
                 key: jax.random.PRNGKey,
                 num_data_points: int,
                 ) -> (optax.OptState, PyTree, OrderedDict):
        (loss, mse), grads = jax.value_and_grad(self.loss, has_aux=True)(
            vmapped_params, inputs, outputs, data_stats, key, num_data_points,
        )
        updates, opt_state = self.tx.update(grads, opt_state, vmapped_params)
        vmapped_params = optax.apply_updates(vmapped_params, updates)
        statiscs = OrderedDict(nll=loss, mse=mse)
        return opt_state, vmapped_params, statiscs

    def _train_model(self,
                     num_training_steps: int,
                     model_state: BNNState,
                     data_stats: DataStats,
                     train_data: Data,
                     eval_data: Data,
                     rng: jax.random.PRNGKey,
                     ) -> BNNState:

        vmapped_params = model_state.vmapped_params
        opt_state = self.tx.init(vmapped_params)
        # convert to numpy array which are cheaper for indexing
        num_data_points = train_data.inputs.shape[0]
        train_data = Data(inputs=np.asarray(train_data.inputs), outputs=np.asarray(train_data.outputs))
        eval_data = Data(inputs=np.asarray(eval_data.inputs), outputs=np.asarray(eval_data.outputs))
        best_statistics = OrderedDict(eval_nll=NO_EVAL_VALUE)
        evaluated_model = False
        best_params = vmapped_params
        for train_step in range(num_training_steps):
            data_rng, sim_rng, rng = jax.random.split(rng, 3)
            data_batch = self.sample_batch(train_data, self.batch_size, data_rng)
            opt_state, vmapped_params, statistics = self.step_jit(opt_state, vmapped_params, data_batch.inputs,
                                                                  data_batch.outputs, data_stats, sim_rng,
                                                                  num_data_points)
            if train_step % self.evaluation_frequency == 0:
                evaluated_model = True
                eval_rng, rng = jax.random.split(rng, 2)
                eval_data_batch = self.sample_batch(eval_data, self.eval_batch_size, eval_rng)
                eval_statistics = self.evaluate_model(vmapped_params=vmapped_params, eval_data=eval_data_batch,
                                                      data_stats=data_stats)
                statistics.update(eval_statistics)
                if best_statistics['eval_nll'] > statistics['eval_nll']:
                    best_statistics = OrderedDict(eval_nll=statistics['eval_nll'])
                    best_params = vmapped_params
            if self.logging_wandb and train_step % self.logging_frequency == 0:
                wandb.log(statistics)

        if self.return_best_model and evaluated_model:
            final_params = best_params
        else:
            final_params = vmapped_params

        calibrate_alpha = jnp.ones(self.output_dim)
        new_model_state = BNNState(data_stats=data_stats, vmapped_params=final_params,
                                   calibration_alpha=calibrate_alpha)
        return new_model_state

if __name__ == "__main__":
    key = random.PRNGKey(0)
    logging_wandb = False
    test_simulator = True
    input_dim = 1
    output_dim = 1
    sim = SinusoidsSim(input_size=input_dim, output_size=output_dim)
    num_train_points = 2
    num_test_points = 1000

    noise_level = 0.001
    d_l, d_u = -5, 5
    xs = jnp.linspace(d_l, d_u, num_train_points + num_test_points).reshape(-1, 1)
    # ys = jnp.concatenate([jnp.sin(xs), jnp.cos(xs)], axis=1)
    sample_key, key = jax.random.split(key, 2)
    ys_co = sim.sample_function_vals(xs, num_samples=1, rng_key=sample_key)
    ys = jnp.squeeze(ys_co, axis=0)
    test_xs = xs
    test_ys = ys
    ys = ys + noise_level * random.normal(key=random.PRNGKey(0), shape=ys.shape)
    test_ys_noisy = ys
    data_std = noise_level * jnp.ones(shape=(output_dim,))
    data_sample_key, key = jax.random.split(key, 2)
    indx = jax.random.randint(key=data_sample_key, shape=(num_train_points, ),
                              minval=0, maxval=num_train_points + num_test_points)

    xs, ys = xs[indx], ys[indx]


    data = Data(inputs=xs, outputs=ys)

    # if test_simulator:
    #     num_samples = 10
    #     x_test = sim.domain.sample_uniformly(key=key, sample_shape=(100,))
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
    model = SimPriorDeterministicEnsemble(
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
        data=data, num_training_steps=10_000, model_state=model_state
    )
    print(f"Training time: {time.time() - start_time:.2f} seconds")

    test_stds = noise_level * jnp.ones(shape=test_ys.shape)
    # test_xs = xs
    # test_ys_noisy = ys
    # test_ys = ys
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
