from abc import ABC, abstractmethod
from typing import Generic
from jax import vmap
import jax.numpy as jnp
import chex

from bsm.bayesian_regression.bayesian_regression_model import BayesianRegressionModel
from bsm.utils.type_aliases import ModelState, StatisticalModelOutput, StatisticalModelState
from bsm.utils.normalization import Data


class StatisticalModel(ABC, Generic[ModelState]):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 model: BayesianRegressionModel[ModelState]):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model = model

    def __call__(self,
                 input: chex.Array,
                 statistical_model_state: StatisticalModelState[ModelState]) -> StatisticalModelOutput[ModelState]:
        assert input.shape == (self.input_dim,)
        outs = self._predict(input, statistical_model_state)
        assert outs.mean.shape == outs.statistical_model_state.beta.shape == (self.output_dim,)
        assert outs.epistemic_std.shape == outs.aleatoric_std.shape == (self.output_dim,)
        return outs

    @staticmethod
    def vmap_input_axis(data_axis: int = 0) -> StatisticalModelState:
        return StatisticalModelState(beta=None, model_state=None)

    @staticmethod
    def vmap_output_axis(data_axis: int = 0) -> StatisticalModelOutput:
        return StatisticalModelOutput(mean=data_axis, epistemic_std=data_axis, aleatoric_std=data_axis,
                                      statistical_model_state=None
                                      )

    def _predict(self,
                 input: chex.Array,
                 statistical_model_state: StatisticalModelState[ModelState]) -> StatisticalModelOutput[ModelState]:
        dist_f, dist_y = self.model.posterior(input, statistical_model_state.model_state)
        statistical_model = StatisticalModelOutput(mean=dist_f.mean(), epistemic_std=dist_f.stddev(),
                                                   aleatoric_std=dist_y.aleatoric_std(),
                                                   statistical_model_state=statistical_model_state)
        return statistical_model

    def predict_batch(self,
                      input: chex.Array,
                      statistical_model_state: StatisticalModelState[ModelState]) -> StatisticalModelOutput[ModelState]:
        preds = vmap(self, in_axes=(0, self.vmap_input_axis(0)),
                     out_axes=self.vmap_output_axis(0))(input, statistical_model_state)
        return preds

    @abstractmethod
    def update(self,
               stats_model_state: StatisticalModelState[ModelState],
               data: Data) -> StatisticalModelState[ModelState]:
        """
        stats_model_state: statistical model state
        data: Data on which we train the statistical model
        """
        pass

    def init(self,
             key: chex.PRNGKey) -> StatisticalModelState[ModelState]:
        model_state = self.model.init(key)
        beta = jnp.ones(self.output_dim)
        return StatisticalModelState(model_state=model_state, beta=beta)
