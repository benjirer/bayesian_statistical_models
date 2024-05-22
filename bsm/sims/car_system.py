import time
from functools import partial
from typing import Dict, Tuple

import chex
import jax.numpy as jnp
import jax.random as jr
from distrax import Distribution
from distrax import Normal
from jax import jit
from mbpo.systems.base_systems import (
    Dynamics,
    Reward,
    System,
    SystemState,
    SystemParams,
    DynamicsParams,
    RewardParams,
)

from bsm.sims.dynamics_models import RaceCar, CarParams
from bsm.sims.envs import RCCarEnvReward
from bsm.sims.util import plot_rc_trajectory, encode_angles


@chex.dataclass
class CarDynamicsParams:
    action_buffer: chex.Array
    car_params: CarParams
    key: chex.PRNGKey


class CarDynamics(Dynamics[CarDynamicsParams]):
    max_steps: int = 200
    _dt: float = 1 / 30.0
    dim_action: Tuple[int] = (2,)
    _goal: jnp.array = jnp.array([0.0, 0.0, -jnp.pi / 2.0])
    # _init_pose: jnp.array = jnp.array([-1.04, -1.42, jnp.pi / 2.])
    _init_pose: jnp.array = jnp.array([1.42, -1.04, jnp.pi])
    # _init_pose: jnp.array = jnp.array([-1.04 - 1, -1.42 + 1, jnp.pi / 2.])
    _angle_idx: int = 2
    _obs_noise_stds: jnp.array = 0.05 * jnp.exp(
        jnp.array(
            [-3.3170326, -3.7336411, -2.7081904, -2.7841284, -2.7067015, -1.4446207]
        )
    )
    _default_car_model_params_bicycle: Dict = {
        "use_blend": 0.0,
        "m": 1.65,
        "l_f": 0.13,
        "l_r": 0.17,
        "angle_offset": 0.02791893,
        "b_f": 2.58,
        "b_r": 3.39,
        "blend_ratio_lb": 0.4472136,
        "blend_ratio_ub": 0.5477226,
        "c_d": -1.8698378e-36,
        "c_f": 1.2,
        "c_m_1": 10.431917,
        "c_m_2": 1.5003588,
        "c_r": 1.27,
        "d_f": 0.02,
        "d_r": 0.017,
        "i_com": 2.78e-05,
        "steering_limit": 0.19989373,
    }

    _default_car_model_params_blend: Dict = {
        "use_blend": 1.0,
        "m": 1.65,
        "l_f": 0.13,
        "l_r": 0.17,
        "angle_offset": 0.00731506,
        "b_f": 2.5134025,
        "b_r": 3.8303657,
        "blend_ratio_lb": -0.00057009,
        "blend_ratio_ub": -0.07274915,
        "c_d": -6.9619144e-37,
        "c_f": 1.2525784,
        "c_m_1": 10.93334,
        "c_m_2": 1.0498677,
        "c_r": 1.2915123,
        "d_f": 0.43698108,
        "d_r": 0.43703166,
        "i_com": 0.06707229,
        "steering_limit": 0.5739077,
    }

    def __init__(
        self,
        encode_angle: bool = False,
        use_tire_model: bool = False,
        action_delay: float = 0.0,
        car_model_params: Dict = None,
        use_obs_noise: bool = True,
        car_id: int = 2,
    ):
        """
        Race car simulator environment

        Args:
            ctrl_cost_weight: weight of the control penalty
            encode_angle: whether to encode the angle as cos(theta), sin(theta)
            use_obs_noise: whether to use observation noise
            use_tire_model: whether to use the (high-fidelity) tire model, if False just uses a kinematic bicycle model
            action_delay: whether to delay the action by a certain amount of time (in seconds)
            car_model_params: dictionary of car model parameters that overwrite the default values
            seed: random number generator seed
        """
        self.dim_state: Tuple[int] = (7,) if encode_angle else (6,)
        Dynamics.__init__(self, self.dim_state[0], 2)
        self.encode_angle: bool = encode_angle

        # initialize dynamics and observation noise models
        self._dynamics_model = RaceCar(dt=self._dt, encode_angle=encode_angle)

        assert car_id in [1, 2]
        self.car_id = car_id
        self._set_default_params()

        _default_params = (
            self._default_car_model_params_blend
            if use_tire_model
            else self._default_car_model_params_bicycle
        )
        self._typical_params = CarParams(**_default_params)

        self.use_tire_model = use_tire_model
        self._dynamics_params = self._typical_params
        self.use_obs_noise = use_obs_noise

        # set up action delay
        assert action_delay >= 0.0, "Action delay must be non-negative"
        self.action_delay = action_delay
        if action_delay % self._dt == 0.0:
            self._act_delay_interpolation_weights = jnp.array([1.0, 0.0])
        else:
            # if action delay is not a multiple of dt, compute weights to interpolate
            # between temporally closest actions
            weight_first = (action_delay % self._dt) / self._dt
            self._act_delay_interpolation_weights = jnp.array(
                [weight_first, 1.0 - weight_first]
            )
        action_delay_buffer_size = int(jnp.ceil(action_delay / self._dt)) + 1
        self._action_buffer = jnp.zeros((action_delay_buffer_size, self.dim_action[0]))

        # initialize state
        self._init_state: jnp.array = jnp.zeros(self.dim_state)

    def init_params(self, key: chex.PRNGKey) -> CarDynamicsParams:
        return CarDynamicsParams(
            car_params=self._dynamics_params, action_buffer=self._action_buffer, key=key
        )

    def _set_default_params(self):
        from bsm.sims.car_sim_config import (
            DEFAULT_PARAMS_BICYCLE_CAR1,
            DEFAULT_PARAMS_BLEND_CAR1,
            BOUNDS_PARAMS_BICYCLE_CAR1,
            BOUNDS_PARAMS_BLEND_CAR1,
            DEFAULT_PARAMS_BICYCLE_CAR2,
            DEFAULT_PARAMS_BLEND_CAR2,
            BOUNDS_PARAMS_BICYCLE_CAR2,
            BOUNDS_PARAMS_BLEND_CAR2,
        )

        if self.car_id == 1:
            self._default_car_model_params_bicycle = DEFAULT_PARAMS_BICYCLE_CAR1
            self._bounds_car_model_params_bicycle = BOUNDS_PARAMS_BICYCLE_CAR1
            self._default_car_model_params_blend = DEFAULT_PARAMS_BLEND_CAR1
            self._bounds_car_model_params_blend = BOUNDS_PARAMS_BLEND_CAR1
        elif self.car_id == 2:
            self._default_car_model_params_bicycle = DEFAULT_PARAMS_BICYCLE_CAR2
            self._bounds_car_model_params_bicycle = BOUNDS_PARAMS_BICYCLE_CAR2
            self._default_car_model_params_blend = DEFAULT_PARAMS_BLEND_CAR2
            self._bounds_car_model_params_blend = BOUNDS_PARAMS_BLEND_CAR2
        else:
            raise ValueError(f"Car id {self.car_id} not supported.")

    def _state_to_obs(self, state: jnp.array, rng_key: chex.PRNGKey) -> jnp.array:
        """Adds observation noise to the state"""
        assert state.shape[-1] == 6
        # add observation noise
        if self.use_obs_noise:
            obs = state + self._obs_noise_stds * jr.normal(rng_key, shape=state.shape)
        else:
            obs = state

        # encode angle to sin(theta) and cos(theta) if desired
        if self.encode_angle:
            obs = encode_angles(obs, self._angle_idx)
        assert (obs.shape[-1] == 7 and self.encode_angle) or (
            obs.shape[-1] == 6 and not self.encode_angle
        )
        return obs

    def next_state(
        self, x: chex.Array, u: chex.Array, dynamics_params: CarDynamicsParams
    ) -> Tuple[Distribution, CarDynamicsParams]:
        assert u.shape == (self.u_dim,) and x.shape == (self.x_dim,)

        action_buffer = dynamics_params.action_buffer
        if self.action_delay > 0.0:
            # pushes action to action buffer and pops the oldest action
            # computes delayed action as a linear interpolation between the relevant actions in the past
            u, action_buffer = self._get_delayed_action(u, action_buffer)

        # Move forward one step in the dynamics using the delayed action and the hidden state
        new_key, key_for_sampling_obs_noise = jr.split(dynamics_params.key)
        x_mean_next = self._dynamics_model.next_step(x, u, dynamics_params.car_params)
        if self.encode_angle:
            x_mean_next = self._dynamics_model.reduce_x(x_mean_next)
        x_next = self._state_to_obs(x_mean_next, key_for_sampling_obs_noise)

        new_params = dynamics_params.replace(action_buffer=action_buffer, key=new_key)
        return Normal(x_next, jnp.zeros_like(x_next)), new_params

    def _get_delayed_action(
        self, action: jnp.array, action_buffer: chex.PRNGKey
    ) -> jnp.array:
        # push action to action buffer
        new_action_buffer = jnp.concatenate(
            [action_buffer[1:], action[None, :]], axis=0
        )

        # get delayed action (interpolate between two actions if the delay is not a multiple of dt)
        delayed_action = jnp.sum(
            new_action_buffer[:2] * self._act_delay_interpolation_weights[:, None],
            axis=0,
        )
        assert delayed_action.shape == self.dim_action
        return delayed_action, new_action_buffer

    def reset(self, key: chex.PRNGKey) -> jnp.array:
        """Resets the environment to a random initial state close to the initial pose"""
        # sample random initial state
        key_pos, key_theta, key_vel, key_obs = jr.split(key, 4)
        init_pos = self._init_pose[:2] + jr.uniform(
            key_pos, shape=(2,), minval=-0.10, maxval=0.10
        )
        init_theta = self._init_pose[2:] + jr.uniform(
            key_pos, shape=(1,), minval=-0.10 * jnp.pi, maxval=0.10 * jnp.pi
        )
        init_vel = jnp.zeros((3,)) + jnp.array([0.005, 0.005, 0.02]) * jr.normal(
            key_vel, shape=(3,)
        )
        init_state = jnp.concatenate([init_pos, init_theta, init_vel])
        return self._state_to_obs(init_state, rng_key=key_obs)
        # return init_state


@chex.dataclass
class CarRewardParams:
    _goal: chex.Array
    key: chex.PRNGKey


class CarReward(Reward[CarRewardParams]):
    _goal: jnp.array = jnp.array([0.0, 0.0, 0.0])

    def __init__(
        self,
        ctrl_cost_weight: float = 0.005,
        encode_angle: bool = False,
        bound: float = 0.1,
        margin_factor: float = 10.0,
        num_frame_stack: int = 0,
        ctrl_diff_weight: float = 0.0,
    ):
        Reward.__init__(self, x_dim=7 if encode_angle else 6, u_dim=2)
        self.num_frame_stack = num_frame_stack
        self.ctrl_diff_weight = ctrl_diff_weight
        self.ctrl_cost_weight = ctrl_cost_weight
        self.encode_angle: bool = encode_angle
        self._reward_model = RCCarEnvReward(
            goal=self._goal,
            ctrl_cost_weight=ctrl_cost_weight,
            encode_angle=self.encode_angle,
            bound=bound,
            margin_factor=margin_factor,
        )

    def init_params(self, key: chex.PRNGKey) -> CarRewardParams:
        return CarRewardParams(_goal=self._goal, key=key)

    def set_goal(self, des_goal: chex.Array):
        self._goal = des_goal

    def __call__(
        self,
        x: chex.Array,
        u: chex.Array,
        reward_params: CarRewardParams,
        x_next: chex.Array | None = None,
    ) -> Tuple[Distribution, CarRewardParams]:
        assert x.shape == (
            self.x_dim + self.num_frame_stack * self.u_dim,
        ) and u.shape == (self.u_dim,)
        assert x_next.shape == (self.x_dim + self.num_frame_stack * self.u_dim,)
        x_state, us_stacked = x[: self.x_dim], x[self.x_dim :]
        x_next_state, us_next_stacked = x_next[: self.x_dim], x_next[self.x_dim :]
        reward = self._reward_model.forward(obs=None, action=u, next_obs=x_next_state)
        # Now we add cost for the control inputs change:
        if self.num_frame_stack > 0:
            u_previous = us_stacked[-self.u_dim :]
            reward -= self.ctrl_diff_weight * jnp.sum((u_previous - u) ** 2)
        return Normal(reward, jnp.zeros_like(reward)), reward_params


class RewardFrameStackWrapper(Reward):
    def __init__(self, reward: Reward, num_frame_stack: int = 1):
        self._reward = reward
        self._num_frame_stack = num_frame_stack
        Reward.__init__(self, x_dim=reward.x_dim * num_frame_stack, u_dim=reward.u_dim)

    def __call__(
        self,
        x: chex.Array,
        u: chex.Array,
        reward_params: RewardParams,
        x_next: chex.Array | None = None,
    ) -> Tuple[Distribution, RewardParams]:
        _x = x[: self._reward.x_dim]
        _x_next = x_next[: self._reward.x_dim]
        return self._reward(_x, u, reward_params, _x_next)

    def init_params(self, key: chex.PRNGKey) -> RewardParams:
        return self._reward.init_params(key)


class DynamicsFrameStackWrapper(Dynamics):
    def __init__(self, dynamics: Dynamics, num_frame_stack: int = 1):
        self._dynamics = dynamics
        self._num_frame_stack = num_frame_stack
        Dynamics.__init__(
            self, x_dim=dynamics.x_dim * num_frame_stack, u_dim=dynamics.u_dim
        )
        """
        We stack the the observations in the forms [x_t, x_{t-1}, ..., x_{t - num_frame_stack + 1}]
        """

    def next_state(
        self, x: chex.Array, u: chex.Array, dynamics_params: DynamicsParams
    ) -> Tuple[Distribution, DynamicsParams]:
        assert x.shape == (self.x_dim,) and u.shape == (self.u_dim,)
        _x_cur = x[: self._dynamics.x_dim]
        _x_next_dist, new_dynamics_params = self._dynamics.next_state(
            _x_cur, u, dynamics_params
        )
        # Here we need to take care of the distribution
        # TODO: for proper handling we need to write a new distribution class, for now we just sample from the
        #  _x_next_dist and return normal distribution
        x_new_tail = x[: -self._dynamics.x_dim]
        new_key, sample_key = jr.split(new_dynamics_params.key)
        _x_next = _x_next_dist.sample(seed=dynamics_params.key)
        x_new = jnp.concatenate([_x_next, x_new_tail], axis=0)
        new_dynamics_params = new_dynamics_params.replace(key=new_key)
        return Normal(x_new, jnp.zeros_like(x_new)), new_dynamics_params

    def init_params(self, key: chex.PRNGKey) -> DynamicsParams:
        return self._dynamics.init_params(key=key)


class FrameStackWrapper(System):
    def __init__(self, system: System, num_frame_stack: int = 0):
        self._system = system
        self._num_frame_stack = num_frame_stack
        System.__init__(
            self, dynamics=self._system.dynamics, reward=self._system.reward
        )
        self.x_dim = self._system.x_dim + self._system.u_dim * num_frame_stack
        self.u_dim = self._system.u_dim

    def system_params_vmap_axes(self, axes: int = 0):
        return self._system.system_params_vmap_axes(axes)

    def step(
        self,
        x: chex.Array,
        u: chex.Array,
        system_params: SystemParams[DynamicsParams, RewardParams],
    ) -> SystemState:
        # Decompose to x and last actions
        _x = x[: self._system.x_dim]
        stacked_us = x[self._system.x_dim :]

        next_sys_step = self._system.step(_x, u, system_params)
        if self._num_frame_stack > 0:
            stacked_us = jnp.concatenate([stacked_us[self._system.u_dim :], u])

        # We add last actions to the state
        x_next = jnp.concatenate([next_sys_step.x_next, stacked_us], axis=0)
        next_sys_step = next_sys_step.replace(x_next=x_next)
        return next_sys_step


class CarSystem(System[CarDynamicsParams, CarRewardParams]):
    def __init__(
        self,
        encode_angle: bool = False,
        use_tire_model: bool = False,
        action_delay: float = 0.0,
        car_model_params: Dict = None,
        ctrl_cost_weight: float = 0.005,
        use_obs_noise: bool = True,
        bound: float = 0.1,
        margin_factor: float = 10.0,
    ):
        System.__init__(
            self,
            dynamics=CarDynamics(
                encode_angle=encode_angle,
                use_tire_model=use_tire_model,
                action_delay=action_delay,
                car_model_params=car_model_params,
                use_obs_noise=use_obs_noise,
            ),
            reward=CarReward(
                ctrl_cost_weight=ctrl_cost_weight,
                encode_angle=encode_angle,
                bound=bound,
                margin_factor=margin_factor,
            ),
        )

    @staticmethod
    def system_params_vmap_axes(axes: int = 0):
        return SystemParams(
            dynamics_params=CarDynamicsParams(
                action_buffer=axes, car_params=None, key=axes
            ),
            reward_params=CarRewardParams(_goal=None, key=axes),
            key=axes,
        )

    @partial(jit, static_argnums=(0,))
    def step(
        self,
        x: chex.Array,
        u: chex.Array,
        system_params: SystemParams[CarDynamicsParams, CarRewardParams],
    ) -> SystemState:
        assert x.shape == (self.x_dim,) and u.shape == (self.u_dim,)
        new_key, key_x_next, key_reward = jr.split(system_params.key, 3)
        x_next_dist, next_dynamics_params = self.dynamics.next_state(
            x, u, system_params.dynamics_params
        )
        x_next = x_next_dist.sample(seed=key_x_next)
        reward_dist, next_reward_params = self.reward(
            x, u, system_params.reward_params, x_next
        )
        return SystemState(
            x_next=x_next,
            reward=reward_dist.sample(seed=key_reward),
            system_params=SystemParams(
                dynamics_params=next_dynamics_params,
                reward_params=next_reward_params,
                key=new_key,
            ),
        )

    def reset(self, key: chex.PRNGKey) -> SystemState:
        return SystemState(
            x_next=self.dynamics.reset(key=key),
            reward=jnp.array([0.0]).squeeze(),
            system_params=self.init_params(key=key),
        )


if __name__ == "__main__":
    ENCODE_ANGLE = False
    system = CarSystem(
        encode_angle=ENCODE_ANGLE,
        action_delay=0.07,
        use_tire_model=True,
    )

    t_start = time.time()
    system_params = system.init_params(key=jr.PRNGKey(0))
    s = system.dynamics.reset(key=jr.PRNGKey(0))

    traj = [s]
    rewards = []
    actions = []
    for i in range(120):
        t = i / 30.0
        a = jnp.array([-1 * jnp.cos(1.0 * t), 0.8 / (t + 1)])
        next_sys_step = system.step(s, a, system_params)
        s = next_sys_step.x_next
        r = next_sys_step.reward
        system_params = next_sys_step.system_params
        traj.append(s)
        actions.append(a)
        rewards.append(r)

    duration = time.time() - t_start
    print(f"Duration of trajectory sim {duration} sec")
    traj = jnp.stack(traj)
    actions = jnp.stack(actions)

    plot_rc_trajectory(traj, actions, encode_angle=ENCODE_ANGLE)
