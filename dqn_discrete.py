"""
Deep Q-learning for discrete action spaces.
"""
import time

import chex
import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
import gymnax
import argparse
from flax.training.train_state import TrainState
from distrax import EpsilonGreedy
from typing import Sequence, NamedTuple, Any, Callable, List, Tuple
from chex import dataclass
from gymnax.wrappers.purerl import FlattenObservationWrapper, LogWrapper
import gymnax.environments.spaces as spaces
import functools
from flax.linen.initializers import he_normal, constant
from jax.experimental import checkify
from matplotlib import pyplot as plt


class Critic(nn.Module):
    action_dim: Sequence[int]

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64, kernel_init=he_normal(), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Dense(64, kernel_init=he_normal(), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim, kernel_init=he_normal(), bias_init=constant(0.0))(x)
        return x


# (state, action, reward, next_state)
class Sample(NamedTuple):
    state: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    next_state: jnp.array
    done: jnp.ndarray


@dataclass(frozen=True)
class ReplayBuffer:
    state: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    next_state: jnp.array
    done: jnp.ndarray
    capacity: int
    ptr: int
    size: int


def init_replay_buffer(capacity, obs_shape, act_shape) -> ReplayBuffer:
    state = jnp.zeros(shape=(capacity, *obs_shape))
    action = jnp.zeros(shape=(capacity, *act_shape), dtype=int)
    reward = jnp.zeros(shape=(capacity,))
    next_state = jnp.zeros(shape=(capacity, *obs_shape))
    done = jnp.zeros(shape=(capacity,))
    ptr = 0
    size = 0
    return ReplayBuffer(state=state, action=action, reward=reward, next_state=next_state, done=done, capacity=capacity,
                        ptr=ptr, size=size)


class State(NamedTuple):
    key: chex.PRNGKey
    env_state: gymnax.EnvState
    train_state: TrainState
    buffer: ReplayBuffer
    eval_params: chex.ArrayTree
    obs: jnp.ndarray
    steps: int
    updates: int


EpsilonPolicy = optax.Schedule

UpdatePolicy = Callable[[State], State]


def soft_update_policy(tau: float):
    def update(state: State):
        return state._replace(eval_params=optax.incremental_update(state.train_state.params, state.eval_params, tau),
                              updates=state.updates + 1)

    return update


def buffer_record(buffer: ReplayBuffer, obs, action, reward, next_obs, done) -> ReplayBuffer:
    idxs = (jnp.arange(obs.shape[0]) + buffer.ptr) % buffer.capacity
    new_state = buffer.state.at[idxs, :].set(obs)
    new_action = buffer.action.at[idxs].set(action)
    new_reward = buffer.reward.at[idxs].set(reward)
    new_next_state = buffer.next_state.at[idxs, :].set(next_obs)
    new_done = buffer.done.at[idxs].set(done)
    return buffer.replace(state=new_state, action=new_action, reward=new_reward, next_state=new_next_state,
                          done=new_done,
                          ptr=(buffer.ptr + obs.shape[0]) % buffer.capacity, capacity=buffer.capacity,
                          size=jnp.minimum(buffer.size + obs.shape[0], buffer.capacity))


def buffer_sample(buffer: ReplayBuffer, batch_size: int, rng_key: chex.PRNGKey):
    # checkify.check(buffer.size >= batch_size, "buffer size {} < batch size {}", buffer.size, batch_size)
    idx = jax.random.randint(rng_key, (batch_size,), 0, buffer.size)
    return Sample(buffer.state[idx], buffer.action[idx], buffer.reward[idx], buffer.next_state[idx], buffer.done[idx])


def epsilon_greedy(rng: chex.PRNGKey, q_vals: jnp.ndarray, epsilon: float):
    actions = EpsilonGreedy(q_vals, epsilon).sample(seed=rng)
    return actions


def q_learning(env_name, num_envs, buffer_size, epsilon_policy: EpsilonPolicy, update_policy: UpdatePolicy,
               batch_size=64, lr=0.001, gamma=0.99, num_steps=int(1e5)):
    env, env_params = gymnax.make(env_name)

    assert isinstance(env.action_space(env_params), spaces.Discrete), "Environment action space is not Discrete"

    env = FlattenObservationWrapper(LogWrapper(env))

    vmap_reset = jax.vmap(env.reset, in_axes=(0, None))
    vmap_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))

    def step(state: State, _):
        key = state.key
        key, eps_key = jax.random.split(key)
        actions = epsilon_greedy(eps_key, state.train_state.apply_fn(state.train_state.params, state.obs),
                                 epsilon_policy(state.steps))
        key, step_key = jax.random.split(key)
        step_keys = jax.random.split(step_key, num_envs)
        n_obs, n_env_state, reward, done, info = vmap_step(step_keys, state.env_state, actions, env_params)
        n_buffer = buffer_record(state.buffer, state.obs, actions, reward, n_obs, done)

        def update(key: chex.PRNGKey, train_state: TrainState, buffer: ReplayBuffer):
            batch = buffer_sample(buffer, batch_size, key)

            def criterion(params):
                targets = batch.reward + gamma * (1 - batch.done) * jnp.max(
                    state.train_state.apply_fn(state.eval_params, batch.next_state), axis=-1)
                # Select Q(s, a) for each sample in batch using the action index and Q value function.
                predictions = jnp.take_along_axis(state.train_state.apply_fn(params, batch.state),
                                                  batch.action[:, None], axis=-1).squeeze(axis=-1)
                return jnp.mean((targets - predictions) ** 2)

            loss, grads = jax.value_and_grad(criterion)(train_state.params)
            train_state = train_state.apply_gradients(grads=grads)
            return train_state, loss

        key, update_key = jax.random.split(key)
        n_train_state, loss = jax.lax.cond(n_buffer.size >= batch_size,
                                           lambda key, train_state: update(update_key, train_state, n_buffer),
                                           lambda _, train_state: (train_state, jnp.array(0.0)),
                                           state.key, state.train_state)
        return update_policy(
            state._replace(key=key, env_state=n_env_state, train_state=n_train_state, buffer=n_buffer, obs=n_obs,
                           steps=state.steps + 1)), info

    def train(key: chex.PRNGKey):
        key, reset_key = jax.random.split(key)
        reset_keys = jax.random.split(reset_key, num_envs)
        obs, env_state = vmap_reset(reset_keys, env_params)

        agent = Critic(env.action_space(env_params).n)
        key, agent_key = jax.random.split(key)
        dummy = jnp.zeros(env.observation_space(env_params).shape)
        agent_params = agent.init(agent_key, dummy)
        optimizer = optax.adam(learning_rate=lr)

        state = State(key=key, env_state=env_state,
                      train_state=TrainState.create(apply_fn=agent.apply, params=agent_params, tx=optimizer),
                      buffer=init_replay_buffer(buffer_size, env.observation_space(env_params).shape,
                                                env.action_space(env_params).shape),
                      eval_params=jax.tree_util.tree_map(jnp.array, agent_params),
                      obs=obs, steps=0, updates=0)
        state, info = jax.lax.scan(step, state, None, num_steps)
        return state, info

    return train


if __name__ == '__main__':
    train = q_learning(env_name='CartPole-v1', num_envs=10, buffer_size=10000,
                       epsilon_policy=optax.piecewise_constant_schedule(0.5,
                                                                        {50: 0.8, 100: 0.75, 300: 1 / 3, 500: 0.5,
                                                                         1000: 0.5, 1500: 0.5, 2000: 0.5, 3000: 0.5}),
                       update_policy=soft_update_policy(tau=1e-2), num_steps=4000)
    train_jit = jax.jit(train)
    rng = jax.random.PRNGKey(2)
    state, info = jax.block_until_ready(train_jit(rng))
    plt.plot(info['returned_episode_returns'].mean(axis=-1))
    plt.show()
