import chex
import jax
import distrax
import jax.numpy as jnp
from flax import linen as nn
import optax
import gymnax

from flax.training.train_state import TrainState
from flax.linen.initializers import he_normal, constant
from typing import NamedTuple, Any
from gymnax.wrappers.purerl import FlattenObservationWrapper, LogWrapper

import gymnax.environments.spaces as spaces
from matplotlib import pyplot as plt


class MLP(nn.Module):
    output_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64, kernel_init=he_normal(), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Dense(64, kernel_init=he_normal(), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Dense(self.output_dim, kernel_init=he_normal(), bias_init=constant(0.0))(x)
        return x


class ActorCritic(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        pi = distrax.Categorical(logits=MLP(self.action_dim)(x))
        v = MLP(1)(x)  # comes out as (n, 1)
        return pi, jnp.squeeze(v, axis=-1)


class Sample(NamedTuple):
    obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    logp: jnp.ndarray
    value: jnp.ndarray
    info: Any


class State(NamedTuple):
    key: chex.PRNGKey
    env_state: gymnax.EnvState
    train_state: TrainState
    obs: jnp.ndarray
    done: jnp.ndarray


class Batch(NamedTuple):
    samples: Sample
    advs: jnp.ndarray
    targets: jnp.ndarray


class UpdateState(NamedTuple):
    state: State
    batch: Batch


def gae_normalized(samples, last_value, gamma, lamda):
    def f(carry, sample: Sample):
        # last_term = gamma * lambda * delta_t+(l+1)
        last_value, last_term = carry
        delta = gamma * (1 - sample.done) * last_value + sample.reward - sample.value
        term = delta + gamma * lamda * (1 - sample.done) * last_term
        return (sample.value, term), term

    _, advs = jax.lax.scan(f, init=(jnp.zeros_like(last_value), last_value), xs=samples, reverse=True)
    return (advs - advs.mean()) / advs.std()


def calculate_discount_return(samples, gamma):
    def f(last_term, sample: Sample):
        term = gamma * last_term * (1 - sample.done) + sample.reward
        return term, term

    _, cumsum = jax.lax.scan(f, init=jnp.zeros_like(samples.reward[0]), xs=samples, reverse=True)
    return cumsum


def ppo(env_name, num_envs, num_epochs, steps_per_epoch, updates_per_epoch, minibatch_size, lr, gamma, lamda,
        clip_ratio,
        value_loss_coeff,
        entropy_coeff):
    env, env_params = gymnax.make(env_name)

    assert isinstance(env.action_space(env_params), spaces.Discrete), "Environment action space is not Discrete"

    assert steps_per_epoch % minibatch_size == 0, "steps_per_epoch must be divisible by minibatch_size"

    env = FlattenObservationWrapper(LogWrapper(env))

    vmap_reset = jax.vmap(env.reset, in_axes=(0, None))
    vmap_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))

    def env_step(state: State, _) -> (State, Sample):
        key = state.key
        key, action_key = jax.random.split(key)
        pi, v = state.train_state.apply_fn(state.train_state.params, state.obs)
        action = pi.sample(seed=action_key)
        logp = pi.log_prob(action)

        key, step_key = jax.random.split(key)
        step_keys = jax.random.split(step_key, num_envs)
        n_obs, n_env_state, reward, done, info = vmap_step(step_keys, state.env_state, action, env_params)
        # state.obs is the last obs so we write (state.obs, action) so we can have state-action pairs
        sample = Sample(obs=state.obs, action=action, reward=reward, done=done, logp=logp, value=v, info=info)
        return state._replace(key=key, obs=n_obs, env_state=n_env_state, done=done), sample

    def train(key: chex.PRNGKey):
        key, reset_key = jax.random.split(key)
        reset_keys = jax.random.split(reset_key, num_envs)
        obs, env_state = vmap_reset(reset_keys, env_params)

        agent = ActorCritic(env.action_space(env_params).n)
        key, agent_key = jax.random.split(key)
        dummy = jnp.zeros(env.observation_space(env_params).shape)
        agent_params = agent.init(agent_key, dummy)
        optimizer = optax.adam(learning_rate=lr)

        state = State(key=key, env_state=env_state,
                      train_state=TrainState.create(apply_fn=agent.apply, params=agent_params, tx=optimizer), obs=obs,
                      done=jnp.zeros((num_envs,), dtype=jnp.bool))

        def step(state: State, _):
            state, samples = jax.lax.scan(env_step, state, None, steps_per_epoch)
            _, v = state.train_state.apply_fn(state.train_state.params, state.obs)
            last_v = jax.lax.select(state.done, jnp.zeros_like(v), v)  # V = 0 if terminal
            advs = gae_normalized(samples, last_v, gamma, lamda)
            targets = calculate_discount_return(samples, gamma)

            def update_minibatch(state: State, minibatch: Batch):
                def loss_fn(params):
                    pis, values = state.train_state.apply_fn(params, minibatch.samples.obs)

                    def pi_loss(pi, samples, gaes):
                        logp = pi.log_prob(samples.action)
                        ratio = jnp.exp(logp - samples.logp)
                        loss = -jnp.minimum(jax.lax.clamp(1 - clip_ratio, ratio, 1 + clip_ratio) * gaes, ratio * gaes)
                        return loss.mean()

                    def v_loss(values, samples, targets):
                        clipped_loss = 0.5 * (targets - (
                                samples.value + jax.lax.clamp(-clip_ratio, values - samples.value, clip_ratio))) ** 2
                        loss = jnp.maximum(clipped_loss, 0.5 * (targets - values) ** 2)
                        return loss.mean()

                    entropy = pis.entropy().mean()
                    return pi_loss(pis, minibatch.samples, minibatch.advs) + \
                        value_loss_coeff * v_loss(values, minibatch.samples,
                                                  minibatch.targets) - entropy_coeff * entropy

                loss, grads = jax.value_and_grad(loss_fn)(state.train_state.params)
                return state._replace(train_state=state.train_state.apply_gradients(grads=grads)), loss

            def generate_minibatch(key: chex.PRNGKey, samples: Sample, advs, targets):
                axis_len = num_envs * steps_per_epoch
                idxs = jax.random.permutation(key, axis_len)
                minibatch = Batch(samples=samples, advs=advs, targets=targets)
                minibatch = jax.tree_map(lambda x: x.reshape((axis_len, *x.shape[2:])).take(idxs, axis=0), minibatch)
                return jax.tree_map(lambda x: x.reshape((-1, minibatch_size, *x.shape[1:])), minibatch)

            def update(update_state: UpdateState, _):
                n_key, minibatch_key = jax.random.split(update_state.state.key)
                state = update_state.state._replace(key=n_key)
                batch = update_state.batch
                minibatch = generate_minibatch(minibatch_key, batch.samples, batch.advs, batch.targets)
                state, loss = jax.lax.scan(update_minibatch, state, minibatch)
                # aggregate statistics in one epoch's update step
                return update_state._replace(state=state), dict(loss=loss.mean(), info=jax.tree_map(jnp.mean,
                                                                                                    samples.info))

            update_state = UpdateState(state=state, batch=Batch(samples=samples, advs=advs, targets=targets))
            update_state, update_info = jax.lax.scan(update, update_state, None, updates_per_epoch)
            return update_state.state, update_info

        state, info = jax.lax.scan(step, state, None, num_epochs)
        return state, info

    return train


if __name__ == '__main__':
    train = ppo(env_name='Pong-misc', num_envs=10, num_epochs=500, steps_per_epoch=400, updates_per_epoch=50,
                minibatch_size=100, lr=3e-4, gamma=0.99, lamda=0.97, clip_ratio=0.2, value_loss_coeff=0.5,
                entropy_coeff=0.01)
    train = jax.jit(train)
    rng = jax.random.PRNGKey(12)
    obs, info = jax.block_until_ready(train(rng))
    plt.plot(info['info']['returned_episode_returns'].mean(axis=-1))
    print(info['info']['returned_episode_returns'].mean(axis=-1))
    print(info['info']['returned_episode_returns'].mean(axis=-1).shape)

    # Add labels and title
    plt.xlabel('Epoch')
    plt.ylabel('Returns')
    plt.title('Episode returns per epoch')

    # Show plot
    plt.grid(True)
    plt.show()
