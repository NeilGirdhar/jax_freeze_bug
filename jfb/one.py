from __future__ import annotations

from collections.abc import Callable
from functools import partial

import jax.numpy as jnp
import numpy as np
from jax import Array, custom_vjp, grad, jit, vjp, vmap
from jax.lax import dot
from jax.nn import softplus
from jax.tree_util import tree_map
from jaxopt import GradientDescent
from tjax import RealArray, RealNumeric, print_generic
from tjax.dataclasses import dataclass


@dataclass
class Weights:
    b1: Array
    w1: Array
    b2: Array
    w2: Array


@dataclass
class AdamState:
    count: RealArray
    mu: Weights
    nu: Weights


def update_moment(updates, moments, decay, order):
  """Compute the exponential moving average of the `order-th` moment."""
  return tree_map(lambda g, t: (1 - decay) * (g ** order) + decay * t, updates, moments)


def update_moment_per_elem_norm(updates, moments, decay, order):
  """Compute the EMA of the `order`-th moment of the element-wise norm."""

  def orderth_norm(g):
    if jnp.isrealobj(g):
      return g ** order
    else:
      half_order = order / 2
      # JAX generates different HLO for int and float `order`
      if half_order.is_integer():
        half_order = int(half_order)
      return jnp.square(g) ** half_order

  return tree_map(lambda g, t: (1 - decay) * orderth_norm(g) + decay * t, updates, moments)


def bias_correction(moment, decay, count):
  bias_correction_ = 1 - decay**count

  # Perform division in the original precision.
  return tree_map(lambda t: t / bias_correction_.astype(t.dtype), moment)


b1 = 0.9
b2 = 0.999
eps = 1e-8
eps_root = 0.0
learning_rate = 1e-2


@dataclass
class Adam:
    b1: RealNumeric = 0.9
    b2: RealNumeric = 0.999
    eps: RealNumeric = 1e-8

    def init(self, parameters: Weights) -> AdamState:
        return AdamState(mu=tree_map(lambda t: jnp.zeros_like(t), parameters),
                         nu=tree_map(lambda t: jnp.zeros_like(t), parameters),
                         count=jnp.zeros([], jnp.int32))

    def update(self,
               gradient: Weights,
               state: AdamState,
               parameters: Weights | None) -> tuple[Weights, AdamState]:
        mu = update_moment(gradient, state.mu, self.b1, 1)
        nu = update_moment_per_elem_norm(gradient, state.nu, self.b2, 2)
        count_inc = state.count + jnp.array(1, dtype=jnp.int32)
        mu_hat = bias_correction(mu, self.b1, count_inc)
        nu_hat = bias_correction(nu, self.b2, count_inc)
        gradient = tree_map(
            lambda m, v: m / (jnp.sqrt(v) + self.eps), mu_hat, nu_hat)
        gradient = tree_map(lambda m: m * -learning_rate, gradient)
        return gradient, AdamState(count=count_inc, mu=mu, nu=nu)


@dataclass
class SolutionState:
    gradient_state: AdamState
    weights: Weights


def cli() -> None:
    gradient_transformation = Adam()
    weights = Weights(b1=jnp.array([0., 0., 0.], dtype=np.float32),
                      w1=jnp.array([[-0.667605, 0.3261746, -0.0785462]], dtype=np.float32),
                      b2=jnp.array([0.], dtype=np.float32),
                      w2=jnp.array([[0.464014], [-0.435685], [0.776788]], dtype=np.float32))
    gradient_state = AdamState(
        count=jnp.asarray(5),
        mu=Weights(b1=jnp.asarray([-15.569108, -8.185916, -18.872583]),
                   w1=jnp.asarray([[-6488.655, -5813.5786, -11111.309]]),
                   b2=jnp.asarray([-16.122942]),
                   w2=jnp.asarray([[-5100.7495], [-6862.2837], [-8967.359]])),
        nu=Weights(b1=jnp.asarray([7.211683, 1.9927658, 10.598419]),
                   w1=jnp.asarray([[1289447., 1035597.7, 3784737.8]]),
                   b2=jnp.asarray([7.749687]),
                   w2=jnp.asarray([[797202.94], [1442427.9], [2465843.5]])))

    state = SolutionState(gradient_state, weights)

    dataset = [2681.0000, 6406.0000, 2098.0000, 5384.0000, 5765.0000, 2273.0000] * 10
    for i, observation in enumerate(dataset):
        observation = jnp.asarray(observation)
        print_generic(iteration=i, weights=state.weights, observation=observation,
                      gs=state.gradient_state)
        state = train_one_episode(observation, state, gradient_transformation)
    print_generic(state)


@jit
def train_one_episode(observation: Array,
                      state: SolutionState,
                      gradient_transformation: Adam,
                      ) -> SolutionState:
    observations = jnp.reshape(observation, (1, 1))
    weights_bar, observation = _v_infer_gradient_and_value(observations, state.weights)
    new_weights_bar, new_gradient_state = gradient_transformation.update(
        weights_bar, state.gradient_state, state.weights)
    new_weights = tree_map(jnp.add, state.weights, new_weights_bar)
    return SolutionState(new_gradient_state, new_weights)


def _infer(observation: Array, weights: Weights) -> tuple[Array, Array]:
    seeker_loss = internal_infer_co(observation, weights)
    return seeker_loss, observation


def _infer_gradient_and_value(observation: Array, weights: Weights) -> tuple[Weights, Array]:
    bound_infer = partial(_infer, observation)
    f: Callable[[Array], tuple[Array, Array]] = grad(bound_infer, has_aux=True)
    return f(weights)


def _v_infer_gradient_and_value(observations: Array, weights: Weights) -> tuple[Weights, Array]:
    f = vmap(_infer_gradient_and_value, in_axes=(0, None), out_axes=(0, 0))
    weights_bars, infer_outputs = f(observations, weights)
    weights_bar = tree_map(partial(jnp.mean, axis=0), weights_bars)
    return weights_bar, infer_outputs


def odd_power(base: Array, exponent: Array) -> Array:
    return jnp.copysign(jnp.abs(base) ** exponent, base)


def energy(natural_explanation: Array, observation: Array, weights: Weights) -> Array:
    p = observation
    q = (dot(softplus(dot(natural_explanation, softplus(weights.w1)) + weights.b1),
             softplus(weights.w2))
         + weights.b2 + 1e-6 * odd_power(natural_explanation, jnp.array(3.0)))
    return dot(p - q, p) + 0.5 * jnp.sum(jnp.square(q)) - 0.5 * jnp.sum(jnp.square(p))


def internal_infer(observation: Array, weights: Weights) -> Array:
    minimizer = GradientDescent(energy, has_aux=False, maxiter=250, tol=0.001, acceleration=False)
    minimizer_result = minimizer.run(jnp.zeros(1), observation=observation, weights=weights)
    return jnp.sum(jnp.square(minimizer_result.params)) * 1e-1


_Weight_VJP = Callable[[Array], tuple[Array]]


@custom_vjp
def internal_infer_co(observation: Array, weights: Weights) -> Array:
    return internal_infer(observation, weights)


def internal_infer_co_fwd(observation: Array, weights: Weights) -> tuple[Array, _Weight_VJP]:
    return vjp(partial(internal_infer, observation), weights)


def internal_infer_co_bwd(weight_vjp: _Weight_VJP, _: Array) -> tuple[None, Array]:
    weights_bar, = weight_vjp(jnp.ones(()))
    return None, weights_bar


internal_infer_co.defvjp(internal_infer_co_fwd, internal_infer_co_bwd)
