from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

import haiku as hk
import jax.numpy as jnp
import numpy as np
from efax import MultivariateUnitNormalNP
from jax import Array, custom_vjp, enable_custom_prng, grad, jit, vjp, vmap
from jax._src.prng import PRNGKeyArray, threefry_prng_impl
from jax.lax import dot, stop_gradient
from jax.nn import softplus
from jax.random import PRNGKey, split
from jax.tree_util import tree_map
from jaxopt import GradientDescent
from tjax import RealArray, get_test_string, print_generic
from tjax.dataclasses import dataclass
from tjax.gradient import Adam, GradientState, GradientTransformation


@dataclass
class Weights:
    b1: RealArray
    w1: RealArray
    b2: RealArray
    w2: RealArray


def cli() -> None:
    with enable_custom_prng():
        encoding = RivalEncoding()
        gradient_transformation = Adam[hk.Params](1e-2)
        rl_inference = RLInference(encoding)
        weight_rng = PRNGKeyArray(threefry_prng_impl,
                                  jnp.array((2634740717, 3214329440), dtype=jnp.uint32))
        weights = Weights(w1=jnp.array([[-0.667605, 0.3261746, -0.0785462]], dtype=np.float32),
                          b1=jnp.array([0., 0., 0.], dtype=np.float32),
                          w2=jnp.array([[0.464014], [-0.435685], [0.776788]], dtype=np.float32),
                          b2=jnp.array([0.], dtype=np.float32))
        gradient_state = gradient_transformation.init(weights)
        state = SolutionState(gradient_state, weights)

        dataset = [6406.0000, 2098.0000, 5384.0000, 5765.0000, 2273.0000, 2681.0000]
        rng = PRNGKey(5)
        example_rng_base, _ = split(rng)
        example_rngs = split(example_rng_base, 5000)
        for i, observation in enumerate(dataset):
            observation = jnp.asarray(observation)
            print_generic(iteration=i, weights=state.weights, observation=observation,
                          gs=state.gradient_state)
            print(get_test_string(weights, 1e-6, 0.0))
            print(get_test_string(state.gradient_state, 1e-6, 0.0))
            state = rl_inference.train_one_episode(observation, state, gradient_transformation)
        print_generic(state)


@dataclass
class SolutionState:
    gradient_state: GradientState
    weights: RealArray


@dataclass
class RLInference:
    encoding: RivalEncoding

    @jit
    def train_one_episode(self,
                          observation: Array,
                          state: SolutionState,
                          gradient_transformation: GradientTransformation[Any, hk.Params],
                          ) -> SolutionState:
        observations = jnp.reshape(observation, (1, 1))
        weights_bar, observation = self._v_infer_gradient_and_value(observations,
                                                                    state.weights)
        new_weights_bar, new_gradient_state = gradient_transformation.update(
            weights_bar, state.gradient_state, state.weights)
        new_weights = tree_map(jnp.add, state.weights, new_weights_bar)
        return SolutionState(new_gradient_state, new_weights)

    def _infer(self,
               observation: Array,
               weights: RealArray) -> tuple[Array, Array]:
        inference_result = infer_encoding_configuration(self.encoding, observation, weights)
        new_model_loss = inference_result.dummy_loss
        return new_model_loss, observation

    def _infer_gradient_and_value(self,
                                  observation: Array,
                                  weights: RealArray) -> (
                                      tuple[hk.Params, Array]):
        bound_infer = partial(self._infer, observation)
        f: Callable[[hk.Params], tuple[hk.Params, Array]]
        f = grad(bound_infer, has_aux=True)
        return f(weights)

    def _v_infer_gradient_and_value(self,
                                    observations: Array,
                                    weights: RealArray) -> (
                                        tuple[hk.Params, Array]):
        f = vmap(self._infer_gradient_and_value, in_axes=(0, None), out_axes=(0, 0))
        weights_bars, infer_outputs = f(observations, weights)
        weights_bar = tree_map(partial(jnp.mean, axis=0), weights_bars)
        return weights_bar, infer_outputs


def odd_power(base: Array, exponent: Array) -> Array:
    return jnp.copysign(jnp.abs(base) ** exponent, base)


@dataclass
class RivalEncoding:
    def explanation_sp_energy(self,
                              natural_explanation: Array,
                              observation: Array,
                              weights: RealArray) -> Array:
        intermediate_explanation = (
            dot(softplus(dot(natural_explanation, softplus(weights.w1)) + weights.b1),
                softplus(weights.w2))
            + weights.b2 + 1e-6 * odd_power(natural_explanation, jnp.array(3.0)))
        exp_cls = MultivariateUnitNormalNP.expectation_parametrization_cls()
        expectation_parametrization = exp_cls.unflattened(observation)
        natural_parametrization = MultivariateUnitNormalNP.unflattened(intermediate_explanation)
        retval = expectation_parametrization.kl_divergence(natural_parametrization)
        assert isinstance(retval, Array)
        return retval


def internal_infer_encoding(encoding: RivalEncoding,
                            observation: Array,
                            weights: RealArray) -> EncodingInferenceResult:
    encoding = stop_gradient(encoding)
    observation = stop_gradient(observation)
    inferred_message = jnp.zeros(1)
    # Stop gradient at the initial inferred message because we don't want to train the variational
    # inference here.
    inferred_message = stop_gradient(inferred_message)

    energy_f = encoding.explanation_sp_energy
    minimizer = GradientDescent(energy_f, has_aux=False, maxiter=250, tol=0.001,
                                acceleration=False)
    # weights = print_cotangent(weights, name='weight_bar')
    # tapped_print_generic(weights=weights)
    # inferred_message = tapped_print_generic(inferred_message=inferred_message,
    #                                         observation=observation, weights=weights,
    #                                         result=inferred_message)
    minimizer_result = minimizer.run(inferred_message, observation=observation,
                                     weights=weights)
    # minimizer_result = print_cotangent(minimizer_result, name='minimizer_result_bar')
    # tapped_print_generic(minimizer_result=minimizer_result)

    # Calculate intermediate quantities.
    inferred_message = minimizer_result.params
    centering_loss = jnp.sum(jnp.square(inferred_message))

    # Calculate seeker loss.
    seeker_loss = SeekerLoss(centering_loss * 1e-1)
    return EncodingInferenceResult(observation, jnp.array(0.0), seeker_loss)


@dataclass
class SeekerLoss:
    centering_loss: Array

    @classmethod
    def cotangent(cls) -> SeekerLoss:
        return SeekerLoss(jnp.array(1.0))


@dataclass
class EncodingInferenceResult:
    observation: Array
    dummy_loss: Array
    seeker_loss: SeekerLoss

    @classmethod
    def cotangent(cls) -> EncodingInferenceResult:
        observation = jnp.zeros(1)
        seeker_loss = SeekerLoss.cotangent()
        return cls(observation, jnp.array(1.0), seeker_loss)


_Weight_VJP = Callable[[EncodingInferenceResult], tuple[hk.Params]]


@custom_vjp
def infer_encoding_configuration(encoding: RivalEncoding,
                                 observation: Array,
                                 weights: RealArray) -> EncodingInferenceResult:
    return internal_infer_encoding(encoding, observation, weights)


def infer_encoding_configuration_fwd(encoding: RivalEncoding,
                                     observation: Array,
                                     weights: RealArray) -> (
                                         tuple[EncodingInferenceResult, _Weight_VJP]):

    inference_result, weight_vjp = vjp(partial(internal_infer_encoding, encoding, observation),
                                       weights)
    return inference_result, weight_vjp


def infer_encoding_configuration_bwd(weight_vjp: _Weight_VJP,
                                     inference_result_bar: EncodingInferenceResult) -> (
                                         tuple[None, None, hk.Params]):
    internal_result_bar = EncodingInferenceResult.cotangent()
    weights_bar, = weight_vjp(internal_result_bar)
    return None, None, weights_bar


infer_encoding_configuration.defvjp(infer_encoding_configuration_fwd,
                                    infer_encoding_configuration_bwd)
