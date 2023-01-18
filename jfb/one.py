from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import KW_ONLY
from dataclasses import dataclass as p_dataclass
from functools import partial
from itertools import chain
from typing import Any, TypeVar

import haiku as hk
import jax.numpy as jnp
import numpy as np
import tensorflow_datasets as tfds
from efax import MultivariateUnitNormalNP, NaturalParametrization
from jax import enable_custom_prng, grad, jit, vjp, vmap
from jax._src.prng import PRNGKeyArray, threefry_prng_impl
from jax.lax import dot, stop_gradient
from jax.nn import softplus
from jax.random import KeyArray, PRNGKey, randint, split
from jax.tree_util import tree_map
from jaxopt import GradientDescent
from more_itertools import mark_ends
from tensorflow_datasets.core import DatasetInfo
from tjax import RealArray, RealNumeric, custom_vjp, print_generic
from tjax.dataclasses import dataclass, field
from tjax.gradient import Adam, GradientState, GradientTransformation


def cli() -> None:
    with enable_custom_prng():
        weight_rng = PRNGKeyArray(threefry_prng_impl,
                                  jnp.array((2634740717, 3214329440), dtype=jnp.uint32))
        distribution_cls = MultivariateUnitNormalNP
        distribution_info = DistributionInfo(distribution_cls)
        encoding = RivalEncoding(distribution_info=distribution_info)
        gradient_transformation = Adam[hk.Params](1e-2)
        rl_inference = RLInference(encoding)
        state = SolutionState.create(gradient_transformation, encoding, weight_rng)

        data_source = create_data_source()
        train_one_episode = jit(RLInference.train_one_episode)
        rng = PRNGKey(5)
        example_rng_base, _ = split(rng)
        example_rngs = split(example_rng_base, 5000)
        for i, example_rng in enumerate(example_rngs):
            observation = data_source.initial_state(example_rng)
            print_generic(iteration=i, observation=observation)
            training_result = train_one_episode(rl_inference, observation,
                                                state.model_weights, state.gradient_state,
                                                gradient_transformation)
            state = SolutionState(training_result.gradient_state, training_result.model_weights)
        print_generic(state)


@p_dataclass
class DeductionDataSource:
    info: DatasetInfo
    dataset: list[tuple[Any, Any]]

    def initial_state(self, example_rng: KeyArray) -> RealArray:
        index = randint(example_rng, (), 0, len(self.dataset))
        _, target = self.dataset[index]
        assert target is not None
        return target


def create_data_source() -> DeductionDataSource:
    ds_map, info = tfds.load('diamonds', as_supervised=True, with_info=True, batch_size=1)
    ds = ds_map['train']
    ds_numpy = tfds.as_numpy(ds)
    assert isinstance(info, DatasetInfo)
    assert isinstance(ds_numpy, Iterable)  # Iterable[tuple[Any, Any]]
    dataset = list(ds_numpy)
    print(f"Loaded dataset of size {len(dataset)}")
    return DeductionDataSource(info, dataset)


SolutionGT = GradientTransformation[Any, hk.Params]
SST = TypeVar('SST', bound='SolutionState')


@dataclass
class SolutionState:
    gradient_state: GradientState
    model_weights: hk.Params

    @classmethod
    def create(cls: type[SST],
               gradient_transformation: SolutionGT,
               element: RivalEncoding,
               weight_rng: KeyArray) -> SST:
        weights = element.create_weights(weight_rng)
        gradient_state = gradient_transformation.init(weights)
        return cls(gradient_state, weights)


@dataclass
class DistributionInfo:
    nat_cls: type[NaturalParametrization[Any, Any]] = field(static=True)

    def value_error(self,
                    expectation_observation: RealArray,
                    natural_explanation: RealArray
                    ) -> RealNumeric:
        exp_cls = self.nat_cls.expectation_parametrization_cls()
        expectation_parametrization = exp_cls.unflattened(expectation_observation)
        natural_parametrization = self.nat_cls.unflattened(natural_explanation)
        return expectation_parametrization.kl_divergence(natural_parametrization)


@dataclass
class RLInference:
    encoding: RivalEncoding

    def train_one_episode(self,
                          observation: RealArray,
                          model_weights: hk.Params,
                          gradient_state: GradientState,
                          gradient_transformation: GradientTransformation[Any, hk.Params],
                          ) -> SolutionState:
        observations = jnp.reshape(observation, (1, 1))
        weights_bar, observation = self._v_infer_gradient_and_value(observations, model_weights)
        new_weights_bar, new_gradient_state = gradient_transformation.update(
            weights_bar, gradient_state, model_weights)
        new_weights = tree_map(jnp.add, model_weights, new_weights_bar)
        return SolutionState(new_gradient_state, new_weights)

    def _infer(self,
               observation: RealArray,
               model_weights: hk.Params) -> tuple[RealNumeric, RealArray]:
        inference_result = infer_encoding_configuration(self.encoding, observation, model_weights)
        new_model_loss = inference_result.dummy_loss
        return new_model_loss, observation

    def _infer_gradient_and_value(self,
                                  observation: RealArray,
                                  model_weights: hk.Params) -> (
                                      tuple[hk.Params, RealArray]):
        bound_infer = partial(self._infer, observation)
        f: Callable[[hk.Params], tuple[hk.Params, RealArray]]
        f = grad(bound_infer, has_aux=True)
        return f(model_weights)

    def _v_infer_gradient_and_value(self,
                                    observations: RealArray,
                                    model_weights: hk.Params) -> (
                                        tuple[hk.Params, RealArray]):
        f = vmap(self._infer_gradient_and_value, in_axes=(0, None), out_axes=(0, 0))
        weights_bars, infer_outputs = f(observations, model_weights)
        weights_bar = tree_map(partial(jnp.mean, axis=0), weights_bars)
        return weights_bar, infer_outputs


def odd_power(base: RealArray, exponent: RealNumeric) -> RealArray:
    return jnp.copysign(jnp.abs(base) ** exponent, base)


@dataclass
class RivalEncoding:
    _: KW_ONLY
    distribution_info: DistributionInfo

    def create_weights(self, rng: KeyArray) -> hk.Params:
        transformed = hk.transform(self.haiku_weight_initializer)
        return transformed.init(rng)

    def explanation_sp_energy(self,
                              natural_explanation: RealArray,
                              observation: RealArray,
                              weights: hk.Params) -> RealNumeric:
        intermediate_explanation_f = hk.transform(self._intermediate_explanation).apply
        intermediate_explanation = intermediate_explanation_f(weights, None, natural_explanation)
        return self.distribution_info.value_error(observation, intermediate_explanation)

    def haiku_weight_initializer(self) -> None:
        natural_explanation = jnp.zeros(1)
        self._intermediate_explanation(natural_explanation)

    def _intermediate_explanation(self, natural_explanation: RealArray) -> RealArray:
        gln_mlp = NoisyMLP((3,), 1, activation=softplus, name='gln')
        return gln_mlp(natural_explanation) + 1e-6 * odd_power(natural_explanation, 3.0)


def internal_infer_encoding(encoding: RivalEncoding,
                            observation: RealArray,
                            weights: hk.Params) -> EncodingInferenceResult:
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
    return EncodingInferenceResult(observation, 0.0, seeker_loss)


@dataclass
class SeekerLoss:
    centering_loss: RealNumeric

    @classmethod
    def cotangent(cls) -> SeekerLoss:
        return SeekerLoss(1.0)


EIRT = TypeVar('EIRT', bound='EncodingInferenceResult')


@dataclass
class EncodingInferenceResult:
    observation: RealArray
    dummy_loss: RealNumeric
    seeker_loss: SeekerLoss

    @classmethod
    def cotangent(cls: type[EIRT], encoding: RivalEncoding) -> EIRT:
        observation = jnp.zeros(1)
        seeker_loss = SeekerLoss.cotangent()
        return cls(observation, 1.0, seeker_loss)


@custom_vjp
def infer_encoding_configuration(encoding: RivalEncoding,
                                 observation: RealArray,
                                 weights: hk.Params) -> EncodingInferenceResult:
    return internal_infer_encoding(encoding, observation, weights)


def infer_encoding_configuration_fwd(encoding: RivalEncoding,
                                     observation: RealArray,
                                     weights: hk.Params) -> (
                                         tuple[EncodingInferenceResult, _EncodingResiduals]):

    inference_result, weight_vjp = vjp(partial(internal_infer_encoding, encoding, observation),
                                       weights)
    residuals = _EncodingResiduals(weight_vjp, encoding)
    return inference_result, residuals


def infer_encoding_configuration_bwd(residuals: _EncodingResiduals,
                                     inference_result_bar: EncodingInferenceResult) -> (
                                         tuple[None, None, hk.Params]):
    internal_result_bar = EncodingInferenceResult.cotangent(residuals.encoding)
    weights_bar, = residuals.weight_vjp(internal_result_bar)
    return None, None, weights_bar


_Weight_VJP = Callable[[EncodingInferenceResult], tuple[hk.Params]]


@dataclass
class _EncodingResiduals:
    weight_vjp: _Weight_VJP  # For the weight cotangent.
    encoding: RivalEncoding


infer_encoding_configuration.defvjp(infer_encoding_configuration_fwd,
                                    infer_encoding_configuration_bwd)


class NoisyMLP(hk.Module):
    def __init__(self,
                 layer_sizes: tuple[int, ...],
                 output_features: int,
                 *,
                 activation: Callable[[RealArray], RealArray],
                 name: None | str = None):
        super().__init__(name=name)
        self.layer_sizes = layer_sizes
        self.output_features = output_features
        self.activation = activation
        layers: list[Dense] = []
        for index, layer_output_features in enumerate(chain(layer_sizes, (output_features,))):
            layers.append(Dense(output_features=layer_output_features, name=f"linear_{index}"))
        self.layers = tuple[Dense, ...](layers)

    def __call__(self, inputs: RealArray) -> RealArray:
        value = inputs
        for _, is_last, layer in mark_ends(self.layers):
            value = layer(value)
            if not is_last:
                value = self.activation(value)
        return value


class Dense(hk.Module):
    def __init__(self, output_features: int, *, name: None | str = None):
        super().__init__(name=name)
        self.output_features = output_features

    def __call__(self, inputs: RealArray) -> RealArray:
        input_features = inputs.shape[-1]
        dtype = inputs.dtype
        stddev = 0. if input_features == 0 else 1. / np.sqrt(input_features)
        kernel_init = hk.initializers.TruncatedNormal(stddev=stddev)
        kernel: RealArray = hk.get_parameter("w", [input_features, self.output_features], dtype,
                                             init=kernel_init)
        kernel = softplus(kernel)
        y = dot(inputs, kernel)  # type: ignore[arg-type]
        b: RealArray = hk.get_parameter("b", [self.output_features], dtype, init=jnp.zeros)
        return y + b
