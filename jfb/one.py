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
from efax import (ExpectationParametrization, MultivariateFixedVarianceNormalNP,
                  NaturalParametrization)
from jax import Array, enable_custom_prng, grad, jit, jvp, vjp, vmap
from jax._src.prng import PRNGKeyArray, threefry_prng_impl
from jax.lax import dot, stop_gradient, while_loop
from jax.nn import softplus
from jax.random import KeyArray, PRNGKey, randint, split
from jax.tree_util import tree_map
from jaxopt import GradientDescent
from more_itertools import mark_ends
from tensorflow_datasets.core import DatasetInfo
from tensorflow_datasets.core.features import Tensor
from tjax import (BooleanNumeric, IntegralNumeric, RealArray, RealNumeric, custom_jvp, custom_vjp,
                  print_generic)
from tjax.dataclasses import dataclass, field
from tjax.gradient import Adam, GradientState, GradientTransformation


def cli() -> None:
    with enable_custom_prng():
        solution = create_deduction_solution()
        examples = 5000
        batch_size = 1

        # TRAIN!
        state = solution.state
        data_source = create_data_source()
        train_one_episode = jit(RLInference.train_one_episode, static_argnums=(3,))
        i = 0
        rng = PRNGKey(5)
        example_rng_base, _ = split(rng)
        example_rngs = split(example_rng_base, examples)
        for example_rng in example_rngs:
            observation = data_source.initial_state(example_rng)
            print_generic(iteration=i, observation=observation)
            training_result = train_one_episode(solution.rl_inference, observation,
                                                state.model_weights, batch_size,
                                                state.gradient_state,
                                                solution.gradient_transformation)
            state = SolutionState(training_result.gradient_state, training_result.model_weights)
            i += 1
        print_generic(state)


def expectation_parameters(feature_info: Any, value: Any) -> RealArray:
    match feature_info:
        case Tensor(shape=shape):
            assert isinstance(value, np.ndarray)
            assert value.shape[1:] == shape
            new_shape = (value.shape[0], -1)
            return np.reshape(value, new_shape)
    raise TypeError


@p_dataclass
class DeductionDataSource:
    info: DatasetInfo
    dataset: list[tuple[Any, Any]]

    def initial_state(self, example_rng: KeyArray) -> RealArray:
        index = randint(example_rng, (), 0, len(self.dataset))
        _, ds_label = self.dataset[index]
        target: None | RealArray = None
        for key, feature_info in self.info.features.items():  # pyright: ignore
            if key == 'price':
                target = expectation_parameters(feature_info, ds_label)
        assert target is not None
        return target


def create_data_source() -> DeductionDataSource:
    ds_map, info = tfds.load('diamonds', as_supervised=True, with_info=True,
                                batch_size=1)
    ds = ds_map['train']
    ds_numpy = tfds.as_numpy(ds)
    assert isinstance(info, DatasetInfo)
    assert isinstance(ds_numpy, Iterable)  # Iterable[tuple[Any, Any]]
    dataset = list(ds_numpy)
    print(f"Loaded dataset of size {len(dataset)}")
    return DeductionDataSource(info, dataset)


def create_deduction_solution() -> TrainingSolution:
    weight_rng = PRNGKeyArray(threefry_prng_impl,
                              jnp.array((2634740717, 3214329440), dtype=jnp.uint32))
    distribution_cls = MultivariateFixedVarianceNormalNP
    kwargs = {'variance': jnp.asarray(1.0)}
    distribution_info = DistributionInfo(1, distribution_cls, kwargs)

    encoding = RivalEncoding(space_features=distribution_info.features,
                             gln_layer_sizes=(3,),
                             distribution_info=distribution_info)
    gradient_transformation = Adam[hk.Params](1e-2)
    return TrainingSolution.create(encoding, gradient_transformation, weight_rng)


@dataclass
class RLTrainingResult:
    gradient_state: GradientState
    model_weights: hk.Params


TST = TypeVar('TST', bound='TrainingSolution')


@dataclass
class TrainingSolution:
    """
    The TrainingSolution is everything needed to keep track of the solution during training:
        rl_inference: The model and the problem.
        gradient_transformation: The way in which the state is updated.
        state: The state of weights, the gradient transformation, and the RNG.
    """
    rl_inference: RLInference
    gradient_transformation: SolutionGT
    state: SolutionState

    @classmethod
    def create(cls: type[TST],
               encoding: RivalEncoding,
               gradient_transformation: SolutionGT,
               weight_rng: KeyArray) -> TST:
        rl_inference = RLInference(encoding)
        solution_state = SolutionState.create(gradient_transformation, encoding, weight_rng)
        return cls(rl_inference, gradient_transformation, solution_state)


SolutionGT = GradientTransformation[Any, hk.Params]
SST = TypeVar('SST', bound='SolutionState')


@dataclass
class SolutionState:
    """
    The SolutionState is iterated by the SolutionTrainer during training.
    """
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
    """
    This class knows how to take flattened natural parameters and create a natural parametrization
    object.
    """
    features: int = field(static=True)
    nat_cls: type[NaturalParametrization[Any, Any]] = field(static=True)
    dist_kwargs: dict[str, Any] = field(static=True)

    # Methods --------------------------------------------------------------------------------------
    def expectation_parameter_cls(self) -> type[ExpectationParametrization[Any]]:
        return self.nat_cls.expectation_parametrization_cls()

    def expectation_parametrization(self, expectation_parameters: RealArray
                                    ) -> ExpectationParametrization[Any]:
        exp_cls = self.expectation_parameter_cls()
        return exp_cls.unflattened(expectation_parameters, **self.dist_kwargs)

    def natural_parametrization(self, flattened_natural_parameters: RealArray
                                ) -> NaturalParametrization[Any, Any]:
        return self.nat_cls.unflattened(flattened_natural_parameters, **self.dist_kwargs)

    def value_error(self,
                    expectation_observation: RealArray,
                    natural_explanation: RealArray
                    ) -> RealNumeric:
        expectation_parametrization = self.expectation_parametrization(expectation_observation)
        natural_parametrization = self.natural_parametrization(natural_explanation)
        return expectation_parametrization.kl_divergence(natural_parametrization)


@dataclass
class _InferenceState:
    step: IntegralNumeric
    done: BooleanNumeric
    observations: RealArray


@dataclass
class _TrainingState:
    inference_state: _InferenceState
    gradient_state: GradientState
    model_weights: hk.Params


@dataclass
class _InferOutput:
    done: BooleanNumeric
    observation: RealArray


@dataclass
class RLInference:
    """
    This class iterates the parameters of the model.  Each step of the iteration corresponds to
    a single step of a reinforcement learning trajectory.

    An RLInference object is stored by SolutionTrainer, which is an instance IteratedFunction.
    """
    encoding: RivalEncoding

    # New methods ----------------------------------------------------------------------------------
    def train_one_episode(self,
                          observation: RealArray,
                          model_weights: hk.Params,
                          batch_size: int,
                          gradient_state: GradientState,
                          gradient_transformation: GradientTransformation[Any, hk.Params],
                          ) -> RLTrainingResult:
        """
        Start batch_size parallel agents and run them for one episode.  The agents all share
        parameters, which trained after every time step.
        """
        inference_state = self._set_up_inference(observation, batch_size)
        training_state = _TrainingState(inference_state, gradient_state, model_weights)
        condition_function = partial(self._training_cond_fun, 1)
        body_function = partial(self._training_body_fun, gradient_transformation)
        training_state = while_loop(condition_function, body_function, training_state)
        return RLTrainingResult(training_state.gradient_state,
                                training_state.model_weights)

    # Private methods ------------------------------------------------------------------------------
    def _set_up_inference(self,
                          observation: RealArray,
                          batch_size: int) -> _InferenceState:
        observations = observation
        return _InferenceState(0, False, observations)

    def _update_inference_state(self,
                                inference_state: _InferenceState,
                                infer_outputs: _InferOutput) -> _InferenceState:
        new_step = inference_state.step + 1
        new_done = jnp.all(infer_outputs.done)
        return _InferenceState(new_step, new_done, infer_outputs.observation)

    def _infer(self,
               observation: RealArray,
               model_weights: hk.Params) -> tuple[RealNumeric, _InferOutput]:
        # Infer action.
        assert isinstance(observation, Array)
        inference_result = infer_encoding_configuration(self.encoding, observation, model_weights)
        new_model_loss = inference_result.dummy_loss

        done: BooleanNumeric = True
        # Produce output.
        infer_output = _InferOutput(done, observation)
        return new_model_loss, infer_output

    def _infer_gradient_and_value(self,
                                  observation: RealArray,
                                  model_weights: hk.Params) -> (
                                      tuple[hk.Params, _InferOutput]):
        bound_infer = partial(self._infer, observation)
        f: Callable[[hk.Params],
                    tuple[hk.Params, _InferOutput]]
        f = grad(bound_infer, has_aux=True)
        return f(model_weights)

    def _v_infer_gradient_and_value(self,
                                    observations: RealArray,
                                    model_weights: hk.Params) -> (
                                        tuple[hk.Params, _InferOutput]):
        f = vmap(self._infer_gradient_and_value,
                 in_axes=(0, None),
                 out_axes=(0, 0))
        weights_bars, infer_outputs = f(observations, model_weights)
        weights_bar = tree_map(partial(jnp.mean, axis=0), weights_bars)
        return weights_bar, infer_outputs

    def _training_cond_fun(self,
                           max_episode_steps: int,
                           training_state: _TrainingState) -> BooleanNumeric:
        return jnp.logical_and(training_state.inference_state.step < max_episode_steps,
                               jnp.logical_not(training_state.inference_state.done))

    def _training_body_fun(self,
                           gradient_transformation: GradientTransformation[Any, hk.Params],
                           training_state: _TrainingState) -> _TrainingState:
        inference_state = training_state.inference_state
        # Infer weight gradient.
        weights_bar, infer_outputs = self._v_infer_gradient_and_value(
            inference_state.observations, training_state.model_weights)

        # Transform the weight gradient using the gradient transformation and update its state.
        new_weights_bar, new_gradient_state = gradient_transformation.update(
            weights_bar, training_state.gradient_state, training_state.model_weights)

        # Update the training state.
        new_inference_state = self._update_inference_state(inference_state, infer_outputs)
        new_weights = tree_map(jnp.add, training_state.model_weights, new_weights_bar)
        return _TrainingState(new_inference_state, new_gradient_state, new_weights)


def odd_power(base: RealArray, exponent: RealNumeric) -> RealArray:
    return jnp.copysign(jnp.abs(base) ** exponent, base)


@dataclass
class RivalEncoding:
    _: KW_ONLY
    space_features: int = field(static=True)
    gln_layer_sizes: tuple[int, ...] = field(static=True)
    distribution_info: DistributionInfo

    # New methods ----------------------------------------------------------------------------------
    def create_weights(self, rng: KeyArray) -> hk.Params:
        transformed = hk.transform(self.haiku_weight_initializer)
        return transformed.init(rng)

    def explanation_sp_energy(self,
                              natural_explanation: RealArray,
                              observation: RealArray,
                              weights: hk.Params) -> RealNumeric:
        assert natural_explanation.shape == (self.space_features,)
        return self._explanation_energy(natural_explanation, observation, weights)

    def intermediate_explanation(self,
                                 natural_explanation: RealArray,
                                 weights: hk.Params,
                                 *,
                                 rng: None | KeyArray = None) -> RealArray:
        """
        Args:
            natural_explanation: The natural parameters of the explanation.
            weights: The model weights.
            rng: The random number generator to generate any noises.
        Returns: The expectation parameters of the explanation.
        """
        assert natural_explanation.shape == (self.space_features,)
        intermediate_explanation_f = hk.transform(self._intermediate_explanation).apply
        return intermediate_explanation_f(weights, rng, natural_explanation)

    # Augmented Haiku methods ----------------------------------------------------------------------
    def haiku_weight_initializer(self) -> None:
        natural_explanation = jnp.zeros(self.space_features)
        self._intermediate_explanation(natural_explanation)

    # Private Haiku methods ------------------------------------------------------------------------
    def _gln_module(self) -> NoisyMLP:
        return NoisyMLP(self.gln_layer_sizes, self.space_features, activation=softplus, name='gln')

    def _intermediate_explanation(self, natural_explanation: RealArray) -> RealArray:
        """
        Args:
            natural_explanation: The natural parameters of the explanation.
        Returns: The intermediate parameters of the explanation.
        """
        gln_mlp = self._gln_module()
        value = gln_mlp(natural_explanation)
        value += 1e-6 * odd_power(natural_explanation, 3.0)
        return value

    # Private methods ------------------------------------------------------------------------------
    def _explanation_energy(self,
                            natural_explanation: RealArray,
                            observation: RealArray,
                            weights: hk.Params) -> RealNumeric:
        intermediate_explanation = self.intermediate_explanation(natural_explanation, weights)
        return self.distribution_info.value_error(observation, intermediate_explanation)

    # TODO: When mypy supports ParamSpec, use a decorator.
    _explanation_energy = custom_jvp(_explanation_energy,  # type: ignore[assignment]
                                     nondiff_argnums=(0,))

    @_explanation_energy.defjvp  # type: ignore[no-redef, attr-defined, misc]
    def _(self,
          primals: tuple[RealArray, RealArray, hk.Params],
          tangents: tuple[RealArray, RealArray, hk.Params]
          ) -> tuple[RealNumeric, RealNumeric]:
        natural_explanation, observation, weights = primals
        natural_explanation_dot, observation_dot, weights_dot = tangents

        # Block natural_explanation_dot cotangent since it is produced below.
        intermediate_explanation, intermediate_explanation_dot = jvp(
            self.intermediate_explanation,
            (natural_explanation, weights),
            (tree_map(jnp.zeros_like, natural_explanation_dot), weights_dot))

        # The intermediate_explanation_dot cotangent is propagated to weights_dot.
        energy_a, energy_dot_a = jvp(self.distribution_info.value_error,
                                     (observation, intermediate_explanation),
                                     (jnp.zeros_like(observation_dot),
                                      intermediate_explanation_dot))
        # Substitute natural_explanation_dot for intermediate_explanation_dot.
        energy_b, energy_dot_b = jvp(self.distribution_info.value_error,
                                     (observation, intermediate_explanation),
                                     (observation_dot, natural_explanation_dot))
        return energy_a + energy_b, energy_dot_a + energy_dot_b


# Un-exported functions ----------------------------------------------------------------------------
def internal_infer_encoding(encoding: RivalEncoding,
                            observation: RealArray,
                            weights: hk.Params) -> EncodingInferenceResult:
    encoding = stop_gradient(encoding)
    observation = stop_gradient(observation)
    seeker_loss = seeker_inference(encoding, observation, weights)
    return EncodingInferenceResult(observation, 0.0, seeker_loss)


# Un-exported types --------------------------------------------------------------------------------
U = TypeVar('U', bound='EncodingInferenceResult')


# Exported classes ---------------------------------------------------------------------------------
@dataclass
class SeekerLoss:
    centering_loss: RealNumeric

    @classmethod
    def zeros(cls) -> SeekerLoss:
        return SeekerLoss(0.0)

    @classmethod
    def cotangent(cls) -> SeekerLoss:
        return SeekerLoss(1.0)


# Un-exported functions ----------------------------------------------------------------------------
def seeker_inference(encoding: RivalEncoding,
                     observation: RealArray,
                     weights: hk.Params
                     ) -> SeekerLoss:
    inferred_message = jnp.zeros(encoding.space_features)
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
    return seeker_loss


EIRT = TypeVar('EIRT', bound='EncodingInferenceResult')


@dataclass
class EncodingInferenceResult:
    observation: RealArray
    dummy_loss: RealNumeric
    seeker_loss: SeekerLoss

    @classmethod
    def zeros(cls: type[EIRT], encoding: RivalEncoding) -> EIRT:
        observation = jnp.zeros(encoding.space_features)
        seeker_loss = SeekerLoss.zeros()
        return cls(observation, 0.0, seeker_loss)

    @classmethod
    def cotangent(cls: type[EIRT], encoding: RivalEncoding) -> EIRT:
        observation = jnp.zeros(encoding.space_features)
        seeker_loss = SeekerLoss.cotangent()
        return cls(observation, 1.0, seeker_loss)


@custom_vjp
def infer_encoding_configuration(encoding: RivalEncoding,
                                 observation: RealArray,
                                 weights: hk.Params) -> EncodingInferenceResult:
    """
    Args:
        encoding: The encoding whose trajectory is inferred.
        observation: The observation at the encoding.
        rng: The random number generator.
        weights: The weights of the encoding.
    Returns:
        inference_result: The encoding's inference result.
    """
    return internal_infer_encoding(encoding, observation, weights)


def infer_encoding_configuration_fwd(encoding: RivalEncoding,
                                     observation: RealArray,
                                     weights: hk.Params) -> (
                                         tuple[EncodingInferenceResult, _EncodingResiduals]):

    # Run inference VJP.
    inference_result, weight_vjp = vjp(partial(internal_infer_encoding, encoding, observation),
                                      weights)
    residuals = _EncodingResiduals(weight_vjp, encoding)
    return inference_result, residuals


def infer_encoding_configuration_bwd(residuals: _EncodingResiduals,
                                     inference_result_bar: EncodingInferenceResult) -> (
                                         tuple[None, None, hk.Params]):
    """
    Create weight cotangents using the weight VJP.
    """
    # This produces a zeroed out internal result cotangent.  The weights within an encoding node do
    # not depend on any cotangents to that node, but the cotangent is needed to run the VJP.
    internal_result_bar = EncodingInferenceResult.cotangent(residuals.encoding)
    weights_bar, = residuals.weight_vjp(internal_result_bar)
    return None, None, weights_bar


# Private types ------------------------------------------------------------------------------------
_Weight_VJP = Callable[[EncodingInferenceResult], tuple[hk.Params]]


@dataclass
class _EncodingResiduals:
    weight_vjp: _Weight_VJP  # For the weight cotangent.
    encoding: RivalEncoding


# Bind vjp -----------------------------------------------------------------------------------------
infer_encoding_configuration.defvjp(infer_encoding_configuration_fwd,
                                    infer_encoding_configuration_bwd)


class NoisyMLP(hk.Module):
    """
    The noisy MLP is an MLP with optional Gaussian noise added to the signals.
    """
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
            layers.append(Dense(output_features=layer_output_features,
                                name=f"linear_{index}"))
        self.layers = tuple[Dense, ...](layers)

    def __call__(self, inputs: RealArray) -> RealArray:
        value = inputs
        for _, is_last, layer in mark_ends(self.layers):
            value = layer(value)
            if not is_last:
                value = self.activation(value)
        return value


class Dense(hk.Module):
    def __init__(self,
                 output_features: int,
                 *,
                 name: None | str = None):
        super().__init__(name=name)
        self.output_features = output_features

    def __call__(self, inputs: RealArray) -> RealArray:
        shape = inputs.shape
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
