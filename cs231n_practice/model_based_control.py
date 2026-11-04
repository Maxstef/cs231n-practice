"""Small NumPy utilities for learned dynamics and model-predictive control."""

from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Callable

import numpy as np


def _positive_integer(value: int, *, name: str) -> int:
    """Validate a positive integer without accepting Boolean values."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _finite_real(value: float, *, name: str) -> float:
    """Validate a finite real scalar."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be numeric")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _state(state: np.ndarray | tuple[float, float], *, name: str) -> np.ndarray:
    """Return one finite floating-point point-mass state with shape (2,)."""
    state = np.asarray(state, dtype=np.float64)
    if state.shape != (2,):
        raise ValueError(f"{name} must have shape (2,)")
    if not np.isfinite(state).all():
        raise ValueError(f"{name} must contain only finite values")
    return state


@dataclass
class PointMass1D:
    """A one-dimensional point mass with damping and quadratic drag.

    The state stores ``(position, velocity)`` and the scalar action represents
    bounded acceleration. Velocity is updated before position.
    """

    time_step: float = 0.2
    damping: float = 0.96
    drag: float = 0.1
    action_limit: float = 1.0
    state: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.time_step = _finite_real(self.time_step, name="time_step")
        self.damping = _finite_real(self.damping, name="damping")
        self.drag = _finite_real(self.drag, name="drag")
        self.action_limit = _finite_real(self.action_limit, name="action_limit")
        if self.time_step <= 0.0:
            raise ValueError("time_step must be positive")
        if not 0.0 <= self.damping <= 1.0:
            raise ValueError("damping must be between 0 and 1")
        if self.drag < 0.0:
            raise ValueError("drag must be nonnegative")
        if self.action_limit <= 0.0:
            raise ValueError("action_limit must be positive")
        self.state = np.zeros(2, dtype=np.float64)

    def reset(self, state: tuple[float, float] | np.ndarray = (-2.0, 0.0)) -> np.ndarray:
        """Set and return a copy of the current ``(position, velocity)`` state."""
        self.state = _state(state, name="state").copy()
        return self.state.copy()

    def step(self, action: float) -> np.ndarray:
        """Apply one bounded acceleration and return the next state."""
        action = _finite_real(action, name="action")
        action = float(np.clip(action, -self.action_limit, self.action_limit))
        position, velocity = self.state
        next_velocity = (
            self.damping * velocity
            + self.time_step * action
            - self.drag * velocity * abs(velocity)
        )
        next_position = position + self.time_step * next_velocity
        self.state = np.array([next_position, next_velocity])
        return self.state.copy()


@dataclass(frozen=True)
class LinearDynamicsModel:
    """Affine one-step model from ``(position, velocity, action)`` to state."""

    weights: np.ndarray

    def __post_init__(self) -> None:
        weights = np.asarray(self.weights, dtype=np.float64)
        if weights.shape != (4, 2):
            raise ValueError("weights must have shape (4, 2)")
        if not np.isfinite(weights).all():
            raise ValueError("weights must contain only finite values")
        object.__setattr__(self, "weights", weights)

    def predict(self, states: np.ndarray, actions: np.ndarray | float) -> np.ndarray:
        """Predict next states for one or more state-action pairs.

        A single state is promoted from shape ``(2,)`` to ``(1, 2)``. Scalar
        actions and action vectors are similarly normalized to ``(N, 1)``.
        The returned value always has shape ``(N, 2)``.
        """
        states = np.asarray(states, dtype=np.float64)
        actions = np.asarray(actions, dtype=np.float64)
        if states.ndim == 1:
            states = states[None, :]
        if states.ndim != 2 or states.shape[1] != 2:
            raise ValueError("states must have shape (2,) or (N, 2)")
        if actions.ndim == 0:
            actions = actions.reshape(1, 1)
        elif actions.ndim == 1:
            actions = actions[:, None]
        if actions.ndim != 2 or actions.shape[1] != 1:
            raise ValueError("actions must be scalar, (N,), or (N, 1)")
        if len(states) != len(actions):
            raise ValueError("states and actions must have equal batch sizes")
        if not np.isfinite(states).all() or not np.isfinite(actions).all():
            raise ValueError("states and actions must contain only finite values")
        features = np.concatenate(
            [states, actions, np.ones((len(states), 1))],
            axis=1,
        )
        return features @ self.weights


def collect_transition_data(
    number_transitions: int,
    generator: np.random.Generator,
    *,
    environment_factory: Callable[[], PointMass1D] = PointMass1D,
    position_range: tuple[float, float] = (-3.0, 3.0),
    velocity_range: tuple[float, float] = (-1.5, 1.5),
    action_range: tuple[float, float] = (-1.0, 1.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample independent states and actions and record true next states."""
    number_transitions = _positive_integer(
        number_transitions,
        name="number_transitions",
    )
    if not isinstance(generator, np.random.Generator):
        raise TypeError("generator must be a numpy.random.Generator")
    ranges = (position_range, velocity_range, action_range)
    for value_range in ranges:
        if len(value_range) != 2 or not value_range[0] < value_range[1]:
            raise ValueError("sampling ranges must contain increasing bounds")

    environment = environment_factory()
    states = np.empty((number_transitions, 2))
    actions = np.empty((number_transitions, 1))
    next_states = np.empty((number_transitions, 2))
    for index in range(number_transitions):
        state = np.array(
            [
                generator.uniform(*position_range),
                generator.uniform(*velocity_range),
            ]
        )
        action = generator.uniform(*action_range)
        environment.reset(state)
        states[index] = state
        actions[index, 0] = action
        next_states[index] = environment.step(action)
    return states, actions, next_states


def fit_linear_dynamics(
    states: np.ndarray,
    actions: np.ndarray,
    next_states: np.ndarray,
) -> LinearDynamicsModel:
    """Fit an affine one-step model by ordinary linear least squares."""
    states = np.asarray(states, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    next_states = np.asarray(next_states, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != 2 or len(states) == 0:
        raise ValueError("states must have nonempty shape (N, 2)")
    if actions.shape != (len(states), 1):
        raise ValueError("actions must have shape (N, 1)")
    if next_states.shape != states.shape:
        raise ValueError("next_states must have shape (N, 2)")
    if not all(np.isfinite(array).all() for array in (states, actions, next_states)):
        raise ValueError("training arrays must contain only finite values")
    features = np.concatenate(
        [states, actions, np.ones((len(states), 1))],
        axis=1,
    )
    weights = np.linalg.lstsq(features, next_states, rcond=None)[0]
    return LinearDynamicsModel(weights)


def compare_rollouts(
    model: LinearDynamicsModel,
    initial_state: tuple[float, float] | np.ndarray,
    action_sequence: np.ndarray,
    *,
    environment_factory: Callable[[], PointMass1D] = PointMass1D,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply identical actions to the true system and learned model."""
    actions = np.asarray(action_sequence, dtype=np.float64)
    if actions.ndim != 1 or len(actions) == 0:
        raise ValueError("action_sequence must be a nonempty vector")
    environment = environment_factory()
    true_state = environment.reset(initial_state)
    model_state = _state(initial_state, name="initial_state").copy()
    true_states = [true_state.copy()]
    model_states = [model_state.copy()]
    for action in actions:
        true_state = environment.step(action)
        model_state = model.predict(model_state, action)[0]
        true_states.append(true_state.copy())
        model_states.append(model_state.copy())
    return np.asarray(true_states), np.asarray(model_states)


def sample_action_sequences(
    number_candidates: int,
    horizon: int,
    block_length: int,
    generator: np.random.Generator,
    *,
    action_limit: float = 1.0,
) -> np.ndarray:
    """Sample piecewise-constant scalar action sequences."""
    number_candidates = _positive_integer(number_candidates, name="number_candidates")
    horizon = _positive_integer(horizon, name="horizon")
    block_length = _positive_integer(block_length, name="block_length")
    action_limit = _finite_real(action_limit, name="action_limit")
    if action_limit <= 0.0:
        raise ValueError("action_limit must be positive")
    if not isinstance(generator, np.random.Generator):
        raise TypeError("generator must be a numpy.random.Generator")
    number_blocks = int(np.ceil(horizon / block_length))
    block_actions = generator.uniform(
        -action_limit,
        action_limit,
        size=(number_candidates, number_blocks),
    )
    return np.repeat(block_actions, block_length, axis=1)[:, :horizon]


def predict_action_sequences(
    model: LinearDynamicsModel,
    initial_state: tuple[float, float] | np.ndarray,
    action_sequences: np.ndarray,
) -> np.ndarray:
    """Predict every candidate trajectory from a shared initial state."""
    action_sequences = np.asarray(action_sequences, dtype=np.float64)
    if action_sequences.ndim != 2 or 0 in action_sequences.shape:
        raise ValueError("action_sequences must have nonempty shape (K, H)")
    number_candidates, horizon = action_sequences.shape
    state = _state(initial_state, name="initial_state")
    states = np.repeat(state[None, :], number_candidates, axis=0)
    predicted_states = np.empty((number_candidates, horizon + 1, 2))
    predicted_states[:, 0] = states
    for step in range(horizon):
        states = model.predict(states, action_sequences[:, step])
        predicted_states[:, step + 1] = states
    return predicted_states


def trajectory_cost(
    predicted_states: np.ndarray,
    action_sequences: np.ndarray,
    target_position: float,
    *,
    running_position_weight: float = 0.05,
    action_weight: float = 0.01,
    terminal_position_weight: float = 5.0,
    terminal_velocity_weight: float = 0.5,
) -> np.ndarray:
    """Score candidate trajectories using position, velocity, and effort."""
    predicted_states = np.asarray(predicted_states, dtype=np.float64)
    action_sequences = np.asarray(action_sequences, dtype=np.float64)
    if action_sequences.ndim != 2:
        raise ValueError("action_sequences must have shape (K, H)")
    expected_shape = (action_sequences.shape[0], action_sequences.shape[1] + 1, 2)
    if predicted_states.shape != expected_shape:
        raise ValueError("predicted_states must have shape (K, H + 1, 2)")
    target_position = _finite_real(target_position, name="target_position")
    weights = [
        _finite_real(running_position_weight, name="running_position_weight"),
        _finite_real(action_weight, name="action_weight"),
        _finite_real(terminal_position_weight, name="terminal_position_weight"),
        _finite_real(terminal_velocity_weight, name="terminal_velocity_weight"),
    ]
    if any(weight < 0.0 for weight in weights):
        raise ValueError("cost weights must be nonnegative")
    position_errors = predicted_states[:, 1:, 0] - target_position
    running_position_cost = weights[0] * np.sum(position_errors**2, axis=1)
    action_cost = weights[1] * np.sum(action_sequences**2, axis=1)
    terminal_position_cost = weights[2] * position_errors[:, -1] ** 2
    terminal_velocity_cost = weights[3] * predicted_states[:, -1, 1] ** 2
    return (
        running_position_cost
        + action_cost
        + terminal_position_cost
        + terminal_velocity_cost
    )


def select_mpc_action(
    model: LinearDynamicsModel,
    state: tuple[float, float] | np.ndarray,
    target_position: float,
    generator: np.random.Generator,
    *,
    number_candidates: int = 2_000,
    horizon: int = 25,
    block_length: int = 5,
) -> tuple[float, np.ndarray, float]:
    """Return the first action from the lowest-cost sampled sequence."""
    action_sequences = sample_action_sequences(
        number_candidates,
        horizon,
        block_length,
        generator,
    )
    predicted_states = predict_action_sequences(model, state, action_sequences)
    costs = trajectory_cost(predicted_states, action_sequences, target_position)
    best_index = int(np.argmin(costs))
    return (
        float(action_sequences[best_index, 0]),
        predicted_states[best_index],
        float(costs[best_index]),
    )


def run_mpc_controller(
    model: LinearDynamicsModel,
    *,
    initial_state: tuple[float, float] = (-2.0, 0.0),
    target_position: float = 2.0,
    number_steps: int = 35,
    disturbance_step: int | None = None,
    disturbance_velocity: float = -1.0,
    seed: int | None = None,
    environment_factory: Callable[[], PointMass1D] = PointMass1D,
) -> tuple[np.ndarray, np.ndarray]:
    """Execute random-shooting MPC with optional instantaneous velocity change."""
    number_steps = _positive_integer(number_steps, name="number_steps")
    disturbance_step = _disturbance_step(disturbance_step, number_steps)
    disturbance_velocity = _finite_real(
        disturbance_velocity,
        name="disturbance_velocity",
    )
    environment = environment_factory()
    state = environment.reset(initial_state)
    generator = np.random.default_rng(seed)
    states = [state.copy()]
    actions: list[float] = []
    for step in range(number_steps):
        action, _, _ = select_mpc_action(
            model,
            state,
            target_position,
            generator,
        )
        state = environment.step(action)
        if disturbance_step is not None and step == disturbance_step:
            environment.state[1] += disturbance_velocity
            state = environment.state.copy()
        actions.append(action)
        states.append(state.copy())
    return np.asarray(states), np.asarray(actions)


def run_open_loop_controller(
    model: LinearDynamicsModel,
    *,
    initial_state: tuple[float, float] = (-2.0, 0.0),
    target_position: float = 2.0,
    number_steps: int = 35,
    disturbance_step: int | None = None,
    disturbance_velocity: float = -1.0,
    number_candidates: int = 5_000,
    block_length: int = 5,
    seed: int | None = None,
    environment_factory: Callable[[], PointMass1D] = PointMass1D,
) -> tuple[np.ndarray, np.ndarray]:
    """Plan once with the learned model and execute the complete sequence."""
    number_steps = _positive_integer(number_steps, name="number_steps")
    disturbance_step = _disturbance_step(disturbance_step, number_steps)
    disturbance_velocity = _finite_real(
        disturbance_velocity,
        name="disturbance_velocity",
    )
    generator = np.random.default_rng(seed)
    candidate_actions = sample_action_sequences(
        number_candidates,
        number_steps,
        block_length,
        generator,
    )
    candidate_states = predict_action_sequences(model, initial_state, candidate_actions)
    costs = trajectory_cost(candidate_states, candidate_actions, target_position)
    planned_actions = candidate_actions[int(np.argmin(costs))]

    environment = environment_factory()
    state = environment.reset(initial_state)
    states = [state.copy()]
    for step, action in enumerate(planned_actions):
        state = environment.step(action)
        if disturbance_step is not None and step == disturbance_step:
            environment.state[1] += disturbance_velocity
            state = environment.state.copy()
        states.append(state.copy())
    return np.asarray(states), planned_actions


def _disturbance_step(value: int | None, number_steps: int) -> int | None:
    """Validate an optional zero-indexed disturbance step."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("disturbance_step must be an integer or None")
    value = int(value)
    if not 0 <= value < number_steps:
        raise ValueError("disturbance_step must lie inside the rollout")
    return value


__all__ = [
    "LinearDynamicsModel",
    "PointMass1D",
    "collect_transition_data",
    "compare_rollouts",
    "fit_linear_dynamics",
    "predict_action_sequences",
    "run_mpc_controller",
    "run_open_loop_controller",
    "sample_action_sequences",
    "select_mpc_action",
    "trajectory_cost",
]
