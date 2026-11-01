"""Tabular imitation-learning helpers for small grid-world experiments."""

from collections import deque
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Callable

import numpy as np

from cs231n_practice.reinforcement_learning import ACTION_DELTAS, GridWorld, State


Policy = Callable[[State, np.random.Generator | None], int]
EnvironmentFactory = Callable[[], GridWorld]


def _positive_integer(value: int, *, name: str) -> int:
    """Validate a positive integer without accepting Boolean values."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _probability(value: float, *, name: str) -> float:
    """Validate a finite probability in the closed interval [0, 1]."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be numeric")
    value = float(value)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return value


def _state_array(states: np.ndarray, *, height: int, width: int) -> np.ndarray:
    """Validate integer grid coordinates with shape (N, 2)."""
    states = np.asarray(states)
    if states.ndim != 2 or states.shape[0] == 0 or states.shape[1] != 2:
        raise ValueError("states must have nonempty shape (N, 2)")
    if not np.issubdtype(states.dtype, np.integer):
        raise TypeError("states must contain integer coordinates")
    if (
        np.any(states[:, 0] < 0)
        or np.any(states[:, 0] >= height)
        or np.any(states[:, 1] < 0)
        or np.any(states[:, 1] >= width)
    ):
        raise ValueError("states must lie inside the grid")
    return states.astype(np.int64, copy=False)


@dataclass(frozen=True)
class TabularClone:
    """A behavioral-cloning policy with a fallback for unlabeled states.

    Attributes:
        actions: Integer array of shape ``(height, width)`` containing the
            most frequently demonstrated action at each grid state.
        known_states: Boolean array of the same shape. ``True`` means at least
            one expert action label was observed for that state.
        fallback_action: Action returned at states absent from demonstrations.
    """

    actions: np.ndarray
    known_states: np.ndarray
    fallback_action: int = 1

    def __post_init__(self) -> None:
        actions = np.asarray(self.actions)
        known_states = np.asarray(self.known_states)
        if actions.ndim != 2 or actions.shape != known_states.shape:
            raise ValueError("actions and known_states must have equal 2D shapes")
        if not np.issubdtype(actions.dtype, np.integer):
            raise TypeError("actions must contain integers")
        if known_states.dtype != np.bool_:
            raise TypeError("known_states must contain Boolean values")
        if isinstance(self.fallback_action, bool) or not isinstance(
            self.fallback_action, Integral
        ):
            raise TypeError("fallback_action must be an integer")
        fallback_action = int(self.fallback_action)
        if not 0 <= fallback_action < len(ACTION_DELTAS):
            raise ValueError("fallback_action is outside the action space")
        if np.any(actions < 0) or np.any(actions >= len(ACTION_DELTAS)):
            raise ValueError("actions contain an index outside the action space")
        object.__setattr__(self, "actions", actions.astype(np.int64, copy=False))
        object.__setattr__(self, "known_states", known_states)
        object.__setattr__(self, "fallback_action", fallback_action)

    def __call__(
        self,
        state: State,
        generator: np.random.Generator | None = None,
    ) -> int:
        """Return the cloned action, or the fallback at an unlabeled state."""
        del generator
        row, column = state
        if not 0 <= row < self.actions.shape[0] or not 0 <= column < self.actions.shape[1]:
            raise ValueError("state is outside the clone's action table")
        if self.known_states[row, column]:
            return int(self.actions[row, column])
        return self.fallback_action


def transition_from_state(
    environment: GridWorld,
    state: State,
    action: int,
) -> State:
    """Return a deterministic grid transition without mutating the environment."""
    if isinstance(action, bool) or not isinstance(action, Integral):
        raise TypeError("action must be an integer")
    action = int(action)
    if not 0 <= action < len(ACTION_DELTAS):
        raise ValueError("action is outside the action space")
    row, column = state
    if not 0 <= row < environment.height or not 0 <= column < environment.width:
        raise ValueError("state is outside the grid")

    candidate_array = np.asarray(state) + ACTION_DELTAS[action]
    candidate = (
        int(np.clip(candidate_array[0], 0, environment.height - 1)),
        int(np.clip(candidate_array[1], 0, environment.width - 1)),
    )
    return state if candidate in environment.obstacle_set else candidate


def shortest_path_distances(environment: GridWorld) -> np.ndarray:
    """Return each reachable cell's shortest step distance to the goal."""
    distances = np.full((environment.height, environment.width), np.inf)
    distances[environment.goal] = 0.0
    queue = deque([environment.goal])

    while queue:
        state = queue.popleft()
        for action in range(len(ACTION_DELTAS)):
            neighbor = transition_from_state(environment, state, action)
            if neighbor != state and np.isinf(distances[neighbor]):
                distances[neighbor] = distances[state] + 1.0
                queue.append(neighbor)
    return distances


def make_shortest_path_expert(
    environment: GridWorld,
) -> tuple[Policy, np.ndarray]:
    """Create a deterministic policy that follows shortest grid distances."""
    distances = shortest_path_distances(environment)

    def expert_policy(
        state: State,
        generator: np.random.Generator | None = None,
    ) -> int:
        del generator
        candidates = [
            distances[transition_from_state(environment, state, action)]
            for action in range(len(ACTION_DELTAS))
        ]
        if np.isinf(candidates).all():
            raise ValueError("the expert cannot reach the goal from this state")
        return int(np.argmin(candidates))

    return expert_policy, distances


def collect_demonstration(
    environment: GridWorld,
    policy: Policy,
    *,
    max_steps: int = 100,
    generator: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, list[State], bool]:
    """Collect pre-action states, actions, and the full trajectory."""
    max_steps = _positive_integer(max_steps, name="max_steps")
    if generator is not None and not isinstance(generator, np.random.Generator):
        raise TypeError("generator must be a numpy.random.Generator or None")
    state = environment.reset()
    states: list[State] = []
    actions: list[int] = []
    trajectory = [state]
    terminated = False

    for _ in range(max_steps):
        action = policy(state, generator)
        next_state, _, terminated = environment.step(action)
        states.append(state)
        actions.append(int(action))
        trajectory.append(next_state)
        state = next_state
        if terminated:
            break

    return (
        np.asarray(states, dtype=np.int64),
        np.asarray(actions, dtype=np.int64),
        trajectory,
        terminated,
    )


def fit_tabular_clone(
    states: np.ndarray,
    actions: np.ndarray,
    height: int,
    width: int,
    *,
    fallback_action: int = 1,
) -> TabularClone:
    """Fit a majority-vote action table from demonstrated state–action pairs."""
    height = _positive_integer(height, name="height")
    width = _positive_integer(width, name="width")
    states = _state_array(states, height=height, width=width)
    actions = np.asarray(actions)
    if actions.ndim != 1:
        raise ValueError("actions must be one-dimensional")
    if len(actions) != len(states):
        raise ValueError("states and actions must contain the same number of examples")
    if not np.issubdtype(actions.dtype, np.integer):
        raise TypeError("actions must contain integers")
    if np.any(actions < 0) or np.any(actions >= len(ACTION_DELTAS)):
        raise ValueError("actions contain an index outside the action space")

    action_counts = np.zeros((height, width, len(ACTION_DELTAS)), dtype=np.int64)
    np.add.at(action_counts, (states[:, 0], states[:, 1], actions), 1)
    known_states = action_counts.sum(axis=2) > 0
    learned_actions = np.argmax(action_counts, axis=2)
    return TabularClone(learned_actions, known_states, fallback_action)


def rollout_with_disturbances(
    environment: GridWorld,
    policy: Policy,
    generator: np.random.Generator,
    *,
    disturbance_probability: float = 0.1,
    max_steps: int = 60,
) -> dict[str, list | bool]:
    """Run a policy while randomly replacing some requested actions."""
    if not isinstance(generator, np.random.Generator):
        raise TypeError("generator must be a numpy.random.Generator")
    disturbance_probability = _probability(
        disturbance_probability,
        name="disturbance_probability",
    )
    max_steps = _positive_integer(max_steps, name="max_steps")
    state = environment.reset()
    states = [state]
    requested_actions: list[int] = []
    executed_actions: list[int] = []
    terminated = False

    for _ in range(max_steps):
        requested_action = int(policy(state, generator))
        if generator.random() < disturbance_probability:
            executed_action = int(generator.integers(len(ACTION_DELTAS)))
        else:
            executed_action = requested_action
        next_state, _, terminated = environment.step(executed_action)
        requested_actions.append(requested_action)
        executed_actions.append(executed_action)
        states.append(next_state)
        state = next_state
        if terminated:
            break

    return {
        "states": states,
        "requested_actions": requested_actions,
        "executed_actions": executed_actions,
        "terminated": terminated,
    }


def evaluate_imitation_policy(
    policy: Policy,
    *,
    environment_factory: EnvironmentFactory = GridWorld,
    number_episodes: int = 200,
    disturbance_probability: float = 0.1,
    max_steps: int = 50,
    seed: int | None = None,
) -> tuple[float, float]:
    """Return closed-loop success rate and mean episode length."""
    number_episodes = _positive_integer(number_episodes, name="number_episodes")
    max_steps = _positive_integer(max_steps, name="max_steps")
    disturbance_probability = _probability(
        disturbance_probability,
        name="disturbance_probability",
    )
    generator = np.random.default_rng(seed)
    successes = np.zeros(number_episodes, dtype=bool)
    lengths = np.zeros(number_episodes, dtype=np.int64)

    for episode_index in range(number_episodes):
        rollout = rollout_with_disturbances(
            environment_factory(),
            policy,
            generator,
            disturbance_probability=disturbance_probability,
            max_steps=max_steps,
        )
        successes[episode_index] = bool(rollout["terminated"])
        lengths[episode_index] = len(rollout["executed_actions"])
    return float(successes.mean()), float(lengths.mean())


def train_tabular_dagger(
    initial_states: np.ndarray,
    initial_actions: np.ndarray,
    expert_policy: Policy,
    *,
    environment_factory: EnvironmentFactory = GridWorld,
    number_iterations: int = 3,
    rollouts_per_iteration: int = 16,
    disturbance_probability: float = 0.15,
    collection_max_steps: int = 15,
    evaluation_episodes: int = 200,
    evaluation_max_steps: int = 50,
    fallback_action: int = 1,
    seed: int | None = None,
) -> tuple[TabularClone, np.ndarray, np.ndarray, np.ndarray]:
    """Train a tabular clone with DAgger-style corrective expert labels.

    The histories include the initial behavioral clone at index zero. Coverage
    snapshots are independent Boolean arrays, so later rounds do not mutate
    earlier visualizations.
    """
    number_iterations = _positive_integer(number_iterations, name="number_iterations")
    rollouts_per_iteration = _positive_integer(
        rollouts_per_iteration,
        name="rollouts_per_iteration",
    )
    collection_max_steps = _positive_integer(
        collection_max_steps,
        name="collection_max_steps",
    )
    evaluation_episodes = _positive_integer(
        evaluation_episodes,
        name="evaluation_episodes",
    )
    evaluation_max_steps = _positive_integer(
        evaluation_max_steps,
        name="evaluation_max_steps",
    )
    disturbance_probability = _probability(
        disturbance_probability,
        name="disturbance_probability",
    )
    reference_environment = environment_factory()
    states = _state_array(
        initial_states,
        height=reference_environment.height,
        width=reference_environment.width,
    )
    actions = np.asarray(initial_actions)
    if actions.ndim != 1 or len(actions) != len(states):
        raise ValueError("initial actions must have shape (N,) matching states")
    if not np.issubdtype(actions.dtype, np.integer):
        raise TypeError("initial_actions must contain integers")

    generator = np.random.default_rng(seed)
    dataset_states = [tuple(state) for state in states]
    dataset_actions = [int(action) for action in actions]
    success_history: list[float] = []
    coverage_history: list[int] = []
    coverage_snapshots: list[np.ndarray] = []

    def fit_current_clone() -> TabularClone:
        return fit_tabular_clone(
            np.asarray(dataset_states),
            np.asarray(dataset_actions),
            reference_environment.height,
            reference_environment.width,
            fallback_action=fallback_action,
        )

    def record(clone: TabularClone) -> None:
        success_rate, _ = evaluate_imitation_policy(
            clone,
            environment_factory=environment_factory,
            number_episodes=evaluation_episodes,
            disturbance_probability=disturbance_probability,
            max_steps=evaluation_max_steps,
            seed=int(generator.integers(1_000_000)),
        )
        success_history.append(success_rate)
        coverage_history.append(int(clone.known_states.sum()))
        coverage_snapshots.append(clone.known_states.copy())

    clone = fit_current_clone()
    record(clone)

    for _ in range(number_iterations):
        for _ in range(rollouts_per_iteration):
            rollout = rollout_with_disturbances(
                environment_factory(),
                clone,
                generator,
                disturbance_probability=disturbance_probability,
                max_steps=collection_max_steps,
            )
            visited_states = [tuple(state) for state in rollout["states"][:-1]]
            corrective_actions = [expert_policy(state, None) for state in visited_states]
            dataset_states.extend(visited_states)
            dataset_actions.extend(corrective_actions)
        clone = fit_current_clone()
        record(clone)

    return (
        clone,
        np.asarray(success_history),
        np.asarray(coverage_history),
        np.asarray(coverage_snapshots),
    )


__all__ = [
    "TabularClone",
    "collect_demonstration",
    "evaluate_imitation_policy",
    "fit_tabular_clone",
    "make_shortest_path_expert",
    "rollout_with_disturbances",
    "shortest_path_distances",
    "train_tabular_dagger",
    "transition_from_state",
]
