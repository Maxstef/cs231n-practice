"""Small NumPy utilities for introductory tabular reinforcement learning."""

from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Callable

import numpy as np


ACTION_DELTAS = np.array(
    [
        [-1, 0],
        [0, 1],
        [1, 0],
        [0, -1],
    ],
    dtype=np.int64,
)
ACTION_NAMES = ("up", "right", "down", "left")

State = tuple[int, int]
Policy = Callable[[State, np.random.Generator], int]


def _probability(value: float, *, name: str) -> float:
    """Validate a finite number in the closed interval from zero to one."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be numeric")
    value = float(value)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return value


def _positive_integer(value: int, *, name: str) -> int:
    """Validate a positive integer without accepting Boolean values."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


@dataclass
class GridWorld:
    """A deterministic rectangular grid with blocked cells and one goal.

    Actions use the order up, right, down, and left. Crossing a boundary or
    entering an obstacle leaves the agent in its current state. Sparse mode
    gives reward one only at the goal. Shaped mode additionally rewards
    progress in Manhattan distance and charges a small cost per step.
    """

    height: int = 5
    width: int = 6
    start: State = (4, 0)
    goal: State = (0, 5)
    obstacles: tuple[State, ...] = (
        (1, 1),
        (1, 2),
        (2, 2),
        (3, 4),
    )
    reward_mode: str = "sparse"
    state: State = field(init=False)
    obstacle_set: frozenset[State] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.height = _positive_integer(self.height, name="height")
        self.width = _positive_integer(self.width, name="width")
        if self.reward_mode not in {"sparse", "shaped"}:
            raise ValueError("reward_mode must be 'sparse' or 'shaped'")

        self.start = self._validated_state(self.start, name="start")
        self.goal = self._validated_state(self.goal, name="goal")
        validated_obstacles = tuple(
            self._validated_state(obstacle, name="obstacle")
            for obstacle in self.obstacles
        )
        self.obstacles = validated_obstacles
        self.obstacle_set = frozenset(validated_obstacles)
        if len(self.obstacle_set) != len(validated_obstacles):
            raise ValueError("obstacles must not contain duplicates")
        if self.start in self.obstacle_set or self.goal in self.obstacle_set:
            raise ValueError("start and goal must not be obstacles")
        self.state = self.start

    def _validated_state(self, state: State, *, name: str) -> State:
        """Return an in-bounds pair of Python integer coordinates."""
        if not isinstance(state, tuple) or len(state) != 2:
            raise TypeError(f"{name} must be a (row, column) tuple")
        row, column = state
        if any(isinstance(value, bool) or not isinstance(value, Integral) for value in state):
            raise TypeError(f"{name} coordinates must be integers")
        row, column = int(row), int(column)
        if not 0 <= row < self.height or not 0 <= column < self.width:
            raise ValueError(f"{name} must lie inside the grid")
        return row, column

    def reset(self) -> State:
        """Move the agent to the start and return that state."""
        self.state = self.start
        return self.state

    def manhattan_distance(self, state: State) -> int:
        """Return the axis-aligned distance from a state to the goal."""
        row, column = self._validated_state(state, name="state")
        return abs(row - self.goal[0]) + abs(column - self.goal[1])

    def step(self, action: int) -> tuple[State, float, bool]:
        """Apply one action and return ``(next_state, reward, terminated)``."""
        if isinstance(action, bool) or not isinstance(action, Integral):
            raise TypeError("action must be an integer")
        action = int(action)
        if not 0 <= action < len(ACTION_DELTAS):
            raise ValueError("action must be an integer from 0 to 3")

        old_state = self.state
        candidate_array = np.asarray(old_state) + ACTION_DELTAS[action]
        candidate = (
            int(np.clip(candidate_array[0], 0, self.height - 1)),
            int(np.clip(candidate_array[1], 0, self.width - 1)),
        )
        next_state = old_state if candidate in self.obstacle_set else candidate
        terminated = next_state == self.goal

        if self.reward_mode == "sparse":
            reward = 1.0 if terminated else 0.0
        else:
            progress = (
                self.manhattan_distance(old_state)
                - self.manhattan_distance(next_state)
            )
            reward = 0.1 * progress - 0.01 + (1.0 if terminated else 0.0)

        self.state = next_state
        return next_state, float(reward), terminated


def discounted_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
    """Return the discounted future reward starting at every time step."""
    gamma = _probability(gamma, name="gamma")
    rewards = np.asarray(rewards)
    if rewards.ndim != 1:
        raise ValueError("rewards must be one-dimensional")
    if not np.issubdtype(rewards.dtype, np.number):
        raise TypeError("rewards must be numeric")
    if not np.isfinite(rewards).all():
        raise ValueError("rewards must contain only finite values")

    returns = np.zeros(rewards.shape, dtype=np.result_type(rewards.dtype, np.float64))
    running_return = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running_return = float(rewards[index]) + gamma * running_return
        returns[index] = running_return
    return returns


def q_learning_update(
    q_values: np.ndarray,
    state: State,
    action: int,
    reward: float,
    next_state: State,
    terminated: bool,
    alpha: float,
    gamma: float,
) -> float:
    """Update one tabular Q-value and return its temporal-difference error."""
    q_values = _q_table(q_values)
    state = _q_state(state, q_values, name="state")
    next_state = _q_state(next_state, q_values, name="next_state")
    if isinstance(action, bool) or not isinstance(action, Integral):
        raise TypeError("action must be an integer")
    action = int(action)
    if not 0 <= action < q_values.shape[2]:
        raise ValueError("action index is outside the Q-table")
    if isinstance(reward, bool) or not isinstance(reward, Real):
        raise TypeError("reward must be numeric")
    reward = float(reward)
    if not np.isfinite(reward):
        raise ValueError("reward must be finite")
    if not isinstance(terminated, (bool, np.bool_)):
        raise TypeError("terminated must be Boolean")
    alpha = _probability(alpha, name="alpha")
    gamma = _probability(gamma, name="gamma")

    current_value = q_values[state[0], state[1], action]
    target = reward
    if not terminated:
        target += gamma * np.max(q_values[next_state[0], next_state[1]])
    td_error = target - current_value
    q_values[state[0], state[1], action] += alpha * td_error
    return float(td_error)


def epsilon_greedy_action(
    q_values: np.ndarray,
    state: State,
    epsilon: float,
    generator: np.random.Generator,
) -> int:
    """Choose a random action with probability epsilon, otherwise greedily."""
    q_values = _q_table(q_values)
    state = _q_state(state, q_values, name="state")
    epsilon = _probability(epsilon, name="epsilon")
    if not isinstance(generator, np.random.Generator):
        raise TypeError("generator must be a numpy.random.Generator")
    if generator.random() < epsilon:
        return int(generator.integers(q_values.shape[2]))
    return int(np.argmax(q_values[state[0], state[1]]))


def rollout_policy(
    environment: GridWorld,
    policy: Policy,
    generator: np.random.Generator,
    *,
    max_steps: int = 50,
) -> dict[str, list | bool]:
    """Run a policy for one episode and record states, actions, and rewards."""
    if not isinstance(generator, np.random.Generator):
        raise TypeError("generator must be a numpy.random.Generator")
    max_steps = _positive_integer(max_steps, name="max_steps")
    state = environment.reset()
    states = [state]
    actions: list[int] = []
    rewards: list[float] = []
    terminated = False

    for _ in range(max_steps):
        action = policy(state, generator)
        next_state, reward, terminated = environment.step(action)
        actions.append(int(action))
        rewards.append(float(reward))
        states.append(next_state)
        state = next_state
        if terminated:
            break

    return {
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "terminated": terminated,
    }


def train_tabular_q_learning(
    environment: GridWorld,
    *,
    number_episodes: int = 600,
    alpha: float = 0.25,
    gamma: float = 0.95,
    epsilon: float = 0.2,
    max_steps: int = 100,
    seed: int | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Train an epsilon-greedy tabular Q-learning agent in a grid world."""
    number_episodes = _positive_integer(number_episodes, name="number_episodes")
    max_steps = _positive_integer(max_steps, name="max_steps")
    alpha = _probability(alpha, name="alpha")
    gamma = _probability(gamma, name="gamma")
    epsilon = _probability(epsilon, name="epsilon")
    generator = np.random.default_rng(seed)
    q_values = np.zeros((environment.height, environment.width, len(ACTION_DELTAS)))
    episode_returns = np.zeros(number_episodes, dtype=np.float64)
    episode_lengths = np.zeros(number_episodes, dtype=np.int64)
    episode_successes = np.zeros(number_episodes, dtype=bool)

    for episode_index in range(number_episodes):
        state = environment.reset()
        total_reward = 0.0
        terminated = False

        for step_index in range(max_steps):
            action = epsilon_greedy_action(
                q_values,
                state,
                epsilon,
                generator,
            )
            next_state, reward, terminated = environment.step(action)
            q_learning_update(
                q_values,
                state,
                action,
                reward,
                next_state,
                terminated,
                alpha,
                gamma,
            )
            total_reward += reward
            state = next_state
            if terminated:
                break

        episode_returns[episode_index] = total_reward
        episode_lengths[episode_index] = step_index + 1
        episode_successes[episode_index] = terminated

    history = {
        "returns": episode_returns,
        "lengths": episode_lengths,
        "successes": episode_successes,
    }
    return q_values, history


def _q_table(q_values: np.ndarray) -> np.ndarray:
    """Validate a writable floating-point table with shape (R, C, A)."""
    if not isinstance(q_values, np.ndarray):
        raise TypeError("q_values must be a numpy array")
    if q_values.ndim != 3 or any(size == 0 for size in q_values.shape):
        raise ValueError("q_values must have nonempty shape (rows, columns, actions)")
    if not np.issubdtype(q_values.dtype, np.floating):
        raise TypeError("q_values must contain floating-point values")
    if not q_values.flags.writeable:
        raise ValueError("q_values must be writable")
    if not np.isfinite(q_values).all():
        raise ValueError("q_values must contain only finite values")
    return q_values


def _q_state(state: State, q_values: np.ndarray, *, name: str) -> State:
    """Validate a state against the spatial dimensions of a Q-table."""
    if not isinstance(state, tuple) or len(state) != 2:
        raise TypeError(f"{name} must be a (row, column) tuple")
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in state):
        raise TypeError(f"{name} coordinates must be integers")
    row, column = int(state[0]), int(state[1])
    if not 0 <= row < q_values.shape[0] or not 0 <= column < q_values.shape[1]:
        raise ValueError(f"{name} is outside the Q-table")
    return row, column


__all__ = [
    "ACTION_DELTAS",
    "ACTION_NAMES",
    "GridWorld",
    "discounted_returns",
    "epsilon_greedy_action",
    "q_learning_update",
    "rollout_policy",
    "train_tabular_q_learning",
]
