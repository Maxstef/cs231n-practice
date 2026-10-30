import numpy as np
import pytest

from cs231n_practice.reinforcement_learning import (
    GridWorld,
    discounted_returns,
    epsilon_greedy_action,
    q_learning_update,
    rollout_policy,
    train_tabular_q_learning,
)


def test_grid_world_handles_movement_boundaries_obstacles_and_goal() -> None:
    environment = GridWorld()

    assert environment.step(1) == ((4, 1), 0.0, False)
    environment.state = (1, 0)
    assert environment.step(1) == ((1, 0), 0.0, False)
    environment.state = (0, 0)
    assert environment.step(0) == ((0, 0), 0.0, False)
    environment.state = (0, 4)
    assert environment.step(1) == ((0, 5), 1.0, True)


def test_shaped_reward_distinguishes_progress_blocking_and_goal() -> None:
    environment = GridWorld(reward_mode="shaped")

    np.testing.assert_allclose(environment.step(1)[1], 0.09)
    environment.state = (1, 0)
    np.testing.assert_allclose(environment.step(1)[1], -0.01)
    environment.state = (0, 4)
    np.testing.assert_allclose(environment.step(1)[1], 1.09)


def test_discounted_returns_propagate_terminal_reward_backward() -> None:
    returns = discounted_returns(np.array([0.0, 0.0, 1.0]), gamma=0.9)

    np.testing.assert_allclose(returns, [0.81, 0.9, 1.0])


def test_q_learning_update_uses_future_value_and_changes_one_entry() -> None:
    q_values = np.zeros((5, 6, 4), dtype=np.float64)
    q_values[4, 1] = [0.2, 0.5, 0.1, 0.0]

    error = q_learning_update(
        q_values,
        state=(4, 0),
        action=1,
        reward=0.0,
        next_state=(4, 1),
        terminated=False,
        alpha=0.5,
        gamma=0.9,
    )

    np.testing.assert_allclose(error, 0.45)
    np.testing.assert_allclose(q_values[4, 0], [0.0, 0.225, 0.0, 0.0])


def test_terminal_q_update_does_not_bootstrap() -> None:
    q_values = np.zeros((2, 2, 2), dtype=np.float64)
    q_values[0, 1] = [10.0, 20.0]

    q_learning_update(q_values, (0, 0), 0, 1.0, (0, 1), True, 1.0, 0.9)

    np.testing.assert_allclose(q_values[0, 0, 0], 1.0)


def test_epsilon_greedy_action_supports_greedy_and_random_selection() -> None:
    q_values = np.array([[[0.1, 0.8, 0.2, 0.3]]])

    greedy = epsilon_greedy_action(
        q_values,
        (0, 0),
        epsilon=0.0,
        generator=np.random.default_rng(2),
    )
    random_actions = {
        epsilon_greedy_action(
            q_values,
            (0, 0),
            epsilon=1.0,
            generator=np.random.default_rng(seed),
        )
        for seed in range(20)
    }

    assert greedy == 1
    assert len(random_actions) > 1


def test_rollout_policy_records_a_complete_goal_trajectory() -> None:
    environment = GridWorld(height=1, width=3, start=(0, 0), goal=(0, 2), obstacles=())

    trajectory = rollout_policy(
        environment,
        lambda state, generator: 1,
        np.random.default_rng(0),
        max_steps=5,
    )

    assert trajectory == {
        "states": [(0, 0), (0, 1), (0, 2)],
        "actions": [1, 1],
        "rewards": [0.0, 1.0],
        "terminated": True,
    }


def test_tabular_training_learns_a_successful_greedy_policy() -> None:
    environment = GridWorld()
    q_values, history = train_tabular_q_learning(
        environment,
        number_episodes=250,
        seed=42,
    )

    def greedy_policy(state, generator):
        del generator
        return int(np.argmax(q_values[state[0], state[1]]))

    trajectory = rollout_policy(
        GridWorld(),
        greedy_policy,
        np.random.default_rng(42),
        max_steps=50,
    )

    assert trajectory["terminated"]
    assert len(trajectory["actions"]) == 9
    assert history["successes"][-100:].mean() == 1.0


def test_reinforcement_learning_helpers_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="reward_mode"):
        GridWorld(reward_mode="unknown")
    with pytest.raises(ValueError, match="between 0 and 1"):
        discounted_returns(np.array([1.0]), gamma=1.1)
    with pytest.raises(TypeError, match="floating-point"):
        q_learning_update(
            np.zeros((2, 2, 2), dtype=np.int64),
            (0, 0),
            0,
            0.0,
            (0, 1),
            False,
            0.5,
            0.9,
        )
