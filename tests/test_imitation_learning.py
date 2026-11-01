import numpy as np
import pytest

from cs231n_practice.imitation_learning import (
    TabularClone,
    collect_demonstration,
    evaluate_imitation_policy,
    fit_tabular_clone,
    make_shortest_path_expert,
    rollout_with_disturbances,
    shortest_path_distances,
    train_tabular_dagger,
    transition_from_state,
)
from cs231n_practice.reinforcement_learning import GridWorld


def test_shortest_path_expert_reaches_goal_in_nine_steps() -> None:
    environment = GridWorld()
    expert, distances = make_shortest_path_expert(environment)

    states, actions, trajectory, terminated = collect_demonstration(
        GridWorld(),
        expert,
    )

    assert distances[environment.start] == 9
    assert states.shape == (9, 2)
    assert actions.shape == (9,)
    assert trajectory[-1] == environment.goal
    assert terminated


def test_transition_helper_does_not_mutate_environment() -> None:
    environment = GridWorld()

    next_state = transition_from_state(environment, (4, 0), 1)

    assert next_state == (4, 1)
    assert environment.state == environment.start


def test_shortest_path_distances_mark_obstacles_unreachable() -> None:
    environment = GridWorld()
    distances = shortest_path_distances(environment)

    assert distances[environment.goal] == 0
    assert np.isinf(distances[1, 1])
    assert distances[environment.start] == 9


def test_tabular_clone_uses_majority_vote_and_fallback() -> None:
    states = np.array([[0, 0], [0, 0], [0, 0], [1, 0]])
    actions = np.array([0, 1, 1, 0])

    clone = fit_tabular_clone(states, actions, height=2, width=2, fallback_action=3)

    assert clone((0, 0)) == 1
    assert clone((1, 0)) == 0
    assert clone((1, 1)) == 3
    assert clone.known_states.sum() == 2


def test_disturbance_free_rollout_executes_requested_actions() -> None:
    environment = GridWorld(height=1, width=3, start=(0, 0), goal=(0, 2), obstacles=())

    rollout = rollout_with_disturbances(
        environment,
        lambda state, generator: 1,
        np.random.default_rng(4),
        disturbance_probability=0.0,
    )

    assert rollout["requested_actions"] == [1, 1]
    assert rollout["executed_actions"] == [1, 1]
    assert rollout["terminated"]


def test_closed_loop_evaluation_returns_success_and_mean_length() -> None:
    expert, _ = make_shortest_path_expert(GridWorld())

    success_rate, mean_length = evaluate_imitation_policy(
        expert,
        number_episodes=10,
        disturbance_probability=0.0,
        seed=3,
    )

    assert success_rate == 1.0
    assert mean_length == 9.0


def test_dagger_expands_coverage_and_improves_disturbed_success() -> None:
    expert, _ = make_shortest_path_expert(GridWorld())
    states, actions, _, _ = collect_demonstration(GridWorld(), expert)

    clone, success, coverage, snapshots = train_tabular_dagger(
        states,
        actions,
        expert,
        number_iterations=3,
        rollouts_per_iteration=16,
        evaluation_episodes=200,
        seed=42,
    )

    np.testing.assert_array_equal(coverage, [9, 15, 23, 24])
    np.testing.assert_array_equal(snapshots.sum(axis=(1, 2)), coverage)
    assert success[-1] > success[0]
    assert clone.known_states.sum() == 24
    assert not clone.known_states[2, 1]


def test_coverage_snapshots_are_independent() -> None:
    expert, _ = make_shortest_path_expert(GridWorld())
    states, actions, _, _ = collect_demonstration(GridWorld(), expert)
    _, _, _, snapshots = train_tabular_dagger(
        states,
        actions,
        expert,
        number_iterations=1,
        rollouts_per_iteration=2,
        evaluation_episodes=2,
        seed=1,
    )

    original = snapshots[0].copy()
    snapshots[1, 0, 0] = ~snapshots[1, 0, 0]
    np.testing.assert_array_equal(snapshots[0], original)


def test_imitation_helpers_reject_misaligned_or_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="same number"):
        fit_tabular_clone(
            np.array([[0, 0], [0, 1]]),
            np.array([1]),
            height=2,
            width=2,
        )
    with pytest.raises(ValueError, match="equal 2D shapes"):
        TabularClone(np.zeros((2, 2), dtype=int), np.zeros((2, 3), dtype=bool))
    with pytest.raises(ValueError, match="between 0 and 1"):
        rollout_with_disturbances(
            GridWorld(),
            lambda state, generator: 0,
            np.random.default_rng(0),
            disturbance_probability=1.1,
        )
