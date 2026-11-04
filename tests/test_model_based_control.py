import numpy as np
import pytest

from cs231n_practice.model_based_control import (
    LinearDynamicsModel,
    PointMass1D,
    collect_transition_data,
    compare_rollouts,
    fit_linear_dynamics,
    predict_action_sequences,
    run_mpc_controller,
    run_open_loop_controller,
    sample_action_sequences,
    select_mpc_action,
    trajectory_cost,
)


@pytest.fixture(scope="module")
def fitted_model() -> LinearDynamicsModel:
    states, actions, next_states = collect_transition_data(
        2_000,
        np.random.default_rng(42),
    )
    return fit_linear_dynamics(states[:1_600], actions[:1_600], next_states[:1_600])


def test_point_mass_step_matches_manual_update_and_clips_action() -> None:
    environment = PointMass1D()
    environment.reset((1.0, 1.0))

    state = environment.step(3.0)

    expected_velocity = 0.96 + 0.2 - 0.1
    expected_position = 1.0 + 0.2 * expected_velocity
    np.testing.assert_allclose(state, [expected_position, expected_velocity])


def test_transition_collection_is_reproducible_and_has_expected_shapes() -> None:
    first = collect_transition_data(20, np.random.default_rng(7))
    second = collect_transition_data(20, np.random.default_rng(7))

    for first_array, second_array in zip(first, second):
        np.testing.assert_allclose(first_array, second_array)
    assert first[0].shape == (20, 2)
    assert first[1].shape == (20, 1)
    assert first[2].shape == (20, 2)


def test_linear_model_supports_single_and_batched_prediction(fitted_model) -> None:
    single = fitted_model.predict(np.array([-2.0, 0.0]), 1.0)
    batch = fitted_model.predict(
        np.array([[-2.0, 0.0], [-1.0, 0.5]]),
        np.array([1.0, -0.5]),
    )

    assert single.shape == (1, 2)
    assert batch.shape == (2, 2)


def test_learned_model_has_small_one_step_error(fitted_model) -> None:
    states, actions, next_states = collect_transition_data(
        400,
        np.random.default_rng(9),
    )
    predictions = fitted_model.predict(states, actions)
    rmse = np.sqrt(np.mean((predictions - next_states) ** 2, axis=0))

    assert rmse[0] < 0.01
    assert rmse[1] < 0.04


def test_multi_step_rollout_preserves_shapes_and_accumulates_error(fitted_model) -> None:
    actions = np.concatenate([np.ones(15), -np.ones(15), np.zeros(10)])

    true_states, model_states = compare_rollouts(fitted_model, (-2.0, 0.0), actions)

    assert true_states.shape == model_states.shape == (41, 2)
    np.testing.assert_allclose(true_states[0], model_states[0])
    assert np.linalg.norm(true_states[-1] - model_states[-1]) > 0.1


def test_block_action_sampling_repeats_each_sampled_value() -> None:
    sequences = sample_action_sequences(
        3,
        horizon=12,
        block_length=5,
        generator=np.random.default_rng(2),
    )

    assert sequences.shape == (3, 12)
    assert np.all(sequences[:, :5] == sequences[:, 0, None])
    assert np.all(sequences[:, 5:10] == sequences[:, 5, None])
    assert np.all(sequences[:, 10:12] == sequences[:, 10, None])


def test_prediction_and_cost_are_candidate_wise(fitted_model) -> None:
    actions = sample_action_sequences(
        30,
        horizon=10,
        block_length=2,
        generator=np.random.default_rng(3),
    )

    states = predict_action_sequences(fitted_model, (-2.0, 0.0), actions)
    costs = trajectory_cost(states, actions, target_position=2.0)

    assert states.shape == (30, 11, 2)
    assert costs.shape == (30,)
    assert np.isfinite(costs).all()
    assert (costs >= 0.0).all()


def test_mpc_action_is_the_first_action_of_selected_trajectory(fitted_model) -> None:
    action, predicted_states, cost = select_mpc_action(
        fitted_model,
        (-2.0, 0.0),
        2.0,
        np.random.default_rng(42),
        number_candidates=100,
        horizon=10,
        block_length=2,
    )

    assert -1.0 <= action <= 1.0
    assert predicted_states.shape == (11, 2)
    assert np.isfinite(cost)


def test_mpc_recovers_from_disturbance_better_than_open_loop(fitted_model) -> None:
    open_states, open_actions = run_open_loop_controller(
        fitted_model,
        disturbance_step=10,
        seed=42,
    )
    mpc_states, mpc_actions = run_mpc_controller(
        fitted_model,
        disturbance_step=10,
        seed=42,
    )

    assert open_states.shape == mpc_states.shape == (36, 2)
    assert open_actions.shape == mpc_actions.shape == (35,)
    open_error = abs(open_states[-1, 0] - 2.0)
    mpc_error = abs(mpc_states[-1, 0] - 2.0)
    assert mpc_error < open_error
    assert mpc_error < 0.3


def test_model_based_helpers_reject_invalid_shapes_and_parameters(fitted_model) -> None:
    with pytest.raises(ValueError, match="equal batch sizes"):
        fitted_model.predict(np.zeros((2, 2)), np.zeros(3))
    with pytest.raises(ValueError, match="positive"):
        sample_action_sequences(2, 0, 1, np.random.default_rng(0))
    with pytest.raises(ValueError, match="shape"):
        trajectory_cost(np.zeros((2, 4, 2)), np.zeros((2, 4)), 0.0)
    with pytest.raises(ValueError, match="inside the rollout"):
        run_mpc_controller(fitted_model, number_steps=5, disturbance_step=5)
