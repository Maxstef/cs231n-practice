import matplotlib
import numpy as np
import pytest
import torch

matplotlib.use("Agg")

from cs231n_practice.goal_conditioning import (
    TinyVisionLanguageActionPolicy,
    VisualGoalGridWorld,
    VisualGoalPolicy,
    build_goal_conditioned_demonstrations,
    evaluate_closed_loop,
    plot_demonstration_examples,
    plot_failed_rollouts,
    plot_policy_action_fields,
    plot_policy_rollouts,
    plot_state_goal_split,
    plot_validation_mistakes,
    rollout_policy,
    split_demonstrations,
)


def test_world_rendering_and_blocked_transitions() -> None:
    world = VisualGoalGridWorld()
    observation = world.render((5, 0), 1)

    assert observation.shape == (3, 6, 6)
    assert observation[0].sum() == 4
    assert observation[1, 5, 0] == 1
    assert observation[2, 0, 5] == 1
    assert world.transition((0, 0), 0, goal_index=0) == (0, 0)
    assert world.transition((2, 1), 1, goal_index=0) == (2, 1)
    assert world.obstacles_for_goal(0) == world.obstacles_for_goal(1)


def test_bfs_and_expert_reduce_shortest_path_distance() -> None:
    world = VisualGoalGridWorld()
    distances = world.shortest_path_distances(2)
    state = (5, 0)
    action = world.expert_action(state, 2)
    next_state = world.transition(state, action, goal_index=2)

    assert len(distances) == len(world.reachable_states(2)) == 32
    assert distances[next_state] == distances[state] - 1


def test_demonstrations_and_split_have_expected_shapes() -> None:
    world = VisualGoalGridWorld()
    data = build_goal_conditioned_demonstrations(world)
    train, validation = split_demonstrations(len(data.actions), seed=7)

    assert data.observations.shape == (93, 3, 6, 6)
    assert data.goal_indices.shape == data.actions.shape == (93,)
    assert len(data.state_goal_pairs) == 93
    assert len(train) == 74
    assert len(validation) == 19
    assert not set(train).intersection(validation)


def test_policy_models_return_four_action_logits() -> None:
    visual_policy = VisualGoalPolicy()
    instruction_policy = TinyVisionLanguageActionPolicy()

    visual_logits = visual_policy(torch.zeros(5, 3, 6, 6))
    instruction_logits = instruction_policy(
        torch.zeros(5, 2, 6, 6), torch.tensor([0, 1, 2, 0, 1])
    )

    assert visual_logits.shape == instruction_logits.shape == (5, 4)


def test_expert_policy_succeeds_from_every_demonstration_pair() -> None:
    world = VisualGoalGridWorld()
    data = build_goal_conditioned_demonstrations(world)
    expert = lambda state, goal_index: world.expert_action(state, goal_index)

    success_rate, rollouts = evaluate_closed_loop(
        world, expert, data.state_goal_pairs
    )

    assert success_rate == 1.0
    assert all(rollout.succeeded for rollout in rollouts)


def test_rollout_step_limit_records_initial_and_resulting_states() -> None:
    world = VisualGoalGridWorld()
    always_up = lambda state, goal_index: 0

    rollout = rollout_policy(world, always_up, (5, 0), 2, maximum_steps=4)

    assert not rollout.succeeded
    assert len(rollout.trajectory) == 5


def test_plot_helpers_accept_shared_policy_interface() -> None:
    world = VisualGoalGridWorld()
    data = build_goal_conditioned_demonstrations(world)
    train, _ = split_demonstrations(len(data.actions), seed=7)
    expert = lambda state, goal_index: world.expert_action(state, goal_index)
    always_up = lambda state, goal_index: 0
    _, failures = evaluate_closed_loop(
        world, always_up, data.state_goal_pairs, maximum_steps=4
    )

    figure_split, axes_split = plot_state_goal_split(
        world, data.state_goal_pairs, train
    )
    validation = np.asarray(
        [index for index in range(len(data.actions)) if index not in set(train)]
    )
    figure_examples, axes_examples = plot_demonstration_examples(
        world, data, {"training": train, "validation": validation}
    )
    predictions = data.actions[validation].clone()
    predictions[0] = (predictions[0] + 1) % len(world.action_names)
    figure_mistakes, axes_mistakes = plot_validation_mistakes(
        world, data, validation, predictions
    )
    figure_paths, axes_paths = plot_policy_rollouts(
        world, {"expert": expert}, (5, 0)
    )
    figure_fields, axes_fields = plot_policy_action_fields(
        world, {"expert": expert}
    )
    figure_failures, axes_failures = plot_failed_rollouts(world, failures)

    assert axes_split.shape == (3,)
    assert axes_examples.shape == (2, 4)
    assert figure_mistakes is not None
    assert axes_mistakes is not None and axes_mistakes.shape == (1,)
    assert axes_paths.shape == axes_fields.shape == (1, 3)
    assert figure_failures is not None
    assert axes_failures is not None and axes_failures.shape[1] == 2
    for figure in (
        figure_split,
        figure_examples,
        figure_mistakes,
        figure_paths,
        figure_fields,
        figure_failures,
    ):
        matplotlib.pyplot.close(figure)


def test_invalid_world_and_split_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        VisualGoalGridWorld(grid_size=0)
    with pytest.raises(ValueError, match="between"):
        split_demonstrations(10, training_fraction=1.0)
