"""Reusable visual goal-conditioning and tiny VLA grid-world utilities."""

from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from torch import nn

State = tuple[int, int]
ActionFunction = Callable[[State, int], int]


@dataclass(frozen=True)
class VisualGoalGridWorld:
    """Small grid world whose observations separate scene semantics by channel.

    Channel 0 marks obstacles, channel 1 marks the agent, and channel 2 marks
    the requested goal. Movement into an obstacle or beyond the grid leaves
    the state unchanged.
    """

    grid_size: int = 6
    obstacles_by_goal: tuple[frozenset[State], ...] = (
        frozenset({(1, 2), (2, 2), (3, 2), (4, 4)}),
        frozenset({(1, 2), (2, 2), (3, 2), (4, 4)}),
        frozenset({(1, 2), (2, 2), (3, 2), (4, 4)}),
    )
    goals: tuple[State, ...] = ((0, 0), (0, 5), (5, 5))
    goal_names: tuple[str, ...] = ("top left", "top right", "bottom right")
    action_deltas: tuple[State, ...] = ((-1, 0), (0, 1), (1, 0), (0, -1))
    action_names: tuple[str, ...] = ("up", "right", "down", "left")

    def __post_init__(self) -> None:
        if isinstance(self.grid_size, bool) or not isinstance(self.grid_size, int):
            raise TypeError("grid_size must be an integer")
        if self.grid_size <= 0:
            raise ValueError("grid_size must be positive")
        if len(self.goals) == 0 or len(self.goals) != len(self.goal_names):
            raise ValueError("goals and goal_names must have equal nonzero lengths")
        if len(self.action_deltas) == 0 or len(self.action_deltas) != len(
            self.action_names
        ):
            raise ValueError(
                "action_deltas and action_names must have equal nonzero lengths"
            )
        if len(self.obstacles_by_goal) != len(self.goals):
            raise ValueError("obstacles_by_goal must contain one layout per goal")
        for state in (
            *[state for layout in self.obstacles_by_goal for state in layout],
            *self.goals,
        ):
            self._validate_state(state)
        if any(
            goal in self.obstacles_by_goal[goal_index]
            for goal_index, goal in enumerate(self.goals)
        ):
            raise ValueError("goals cannot overlap obstacles")

    def _validate_state(self, state: State) -> None:
        if len(state) != 2 or not all(isinstance(value, int) for value in state):
            raise TypeError("states must contain two integer coordinates")
        if not all(0 <= value < self.grid_size for value in state):
            raise ValueError("state coordinates must lie inside the grid")

    def obstacles_for_goal(self, goal_index: int) -> frozenset[State]:
        """Return the obstacle layout associated with one requested goal."""
        if not 0 <= goal_index < len(self.goals):
            raise ValueError("goal_index is out of range")
        return self.obstacles_by_goal[goal_index]

    def reachable_states(self, goal_index: int) -> tuple[State, ...]:
        """Return every non-obstacle state for one goal in row-major order."""
        obstacles = self.obstacles_for_goal(goal_index)
        return tuple(
            (row, column)
            for row in range(self.grid_size)
            for column in range(self.grid_size)
            if (row, column) not in obstacles
        )

    def transition(self, state: State, action: int, goal_index: int) -> State:
        """Apply one discrete action, retaining the state when movement is blocked."""
        self._validate_state(state)
        obstacles = self.obstacles_for_goal(goal_index)
        if state in obstacles:
            raise ValueError("the agent cannot occupy an obstacle")
        if isinstance(action, bool) or not isinstance(action, (int, np.integer)):
            raise TypeError("action must be an integer")
        action = int(action)
        if not 0 <= action < len(self.action_deltas):
            raise ValueError("action is out of range")
        row, column = state
        delta_row, delta_column = self.action_deltas[action]
        candidate = (row + delta_row, column + delta_column)
        if (
            0 <= candidate[0] < self.grid_size
            and 0 <= candidate[1] < self.grid_size
            and candidate not in obstacles
        ):
            return candidate
        return state

    def render(self, state: State, goal_index: int) -> np.ndarray:
        """Return a channel-first ``(3, H, W)`` obstacle/agent/goal observation."""
        self._validate_state(state)
        obstacles = self.obstacles_for_goal(goal_index)
        if state in obstacles:
            raise ValueError("the agent cannot occupy an obstacle")
        observation = np.zeros(
            (3, self.grid_size, self.grid_size), dtype=np.float32
        )
        for row, column in obstacles:
            observation[0, row, column] = 1.0
        observation[1, state[0], state[1]] = 1.0
        goal = self.goals[goal_index]
        observation[2, goal[0], goal[1]] = 1.0
        return observation

    def shortest_path_distances(self, goal_index: int) -> dict[State, int]:
        """Return BFS shortest-path distance from each reachable state to a goal."""
        if not 0 <= goal_index < len(self.goals):
            raise ValueError("goal_index is out of range")
        goal = self.goals[goal_index]
        distances = {goal: 0}
        queue = deque([goal])
        while queue:
            state = queue.popleft()
            for action in range(len(self.action_deltas)):
                neighbor = self.transition(state, action, goal_index)
                if neighbor not in distances:
                    distances[neighbor] = distances[state] + 1
                    queue.append(neighbor)
        return distances

    def expert_action(self, state: State, goal_index: int) -> int:
        """Return an action leading to a smallest BFS distance to the goal."""
        distances = self.shortest_path_distances(goal_index)
        if state not in distances:
            raise ValueError("state is not reachable from the requested goal")
        next_distances = [
            distances.get(self.transition(state, action, goal_index), np.inf)
            for action in range(len(self.action_deltas))
        ]
        return int(np.argmin(next_distances))


@dataclass(frozen=True)
class GoalConditionedDemonstrations:
    """Aligned arrays and original pairs for expert state-goal demonstrations."""

    observations: torch.Tensor
    goal_indices: torch.Tensor
    actions: torch.Tensor
    state_goal_pairs: tuple[tuple[State, int], ...]


def build_goal_conditioned_demonstrations(
    world: VisualGoalGridWorld,
) -> GoalConditionedDemonstrations:
    """Enumerate every reachable non-goal state for every requested goal."""
    observations: list[np.ndarray] = []
    goal_indices: list[int] = []
    actions: list[int] = []
    pairs: list[tuple[State, int]] = []
    for goal_index, goal in enumerate(world.goals):
        for state in world.reachable_states(goal_index):
            if state == goal:
                continue
            observations.append(world.render(state, goal_index))
            goal_indices.append(goal_index)
            actions.append(world.expert_action(state, goal_index))
            pairs.append((state, goal_index))
    return GoalConditionedDemonstrations(
        observations=torch.from_numpy(np.stack(observations)),
        goal_indices=torch.tensor(goal_indices, dtype=torch.long),
        actions=torch.tensor(actions, dtype=torch.long),
        state_goal_pairs=tuple(pairs),
    )


def split_demonstrations(
    number_examples: int,
    *,
    training_fraction: float = 0.8,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """Return reproducibly shuffled training and validation indices."""
    if number_examples <= 1:
        raise ValueError("number_examples must exceed one")
    if not 0.0 < training_fraction < 1.0:
        raise ValueError("training_fraction must lie between zero and one")
    permutation = np.random.default_rng(seed).permutation(number_examples)
    split = int(training_fraction * number_examples)
    if split == 0 or split == number_examples:
        raise ValueError("training_fraction produces an empty split")
    return permutation[:split], permutation[split:]


class VisualGoalPolicy(nn.Module):
    """CNN policy receiving obstacle, agent, and visual-goal channels."""

    def __init__(self, grid_size: int = 6, number_actions: int = 4) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 24, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(24, 32, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * grid_size * grid_size, 64),
            nn.ReLU(),
            nn.Linear(64, number_actions),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Return one action-logit vector for every channel-first observation."""
        return self.classifier(self.encoder(observation))


class TinyVisionLanguageActionPolicy(nn.Module):
    """Tiny VLA-like policy combining scene features and instruction IDs."""

    def __init__(
        self,
        grid_size: int = 6,
        number_instructions: int = 3,
        embedding_dimension: int = 12,
        number_actions: int = 4,
    ) -> None:
        super().__init__()
        self.visual_encoder = nn.Sequential(
            nn.Conv2d(2, 24, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(24, 32, 3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * grid_size * grid_size, 64),
            nn.ReLU(),
        )
        self.instruction_embedding = nn.Embedding(
            number_instructions, embedding_dimension
        )
        self.action_head = nn.Sequential(
            nn.Linear(64 + embedding_dimension, 64),
            nn.ReLU(),
            nn.Linear(64, number_actions),
        )

    def forward(
        self,
        visual_observation: torch.Tensor,
        instruction_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Return logits after concatenating visual and instruction features."""
        visual_features = self.visual_encoder(visual_observation)
        instruction_features = self.instruction_embedding(instruction_ids)
        combined_features = torch.cat(
            (visual_features, instruction_features), dim=1
        )
        return self.action_head(combined_features)


def _model_device(model: nn.Module) -> torch.device:
    """Return the device of the first model parameter."""
    return next(model.parameters()).device


@torch.no_grad()
def visual_goal_policy_action(
    model: VisualGoalPolicy,
    world: VisualGoalGridWorld,
    state: State,
    goal_index: int,
) -> int:
    """Return the visual-goal policy's greedy action for one state-goal pair."""
    observation = torch.from_numpy(world.render(state, goal_index))[None]
    logits = model(observation.to(_model_device(model)))
    return int(logits.argmax(dim=1).item())


@torch.no_grad()
def instruction_policy_action(
    model: TinyVisionLanguageActionPolicy,
    world: VisualGoalGridWorld,
    state: State,
    goal_index: int,
) -> int:
    """Return the instruction-conditioned policy's greedy action."""
    visual = torch.from_numpy(world.render(state, goal_index)[:2])[None]
    instruction = torch.tensor([goal_index], device=_model_device(model))
    logits = model(visual.to(_model_device(model)), instruction)
    return int(logits.argmax(dim=1).item())


@dataclass(frozen=True)
class PolicyRollout:
    """One closed-loop policy trajectory and its task outcome."""

    start: State
    goal_index: int
    trajectory: tuple[State, ...]
    succeeded: bool


def rollout_policy(
    world: VisualGoalGridWorld,
    action_function: ActionFunction,
    start: State,
    goal_index: int,
    *,
    maximum_steps: int = 20,
) -> PolicyRollout:
    """Run a greedy policy in closed loop until success or the step limit."""
    if maximum_steps <= 0:
        raise ValueError("maximum_steps must be positive")
    state = start
    trajectory = [state]
    goal = world.goals[goal_index]
    for _ in range(maximum_steps):
        if state == goal:
            break
        action = action_function(state, goal_index)
        state = world.transition(state, action, goal_index)
        trajectory.append(state)
    return PolicyRollout(
        start=start,
        goal_index=goal_index,
        trajectory=tuple(trajectory),
        succeeded=state == goal,
    )


def evaluate_closed_loop(
    world: VisualGoalGridWorld,
    action_function: ActionFunction,
    state_goal_pairs: Sequence[tuple[State, int]],
    *,
    maximum_steps: int = 20,
) -> tuple[float, list[PolicyRollout]]:
    """Evaluate every pair and return success rate plus complete rollouts."""
    rollouts = [
        rollout_policy(
            world,
            action_function,
            state,
            goal_index,
            maximum_steps=maximum_steps,
        )
        for state, goal_index in state_goal_pairs
    ]
    if not rollouts:
        raise ValueError("state_goal_pairs cannot be empty")
    return float(np.mean([rollout.succeeded for rollout in rollouts])), rollouts


def plot_policy_rollouts(
    world: VisualGoalGridWorld,
    policies: Mapping[str, ActionFunction],
    start: State,
    *,
    maximum_steps: int = 20,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot one complete trajectory per policy and requested goal."""
    if not policies:
        raise ValueError("policies cannot be empty")
    figure, axes = plt.subplots(
        len(policies),
        len(world.goals),
        figsize=(3.5 * len(world.goals), 3.5 * len(policies)),
        squeeze=False,
    )
    for row, (policy_name, action_function) in enumerate(policies.items()):
        for goal_index, goal in enumerate(world.goals):
            obstacle_map = _obstacle_map(world, goal_index)
            rollout = rollout_policy(
                world,
                action_function,
                start,
                goal_index,
                maximum_steps=maximum_steps,
            )
            axis = axes[row, goal_index]
            rows = [state[0] for state in rollout.trajectory]
            columns = [state[1] for state in rollout.trajectory]
            axis.imshow(
                obstacle_map,
                cmap=ListedColormap(["white", "#333333"]),
                vmin=0,
                vmax=1,
            )
            axis.plot(columns, rows, color="tab:red", marker="o", linewidth=2)
            axis.scatter(
                start[1], start[0], c="limegreen", s=100, edgecolors="black"
            )
            axis.scatter(
                goal[1], goal[0], c="gold", marker="*", s=200,
                edgecolors="black",
            )
            status = "success" if rollout.succeeded else "failure"
            axis.set_title(
                f"{policy_name} → {world.goal_names[goal_index]}\n"
                f"{len(rollout.trajectory) - 1} actions; {status}"
            )
            _format_grid_axis(axis, world.grid_size)
    figure.tight_layout()
    return figure, axes


def plot_policy_action_fields(
    world: VisualGoalGridWorld,
    policies: Mapping[str, ActionFunction],
) -> tuple[plt.Figure, np.ndarray]:
    """Plot greedy actions at every non-goal state for one or more policies."""
    if not policies:
        raise ValueError("policies cannot be empty")
    figure, axes = plt.subplots(
        len(policies),
        len(world.goals),
        figsize=(3.7 * len(world.goals), 3.7 * len(policies)),
        squeeze=False,
    )
    for row, (policy_name, action_function) in enumerate(policies.items()):
        for goal_index, goal in enumerate(world.goals):
            obstacle_map = _obstacle_map(world, goal_index)
            axis = axes[row, goal_index]
            axis.imshow(
                obstacle_map,
                cmap=ListedColormap(["white", "#333333"]),
                vmin=0,
                vmax=1,
            )
            matches = 0
            total = 0
            for state in world.reachable_states(goal_index):
                if state == goal:
                    continue
                predicted = action_function(state, goal_index)
                target = world.expert_action(state, goal_index)
                agrees = predicted == target
                matches += int(agrees)
                total += 1
                grid_row, column = state
                delta_row, delta_column = world.action_deltas[predicted]
                axis.arrow(
                    column,
                    grid_row,
                    0.32 * delta_column,
                    0.32 * delta_row,
                    width=0.035,
                    head_width=0.18,
                    head_length=0.14,
                    length_includes_head=True,
                    color="tab:blue" if agrees else "tab:orange",
                )
            axis.scatter(
                goal[1], goal[0], marker="*", s=230, c="gold",
                edgecolors="black", zorder=3,
            )
            axis.set_title(
                f"{policy_name} → {world.goal_names[goal_index]}\n"
                f"expert agreement: {matches / total:.1%}"
            )
            _format_grid_axis(axis, world.grid_size)
    legend = [
        Line2D([], [], color="tab:blue", linewidth=3, label="matches expert"),
        Line2D([], [], color="tab:orange", linewidth=3, label="different action"),
        Patch(color="#333333", label="obstacle"),
        Line2D(
            [], [], marker="*", markersize=13, markerfacecolor="gold",
            markeredgecolor="black", linestyle="None", label="goal",
        ),
    ]
    figure.legend(handles=legend, loc="lower center", ncol=4, frameon=False)
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    return figure, axes


def plot_failed_rollouts(
    world: VisualGoalGridWorld,
    rollouts: Sequence[PolicyRollout],
    *,
    maximum_examples: int = 3,
) -> tuple[plt.Figure | None, np.ndarray | None]:
    """Plot representative failed paths and their per-state visit counts."""
    failures = [rollout for rollout in rollouts if not rollout.succeeded]
    if not failures:
        return None, None
    failures.sort(key=lambda item: len(set(item.trajectory)), reverse=True)
    selected: list[PolicyRollout] = []
    selected_goals: set[int] = set()
    for rollout in failures:
        if rollout.goal_index not in selected_goals:
            selected.append(rollout)
            selected_goals.add(rollout.goal_index)
        if len(selected) == maximum_examples:
            break
    for rollout in failures:
        if len(selected) == maximum_examples:
            break
        if rollout not in selected:
            selected.append(rollout)

    figure, axes = plt.subplots(
        len(selected), 2, figsize=(9, 3.8 * len(selected)), squeeze=False
    )
    for row_index, rollout in enumerate(selected):
        obstacle_map = _obstacle_map(world, rollout.goal_index)
        goal = world.goals[rollout.goal_index]
        rows = [state[0] for state in rollout.trajectory]
        columns = [state[1] for state in rollout.trajectory]
        visit_counts = np.zeros((world.grid_size, world.grid_size), dtype=float)
        for grid_row, column in rollout.trajectory:
            visit_counts[grid_row, column] += 1
        for obstacle_row, obstacle_column in world.obstacles_for_goal(
            rollout.goal_index
        ):
            visit_counts[obstacle_row, obstacle_column] = np.nan

        path_axis, count_axis = axes[row_index]
        path_axis.imshow(
            obstacle_map,
            cmap=ListedColormap(["white", "#333333"]),
            vmin=0,
            vmax=1,
        )
        path_axis.plot(columns, rows, color="tab:red", marker="o", linewidth=2)
        path_axis.scatter(
            rollout.start[1], rollout.start[0], c="limegreen", s=100,
            edgecolors="black",
        )
        path_axis.scatter(
            goal[1], goal[0], c="gold", marker="*", s=200,
            edgecolors="black",
        )
        path_axis.set_title(
            f"goal: {world.goal_names[rollout.goal_index]}\n"
            f"{len(rollout.trajectory) - 1} actions; "
            f"{len(set(rollout.trajectory))} unique states"
        )

        visit_colormap = plt.colormaps["Reds"].with_extremes(bad="#333333")
        image = count_axis.imshow(visit_counts, cmap=visit_colormap, vmin=0)
        count_axis.scatter(
            goal[1], goal[0], c="gold", marker="*", s=200,
            edgecolors="black",
        )
        count_axis.set_title("Visits to each state")
        figure.colorbar(image, ax=count_axis, fraction=0.046, pad=0.04)
        _format_grid_axis(path_axis, world.grid_size)
        _format_grid_axis(count_axis, world.grid_size)
    figure.tight_layout()
    return figure, axes


def plot_state_goal_split(
    world: VisualGoalGridWorld,
    pairs: Sequence[tuple[State, int]],
    training_indices: Sequence[int],
) -> tuple[plt.Figure, np.ndarray]:
    """Plot spatial training/validation membership separately for each goal."""
    training = {int(index) for index in training_indices}
    colormap = ListedColormap(["#eeeeee", "tab:blue", "tab:orange", "#333333"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], colormap.N)
    figure, axes = plt.subplots(1, len(world.goals), figsize=(11, 3.8))
    axes = np.atleast_1d(axes)
    for goal_index, axis in enumerate(axes):
        split_map = np.zeros((world.grid_size, world.grid_size), dtype=int)
        for row, column in world.obstacles_for_goal(goal_index):
            split_map[row, column] = 3
        for sample_index, (state, pair_goal) in enumerate(pairs):
            if pair_goal == goal_index:
                split_map[state] = 1 if sample_index in training else 2
        axis.imshow(split_map, cmap=colormap, norm=norm)
        goal = world.goals[goal_index]
        axis.scatter(
            goal[1], goal[0], marker="*", s=220, c="gold",
            edgecolors="black",
        )
        axis.set_title(f"goal: {world.goal_names[goal_index]}")
        _format_grid_axis(axis, world.grid_size, grid_color="white")
    legend = [
        Patch(color="tab:blue", label="training pair"),
        Patch(color="tab:orange", label="validation pair"),
        Patch(color="#333333", label="obstacle"),
        Line2D(
            [], [], marker="*", markersize=13, markerfacecolor="gold",
            markeredgecolor="black", linestyle="None", label="goal",
        ),
    ]
    figure.legend(handles=legend, loc="lower center", ncol=4, frameon=False)
    figure.tight_layout(rect=(0, 0.12, 1, 1))
    return figure, axes


def plot_demonstration_examples(
    world: VisualGoalGridWorld,
    demonstrations: GoalConditionedDemonstrations,
    splits: Mapping[str, Sequence[int]],
    *,
    examples_per_split: int = 4,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot model inputs and expert-action targets from named data splits."""
    if not splits:
        raise ValueError("splits cannot be empty")
    if examples_per_split <= 0:
        raise ValueError("examples_per_split must be positive")
    figure, axes = plt.subplots(
        len(splits),
        examples_per_split,
        figsize=(3 * examples_per_split, 3 * len(splits)),
        squeeze=False,
    )
    for row, (split_name, indices) in enumerate(splits.items()):
        selected = list(indices)[:examples_per_split]
        if len(selected) < examples_per_split:
            raise ValueError("each split must contain enough examples")
        for column, sample_index in enumerate(selected):
            goal_index = int(demonstrations.goal_indices[sample_index])
            target = int(demonstrations.actions[sample_index])
            image = demonstrations.observations[sample_index].numpy()
            axis = axes[row, column]
            axis.imshow(image.transpose(1, 2, 0))
            axis.set_title(
                f"goal: {world.goal_names[goal_index]}\n"
                f"target: {world.action_names[target]}"
            )
            axis.set_xticks([])
            axis.set_yticks([])
        axes[row, 0].set_ylabel(split_name, fontsize=11, labelpad=12)
    figure.suptitle("Example inputs and expert-action targets", fontsize=14)
    figure.tight_layout()
    return figure, axes


def plot_validation_mistakes(
    world: VisualGoalGridWorld,
    demonstrations: GoalConditionedDemonstrations,
    validation_indices: Sequence[int],
    predictions: torch.Tensor,
    *,
    maximum_examples: int = 4,
    seed: int = 7,
) -> tuple[plt.Figure | None, np.ndarray | None]:
    """Plot reproducibly sampled validation examples misclassified by a policy."""
    indices = np.asarray(validation_indices, dtype=int)
    predictions = predictions.detach().cpu()
    if predictions.shape != (len(indices),):
        raise ValueError("predictions must have one value per validation index")
    targets = demonstrations.actions[indices]
    wrong_local = torch.nonzero(
        predictions != targets, as_tuple=False
    ).squeeze(1).numpy()
    if len(wrong_local) == 0:
        return None, None
    number_to_show = min(maximum_examples, len(wrong_local))
    selected = np.random.default_rng(seed).choice(
        wrong_local, size=number_to_show, replace=False
    )
    figure, axes = plt.subplots(
        1, number_to_show, figsize=(3 * number_to_show, 3.3)
    )
    axes = np.atleast_1d(axes)
    for axis, local_index in zip(axes, selected):
        sample_index = indices[local_index]
        target = int(targets[local_index])
        prediction = int(predictions[local_index])
        image = demonstrations.observations[sample_index].numpy()
        axis.imshow(image.transpose(1, 2, 0))
        axis.set_title(
            f"target: {world.action_names[target]}\n"
            f"predicted: {world.action_names[prediction]}"
        )
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle("Incorrect one-step validation predictions", fontsize=14)
    figure.tight_layout()
    return figure, axes


def _obstacle_map(
    world: VisualGoalGridWorld,
    goal_index: int,
) -> np.ndarray:
    """Return the binary obstacle map associated with one requested goal."""
    obstacle_map = np.zeros((world.grid_size, world.grid_size), dtype=int)
    for row, column in world.obstacles_for_goal(goal_index):
        obstacle_map[row, column] = 1
    return obstacle_map


def _format_grid_axis(
    axis: plt.Axes,
    grid_size: int,
    *,
    grid_color: str = "lightgray",
) -> None:
    """Apply shared cell boundaries and hide coordinate tick labels."""
    axis.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
    axis.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
    axis.grid(which="minor", color=grid_color, linewidth=0.7)
    axis.tick_params(
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )
