import pytest
import torch
from torch import nn

from cs231n_practice.classifiers import ConditionalVelocityMLP
from cs231n_practice.generative import (
    classifier_free_guided_velocity,
    conditional_rectified_flow_loss,
    conditional_rectified_flow_training_step,
    drop_conditions,
    sample_conditional_rectified_flow,
)


def test_conditional_velocity_mlp_shapes_and_null_embedding() -> None:
    model = ConditionalVelocityMLP(
        input_dim=3,
        number_conditions=4,
        condition_dim=5,
        hidden_dim=12,
    )
    positions = torch.randn(6, 3, requires_grad=True)
    times = torch.rand(6, 1)
    conditions = torch.tensor([0, 1, 2, 3, 4, 0])

    velocities = model(positions, times, conditions)
    velocities.square().mean().backward()

    assert model.null_condition == 4
    assert model.condition_embedding.num_embeddings == 5
    assert velocities.shape == positions.shape
    assert positions.grad is not None
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_condition_dropout_handles_endpoint_probabilities() -> None:
    conditions = torch.tensor([0, 1, 2, 3])

    kept = drop_conditions(conditions, 0.0, null_condition=4)
    dropped = drop_conditions(conditions, 1.0, null_condition=4)

    torch.testing.assert_close(kept, conditions)
    torch.testing.assert_close(dropped, torch.full_like(conditions, 4))
    torch.testing.assert_close(conditions, torch.tensor([0, 1, 2, 3]))


def test_condition_dropout_is_reproducible_with_local_generator() -> None:
    conditions = torch.arange(100) % 4
    first_generator = torch.Generator().manual_seed(7)
    second_generator = torch.Generator().manual_seed(7)

    first = drop_conditions(
        conditions,
        0.3,
        null_condition=4,
        random_generator=first_generator,
    )
    second = drop_conditions(
        conditions,
        0.3,
        null_condition=4,
        random_generator=second_generator,
    )

    torch.testing.assert_close(first, second)


def test_classifier_free_guidance_reference_scales() -> None:
    class ConditionVelocity(nn.Module):
        def forward(self, positions, times, conditions):
            return conditions.to(positions.dtype).unsqueeze(1).expand_as(positions)

    positions = torch.zeros(3, 2)
    times = torch.zeros(3, 1)
    conditions = torch.tensor([0, 1, 2])
    model = ConditionVelocity()

    unconditional = classifier_free_guided_velocity(
        model,
        positions,
        times,
        conditions,
        null_condition=3,
        guidance_scale=0.0,
    )
    conditional = classifier_free_guided_velocity(
        model,
        positions,
        times,
        conditions,
        null_condition=3,
        guidance_scale=1.0,
    )
    stronger = classifier_free_guided_velocity(
        model,
        positions,
        times,
        conditions,
        null_condition=3,
        guidance_scale=2.0,
    )

    torch.testing.assert_close(unconditional, torch.full_like(positions, 3.0))
    torch.testing.assert_close(
        conditional,
        conditions.float().unsqueeze(1).expand_as(positions),
    )
    torch.testing.assert_close(stronger, 2.0 * conditional - unconditional)


def test_conditional_flow_loss_matches_manual_mse() -> None:
    class ConditionalOffset(nn.Module):
        def forward(self, positions, times, conditions):
            return positions + conditions.float().unsqueeze(1)

    positions = torch.tensor([[1.0, -1.0], [0.5, 2.0]])
    times = torch.tensor([[0.25], [0.75]])
    conditions = torch.tensor([0, 2])
    targets = torch.tensor([[1.0, 0.0], [-1.0, 2.0]])

    actual = conditional_rectified_flow_loss(
        ConditionalOffset(),
        positions,
        times,
        targets,
        conditions,
    )
    predicted = positions + conditions.float().unsqueeze(1)
    torch.testing.assert_close(actual, torch.mean((predicted - targets).square()))


def test_conditional_training_step_updates_model() -> None:
    torch.manual_seed(0)
    model = ConditionalVelocityMLP(
        number_conditions=3,
        hidden_dim=8,
        condition_dim=4,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    data = torch.randn(24, 2)
    conditions = torch.arange(24) % 3
    before = [parameter.detach().clone() for parameter in model.parameters()]

    loss = conditional_rectified_flow_training_step(
        model,
        optimizer,
        data,
        conditions,
        null_condition=model.null_condition,
    )

    assert loss.ndim == 0
    assert not loss.requires_grad
    assert any(
        not torch.equal(old, current)
        for old, current in zip(before, model.parameters())
    )


def test_conditional_sampler_integrates_guided_velocity() -> None:
    class ConstantConditionVelocity(nn.Module):
        def forward(self, positions, times, conditions):
            values = conditions.to(positions.dtype)
            return torch.stack((values, -values), dim=1)

    initial = torch.zeros(3, 2, dtype=torch.float64)
    conditions = torch.tensor([0, 1, 2])
    final = sample_conditional_rectified_flow(
        ConstantConditionVelocity(),
        initial,
        conditions,
        5,
        null_condition=3,
        guidance_scale=1.0,
    )
    expected = torch.tensor(
        [[0.0, 0.0], [1.0, -1.0], [2.0, -2.0]],
        dtype=initial.dtype,
    )

    torch.testing.assert_close(final, expected)
    assert final.dtype == initial.dtype
    assert final.device == initial.device


def test_conditional_flow_rejects_invalid_inputs() -> None:
    model = ConditionalVelocityMLP(number_conditions=3)
    with pytest.raises(ValueError, match="outside"):
        model(torch.randn(2, 2), torch.rand(2, 1), torch.tensor([0, 4]))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        drop_conditions(torch.tensor([0, 1]), 1.5, null_condition=3)
    with pytest.raises(ValueError, match="shape"):
        sample_conditional_rectified_flow(
            model,
            torch.randn(2, 2),
            torch.tensor([0]),
            5,
            null_condition=3,
        )
