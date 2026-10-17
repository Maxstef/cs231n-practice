import pytest
import torch
from torch import nn

from cs231n_practice.classifiers import VelocityMLP
from cs231n_practice.generative import (
    diffuse_samples,
    make_linear_diffusion_schedule,
    make_rectified_flow_pairs,
    reconstruct_clean_samples,
    rectified_flow_loss,
    rectified_flow_training_step,
    sample_rectified_flow,
)


def test_linear_diffusion_schedule_has_expected_relationships() -> None:
    betas, alphas, alpha_bars = make_linear_diffusion_schedule(
        4,
        beta_start=0.1,
        beta_end=0.4,
        dtype=torch.float64,
    )

    torch.testing.assert_close(alphas, 1.0 - betas)
    torch.testing.assert_close(alpha_bars, torch.cumprod(alphas, dim=0))
    assert alpha_bars.dtype == torch.float64
    assert torch.all(alpha_bars[1:] < alpha_bars[:-1])


def test_diffusion_and_exact_noise_reconstruct_clean_samples() -> None:
    _, _, alpha_bars = make_linear_diffusion_schedule(10)
    clean = torch.tensor([[1.0, 2.0], [-1.0, 0.5]])
    noise = torch.tensor([[0.5, -0.5], [1.0, -1.0]])
    times = torch.tensor([0, 9])

    noisy, returned_noise = diffuse_samples(
        clean,
        times,
        alpha_bars,
        noise=noise,
    )
    recovered = reconstruct_clean_samples(
        noisy,
        returned_noise,
        alpha_bars[times],
    )

    torch.testing.assert_close(returned_noise, noise)
    torch.testing.assert_close(recovered, clean)


def test_rectified_flow_pairs_reach_both_endpoints() -> None:
    data = torch.tensor([[5.0, 3.0], [-2.0, 4.0]])
    noise = torch.tensor([[1.0, -1.0], [2.0, 2.0]])
    times = torch.tensor([[0.0], [1.0]])

    interpolated, returned_times, velocity = make_rectified_flow_pairs(
        data,
        noise=noise,
        times=times,
    )

    torch.testing.assert_close(interpolated[0], noise[0])
    torch.testing.assert_close(interpolated[1], data[1])
    torch.testing.assert_close(returned_times, times)
    torch.testing.assert_close(velocity, data - noise)


def test_velocity_mlp_shapes_and_backpropagates() -> None:
    model = VelocityMLP(input_dim=3, hidden_dim=12)
    positions = torch.randn(5, 3, requires_grad=True)
    times = torch.rand(5, 1)

    velocities = model(positions, times)
    velocities.square().mean().backward()

    assert velocities.shape == positions.shape
    assert positions.grad is not None
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_rectified_flow_loss_matches_manual_mse() -> None:
    class DoublePosition(nn.Module):
        def forward(self, positions, times):
            return 2.0 * positions

    positions = torch.tensor([[1.0, -1.0], [0.5, 2.0]])
    times = torch.tensor([[0.25], [0.75]])
    targets = torch.tensor([[1.0, 0.0], [-1.0, 2.0]])

    actual = rectified_flow_loss(DoublePosition(), positions, times, targets)
    expected = torch.mean((2.0 * positions - targets).square())

    torch.testing.assert_close(actual, expected)


def test_training_step_updates_velocity_model() -> None:
    torch.manual_seed(0)
    model = VelocityMLP(hidden_dim=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    data = torch.randn(32, 2) + 2.0
    before = [parameter.detach().clone() for parameter in model.parameters()]

    loss = rectified_flow_training_step(model, optimizer, data)

    assert loss.ndim == 0
    assert not loss.requires_grad
    assert any(
        not torch.equal(old, current)
        for old, current in zip(before, model.parameters())
    )


def test_euler_sampler_uses_time_and_preserves_device_and_dtype() -> None:
    class ConstantVelocity(nn.Module):
        def forward(self, positions, times):
            return torch.cat((torch.ones_like(times), 2.0 * torch.ones_like(times)), dim=1)

    initial = torch.zeros(3, 2, dtype=torch.float64)
    final, snapshots = sample_rectified_flow(
        ConstantVelocity(),
        initial,
        4,
        snapshot_indices=(1, 2),
    )

    expected = torch.tensor([[1.0, 2.0]], dtype=initial.dtype).repeat(3, 1)
    torch.testing.assert_close(final, expected)
    assert set(snapshots) == {0, 1, 2, 4}
    assert final.dtype == initial.dtype
    assert final.device == initial.device


def test_diffusion_and_flow_helpers_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="betas"):
        make_linear_diffusion_schedule(5, beta_start=0.2, beta_end=0.1)
    with pytest.raises(ValueError, match="outside"):
        diffuse_samples(
            torch.randn(2, 3),
            torch.tensor([0, 3]),
            torch.tensor([0.9, 0.8, 0.7]),
        )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        make_rectified_flow_pairs(
            torch.randn(2, 3),
            times=torch.tensor([[0.5], [1.5]]),
        )
    with pytest.raises(ValueError, match="between zero"):
        sample_rectified_flow(
            VelocityMLP(input_dim=2),
            torch.randn(2, 2),
            5,
            snapshot_indices=(6,),
        )
