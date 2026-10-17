import pytest
import torch
import torch.nn.functional as F

from cs231n_practice.classifiers import MLPDiscriminator, MLPGenerator
from cs231n_practice.generative import (
    gan_discriminator_loss,
    gan_discriminator_step,
    gan_generator_loss,
    gan_generator_step,
    sample_latent,
    set_requires_grad,
)


def _parameters_changed(model, before):
    return any(
        not torch.equal(old, current)
        for old, current in zip(before, model.parameters())
    )


def test_mlp_gan_models_have_expected_shapes_and_gradients() -> None:
    generator = MLPGenerator(latent_dim=3, output_dim=4, hidden_dim=8)
    discriminator = MLPDiscriminator(input_dim=4, hidden_dim=8)
    latent_vectors = torch.randn(5, 3, requires_grad=True)

    generated = generator(latent_vectors)
    logits = discriminator(generated)
    logits.mean().backward()

    assert generated.shape == (5, 4)
    assert logits.shape == (5, 1)
    assert latent_vectors.grad is not None
    assert all(parameter.grad is not None for parameter in generator.parameters())
    assert all(parameter.grad is not None for parameter in discriminator.parameters())


def test_gan_losses_match_binary_cross_entropy_with_logits() -> None:
    real_logits = torch.tensor([[2.0], [-0.5]])
    generated_logits = torch.tensor([[-1.0], [0.25]])

    expected_discriminator = F.binary_cross_entropy_with_logits(
        real_logits,
        torch.ones_like(real_logits),
    ) + F.binary_cross_entropy_with_logits(
        generated_logits,
        torch.zeros_like(generated_logits),
    )
    expected_generator = F.binary_cross_entropy_with_logits(
        generated_logits,
        torch.ones_like(generated_logits),
    )

    torch.testing.assert_close(
        gan_discriminator_loss(real_logits, generated_logits),
        expected_discriminator,
    )
    torch.testing.assert_close(
        gan_generator_loss(generated_logits),
        expected_generator,
    )


def test_sample_latent_is_reproducible_with_local_generators() -> None:
    first_generator = torch.Generator().manual_seed(7)
    second_generator = torch.Generator().manual_seed(7)

    first = sample_latent(6, 3, random_generator=first_generator)
    second = sample_latent(6, 3, random_generator=second_generator)

    assert first.shape == (6, 3)
    torch.testing.assert_close(first, second)


def test_discriminator_step_updates_only_discriminator() -> None:
    torch.manual_seed(0)
    generator = MLPGenerator(hidden_dim=8)
    discriminator = MLPDiscriminator(hidden_dim=8)
    optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-2)
    generator_before = [parameter.detach().clone() for parameter in generator.parameters()]
    discriminator_before = [
        parameter.detach().clone() for parameter in discriminator.parameters()
    ]

    loss = gan_discriminator_step(
        generator,
        discriminator,
        optimizer,
        real_samples=torch.randn(16, 2),
        latent_vectors=torch.randn(16, 2),
    )

    assert loss.ndim == 0
    assert not loss.requires_grad
    assert not _parameters_changed(generator, generator_before)
    assert _parameters_changed(discriminator, discriminator_before)
    assert all(parameter.grad is None for parameter in generator.parameters())


def test_generator_step_updates_only_generator_and_restores_freezing() -> None:
    torch.manual_seed(1)
    generator = MLPGenerator(hidden_dim=8)
    discriminator = MLPDiscriminator(hidden_dim=8)
    discriminator.network[0].bias.requires_grad_(False)
    original_states = [
        parameter.requires_grad for parameter in discriminator.parameters()
    ]
    optimizer = torch.optim.Adam(generator.parameters(), lr=1e-2)
    generator_before = [parameter.detach().clone() for parameter in generator.parameters()]
    discriminator_before = [
        parameter.detach().clone() for parameter in discriminator.parameters()
    ]

    loss = gan_generator_step(
        generator,
        discriminator,
        optimizer,
        latent_vectors=torch.randn(16, 2),
    )

    assert loss.ndim == 0
    assert not loss.requires_grad
    assert _parameters_changed(generator, generator_before)
    assert not _parameters_changed(discriminator, discriminator_before)
    assert [parameter.requires_grad for parameter in discriminator.parameters()] == original_states
    assert all(parameter.grad is None for parameter in discriminator.parameters())


def test_gan_reusables_reject_invalid_inputs() -> None:
    with pytest.raises(TypeError, match="integer"):
        MLPGenerator(latent_dim=True)
    with pytest.raises(ValueError, match="shape"):
        MLPGenerator(latent_dim=2)(torch.randn(3, 4))
    with pytest.raises(TypeError, match="floating-point dtype"):
        sample_latent(2, 3, dtype=torch.long)
    with pytest.raises(ValueError, match="nonempty"):
        gan_generator_loss(torch.empty(0, 1))
    with pytest.raises(TypeError, match="Boolean"):
        set_requires_grad(MLPDiscriminator(), 1)  # type: ignore[arg-type]
