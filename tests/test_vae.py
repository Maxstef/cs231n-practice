import pytest
import torch
import torch.nn.functional as F

from cs231n_practice.classifiers import (
    ConvolutionalAutoencoder,
    ConvolutionalVAE,
    TinyVAE,
    VAEEncoderView,
)
from cs231n_practice.generative import (
    encode_vae_means,
    gaussian_kl_divergence,
    reconstruct_with_vae,
    reparameterize,
    sample_from_vae,
    train_vae,
    vae_loss,
)


def test_reparameterization_matches_formula_and_preserves_gradients() -> None:
    mean = torch.tensor([[1.0, -1.0]], requires_grad=True)
    log_variance = torch.tensor([[0.0, torch.log(torch.tensor(4.0))]], requires_grad=True)
    epsilon = torch.tensor([[0.5, -1.0]])

    latent_codes = reparameterize(mean, log_variance, epsilon)

    torch.testing.assert_close(latent_codes, torch.tensor([[1.5, -3.0]]))
    latent_codes.sum().backward()
    assert mean.grad is not None
    assert log_variance.grad is not None


def test_gaussian_kl_matches_standard_normal_cases() -> None:
    zero_mean = torch.zeros(3, 2)
    zero_log_variance = torch.zeros(3, 2)
    shifted_mean = torch.ones(3, 2)

    torch.testing.assert_close(
        gaussian_kl_divergence(zero_mean, zero_log_variance),
        torch.zeros(3),
    )
    torch.testing.assert_close(
        gaussian_kl_divergence(shifted_mean, zero_log_variance),
        torch.ones(3),
    )


def test_tiny_vae_shapes_and_gradients() -> None:
    torch.manual_seed(0)
    model = TinyVAE(image_size=4, hidden_dim=10, latent_dim=3)
    images = torch.rand(5, 4, 4)

    logits, mean, log_variance, latent_codes = model(images)
    total, reconstruction, kl = vae_loss(
        logits,
        images,
        mean,
        log_variance,
    )
    total.backward()

    assert logits.shape == images.shape
    assert mean.shape == log_variance.shape == latent_codes.shape == (5, 3)
    assert total.ndim == reconstruction.ndim == kl.ndim == 0
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_vae_loss_has_per_image_scaling_and_beta_weight() -> None:
    logits = torch.zeros(2, 2, 2)
    targets = torch.tensor(
        [
            [[0.0, 1.0], [1.0, 0.0]],
            [[1.0, 1.0], [0.0, 0.0]],
        ]
    )
    mean = torch.ones(2, 2)
    log_variance = torch.zeros_like(mean)

    total, reconstruction, kl = vae_loss(
        logits,
        targets,
        mean,
        log_variance,
        beta=0.25,
    )
    expected_reconstruction = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="sum",
    ) / targets.shape[0]

    torch.testing.assert_close(reconstruction, expected_reconstruction)
    torch.testing.assert_close(kl, torch.tensor(1.0))
    torch.testing.assert_close(total, reconstruction + 0.25 * kl)


def test_vae_loss_accepts_color_image_batches() -> None:
    logits = torch.randn(3, 3, 4, 4)
    targets = torch.rand(3, 3, 4, 4)
    mean = torch.zeros(3, 2)
    log_variance = torch.zeros_like(mean)

    total, reconstruction, kl = vae_loss(
        logits,
        targets,
        mean,
        log_variance,
    )

    expected = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="sum",
    ) / targets.shape[0]
    torch.testing.assert_close(reconstruction, expected)
    torch.testing.assert_close(kl, torch.zeros(()))
    torch.testing.assert_close(total, expected)


def test_vae_reconstruction_and_prior_sampling_helpers() -> None:
    torch.manual_seed(1)
    model = TinyVAE(image_size=4, hidden_dim=8, latent_dim=2)
    images = torch.rand(6, 4, 4)

    deterministic_first = reconstruct_with_vae(model, images)
    deterministic_second = reconstruct_with_vae(model, images)
    stochastic_first = reconstruct_with_vae(model, images, sample=True)
    stochastic_second = reconstruct_with_vae(model, images, sample=True)
    generated_first = sample_from_vae(model, 7, seed=4)
    generated_second = sample_from_vae(model, 7, seed=4)
    means = encode_vae_means(model, images)

    torch.testing.assert_close(deterministic_first, deterministic_second)
    assert not torch.equal(stochastic_first, stochastic_second)
    torch.testing.assert_close(generated_first, generated_second)
    assert generated_first.shape == (7, 4, 4)
    assert means.shape == (6, 2)
    assert means.device.type == generated_first.device.type == "cpu"
    assert torch.all((generated_first >= 0.0) & (generated_first <= 1.0))


def test_train_vae_returns_all_loss_histories() -> None:
    torch.manual_seed(2)
    model = TinyVAE(image_size=4, hidden_dim=12, latent_dim=2)
    diagonal = torch.eye(4).repeat(32, 1, 1)
    images = torch.cat((diagonal, diagonal.flip(1)), dim=0)

    history = train_vae(
        model,
        images,
        epochs=8,
        batch_size=16,
        learning_rate=1e-2,
    )

    assert set(history) == {"total", "reconstruction", "kl"}
    assert all(len(values) == 8 for values in history.values())
    assert history["total"][-1] < history["total"][0]


def test_vae_components_reject_invalid_inputs() -> None:
    model = TinyVAE(image_size=4, latent_dim=2)
    with pytest.raises(ValueError, match="shape"):
        model(torch.zeros(2, 3, 3))
    with pytest.raises(ValueError, match="equal shapes"):
        reparameterize(torch.zeros(2, 2), torch.zeros(2, 3))
    with pytest.raises(ValueError, match="nonnegative"):
        vae_loss(
            torch.zeros(2, 4, 4),
            torch.zeros(2, 4, 4),
            torch.zeros(2, 2),
            torch.zeros(2, 2),
            beta=-1.0,
        )
    with pytest.raises(TypeError, match="Boolean"):
        reconstruct_with_vae(model, torch.zeros(2, 4, 4), sample=1)


def test_convolutional_autoencoder_and_vae_shapes_and_gradients() -> None:
    torch.manual_seed(5)
    images = torch.rand(4, 3, 16, 16)
    autoencoder = ConvolutionalAutoencoder(latent_dim=7)
    vae = ConvolutionalVAE(latent_dim=7)

    ae_codes = autoencoder.encode(images)
    ae_logits = autoencoder(images)
    vae_logits, mean, log_variance, latent_codes = vae(images)
    total, _, _ = vae_loss(vae_logits, images, mean, log_variance)
    (ae_logits.square().mean() + total).backward()

    assert ae_codes.shape == (4, 7)
    assert ae_logits.shape == vae_logits.shape == images.shape
    assert mean.shape == log_variance.shape == latent_codes.shape == (4, 7)
    assert all(parameter.grad is not None for parameter in autoencoder.parameters())
    assert all(parameter.grad is not None for parameter in vae.parameters())


def test_vae_encoder_view_returns_deterministic_posterior_means() -> None:
    vae = ConvolutionalVAE(latent_dim=5)
    encoder = VAEEncoderView(vae)
    images = torch.rand(3, 3, 16, 16)

    first = encoder(images)
    second = encoder(images)
    expected, _ = vae.encode(images)

    torch.testing.assert_close(first, second)
    torch.testing.assert_close(first, expected)
    assert first.shape == (3, 5)
    assert encoder.feature_dim == 5


def test_color_vae_helpers_reconstruct_encode_sample_and_train() -> None:
    torch.manual_seed(6)
    model = ConvolutionalVAE(
        image_size=8,
        base_channels=4,
        latent_dim=3,
    )
    images = torch.rand(8, 3, 8, 8)

    reconstructions = reconstruct_with_vae(model, images)
    means = encode_vae_means(model, images)
    samples = sample_from_vae(model, 5, seed=9)
    history = train_vae(
        model,
        images,
        epochs=1,
        batch_size=4,
        learning_rate=1e-3,
    )

    assert reconstructions.shape == images.shape
    assert means.shape == (8, 3)
    assert samples.shape == (5, 3, 8, 8)
    assert all(len(values) == 1 for values in history.values())
    assert torch.all((samples >= 0.0) & (samples <= 1.0))


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"image_size": 10}, ValueError),
        ({"input_channels": 0}, ValueError),
        ({"base_channels": True}, TypeError),
        ({"latent_dim": 0}, ValueError),
    ],
)
def test_convolutional_vae_rejects_invalid_configuration(
    kwargs: dict[str, object],
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        ConvolutionalVAE(**kwargs)  # type: ignore[arg-type]
