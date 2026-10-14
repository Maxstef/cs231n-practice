import pytest
import torch
import torch.nn.functional as F

from cs231n_practice.classifiers import TinyAutoencoder
from cs231n_practice.generative import (
    autoencoder_training_step,
    binary_reconstruction_loss,
    decode_latent_grid,
    encode_images,
    interpolate_latent_codes,
    reconstruct_images,
    spatial_sharpness,
    train_autoencoder,
)


def test_tiny_autoencoder_shapes_and_gradients() -> None:
    model = TinyAutoencoder(image_size=4, hidden_dim=10, latent_dim=3)
    images = torch.rand(5, 4, 4, requires_grad=True)

    codes = model.encode(images)
    logits = model(images)
    logits.square().mean().backward()

    assert codes.shape == (5, 3)
    assert logits.shape == images.shape
    assert images.grad is not None
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_binary_reconstruction_loss_matches_pytorch() -> None:
    logits = torch.randn(3, 4, 4)
    targets = torch.rand(3, 4, 4)

    actual = binary_reconstruction_loss(logits, targets)
    expected = F.binary_cross_entropy_with_logits(logits, targets)

    torch.testing.assert_close(actual, expected)


def test_autoencoder_training_step_updates_encoder_and_decoder() -> None:
    torch.manual_seed(0)
    model = TinyAutoencoder(image_size=4, hidden_dim=8, latent_dim=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    images = torch.rand(16, 4, 4)
    before = [parameter.detach().clone() for parameter in model.parameters()]

    loss = autoencoder_training_step(model, images, optimizer)

    assert loss > 0.0
    assert all(
        not torch.equal(old, new)
        for old, new in zip(before, model.parameters())
    )


def test_train_autoencoder_returns_decreasing_history() -> None:
    torch.manual_seed(1)
    model = TinyAutoencoder(image_size=4, hidden_dim=12, latent_dim=2)
    base = torch.eye(4).repeat(32, 1, 1)
    images = torch.cat((base, base.flip(1)), dim=0)

    history = train_autoencoder(
        model,
        images,
        epochs=8,
        batch_size=16,
        learning_rate=1e-2,
    )

    assert len(history) == 8
    assert history[-1] < history[0]


def test_reconstruction_encoding_and_interpolation_helpers() -> None:
    torch.manual_seed(2)
    model = TinyAutoencoder(image_size=4, hidden_dim=8, latent_dim=2)
    images = torch.rand(6, 4, 4)

    reconstructions = reconstruct_images(model, images)
    codes = encode_images(model, images)
    interpolations = interpolate_latent_codes(
        model,
        images[0],
        images[1],
        number_steps=5,
    )

    assert reconstructions.shape == images.shape
    assert torch.all((reconstructions >= 0.0) & (reconstructions <= 1.0))
    assert codes.shape == (6, 2)
    assert codes.device.type == "cpu"
    assert interpolations.shape == (5, 4, 4)
    with torch.no_grad():
        first = torch.sigmoid(model.decode(model.encode(images[:1])))[0]
        second = torch.sigmoid(model.decode(model.encode(images[1:2])))[0]
    torch.testing.assert_close(interpolations[0], first)
    torch.testing.assert_close(interpolations[-1], second)


def test_color_vae_interpolation_uses_posterior_means() -> None:
    from cs231n_practice.classifiers import ConvolutionalVAE

    model = ConvolutionalVAE(image_size=8, base_channels=4, latent_dim=3)
    first = torch.rand(3, 8, 8)
    second = torch.rand(3, 8, 8)

    interpolations = interpolate_latent_codes(
        model,
        first,
        second,
        number_steps=5,
    )

    assert interpolations.shape == (5, 3, 8, 8)
    with torch.no_grad():
        endpoint_means, _ = model.encode(torch.stack((first, second)))
        expected_first = torch.sigmoid(model.decode(endpoint_means[:1]))[0]
        expected_second = torch.sigmoid(model.decode(endpoint_means[1:]))[0]
    torch.testing.assert_close(interpolations[0], expected_first)
    torch.testing.assert_close(interpolations[-1], expected_second)


def test_spatial_sharpness_measures_neighbor_differences() -> None:
    constant = torch.zeros(2, 3, 4, 4)
    horizontal_edge = constant.clone()
    horizontal_edge[:, :, 2:, :] = 1.0

    torch.testing.assert_close(spatial_sharpness(constant), torch.zeros(()))
    assert spatial_sharpness(horizontal_edge) > 0.0


def test_decode_latent_grid_returns_axes_and_row_major_images() -> None:
    model = TinyAutoencoder(image_size=4, hidden_dim=8, latent_dim=2)
    observed_codes = torch.tensor([[-1.0, -2.0], [1.0, 2.0]])

    decoded, first_axis, second_axis = decode_latent_grid(
        model,
        observed_codes,
        grid_size=3,
        margin=0.5,
    )

    assert decoded.shape == (9, 4, 4)
    torch.testing.assert_close(first_axis, torch.tensor([-2.0, 0.0, 2.0]))
    torch.testing.assert_close(second_axis, torch.tensor([-4.0, 0.0, 4.0]))
    assert torch.all((decoded >= 0.0) & (decoded <= 1.0))


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"image_size": 0}, ValueError),
        ({"hidden_dim": 0}, ValueError),
        ({"latent_dim": True}, TypeError),
    ],
)
def test_tiny_autoencoder_rejects_invalid_configuration(
    kwargs: dict[str, object],
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        TinyAutoencoder(**kwargs)  # type: ignore[arg-type]


def test_autoencoder_helpers_reject_invalid_inputs() -> None:
    model = TinyAutoencoder(image_size=4, latent_dim=2)
    with pytest.raises(ValueError, match="shape"):
        model(torch.zeros(2, 3, 3))
    with pytest.raises(TypeError, match="floating"):
        model(torch.zeros(2, 4, 4, dtype=torch.long))
    with pytest.raises(ValueError, match="interval"):
        binary_reconstruction_loss(
            torch.zeros(2, 4, 4),
            torch.full((2, 4, 4), 2.0),
        )
    with pytest.raises(ValueError, match="at least two"):
        interpolate_latent_codes(
            model,
            torch.zeros(4, 4),
            torch.ones(4, 4),
            number_steps=1,
        )
    with pytest.raises(ValueError, match=r"\(N, 2\)"):
        decode_latent_grid(model, torch.zeros(4, 3))
