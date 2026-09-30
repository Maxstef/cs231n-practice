import pytest
import torch

from cs231n_practice.classifiers.self_supervised import (
    RotationPredictionModel,
    SmallEncoder,
)


def test_small_encoder_returns_one_feature_vector_per_image() -> None:
    encoder = SmallEncoder(input_channels=3, feature_dim=40)

    features = encoder(torch.randn(5, 3, 32, 32))

    assert features.shape == (5, 40)


def test_rotation_prediction_model_returns_rotation_scores() -> None:
    model = RotationPredictionModel(feature_dim=32, num_rotations=4)
    images = torch.randn(6, 3, 16, 16, requires_grad=True)

    scores = model(images)
    scores.square().mean().backward()

    assert scores.shape == (6, 4)
    assert images.grad is not None
    assert torch.isfinite(images.grad).all()
    assert all(parameter.grad is not None for parameter in model.parameters())


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"input_channels": 0}, ValueError),
        ({"feature_dim": 0}, ValueError),
        ({"feature_dim": True}, TypeError),
    ],
)
def test_small_encoder_rejects_invalid_configuration(
    kwargs: dict[str, object], error: type[Exception]
) -> None:
    with pytest.raises(error):
        SmallEncoder(**kwargs)  # type: ignore[arg-type]


def test_small_encoder_rejects_invalid_images() -> None:
    encoder = SmallEncoder()
    with pytest.raises(ValueError, match="shape"):
        encoder(torch.zeros(3, 32, 32))
    with pytest.raises(ValueError, match="3 channels"):
        encoder(torch.zeros(2, 1, 32, 32))
    with pytest.raises(ValueError, match="at least 4"):
        encoder(torch.zeros(2, 3, 3, 8))
    with pytest.raises(TypeError, match="floating-point"):
        encoder(torch.zeros(2, 3, 8, 8, dtype=torch.int64))


def test_rotation_model_rejects_invalid_number_of_rotations() -> None:
    with pytest.raises(ValueError, match="positive"):
        RotationPredictionModel(num_rotations=0)
