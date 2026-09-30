import pytest
import torch
from torch import nn

from cs231n_practice.self_supervised import (
    extract_features,
    freeze_encoder,
    make_rotation_batch,
)


def test_make_rotation_batch_matches_every_sampled_target() -> None:
    images = torch.arange(8 * 3 * 5 * 5).reshape(8, 3, 5, 5)

    rotated, targets = make_rotation_batch(
        images,
        generator=torch.Generator().manual_seed(7),
    )

    assert rotated.shape == images.shape
    assert rotated.dtype == images.dtype
    assert targets.shape == (8,)
    assert targets.dtype == torch.long
    assert bool(((0 <= targets) & (targets < 4)).all())
    for index, target in enumerate(targets):
        expected = torch.rot90(images[index], int(target), dims=(-2, -1))
        torch.testing.assert_close(rotated[index], expected)


def test_make_rotation_batch_is_reproducible_without_modifying_input() -> None:
    images = torch.randn(12, 3, 6, 6)
    original = images.clone()

    first = make_rotation_batch(images, torch.Generator().manual_seed(11))
    second = make_rotation_batch(images, torch.Generator().manual_seed(11))

    torch.testing.assert_close(first[0], second[0])
    torch.testing.assert_close(first[1], second[1])
    torch.testing.assert_close(images, original)


def test_make_rotation_batch_rejects_invalid_inputs() -> None:
    with pytest.raises(TypeError, match="tensor"):
        make_rotation_batch([[1.0]])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="shape"):
        make_rotation_batch(torch.zeros(3, 8, 8))
    with pytest.raises(ValueError, match="square"):
        make_rotation_batch(torch.zeros(2, 3, 8, 7))
    with pytest.raises(TypeError, match="Generator"):
        make_rotation_batch(torch.zeros(2, 3, 8, 8), generator=3)  # type: ignore[arg-type]


def test_freeze_encoder_disables_gradients_and_evaluation_behavior() -> None:
    encoder = nn.Sequential(nn.Linear(4, 5), nn.Dropout(0.5))
    encoder.train()

    returned = freeze_encoder(encoder)

    assert returned is encoder
    assert not encoder.training
    assert all(not parameter.requires_grad for parameter in encoder.parameters())


def test_freeze_encoder_rejects_non_module() -> None:
    with pytest.raises(TypeError, match="Module"):
        freeze_encoder(lambda x: x)  # type: ignore[arg-type]


def test_extract_features_batches_results_and_detaches_them() -> None:
    encoder = nn.Linear(4, 3, bias=False)
    images = torch.randn(7, 4)
    expected = encoder(images).detach()

    features = extract_features(encoder, images, batch_size=3)

    assert features.shape == (7, 3)
    assert features.device.type == "cpu"
    assert not features.requires_grad
    assert not encoder.training
    torch.testing.assert_close(features, expected)
    assert all(parameter.grad is None for parameter in encoder.parameters())


def test_extract_features_rejects_invalid_arguments() -> None:
    encoder = nn.Linear(4, 3)
    with pytest.raises(TypeError, match="Module"):
        extract_features(lambda x: x, torch.zeros(2, 4))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="tensor"):
        extract_features(encoder, [[1.0, 2.0, 3.0, 4.0]])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="nonempty"):
        extract_features(encoder, torch.zeros(0, 4))
    with pytest.raises(ValueError, match="positive"):
        extract_features(encoder, torch.zeros(2, 4), batch_size=0)
    with pytest.raises(TypeError, match="integer"):
        extract_features(encoder, torch.zeros(2, 4), batch_size=True)
