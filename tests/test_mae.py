import pytest
import torch

from cs231n_practice.classifiers.self_supervised import (
    MAEEncoderView,
    TinyMaskedAutoencoder,
)
from cs231n_practice.self_supervised import (
    masked_reconstruction_loss,
    patchify,
    random_mask_patches,
    restore_mask_tokens,
    unpatchify,
)


def test_patchify_uses_row_major_patch_order_and_round_trips() -> None:
    images = torch.arange(16).reshape(1, 1, 4, 4)

    patches = patchify(images, patch_size=2)

    expected = torch.tensor([[
        [0, 1, 4, 5],
        [2, 3, 6, 7],
        [8, 9, 12, 13],
        [10, 11, 14, 15],
    ]])
    torch.testing.assert_close(patches, expected)
    restored = unpatchify(patches, 2, 1, 4, 4)
    torch.testing.assert_close(restored, images)


def test_patch_conversion_rejects_incompatible_shapes() -> None:
    with pytest.raises(ValueError, match="divisible"):
        patchify(torch.zeros(2, 3, 7, 8), patch_size=4)
    with pytest.raises(ValueError, match="patch count"):
        unpatchify(torch.zeros(2, 3, 12), 2, 3, 4, 4)
    with pytest.raises(ValueError, match="patch_dim"):
        unpatchify(torch.zeros(2, 4, 10), 2, 3, 4, 4)


def test_random_mask_patches_selects_and_marks_the_same_positions() -> None:
    patches = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)

    visible, mask, ids_keep, ids_restore = random_mask_patches(
        patches,
        0.5,
        generator=torch.Generator().manual_seed(9),
    )

    assert visible.shape == (2, 2, 3)
    assert mask.shape == (2, 4)
    assert ids_keep.shape == (2, 2)
    assert ids_restore.shape == (2, 4)
    torch.testing.assert_close(mask.sum(dim=1), torch.tensor([2.0, 2.0]))
    torch.testing.assert_close(mask.gather(1, ids_keep), torch.zeros(2, 2))
    torch.testing.assert_close(
        visible,
        patches.gather(1, ids_keep.unsqueeze(-1).expand(-1, -1, 3)),
    )


def test_random_mask_patches_is_reproducible_with_a_generator() -> None:
    patches = torch.randn(3, 8, 5)

    first = random_mask_patches(
        patches,
        0.75,
        generator=torch.Generator().manual_seed(17),
    )
    second = random_mask_patches(
        patches,
        0.75,
        generator=torch.Generator().manual_seed(17),
    )

    for first_value, second_value in zip(first, second):
        torch.testing.assert_close(first_value, second_value)


def test_restore_mask_tokens_returns_original_spatial_order() -> None:
    visible = torch.tensor([[[20.0], [0.0]]])
    ids_restore = torch.tensor([[1, 3, 0, 2]])
    mask_token = torch.tensor([[[-1.0]]])

    restored = restore_mask_tokens(visible, ids_restore, mask_token)

    expected = torch.tensor([[[0.0], [-1.0], [20.0], [-1.0]]])
    torch.testing.assert_close(restored, expected)


def test_masked_reconstruction_loss_uses_only_masked_values() -> None:
    predictions = torch.tensor(
        [[[1.0, 2.0], [5.0, 8.0]]],
        requires_grad=True,
    )
    targets = torch.tensor([[[1.0, 0.0], [2.0, 4.0]]])
    mask = torch.tensor([[0.0, 1.0]])

    loss = masked_reconstruction_loss(predictions, targets, mask)
    loss.backward()

    torch.testing.assert_close(loss, torch.tensor(12.5))
    torch.testing.assert_close(predictions.grad[:, 0], torch.zeros(1, 2))
    assert predictions.grad[:, 1].abs().sum() > 0


def test_masked_reconstruction_loss_rejects_invalid_masks() -> None:
    predictions = torch.zeros(2, 4, 3)
    targets = torch.zeros_like(predictions)
    with pytest.raises(ValueError, match="binary"):
        masked_reconstruction_loss(
            predictions,
            targets,
            torch.full((2, 4), 0.5),
        )
    with pytest.raises(ValueError, match="at least one"):
        masked_reconstruction_loss(predictions, targets, torch.zeros(2, 4))


def test_tiny_mae_forward_loss_and_encoder_shapes() -> None:
    torch.manual_seed(23)
    model = TinyMaskedAutoencoder(
        image_size=8,
        patch_size=4,
        encoder_dim=12,
        decoder_dim=8,
        num_heads=4,
        encoder_layers=1,
    )
    images = torch.rand(2, 3, 8, 8, requires_grad=True)

    predictions, targets, mask = model(
        images,
        mask_ratio=0.5,
        generator=torch.Generator().manual_seed(29),
    )
    loss = masked_reconstruction_loss(predictions, targets, mask)
    loss.backward()

    assert predictions.shape == (2, 4, 48)
    assert targets.shape == predictions.shape
    assert mask.shape == (2, 4)
    assert model.encode_all(images.detach()).shape == (2, 12)
    assert images.grad is not None
    assert torch.isfinite(images.grad).all()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_mae_encoder_view_exposes_mean_pooled_features() -> None:
    model = TinyMaskedAutoencoder(
        image_size=8,
        patch_size=4,
        encoder_dim=12,
        decoder_dim=8,
        num_heads=4,
        encoder_layers=1,
    )
    encoder = MAEEncoderView(model)
    images = torch.rand(3, 3, 8, 8)

    torch.testing.assert_close(encoder(images), model.encode_all(images))
    assert encoder.feature_dim == 12


def test_tiny_mae_rejects_invalid_configuration_and_images() -> None:
    with pytest.raises(ValueError, match="divisible by patch_size"):
        TinyMaskedAutoencoder(image_size=10, patch_size=4)
    with pytest.raises(ValueError, match="encoder_dim"):
        TinyMaskedAutoencoder(encoder_dim=10, num_heads=4)
    model = TinyMaskedAutoencoder()
    with pytest.raises(ValueError, match="shape"):
        model(torch.zeros(2, 3, 16, 16))
    with pytest.raises(TypeError, match="floating-point"):
        model(torch.zeros(2, 3, 32, 32, dtype=torch.int64))
