import pytest
import torch

from cs231n_practice.classifiers.vision_transformer import TinyVisionTransformer


def test_tiny_vision_transformer_shapes_and_patch_metadata() -> None:
    model = TinyVisionTransformer(
        image_size=16,
        patch_size=2,
        model_dim=12,
        num_heads=3,
        feed_forward_dim=20,
        num_classes=5,
    )

    scores = model(torch.randn(4, 3, 16, 16))

    assert scores.shape == (4, 5)
    assert model.num_patches == 64
    assert model.sequence_length == 65
    assert model.patch_projection.weight.shape == (12, 3, 2, 2)
    assert model.position_embedding.shape == (1, 65, 12)


def test_tiny_vision_transformer_backpropagates_to_all_parameters() -> None:
    torch.manual_seed(3)
    model = TinyVisionTransformer(
        image_size=8,
        patch_size=4,
        model_dim=8,
        num_heads=2,
        feed_forward_dim=12,
        num_classes=3,
    )
    images = torch.randn(3, 3, 8, 8)
    labels = torch.tensor([0, 1, 2])

    loss = torch.nn.functional.cross_entropy(model(images), labels)
    loss.backward()

    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_tiny_vision_transformer_preserves_batch_independence_in_eval_mode() -> None:
    torch.manual_seed(5)
    model = TinyVisionTransformer(image_size=8, patch_size=4, dropout=0.2)
    model.eval()
    images = torch.randn(2, 3, 8, 8)

    with torch.no_grad():
        batched_scores = model(images)
        separate_scores = torch.cat(
            (model(images[:1]), model(images[1:])), dim=0
        )

    torch.testing.assert_close(batched_scores, separate_scores)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"image_size": 15, "patch_size": 4}, "divisible"),
        ({"model_dim": 10, "num_heads": 4}, "divisible"),
        ({"patch_size": 0}, "positive"),
        ({"dropout": 1.0}, "dropout"),
    ],
)
def test_tiny_vision_transformer_rejects_invalid_configuration(
    arguments: dict[str, object], message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        TinyVisionTransformer(**arguments)


def test_tiny_vision_transformer_rejects_invalid_images() -> None:
    model = TinyVisionTransformer(image_size=8, patch_size=4)

    with pytest.raises(ValueError, match="shape"):
        model(torch.randn(2, 3, 7, 8))
    with pytest.raises(TypeError, match="floating-point"):
        model(torch.ones(2, 3, 8, 8, dtype=torch.int64))
    with pytest.raises(TypeError, match="torch.Tensor"):
        model(None)  # type: ignore[arg-type]
