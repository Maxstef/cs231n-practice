import pytest
import torch

from cs231n_practice.vision_language import (
    clip_contrastive_loss,
    image_text_similarity_matrix,
    retrieve_top_k,
)


def test_similarity_matrix_normalizes_rows_and_matches_manual_cosines() -> None:
    images = torch.tensor([[3.0, 0.0], [0.0, 2.0]])
    texts = torch.tensor([[5.0, 0.0], [1.0, 1.0], [0.0, 4.0]])

    similarities = image_text_similarity_matrix(images, texts)

    expected = torch.tensor(
        [[1.0, 2**-0.5, 0.0], [0.0, 2**-0.5, 1.0]]
    )
    torch.testing.assert_close(similarities, expected)


def test_clip_loss_is_small_for_clear_diagonal_matches() -> None:
    embeddings = torch.eye(4)

    loss, similarities = clip_contrastive_loss(
        embeddings, embeddings.clone(), temperature=0.1
    )

    torch.testing.assert_close(similarities, torch.eye(4))
    assert loss.item() < 0.001


def test_clip_loss_matches_average_of_both_directions() -> None:
    images = torch.tensor([[1.0, 0.0], [0.2, 1.0], [-1.0, 0.1]])
    texts = torch.tensor([[0.8, 0.2], [0.0, 1.0], [-0.8, -0.2]])
    temperature = 0.3

    loss, similarities = clip_contrastive_loss(images, texts, temperature)
    targets = torch.arange(3)
    logits = similarities / temperature
    expected = 0.5 * (
        torch.nn.functional.cross_entropy(logits, targets)
        + torch.nn.functional.cross_entropy(logits.T, targets)
    )

    torch.testing.assert_close(loss, expected)


def test_clip_loss_propagates_to_both_modalities() -> None:
    images = torch.randn(5, 4, requires_grad=True)
    texts = torch.randn(5, 4, requires_grad=True)

    loss, _ = clip_contrastive_loss(images, texts)
    loss.backward()

    assert images.grad is not None and torch.isfinite(images.grad).all()
    assert texts.grad is not None and torch.isfinite(texts.grad).all()


def test_retrieval_selects_rows_and_columns_in_descending_order() -> None:
    similarities = torch.tensor(
        [[0.9, 0.1, 0.2], [0.3, 0.8, 0.7]]
    )

    image_to_text = retrieve_top_k(
        similarities, 1, 2, direction="image_to_text"
    )
    text_to_image = retrieve_top_k(
        similarities, 2, 2, direction="text_to_image"
    )

    torch.testing.assert_close(image_to_text, torch.tensor([1, 2]))
    torch.testing.assert_close(text_to_image, torch.tensor([1, 0]))


@pytest.mark.parametrize("temperature", [0.0, -1.0, float("inf")])
def test_clip_loss_rejects_invalid_temperature(temperature: float) -> None:
    with pytest.raises(ValueError, match="temperature"):
        clip_contrastive_loss(torch.eye(2), torch.eye(2), temperature)


def test_vision_language_helpers_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="widths"):
        image_text_similarity_matrix(torch.ones(2, 3), torch.ones(4, 2))
    with pytest.raises(ValueError, match="same size"):
        clip_contrastive_loss(torch.ones(2, 3), torch.ones(4, 3))
    with pytest.raises(ValueError, match="direction"):
        retrieve_top_k(torch.eye(2), 0, 1, direction="unknown")
    with pytest.raises(ValueError, match="cannot exceed"):
        retrieve_top_k(torch.eye(2), 0, 3, direction="image_to_text")
