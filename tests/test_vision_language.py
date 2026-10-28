import pytest
import torch

from cs231n_practice.vision_language import (
    assemble_multimodal_sequence,
    clip_contrastive_loss,
    image_text_similarity_matrix,
    prompt_ensemble_prototypes,
    retrieve_top_k,
    zero_shot_classify,
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


def test_prompt_ensemble_normalizes_prompts_and_final_prototypes() -> None:
    prompt_embeddings = torch.tensor(
        [
            [[2.0, 0.0], [0.0, 4.0]],
            [[-3.0, 0.0], [-2.0, 0.0]],
        ]
    )

    prototypes = prompt_ensemble_prototypes(prompt_embeddings)

    expected = torch.tensor(
        [[2**-0.5, 2**-0.5], [-1.0, 0.0]]
    )
    torch.testing.assert_close(prototypes, expected)
    torch.testing.assert_close(prototypes.norm(dim=1), torch.ones(2))


def test_zero_shot_classify_returns_cosine_scores_and_predictions() -> None:
    images = torch.tensor([[4.0, 0.0], [0.0, 3.0], [-2.0, 0.0]])
    prototypes = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    scores, predictions = zero_shot_classify(images, prototypes)

    expected_scores = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
    )
    torch.testing.assert_close(scores, expected_scores)
    torch.testing.assert_close(predictions, torch.tensor([0, 1, 1]))


def test_prompt_ensemble_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="nonempty shape"):
        prompt_ensemble_prototypes(torch.ones(2, 3))
    with pytest.raises(TypeError, match="floating-point"):
        prompt_ensemble_prototypes(torch.ones(2, 3, 4, dtype=torch.long))


def test_assemble_multimodal_sequence_tags_and_concatenates_tokens() -> None:
    visual_tokens = torch.zeros(2, 3, 4)
    text_tokens = torch.ones(2, 2, 4)
    visual_valid = torch.ones(2, 3, dtype=torch.bool)
    text_valid = torch.tensor([[True, True], [True, False]])
    visual_modality = torch.tensor([1.0, 2.0, 3.0, 4.0])
    text_modality = torch.tensor([-1.0, -2.0, -3.0, -4.0])

    joint_tokens, joint_valid = assemble_multimodal_sequence(
        visual_tokens,
        text_tokens,
        visual_valid,
        text_valid,
        visual_modality_embedding=visual_modality,
        text_modality_embedding=text_modality,
    )

    assert joint_tokens.shape == (2, 5, 4)
    torch.testing.assert_close(
        joint_tokens[:, :3],
        visual_modality.expand(2, 3, 4),
    )
    torch.testing.assert_close(
        joint_tokens[:, 3:],
        (torch.ones(4) + text_modality).expand(2, 2, 4),
    )
    torch.testing.assert_close(
        joint_valid,
        torch.cat((visual_valid, text_valid), dim=1),
    )


def test_assemble_multimodal_sequence_rejects_incompatible_inputs() -> None:
    visual_tokens = torch.zeros(2, 3, 4)
    text_tokens = torch.zeros(2, 2, 4)
    visual_valid = torch.ones(2, 3, dtype=torch.bool)
    text_valid = torch.ones(2, 2, dtype=torch.bool)

    with pytest.raises(ValueError, match="widths"):
        assemble_multimodal_sequence(
            visual_tokens,
            torch.zeros(2, 2, 5),
            visual_valid,
            text_valid,
        )
    with pytest.raises(TypeError, match="Boolean"):
        assemble_multimodal_sequence(
            visual_tokens,
            text_tokens,
            visual_valid.float(),
            text_valid,
        )
    with pytest.raises(ValueError, match="shape"):
        assemble_multimodal_sequence(
            visual_tokens,
            text_tokens,
            visual_valid,
            text_valid,
            visual_modality_embedding=torch.zeros(5),
        )
