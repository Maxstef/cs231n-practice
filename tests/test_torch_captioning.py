import pytest
import torch
from torch import nn

from cs231n_practice.classifiers import (
    RecurrentCaptionDecoder,
    TransformerCaptionDecoder,
    caption_accuracies,
    causal_attention_mask,
    generate_captions,
)


@pytest.mark.parametrize(
    "model",
    [
        RecurrentCaptionDecoder(6, 10, model_dim=8),
        TransformerCaptionDecoder(
            6,
            10,
            model_dim=8,
            num_heads=2,
            feed_forward_dim=16,
            max_caption_length=4,
            num_layers=1,
        ),
    ],
)
def test_caption_decoders_return_token_logits_and_backpropagate(model) -> None:
    image_features = torch.randn(3, 6, requires_grad=True)
    caption_inputs = torch.randint(0, 10, (3, 4))

    logits = model(image_features, caption_inputs)
    logits.square().mean().backward()

    assert logits.shape == (3, 4, 10)
    assert image_features.grad is not None
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_transformer_caption_decoder_cannot_read_future_tokens() -> None:
    torch.manual_seed(2)
    model = TransformerCaptionDecoder(
        5,
        9,
        model_dim=8,
        num_heads=2,
        feed_forward_dim=16,
        max_caption_length=4,
        num_layers=1,
    ).eval()
    image_features = torch.randn(2, 5)
    first_inputs = torch.tensor([[1, 2, 3, 4], [1, 4, 3, 2]])
    changed_future = first_inputs.clone()
    changed_future[:, 3] = torch.tensor([7, 8])

    first_logits = model(image_features, first_inputs)
    changed_logits = model(image_features, changed_future)

    torch.testing.assert_close(first_logits[:, :3], changed_logits[:, :3])
    assert not torch.allclose(first_logits[:, 3], changed_logits[:, 3])


def test_causal_attention_mask_marks_only_future_keys() -> None:
    mask = causal_attention_mask(4, dtype=torch.float64)

    assert mask.shape == (4, 4)
    assert mask.dtype == torch.float64
    assert torch.all(mask.diagonal() == 0)
    assert torch.isneginf(mask[0, 1:]).all()
    assert torch.all(mask[3] == 0)


def test_greedy_caption_generation_handles_finished_sequences() -> None:
    class ScheduledDecoder(nn.Module):
        def forward(self, image_features, caption_inputs):
            batch_size, length = caption_inputs.shape
            logits = torch.zeros(batch_size, length, 6)
            if length == 1:
                logits[0, -1, 2] = 5.0  # First sample ends immediately.
                logits[1, -1, 3] = 5.0
            else:
                logits[:, -1, 2] = 5.0
            return logits

    generated = generate_captions(
        ScheduledDecoder(),
        torch.randn(2, 4),
        start_token_id=1,
        end_token_id=2,
        max_length=3,
        forbidden_token_ids=(0,),
    )

    torch.testing.assert_close(generated, torch.tensor([[2, 2], [3, 2]]))


def test_caption_accuracies_separate_token_and_exact_matches() -> None:
    generated = torch.tensor([[3, 4, 2], [3, 7, 2]])
    targets = torch.tensor([[3, 4, 2], [3, 5, 2]])

    token_accuracy, exact_accuracy = caption_accuracies(generated, targets)

    assert token_accuracy == pytest.approx(5 / 6)
    assert exact_accuracy == pytest.approx(1 / 2)


def test_captioning_reusables_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="divisible"):
        TransformerCaptionDecoder(4, 8, model_dim=7, num_heads=2)
    with pytest.raises(ValueError, match="max_caption_length"):
        TransformerCaptionDecoder(4, 8, max_caption_length=2)(
            torch.randn(2, 4),
            torch.ones(2, 3, dtype=torch.long),
        )
    with pytest.raises(ValueError, match="equal shapes"):
        caption_accuracies(torch.ones(2, 3), torch.ones(2, 2))
