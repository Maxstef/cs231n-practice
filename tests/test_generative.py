import math

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from cs231n_practice.generative import (
    autoregressive_cross_entropy,
    evaluate_autoregressive_model,
    images_to_sequences,
    make_autoregressive_inputs,
    nats_to_bits,
    sample_autoregressive_sequences,
    sequences_to_images,
)


class UniformAutoregressiveModel(nn.Module):
    """Return equal logits for every vocabulary class."""

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, token_ids, hidden=None):
        logits = torch.zeros(
            *token_ids.shape,
            2,
            device=token_ids.device,
        ) + self.anchor
        return logits, hidden


class HiddenTrackingModel(nn.Module):
    """Fail if a sampler does not feed the returned state into the next step."""

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.calls = 0

    def forward(self, token_ids, hidden=None):
        expected = self.calls
        if expected == 0:
            assert hidden is None
        else:
            torch.testing.assert_close(hidden, torch.tensor(float(expected)))
        self.calls += 1
        logits = torch.zeros(
            token_ids.shape[0],
            token_ids.shape[1],
            2,
            device=token_ids.device,
        ) + self.anchor
        return logits, torch.tensor(float(self.calls), device=token_ids.device)


def test_images_and_sequences_round_trip_in_raster_order() -> None:
    images = torch.arange(24).reshape(2, 3, 4)

    sequences = images_to_sequences(images)
    restored = sequences.reshape(2, 3, 4)

    assert sequences.shape == (2, 12)
    torch.testing.assert_close(restored, images)

    square_images = torch.arange(18).reshape(2, 3, 3)
    torch.testing.assert_close(
        sequences_to_images(images_to_sequences(square_images), 3),
        square_images,
    )


def test_make_autoregressive_inputs_shifts_targets() -> None:
    targets = torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.long)

    inputs = make_autoregressive_inputs(targets, bos_token=2)

    expected = torch.tensor([[2, 1, 0], [2, 0, 1]])
    torch.testing.assert_close(inputs, expected)
    assert inputs.dtype == targets.dtype
    assert inputs.device == targets.device


def test_autoregressive_cross_entropy_matches_pytorch() -> None:
    logits = torch.tensor(
        [
            [[2.0, -1.0], [0.5, 1.5]],
            [[-0.5, 0.5], [1.0, 0.0]],
        ]
    )
    targets = torch.tensor([[0, 1], [1, 0]])

    actual = autoregressive_cross_entropy(logits, targets, reduction="none")
    expected = F.cross_entropy(
        logits.reshape(-1, 2),
        targets.reshape(-1),
        reduction="none",
    ).reshape(2, 2)

    torch.testing.assert_close(actual, expected)


def test_uniform_binary_model_has_one_bit_per_token() -> None:
    model = UniformAutoregressiveModel()
    sequences = torch.tensor([[0, 1, 0], [1, 1, 0]])

    nll, bits = evaluate_autoregressive_model(
        model,
        sequences,
        bos_token=2,
        batch_size=1,
    )

    assert nll == pytest.approx(math.log(2.0))
    assert bits == pytest.approx(1.0)
    assert nats_to_bits(math.log(2.0)) == pytest.approx(1.0)


def test_sampling_is_reproducible_and_carries_hidden_state() -> None:
    first_model = HiddenTrackingModel()
    second_model = HiddenTrackingModel()

    first = sample_autoregressive_sequences(
        first_model,
        number_sequences=4,
        sequence_length=5,
        bos_token=2,
        seed=7,
    )
    second = sample_autoregressive_sequences(
        second_model,
        number_sequences=4,
        sequence_length=5,
        bos_token=2,
        seed=7,
    )

    assert first.shape == (4, 5)
    assert first_model.calls == 5
    torch.testing.assert_close(first, second)


def test_generative_utilities_reject_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="shape"):
        images_to_sequences(torch.zeros(2, 3))
    with pytest.raises(ValueError, match="squared"):
        sequences_to_images(torch.zeros(2, 8, dtype=torch.long), 3)
    with pytest.raises(TypeError, match="integer"):
        make_autoregressive_inputs(torch.zeros(2, 3), bos_token=2)
    with pytest.raises(ValueError, match="positive"):
        sample_autoregressive_sequences(
            UniformAutoregressiveModel(),
            number_sequences=2,
            sequence_length=3,
            bos_token=2,
            temperature=0.0,
        )
