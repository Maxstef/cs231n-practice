"""Reusable components for image-conditioned recurrent captioning models."""

from collections.abc import Mapping

import numpy as np

from cs231n_practice.layers import affine_backward, affine_forward
from cs231n_practice.rnn_layers import rnn_backward, rnn_forward, rnn_step_forward
from cs231n_practice.sequence_layers import (
    embedding_backward,
    embedding_forward,
    temporal_affine_backward,
    temporal_affine_forward,
    temporal_softmax_loss,
)

CAPTIONING_PARAMETER_NAMES = (
    "embedding_matrix",
    "image_weights",
    "image_bias",
    "rnn_weights_x",
    "rnn_weights_h",
    "rnn_bias",
    "output_weights",
    "output_bias",
)


def _token_id(value: int, *, name: str, vocabulary_size: int | None = None) -> int:
    """Validate one scalar integer token ID."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer token ID")
    token_id = int(value)
    if token_id < 0 or (
        vocabulary_size is not None and token_id >= vocabulary_size
    ):
        raise ValueError(f"{name} is outside the vocabulary")
    return token_id


def _parameters(parameters: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return the required captioning parameters after checking their names."""
    if not isinstance(parameters, Mapping):
        raise TypeError("parameters must be a mapping")
    missing = set(CAPTIONING_PARAMETER_NAMES) - parameters.keys()
    extra = parameters.keys() - set(CAPTIONING_PARAMETER_NAMES)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing {sorted(missing)}")
        if extra:
            details.append(f"unexpected {sorted(extra)}")
        raise ValueError("invalid captioning parameters: " + ", ".join(details))
    return {name: np.asarray(parameters[name]) for name in CAPTIONING_PARAMETER_NAMES}


def prepare_caption_data(
    captions: np.ndarray,
    pad_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shift stored captions into model inputs, next-token targets, and a mask.

    Args:
        captions: Integer token IDs with shape ``(N, T + 1)``. A row normally
            starts with ``<START>`` and contains ``<END>`` before any padding.
        pad_id: Vocabulary ID used for padding.

    Returns:
        ``caption_inputs``, ``caption_targets``, and a Boolean target mask,
        each with shape ``(N, T)``.
    """
    captions = np.asarray(captions)
    if not np.issubdtype(captions.dtype, np.integer) or np.issubdtype(
        captions.dtype, np.bool_
    ):
        raise TypeError("captions must contain integer token IDs")
    if captions.ndim != 2 or captions.shape[0] == 0 or captions.shape[1] < 2:
        raise ValueError("captions must have shape (N, T + 1) with T >= 1")
    pad_id = _token_id(pad_id, name="pad_id")
    if np.any(captions < 0):
        raise ValueError("captions must contain nonnegative token IDs")

    caption_inputs = captions[:, :-1]
    caption_targets = captions[:, 1:]
    mask = caption_targets != pad_id
    return caption_inputs, caption_targets, mask


def rnn_captioning_loss(
    features: np.ndarray,
    captions: np.ndarray,
    parameters: Mapping[str, np.ndarray],
    pad_id: int,
) -> tuple[float, dict[str, np.ndarray]]:
    """Return masked captioning loss and gradients for a vanilla-RNN model.

    Image features are projected into ``h0``. Shifted caption inputs pass
    through an embedding layer and the RNN, and every hidden state is projected
    into vocabulary scores. The loss is averaged over the ``N`` images.
    """
    parameters = _parameters(parameters)
    caption_inputs, caption_targets, mask = prepare_caption_data(captions, pad_id)
    features = np.asarray(features)
    if features.ndim != 2 or features.shape[0] != captions.shape[0]:
        raise ValueError("features must have shape (N, D_image) matching captions")

    h0, image_cache = affine_forward(
        features,
        parameters["image_weights"],
        parameters["image_bias"],
    )
    embeddings, embedding_cache = embedding_forward(
        caption_inputs,
        parameters["embedding_matrix"],
    )
    hidden_states, rnn_cache = rnn_forward(
        embeddings,
        h0,
        parameters["rnn_weights_x"],
        parameters["rnn_weights_h"],
        parameters["rnn_bias"],
    )
    scores, output_cache = temporal_affine_forward(
        hidden_states,
        parameters["output_weights"],
        parameters["output_bias"],
    )
    loss, dscores = temporal_softmax_loss(scores, caption_targets, mask)

    dhidden, doutput_weights, doutput_bias = temporal_affine_backward(
        dscores, output_cache
    )
    dembeddings, dh0, drnn_weights_x, drnn_weights_h, drnn_bias = rnn_backward(
        dhidden, rnn_cache
    )
    dembedding_matrix = embedding_backward(dembeddings, embedding_cache)
    _, dimage_weights, dimage_bias = affine_backward(dh0, image_cache)

    gradients = {
        "embedding_matrix": dembedding_matrix,
        "image_weights": dimage_weights,
        "image_bias": dimage_bias,
        "rnn_weights_x": drnn_weights_x,
        "rnn_weights_h": drnn_weights_h,
        "rnn_bias": drnn_bias,
        "output_weights": doutput_weights,
        "output_bias": doutput_bias,
    }
    return loss, gradients


def sample_rnn_captions(
    features: np.ndarray,
    parameters: Mapping[str, np.ndarray],
    *,
    start_id: int,
    end_id: int,
    pad_id: int,
    max_length: int,
) -> np.ndarray:
    """Generate one caption per image using batched greedy decoding.

    ``<START>`` and ``<PAD>`` are forbidden as generated outputs. Positions
    after a generated ``<END>`` remain filled with ``end_id``.
    """
    parameters = _parameters(parameters)
    features = np.asarray(features)
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError("features must have nonempty shape (N, D_image)")
    if isinstance(max_length, (bool, np.bool_)) or not isinstance(
        max_length, (int, np.integer)
    ):
        raise TypeError("max_length must be an integer")
    if max_length <= 0:
        raise ValueError("max_length must be positive")

    vocabulary_size = parameters["embedding_matrix"].shape[0]
    start_id = _token_id(start_id, name="start_id", vocabulary_size=vocabulary_size)
    end_id = _token_id(end_id, name="end_id", vocabulary_size=vocabulary_size)
    pad_id = _token_id(pad_id, name="pad_id", vocabulary_size=vocabulary_size)
    if len({start_id, end_id, pad_id}) != 3:
        raise ValueError("start_id, end_id, and pad_id must be distinct")

    hidden, _ = affine_forward(
        features,
        parameters["image_weights"],
        parameters["image_bias"],
    )
    num_examples = features.shape[0]
    current_ids = np.full(num_examples, start_id, dtype=np.int64)
    sampled_ids = np.full((num_examples, max_length), end_id, dtype=np.int64)
    finished = np.zeros(num_examples, dtype=bool)

    for time_index in range(max_length):
        embedded, _ = embedding_forward(
            current_ids[:, None], parameters["embedding_matrix"]
        )
        hidden, _ = rnn_step_forward(
            embedded[:, 0, :],
            hidden,
            parameters["rnn_weights_x"],
            parameters["rnn_weights_h"],
            parameters["rnn_bias"],
        )
        scores = hidden @ parameters["output_weights"] + parameters["output_bias"]
        scores[:, [start_id, pad_id]] = -np.inf
        next_ids = np.argmax(scores, axis=1)
        next_ids = np.where(finished, end_id, next_ids)
        sampled_ids[:, time_index] = next_ids
        finished |= next_ids == end_id
        current_ids = next_ids
        if np.all(finished):
            break

    return sampled_ids
