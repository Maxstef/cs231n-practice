import numpy as np
import pytest

from cs231n_practice.captioning import (
    prepare_caption_data,
    rnn_captioning_loss,
    sample_rnn_captions,
)
from cs231n_practice.gradient_check import eval_numerical_gradient


def _parameters(generator: np.random.Generator) -> dict[str, np.ndarray]:
    vocabulary_size = 6
    feature_dim = 3
    embedding_dim = 2
    hidden_dim = 2
    return {
        "embedding_matrix": generator.normal(
            scale=0.1, size=(vocabulary_size, embedding_dim)
        ),
        "image_weights": generator.normal(
            scale=0.1, size=(feature_dim, hidden_dim)
        ),
        "image_bias": generator.normal(scale=0.1, size=hidden_dim),
        "rnn_weights_x": generator.normal(
            scale=0.1, size=(embedding_dim, hidden_dim)
        ),
        "rnn_weights_h": generator.normal(
            scale=0.1, size=(hidden_dim, hidden_dim)
        ),
        "rnn_bias": generator.normal(scale=0.1, size=hidden_dim),
        "output_weights": generator.normal(
            scale=0.1, size=(hidden_dim, vocabulary_size)
        ),
        "output_bias": generator.normal(scale=0.1, size=vocabulary_size),
    }


def test_prepare_caption_data_shifts_tokens_and_masks_targets() -> None:
    captions = np.array([[1, 3, 2, 0], [1, 4, 5, 2]])

    inputs, targets, mask = prepare_caption_data(captions, pad_id=0)

    np.testing.assert_array_equal(inputs, [[1, 3, 2], [1, 4, 5]])
    np.testing.assert_array_equal(targets, [[3, 2, 0], [4, 5, 2]])
    np.testing.assert_array_equal(
        mask, [[True, True, False], [True, True, True]]
    )


def test_rnn_captioning_gradients_match_numerical_gradients() -> None:
    generator = np.random.default_rng(73)
    features = generator.normal(size=(2, 3))
    captions = np.array([[1, 3, 2, 0], [1, 4, 5, 2]])
    parameters = _parameters(generator)

    loss, analytical = rnn_captioning_loss(
        features, captions, parameters, pad_id=0
    )

    assert np.isfinite(loss)
    assert analytical.keys() == parameters.keys()
    for name, value in parameters.items():
        numerical = eval_numerical_gradient(
            lambda candidate: rnn_captioning_loss(
                features,
                captions,
                {**parameters, name: candidate},
                pad_id=0,
            )[0],
            value,
        )
        np.testing.assert_allclose(
            analytical[name], numerical, rtol=2e-5, atol=1e-7
        )


def test_sampling_forbids_start_and_padding_tokens() -> None:
    generator = np.random.default_rng(79)
    parameters = _parameters(generator)
    parameters["output_weights"] = np.zeros_like(parameters["output_weights"])
    parameters["output_bias"] = np.array([10.0, 9.0, 0.0, 8.0, 1.0, 2.0])
    features = generator.normal(size=(2, 3))

    sampled = sample_rnn_captions(
        features,
        parameters,
        start_id=1,
        end_id=2,
        pad_id=0,
        max_length=4,
    )

    # IDs 0 and 1 have the largest raw biases but are forbidden at output.
    np.testing.assert_array_equal(sampled, np.full((2, 4), 3))


def test_sampling_keeps_end_token_after_each_caption_finishes() -> None:
    generator = np.random.default_rng(83)
    parameters = _parameters(generator)
    parameters["output_weights"] = np.zeros_like(parameters["output_weights"])
    parameters["output_bias"] = np.array([0.0, 0.0, 5.0, 1.0, 1.0, 1.0])

    sampled = sample_rnn_captions(
        generator.normal(size=(3, 3)),
        parameters,
        start_id=1,
        end_id=2,
        pad_id=0,
        max_length=5,
    )

    np.testing.assert_array_equal(sampled, np.full((3, 5), 2))


def test_captioning_rejects_mismatched_batch_sizes() -> None:
    generator = np.random.default_rng(89)
    with pytest.raises(ValueError, match="matching captions"):
        rnn_captioning_loss(
            generator.normal(size=(3, 3)),
            np.array([[1, 3, 2], [1, 4, 2]]),
            _parameters(generator),
            pad_id=0,
        )


@pytest.mark.parametrize("invalid_length", [0, -1, 1.5, True])
def test_sampling_rejects_invalid_max_length(invalid_length) -> None:
    generator = np.random.default_rng(97)
    with pytest.raises((TypeError, ValueError)):
        sample_rnn_captions(
            generator.normal(size=(2, 3)),
            _parameters(generator),
            start_id=1,
            end_id=2,
            pad_id=0,
            max_length=invalid_length,
        )
