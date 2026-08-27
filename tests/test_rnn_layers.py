import numpy as np
import pytest

from cs231n_practice.rnn_layers import (
    rnn_forward,
    rnn_forward_masked,
    rnn_step_forward,
)


def test_rnn_step_forward_matches_direct_calculation() -> None:
    x_t = np.array([[1.0, -2.0], [0.5, 1.0]])
    h_previous = np.array([[0.2, -0.1], [0.0, 0.3]])
    weights_x = np.array([[0.5, -0.25], [1.0, 0.75]])
    weights_h = np.array([[0.4, 0.2], [-0.5, 0.3]])
    bias = np.array([0.1, -0.2])

    h_next, cache = rnn_step_forward(
        x_t, h_previous, weights_x, weights_h, bias
    )

    expected = np.tanh(
        x_t @ weights_x + h_previous @ weights_h + bias
    )
    np.testing.assert_allclose(h_next, expected)
    assert h_next.shape == h_previous.shape
    assert len(cache) == 5


def test_rnn_forward_matches_manual_unrolling_and_caches_every_step() -> None:
    generator = np.random.default_rng(7)
    x = generator.normal(size=(2, 4, 3))
    h0 = generator.normal(size=(2, 5))
    weights_x = generator.normal(size=(3, 5))
    weights_h = generator.normal(size=(5, 5))
    bias = generator.normal(size=5)

    hidden_states, caches = rnn_forward(
        x, h0, weights_x, weights_h, bias
    )

    expected_steps = []
    h_previous = h0
    for time_index in range(x.shape[1]):
        h_previous = np.tanh(
            x[:, time_index, :] @ weights_x
            + h_previous @ weights_h
            + bias
        )
        expected_steps.append(h_previous)
    expected = np.stack(expected_steps, axis=1)

    np.testing.assert_allclose(hidden_states, expected)
    assert hidden_states.shape == (2, 4, 5)
    assert len(caches) == x.shape[1]


def test_rnn_forward_does_not_mutate_inputs_or_parameters() -> None:
    generator = np.random.default_rng(11)
    arrays = [
        generator.normal(size=(2, 3, 4)),
        generator.normal(size=(2, 5)),
        generator.normal(size=(4, 5)),
        generator.normal(size=(5, 5)),
        generator.normal(size=5),
    ]
    originals = [array.copy() for array in arrays]

    rnn_forward(*arrays)

    for array, original in zip(arrays, originals):
        np.testing.assert_array_equal(array, original)


def test_rnn_forward_promotes_integer_inputs_to_floating_point() -> None:
    x = np.ones((2, 3, 2), dtype=np.int64)
    h0 = np.zeros((2, 4), dtype=np.int64)
    weights_x = np.ones((2, 4), dtype=np.int64)
    weights_h = np.eye(4, dtype=np.int64)
    bias = np.zeros(4, dtype=np.int64)

    hidden_states, _ = rnn_forward(x, h0, weights_x, weights_h, bias)

    assert np.issubdtype(hidden_states.dtype, np.floating)
    assert np.all(np.isfinite(hidden_states))


def test_rnn_forward_masked_preserves_state_after_each_sequence_ends() -> None:
    generator = np.random.default_rng(19)
    x = generator.normal(size=(3, 4, 2))
    h0 = generator.normal(size=(3, 3))
    weights_x = generator.normal(size=(2, 3))
    weights_h = generator.normal(size=(3, 3))
    bias = generator.normal(size=3)
    lengths = np.array([0, 2, 4])

    hidden_states, caches = rnn_forward_masked(
        x, h0, lengths, weights_x, weights_h, bias
    )

    np.testing.assert_allclose(hidden_states[0], np.broadcast_to(h0[0], (4, 3)))
    np.testing.assert_allclose(hidden_states[1, 2], hidden_states[1, 1])
    np.testing.assert_allclose(hidden_states[1, 3], hidden_states[1, 1])
    unmasked, _ = rnn_forward(
        x[2:3], h0[2:3], weights_x, weights_h, bias
    )
    np.testing.assert_allclose(hidden_states[2:3], unmasked)
    assert len(caches) == x.shape[1]
    assert all(mask.shape == (3, 1) for _, mask in caches)


@pytest.mark.parametrize(
    ("lengths", "error_type", "message"),
    [
        (np.array([[2], [3]]), ValueError, "shape"),
        (np.array([2.0, 3.0]), TypeError, "integers"),
        (np.array([-1, 3]), ValueError, "between"),
        (np.array([2, 5]), ValueError, "between"),
    ],
)
def test_rnn_forward_masked_rejects_invalid_lengths(
    lengths: np.ndarray,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        rnn_forward_masked(
            np.ones((2, 4, 3)),
            np.ones((2, 5)),
            lengths,
            np.ones((3, 5)),
            np.ones((5, 5)),
            np.ones(5),
        )


@pytest.mark.parametrize(
    ("x_t", "h_previous", "weights_x", "weights_h", "bias", "message"),
    [
        (
            np.ones((2, 3)),
            np.ones((4, 5)),
            np.ones((3, 5)),
            np.ones((5, 5)),
            np.ones(5),
            "batch sizes",
        ),
        (
            np.ones((2, 3)),
            np.ones((2, 5)),
            np.ones((4, 5)),
            np.ones((5, 5)),
            np.ones(5),
            "weights_x",
        ),
        (
            np.ones((2, 3)),
            np.ones((2, 5)),
            np.ones((3, 5)),
            np.ones((4, 5)),
            np.ones(5),
            "weights_h",
        ),
        (
            np.ones((2, 3)),
            np.ones((2, 5)),
            np.ones((3, 5)),
            np.ones((5, 5)),
            np.ones(4),
            "bias",
        ),
    ],
)
def test_rnn_step_forward_rejects_incompatible_shapes(
    x_t: np.ndarray,
    h_previous: np.ndarray,
    weights_x: np.ndarray,
    weights_h: np.ndarray,
    bias: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        rnn_step_forward(x_t, h_previous, weights_x, weights_h, bias)


def test_rnn_forward_rejects_invalid_sequence_shape() -> None:
    with pytest.raises(ValueError, match="x must have shape"):
        rnn_forward(
            np.ones((2, 3)),
            np.ones((2, 4)),
            np.ones((3, 4)),
            np.ones((4, 4)),
            np.ones(4),
        )


def test_rnn_layers_reject_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        rnn_step_forward(
            np.array([[np.nan, 1.0]]),
            np.ones((1, 2)),
            np.ones((2, 2)),
            np.ones((2, 2)),
            np.ones(2),
        )
