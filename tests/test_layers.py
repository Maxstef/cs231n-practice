import numpy as np
import pytest

from cs231n_practice.gradient_check import eval_numerical_gradient_array
from cs231n_practice.layers import (
    affine_backward,
    affine_forward,
    affine_relu_backward,
    affine_relu_forward,
    relu_backward,
    relu_forward,
)


def test_affine_forward_computes_batched_outputs() -> None:
    x = np.array([[1.0, 2.0], [-1.0, 3.0]])
    weights = np.array([[2.0, -1.0, 0.5], [0.0, 4.0, -2.0]])
    bias = np.array([0.1, -0.2, 0.3])

    output, _ = affine_forward(x, weights, bias)

    np.testing.assert_allclose(
        output,
        [[2.1, 6.8, -3.2], [-1.9, 12.8, -6.2]],
    )


def test_affine_backward_matches_hand_computed_gradients() -> None:
    x = np.array([[1.0, 2.0], [-1.0, 3.0]])
    weights = np.array([[2.0, -1.0, 0.5], [0.0, 4.0, -2.0]])
    bias = np.array([0.1, -0.2, 0.3])
    dout = np.array([[1.0, -2.0, 0.5], [-1.0, 3.0, 2.0]])
    _, cache = affine_forward(x, weights, bias)

    dx, dweights, dbias = affine_backward(dout, cache)

    np.testing.assert_allclose(dx, [[4.25, -9.0], [-4.0, 8.0]])
    np.testing.assert_allclose(
        dweights,
        [[2.0, -5.0, -1.5], [-1.0, 5.0, 7.0]],
    )
    np.testing.assert_allclose(dbias, [0.0, 1.0, 2.5])
    assert dx.shape == x.shape
    assert dweights.shape == weights.shape
    assert dbias.shape == bias.shape


def test_affine_gradients_match_numerical_gradients() -> None:
    generator = np.random.default_rng(7)
    x = generator.normal(size=(3, 4))
    weights = generator.normal(size=(4, 5))
    bias = generator.normal(size=5)
    dout = generator.normal(size=(3, 5))
    _, cache = affine_forward(x, weights, bias)
    dx, dweights, dbias = affine_backward(dout, cache)

    numerical_dx = eval_numerical_gradient_array(
        lambda candidate: affine_forward(candidate, weights, bias)[0],
        x,
        dout,
    )
    numerical_dweights = eval_numerical_gradient_array(
        lambda candidate: affine_forward(x, candidate, bias)[0],
        weights,
        dout,
    )
    numerical_dbias = eval_numerical_gradient_array(
        lambda candidate: affine_forward(x, weights, candidate)[0],
        bias,
        dout,
    )

    np.testing.assert_allclose(dx, numerical_dx, rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(
        dweights,
        numerical_dweights,
        rtol=1e-8,
        atol=1e-9,
    )
    np.testing.assert_allclose(dbias, numerical_dbias, rtol=1e-8, atol=1e-9)


def test_relu_forward_and_backward_preserve_shape_and_upstream() -> None:
    x = np.array([[-2.0, 0.0], [3.0, 1.5]])
    dout = np.array([[10.0, 20.0], [30.0, -4.0]])
    dout_before = dout.copy()

    output, cache = relu_forward(x)
    dx = relu_backward(dout, cache)

    np.testing.assert_allclose(output, [[0.0, 0.0], [3.0, 1.5]])
    np.testing.assert_allclose(dx, [[0.0, 0.0], [30.0, -4.0]])
    np.testing.assert_array_equal(dout, dout_before)
    assert output.shape == x.shape
    assert dx.shape == x.shape


def test_relu_backward_matches_numerical_gradient_away_from_zero() -> None:
    x = np.array([[-1.5, 0.4, 2.0], [3.0, -0.7, -2.0]])
    dout = np.array([[2.0, -3.0, 0.5], [-1.0, 4.0, 2.5]])
    _, cache = relu_forward(x)

    dx = relu_backward(dout, cache)
    numerical_dx = eval_numerical_gradient_array(
        lambda candidate: relu_forward(candidate)[0],
        x,
        dout,
    )

    np.testing.assert_allclose(dx, numerical_dx, rtol=1e-8, atol=1e-9)


def test_affine_relu_gradients_match_numerical_gradients() -> None:
    x = np.array([[0.5, -1.0], [1.5, 0.25]])
    weights = np.array([[0.8, -0.4, 1.2], [-0.6, 0.9, 0.5]])
    bias = np.array([0.2, -0.1, 0.3])
    dout = np.array([[1.0, -2.0, 0.5], [-0.5, 1.5, 2.0]])
    _, cache = affine_relu_forward(x, weights, bias)

    dx, dweights, dbias = affine_relu_backward(dout, cache)
    numerical_dx = eval_numerical_gradient_array(
        lambda candidate: affine_relu_forward(candidate, weights, bias)[0],
        x,
        dout,
    )
    numerical_dweights = eval_numerical_gradient_array(
        lambda candidate: affine_relu_forward(x, candidate, bias)[0],
        weights,
        dout,
    )
    numerical_dbias = eval_numerical_gradient_array(
        lambda candidate: affine_relu_forward(x, weights, candidate)[0],
        bias,
        dout,
    )

    np.testing.assert_allclose(dx, numerical_dx, rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(
        dweights,
        numerical_dweights,
        rtol=1e-8,
        atol=1e-9,
    )
    np.testing.assert_allclose(dbias, numerical_dbias, rtol=1e-8, atol=1e-9)


def test_affine_forward_rejects_incompatible_dimensions() -> None:
    with pytest.raises(ValueError, match="incompatible feature"):
        affine_forward(
            np.ones((2, 3)),
            np.ones((2, 4)),
            np.zeros(4),
        )


def test_affine_backward_rejects_incompatible_upstream_shape() -> None:
    _, cache = affine_forward(
        np.ones((2, 3)),
        np.ones((3, 4)),
        np.zeros(4),
    )

    with pytest.raises(ValueError, match="dout shape"):
        affine_backward(np.ones((2, 3)), cache)


def test_relu_backward_rejects_shape_mismatch() -> None:
    _, cache = relu_forward(np.ones((2, 3)))

    with pytest.raises(ValueError, match="same shape"):
        relu_backward(np.ones((3, 2)), cache)
