import numpy as np
import pytest

from cs231n_practice.gradient_check import eval_numerical_gradient_array
from cs231n_practice.residual import residual_backward, residual_forward


def test_residual_forward_adds_identity_shortcut() -> None:
    x = np.array([[1.0, -2.0], [0.5, 3.0]])
    weights1 = np.array([[1.0, -1.0, 0.5], [0.5, 2.0, -1.0]])
    bias1 = np.array([0.0, 1.0, 0.5])
    weights2 = np.array([[1.0, 0.0], [-0.5, 2.0], [1.5, -1.0]])
    bias2 = np.array([0.25, -0.5])

    output, _ = residual_forward(x, weights1, bias1, weights2, bias2)

    hidden = np.maximum(0.0, x @ weights1 + bias1)
    expected = x + hidden @ weights2 + bias2
    np.testing.assert_allclose(output, expected)
    assert output.shape == x.shape


def test_residual_gradients_match_numerical_gradients() -> None:
    generator = np.random.default_rng(17)
    x = generator.normal(size=(3, 4))
    weights1 = generator.normal(scale=0.2, size=(4, 6))
    bias1 = generator.normal(loc=0.4, scale=0.1, size=6)
    weights2 = generator.normal(scale=0.2, size=(6, 4))
    bias2 = generator.normal(scale=0.1, size=4)
    dout = generator.normal(size=x.shape)

    _, cache = residual_forward(x, weights1, bias1, weights2, bias2)
    analytical = residual_backward(dout, cache)
    arguments = [x, weights1, bias1, weights2, bias2]

    for index, (value, gradient) in enumerate(zip(arguments, analytical)):
        def forward(candidate: np.ndarray) -> np.ndarray:
            current = arguments.copy()
            current[index] = candidate
            return residual_forward(*current)[0]

        numerical = eval_numerical_gradient_array(forward, value, dout)
        np.testing.assert_allclose(
            gradient,
            numerical,
            rtol=1e-7,
            atol=1e-8,
        )


def test_zero_residual_branch_is_identity_forward_and_backward() -> None:
    generator = np.random.default_rng(23)
    x = generator.normal(size=(3, 4))
    weights1 = generator.normal(size=(4, 5))
    bias1 = generator.normal(size=5)
    weights2 = np.zeros((5, 4))
    bias2 = np.zeros(4)
    dout = generator.normal(size=x.shape)

    output, cache = residual_forward(x, weights1, bias1, weights2, bias2)
    dx, _, _, _, _ = residual_backward(dout, cache)

    np.testing.assert_allclose(output, x)
    np.testing.assert_allclose(dx, dout)


def test_residual_forward_rejects_incompatible_shortcut_shape() -> None:
    x = np.ones((2, 3))
    weights1 = np.ones((3, 4))
    bias1 = np.zeros(4)
    weights2 = np.ones((4, 5))
    bias2 = np.zeros(5)

    with pytest.raises(ValueError, match="same shape"):
        residual_forward(x, weights1, bias1, weights2, bias2)

