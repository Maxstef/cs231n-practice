import numpy as np
import pytest

from cs231n_practice.gradient_check import (
    eval_numerical_gradient,
    eval_numerical_gradient_array,
    relative_error,
)


def test_scalar_output_gradient_supports_multidimensional_inputs() -> None:
    x = np.array([[1.5, -2.0], [0.5, 3.0]])

    numerical = eval_numerical_gradient(lambda value: np.sum(value**3), x)

    np.testing.assert_allclose(numerical, 3 * x**2, rtol=1e-9, atol=1e-9)
    assert numerical.shape == x.shape


def test_scalar_output_gradient_handles_interacting_coordinates() -> None:
    x = np.array([2.0, -3.0])

    numerical = eval_numerical_gradient(lambda value: value[0] * value[1], x)

    np.testing.assert_allclose(numerical, [x[1], x[0]], rtol=1e-10, atol=1e-10)


def test_numerical_gradient_does_not_mutate_its_input() -> None:
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    before = x.copy()

    eval_numerical_gradient(lambda value: np.sum(value**2), x)

    np.testing.assert_array_equal(x, before)


def test_array_output_gradient_applies_upstream_gradient() -> None:
    x = np.array([1.5, -2.0, 0.5])
    upstream = np.array([2.0, -3.0, 4.0])

    numerical = eval_numerical_gradient_array(
        lambda value: value**2,
        x,
        upstream,
    )

    expected = 2 * x * upstream
    np.testing.assert_allclose(numerical, expected, rtol=1e-9, atol=1e-9)


def test_relative_error_is_zero_for_equal_gradients() -> None:
    gradient = np.array([0.0, -2.0, 3.0])

    assert relative_error(gradient, gradient.copy()) == 0.0


def test_relative_error_is_scale_aware() -> None:
    analytical = np.array([1000.0, 1e-10])
    numerical = np.array([1001.0, 0.0])

    assert relative_error(analytical, numerical) == pytest.approx(1e-2)


def test_scalar_gradient_rejects_array_function_output() -> None:
    with pytest.raises(ValueError, match="return a scalar"):
        eval_numerical_gradient(lambda value: value**2, np.array([1.0, 2.0]))


def test_array_gradient_rejects_upstream_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="same shape"):
        eval_numerical_gradient_array(
            lambda value: value**2,
            np.array([1.0, 2.0]),
            np.ones((1, 2)),
        )


@pytest.mark.parametrize("step", [0.0, -1e-5, np.inf, np.nan])
def test_numerical_gradient_rejects_invalid_step(step: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        eval_numerical_gradient(lambda value: np.sum(value), np.ones(2), step=step)
