import numpy as np
import pytest

from cs231n_practice.gradient_check import eval_numerical_gradient_array
from cs231n_practice.normalization import (
    batchnorm_backward,
    batchnorm_forward,
    dropout_backward,
    dropout_forward,
)


def test_batchnorm_training_normalizes_each_feature() -> None:
    x = np.array([[1.0, 10.0], [3.0, 14.0], [5.0, 18.0]])
    state = {"mode": "train", "momentum": 0.9}

    output, cache = batchnorm_forward(x, np.ones(2), np.zeros(2), state)

    np.testing.assert_allclose(output.mean(axis=0), 0.0, atol=1e-7)
    np.testing.assert_allclose(output.var(axis=0), 1.0, atol=1e-5)
    assert cache is not None
    assert state["running_mean"].shape == (2,)
    assert state["running_variance"].shape == (2,)


def test_batchnorm_test_uses_running_statistics_without_mutating_them() -> None:
    state = {
        "mode": "test",
        "epsilon": 1e-5,
        "running_mean": np.array([1.0, 4.0]),
        "running_variance": np.array([4.0, 9.0]),
    }
    original_mean = state["running_mean"].copy()
    original_variance = state["running_variance"].copy()

    output, cache = batchnorm_forward(
        np.array([[3.0, 1.0]]),
        np.array([2.0, 3.0]),
        np.array([-1.0, 0.5]),
        state,
    )

    expected = np.array([[1.0, -2.5]])
    np.testing.assert_allclose(output, expected, atol=1e-5)
    np.testing.assert_array_equal(state["running_mean"], original_mean)
    np.testing.assert_array_equal(state["running_variance"], original_variance)
    assert cache is None


def test_batchnorm_backward_matches_numerical_gradients() -> None:
    generator = np.random.default_rng(42)
    x = generator.normal(size=(4, 3))
    gamma = generator.normal(size=3)
    beta = generator.normal(size=3)
    dout = generator.normal(size=(4, 3))
    _, cache = batchnorm_forward(x, gamma, beta, {"mode": "train"})
    assert cache is not None

    dx, dgamma, dbeta = batchnorm_backward(dout, cache)
    numerical_dx = eval_numerical_gradient_array(
        lambda candidate: batchnorm_forward(
            candidate, gamma, beta, {"mode": "train"}
        )[0],
        x,
        dout,
    )
    numerical_dgamma = eval_numerical_gradient_array(
        lambda candidate: batchnorm_forward(
            x, candidate, beta, {"mode": "train"}
        )[0],
        gamma,
        dout,
    )
    numerical_dbeta = eval_numerical_gradient_array(
        lambda candidate: batchnorm_forward(
            x, gamma, candidate, {"mode": "train"}
        )[0],
        beta,
        dout,
    )

    np.testing.assert_allclose(dx, numerical_dx, rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(dgamma, numerical_dgamma, rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(dbeta, numerical_dbeta, rtol=1e-8, atol=1e-9)


def test_dropout_preserves_activations_in_expectation() -> None:
    x = np.linspace(0.5, 2.0, 12).reshape(3, 4)
    generator = np.random.default_rng(42)
    samples = np.stack(
        [
            dropout_forward(
                x,
                keep_probability=0.6,
                mode="train",
                generator=generator,
            )[0]
            for _ in range(10_000)
        ]
    )

    np.testing.assert_allclose(samples.mean(axis=0), x, rtol=0.03, atol=0.03)


def test_dropout_backward_reuses_training_mask() -> None:
    x = np.ones((3, 4))
    output, cache = dropout_forward(
        x,
        keep_probability=0.5,
        mode="train",
        generator=np.random.default_rng(7),
    )

    dx = dropout_backward(np.ones_like(x), cache)

    np.testing.assert_array_equal(dx, output)
    assert set(np.unique(output)) <= {0.0, 2.0}


def test_dropout_test_mode_is_identity() -> None:
    x = np.arange(6.0).reshape(2, 3)
    output, cache = dropout_forward(
        x,
        keep_probability=0.5,
        mode="test",
        generator=np.random.default_rng(1),
    )

    np.testing.assert_array_equal(output, x)
    np.testing.assert_array_equal(dropout_backward(np.ones_like(x), cache), 1.0)


def test_normalization_and_dropout_reject_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="one value per feature"):
        batchnorm_forward(
            np.ones((2, 3)), np.ones(2), np.zeros(3), {"mode": "train"}
        )
    with pytest.raises(ValueError, match="mode"):
        batchnorm_forward(
            np.ones((2, 3)), np.ones(3), np.zeros(3), {"mode": "invalid"}
        )
    with pytest.raises(ValueError, match="keep_probability"):
        dropout_forward(
            np.ones(2),
            keep_probability=0.0,
            mode="train",
            generator=np.random.default_rng(1),
        )
    with pytest.raises(TypeError, match="keep_probability"):
        dropout_forward(
            np.ones(2),
            keep_probability=True,
            mode="train",
            generator=np.random.default_rng(1),
        )
