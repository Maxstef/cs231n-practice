import numpy as np
import pytest

from cs231n_practice.gradient_check import eval_numerical_gradient_array
from cs231n_practice.transformer import (
    feed_forward_backward,
    feed_forward_forward,
    residual_add,
    transformer_encoder_block_backward,
    transformer_encoder_block_forward,
)


def _parameters(generator: np.random.Generator) -> dict[str, np.ndarray]:
    model_dim, feed_forward_dim = 4, 5
    return {
        "gamma1": generator.normal(size=model_dim),
        "beta1": generator.normal(size=model_dim),
        "W_query": generator.normal(scale=0.2, size=(model_dim, model_dim)),
        "W_key": generator.normal(scale=0.2, size=(model_dim, model_dim)),
        "W_value": generator.normal(scale=0.2, size=(model_dim, model_dim)),
        "W_output": generator.normal(scale=0.2, size=(model_dim, model_dim)),
        "b_query": generator.normal(scale=0.1, size=model_dim),
        "b_key": generator.normal(scale=0.1, size=model_dim),
        "b_value": generator.normal(scale=0.1, size=model_dim),
        "b_output": generator.normal(scale=0.1, size=model_dim),
        "gamma2": generator.normal(size=model_dim),
        "beta2": generator.normal(size=model_dim),
        "W1": generator.normal(scale=0.2, size=(model_dim, feed_forward_dim)),
        "b1": generator.normal(scale=0.1, size=feed_forward_dim),
        "W2": generator.normal(scale=0.2, size=(feed_forward_dim, model_dim)),
        "b2": generator.normal(scale=0.1, size=model_dim),
    }


def test_feed_forward_backward_matches_numerical_gradients() -> None:
    generator = np.random.default_rng(131)
    arguments = [
        generator.normal(size=(2, 3, 4)),
        generator.normal(size=(4, 5)),
        generator.normal(size=5),
        generator.normal(size=(5, 4)),
        generator.normal(size=4),
    ]
    output, cache = feed_forward_forward(*arguments)
    dout = generator.normal(size=output.shape)
    analytical = feed_forward_backward(dout, cache)

    for index, (value, gradient) in enumerate(zip(arguments, analytical)):
        def forward(candidate: np.ndarray) -> np.ndarray:
            current = arguments.copy()
            current[index] = candidate
            return feed_forward_forward(*current)[0]

        numerical = eval_numerical_gradient_array(forward, value, dout)
        np.testing.assert_allclose(gradient, numerical, rtol=1e-7, atol=1e-8)


def test_encoder_forward_preserves_shape_and_applies_mask() -> None:
    generator = np.random.default_rng(137)
    x = generator.normal(size=(2, 3, 4))
    parameters = _parameters(generator)
    mask = np.array([
        [[[True, True, True]]],
        [[[True, True, False]]],
    ])

    output, weights, _ = transformer_encoder_block_forward(
        x, parameters, num_heads=2, mask=mask
    )

    assert output.shape == x.shape
    assert weights.shape == (2, 2, 3, 3)
    np.testing.assert_allclose(weights.sum(axis=-1), 1.0)
    np.testing.assert_array_equal(weights[1, :, :, 2], 0.0)


def test_encoder_backward_matches_all_numerical_gradients() -> None:
    generator = np.random.default_rng(139)
    x = generator.normal(size=(1, 2, 4))
    parameters = _parameters(generator)
    mask = np.array([[[[True, False], [True, True]]]])
    output, _, cache = transformer_encoder_block_forward(
        x, parameters, num_heads=2, mask=mask
    )
    dout = generator.normal(size=output.shape)
    dx, gradients = transformer_encoder_block_backward(dout, cache)

    numerical_dx = eval_numerical_gradient_array(
        lambda candidate: transformer_encoder_block_forward(
            candidate, parameters, num_heads=2, mask=mask
        )[0],
        x,
        dout,
    )
    np.testing.assert_allclose(dx, numerical_dx, rtol=3e-6, atol=1e-8)

    for name, value in parameters.items():
        def forward(candidate: np.ndarray) -> np.ndarray:
            current = parameters.copy()
            current[name] = candidate
            return transformer_encoder_block_forward(
                x, current, num_heads=2, mask=mask
            )[0]

        numerical = eval_numerical_gradient_array(forward, value, dout)
        np.testing.assert_allclose(
            gradients[name], numerical, rtol=3e-6, atol=1e-8
        )


def test_residual_add_rejects_different_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        residual_add(np.zeros((2, 3, 4)), np.zeros((2, 3, 5)))


def test_encoder_rejects_missing_parameters() -> None:
    with pytest.raises(ValueError, match="missing Transformer parameters"):
        transformer_encoder_block_forward(
            np.zeros((1, 2, 4)), {}, num_heads=2
        )
