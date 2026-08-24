"""A small convolutional image classifier implemented with NumPy."""

import numpy as np

from cs231n_practice.cnn_layers import (
    conv_backward_naive,
    conv_forward_naive,
    convolution_output_shape,
    max_pool_backward_naive,
    max_pool_forward_naive,
    pooling_output_shape,
)
from cs231n_practice.layers import (
    affine_backward,
    affine_forward,
    affine_relu_backward,
    affine_relu_forward,
    relu_backward,
    relu_forward,
    softmax_loss,
)
from cs231n_practice.training import sample_minibatch


def _positive_integer(value: int, *, name: str) -> int:
    """Return a positive integer while rejecting Boolean values."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _nonnegative_float(value: float, *, name: str) -> float:
    """Return a finite, nonnegative floating-point value."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.number)
    ):
        raise TypeError(f"{name} must be numeric")
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


class SmallConvNet:
    """A conv-ReLU-pool-affine-ReLU-affine classifier.

    The convolution uses stride 1 and same-size padding. Max pooling uses a
    ``2x2`` window with stride 2. The naive convolution and pooling layers make
    this model suitable for small educational experiments rather than
    large-scale training.
    """

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        num_filters: int,
        filter_size: int,
        hidden_dim: int,
        num_classes: int,
        *,
        seed: int | None = None,
    ) -> None:
        if not isinstance(input_shape, tuple) or len(input_shape) != 3:
            raise TypeError("input_shape must be a (channels, height, width) tuple")
        input_channels, input_height, input_width = (
            _positive_integer(value, name=f"input_shape[{index}]")
            for index, value in enumerate(input_shape)
        )
        num_filters = _positive_integer(num_filters, name="num_filters")
        filter_size = _positive_integer(filter_size, name="filter_size")
        hidden_dim = _positive_integer(hidden_dim, name="hidden_dim")
        num_classes = _positive_integer(num_classes, name="num_classes")
        if filter_size % 2 == 0:
            raise ValueError("filter_size must be odd for same-size padding")

        self.input_shape = (input_channels, input_height, input_width)
        self.padding = (filter_size - 1) // 2
        conv_height, conv_width = convolution_output_shape(
            input_height,
            input_width,
            filter_size,
            filter_size,
            stride=1,
            padding=self.padding,
        )
        pooled_height, pooled_width = pooling_output_shape(
            conv_height,
            conv_width,
            2,
            2,
            stride=2,
        )
        flattened_dim = num_filters * pooled_height * pooled_width

        generator = np.random.default_rng(seed)
        conv_scale = np.sqrt(
            2.0 / (input_channels * filter_size * filter_size)
        )
        hidden_scale = np.sqrt(2.0 / flattened_dim)
        output_scale = np.sqrt(2.0 / hidden_dim)
        self.parameters = {
            "W1": generator.normal(
                scale=conv_scale,
                size=(num_filters, input_channels, filter_size, filter_size),
            ),
            "b1": np.zeros(num_filters),
            "W2": generator.normal(
                scale=hidden_scale,
                size=(flattened_dim, hidden_dim),
            ),
            "b2": np.zeros(hidden_dim),
            "W3": generator.normal(
                scale=output_scale,
                size=(hidden_dim, num_classes),
            ),
            "b3": np.zeros(num_classes),
        }

    def loss(
        self,
        features: np.ndarray,
        labels: np.ndarray | None = None,
        *,
        regularization_strength: float = 0.0,
    ) -> np.ndarray | tuple[float, dict[str, np.ndarray]]:
        """Return class scores, or loss and gradients when labels are given."""
        features = np.asarray(features)
        if features.ndim != 4 or features.shape[1:] != self.input_shape:
            raise ValueError(
                f"features must have shape (N, {self.input_shape[0]}, "
                f"{self.input_shape[1]}, {self.input_shape[2]})"
            )
        regularization_strength = _nonnegative_float(
            regularization_strength,
            name="regularization_strength",
        )

        conv, conv_cache = conv_forward_naive(
            features,
            self.parameters["W1"],
            self.parameters["b1"],
            stride=1,
            padding=self.padding,
        )
        activated, relu_cache = relu_forward(conv)
        pooled, pool_cache = max_pool_forward_naive(
            activated,
            pool_height=2,
            pool_width=2,
            stride=2,
        )
        flat = pooled.reshape(features.shape[0], -1)
        hidden, hidden_cache = affine_relu_forward(
            flat, self.parameters["W2"], self.parameters["b2"]
        )
        scores, scores_cache = affine_forward(
            hidden, self.parameters["W3"], self.parameters["b3"]
        )
        if labels is None:
            return scores

        data_loss, dscores = softmax_loss(scores, labels)
        weight_penalty = sum(
            np.sum(self.parameters[name] ** 2) for name in ("W1", "W2", "W3")
        )
        loss = data_loss + regularization_strength * weight_penalty

        dhidden, dW3, db3 = affine_backward(dscores, scores_cache)
        dflat, dW2, db2 = affine_relu_backward(dhidden, hidden_cache)
        dpooled = dflat.reshape(pooled.shape)
        dactivated = max_pool_backward_naive(dpooled, pool_cache)
        dconv = relu_backward(dactivated, relu_cache)
        _, dW1, db1 = conv_backward_naive(dconv, conv_cache)

        dW1 += 2 * regularization_strength * self.parameters["W1"]
        dW2 += 2 * regularization_strength * self.parameters["W2"]
        dW3 += 2 * regularization_strength * self.parameters["W3"]
        gradients = {
            "W1": dW1,
            "b1": db1,
            "W2": dW2,
            "b2": db2,
            "W3": dW3,
            "b3": db3,
        }
        return float(loss), gradients

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return the highest-scoring class for each input image."""
        return np.argmax(self.loss(features), axis=1)

    def train(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        *,
        learning_rate: float,
        regularization_strength: float = 0.0,
        batch_size: int = 32,
        num_iterations: int = 100,
        seed: int | None = None,
    ) -> np.ndarray:
        """Update parameters with vanilla minibatch SGD and return losses."""
        if isinstance(learning_rate, (bool, np.bool_)) or not isinstance(
            learning_rate, (int, float, np.number)
        ):
            raise TypeError("learning_rate must be numeric")
        learning_rate = float(learning_rate)
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        num_iterations = _positive_integer(
            num_iterations, name="num_iterations"
        )

        generator = np.random.default_rng(seed)
        history = np.empty(num_iterations, dtype=np.float64)
        for iteration in range(num_iterations):
            batch_features, batch_labels = sample_minibatch(
                features, labels, batch_size, generator
            )
            loss, gradients = self.loss(
                batch_features,
                batch_labels,
                regularization_strength=regularization_strength,
            )
            for name in self.parameters:
                self.parameters[name] -= learning_rate * gradients[name]
            history[iteration] = loss
        return history
