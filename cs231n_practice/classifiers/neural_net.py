"""A two-layer fully connected classifier implemented with NumPy."""

import numpy as np

from cs231n_practice.layers import (
    affine_backward,
    affine_forward,
    affine_relu_backward,
    affine_relu_forward,
    softmax_loss,
)
from cs231n_practice.training import sample_minibatch


class TwoLayerNet:
    """An affine-ReLU-affine classifier trained with vanilla minibatch SGD."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        *,
        weight_scale: float = 1e-3,
        seed: int | None = None,
    ) -> None:
        for name, value in (
            ("input_dim", input_dim),
            ("hidden_dim", hidden_dim),
            ("num_classes", num_classes),
        ):
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)
            ):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if isinstance(weight_scale, (bool, np.bool_)) or not isinstance(
            weight_scale, (int, float, np.number)
        ):
            raise TypeError("weight_scale must be numeric")
        weight_scale = float(weight_scale)
        if not np.isfinite(weight_scale) or weight_scale <= 0:
            raise ValueError("weight_scale must be finite and positive")

        generator = np.random.default_rng(seed)
        self.parameters = {
            "W1": generator.normal(
                scale=weight_scale, size=(input_dim, hidden_dim)
            ),
            "b1": np.zeros(hidden_dim),
            "W2": generator.normal(
                scale=weight_scale, size=(hidden_dim, num_classes)
            ),
            "b2": np.zeros(num_classes),
        }

    def loss(
        self,
        features: np.ndarray,
        labels: np.ndarray | None = None,
        *,
        regularization_strength: float = 0.0,
    ) -> np.ndarray | tuple[float, dict[str, np.ndarray]]:
        """Return scores for inference, or loss and gradients for training."""
        if isinstance(regularization_strength, (bool, np.bool_)) or not isinstance(
            regularization_strength, (int, float, np.number)
        ):
            raise TypeError("regularization_strength must be numeric")
        regularization_strength = float(regularization_strength)
        if not np.isfinite(regularization_strength) or regularization_strength < 0:
            raise ValueError("regularization_strength must be finite and nonnegative")

        hidden, hidden_cache = affine_relu_forward(
            features, self.parameters["W1"], self.parameters["b1"]
        )
        scores, output_cache = affine_forward(
            hidden, self.parameters["W2"], self.parameters["b2"]
        )
        if labels is None:
            return scores

        data_loss, dscores = softmax_loss(scores, labels)
        weight_penalty = np.sum(self.parameters["W1"] ** 2) + np.sum(
            self.parameters["W2"] ** 2
        )
        loss = data_loss + regularization_strength * weight_penalty

        dhidden, dW2, db2 = affine_backward(dscores, output_cache)
        _, dW1, db1 = affine_relu_backward(dhidden, hidden_cache)
        dW1 += 2 * regularization_strength * self.parameters["W1"]
        dW2 += 2 * regularization_strength * self.parameters["W2"]
        gradients = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
        return float(loss), gradients

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return the highest-scoring class for each example."""
        scores = self.loss(features)
        return np.argmax(scores, axis=1)

    def train(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        *,
        learning_rate: float,
        regularization_strength: float = 0.0,
        batch_size: int = 128,
        num_iterations: int = 100,
        seed: int | None = None,
    ) -> np.ndarray:
        """Update this model with minibatch SGD and return its loss history."""
        if isinstance(learning_rate, (bool, np.bool_)) or not isinstance(
            learning_rate, (int, float, np.number)
        ):
            raise TypeError("learning_rate must be numeric")
        learning_rate = float(learning_rate)
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if isinstance(num_iterations, (bool, np.bool_)) or not isinstance(
            num_iterations, (int, np.integer)
        ):
            raise TypeError("num_iterations must be an integer")
        if num_iterations <= 0:
            raise ValueError("num_iterations must be positive")

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
