"""Linear image-classifier building blocks implemented with NumPy.

The module follows the class-major convention used throughout the notebooks:

``features`` has shape ``(num_examples, num_features)`` and ``weights`` has
shape ``(num_classes, num_features)``. Scores are therefore calculated as
``features @ weights.T + bias`` and have shape
``(num_examples, num_classes)``.

SVM and softmax use different objectives and score gradients, but the same
linear score function, parameter-gradient formulas, and SGD optimizer. Keeping
these concerns separate makes their shared structure explicit and testable.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

LossFunction = Callable[
    [np.ndarray, np.ndarray, np.ndarray, np.ndarray, float],
    tuple[float, np.ndarray, np.ndarray],
]


@dataclass(frozen=True)
class TrainingResult:
    """Parameters and minibatch losses produced by linear-classifier SGD.

    Attributes:
        weights: Trained class-major weights with shape
            ``(num_classes, num_features)``.
        bias: Trained class biases with shape ``(num_classes,)``.
        loss_history: One scalar minibatch loss per SGD iteration, stored in a
            one-dimensional array.
    """

    weights: np.ndarray
    bias: np.ndarray
    loss_history: np.ndarray


def _as_calculation_arrays(
    features: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate score inputs and convert them to a safe floating dtype."""
    features = np.asarray(features)
    weights = np.asarray(weights)
    bias = np.asarray(bias)

    if features.ndim != 2 or weights.ndim != 2:
        raise ValueError("Features and weights must be two-dimensional")
    if bias.ndim != 1:
        raise ValueError("Bias must be one-dimensional")
    if features.shape[1] != weights.shape[1]:
        raise ValueError("Features and weights must have equal feature dimensions")
    if weights.shape[0] != bias.shape[0]:
        raise ValueError("Weights and bias must describe the same number of classes")
    if weights.shape[0] == 0:
        raise ValueError("At least one class is required")
    if not all(
        np.issubdtype(array.dtype, np.number)
        for array in (features, weights, bias)
    ):
        raise TypeError("Features, weights, and bias must be numeric")
    if not all(np.all(np.isfinite(array)) for array in (features, weights, bias)):
        raise ValueError("Features, weights, and bias must contain finite values")

    calculation_dtype = np.result_type(
        features.dtype,
        weights.dtype,
        bias.dtype,
        np.float32,
    )
    return (
        features.astype(calculation_dtype, copy=False),
        weights.astype(calculation_dtype, copy=False),
        bias.astype(calculation_dtype, copy=False),
    )


def _validate_labels(labels: np.ndarray, num_examples: int, num_classes: int) -> np.ndarray:
    """Return validated integer labels for a nonempty batch."""
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("Labels must be one-dimensional")
    if num_examples == 0:
        raise ValueError("The example batch must not be empty")
    if labels.shape[0] != num_examples:
        raise ValueError("Each example must have one label")
    if not np.issubdtype(labels.dtype, np.integer):
        raise TypeError("Labels must be integers")
    if np.any((labels < 0) | (labels >= num_classes)):
        raise ValueError("Labels must be valid class indices")
    return labels


def _validate_regularization(regularization_strength: float) -> float:
    """Return a finite, nonnegative regularization strength."""
    if isinstance(regularization_strength, (bool, np.bool_)) or not isinstance(
        regularization_strength, (int, float, np.number)
    ):
        raise TypeError("Regularization strength must be numeric")
    regularization_strength = float(regularization_strength)
    if not np.isfinite(regularization_strength) or regularization_strength < 0:
        raise ValueError("Regularization strength must be finite and nonnegative")
    return regularization_strength


def linear_scores(
    features: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    """Calculate one class score per example and class.

    Args:
        features: Feature matrix with shape ``(N, D)``.
        weights: Class-major weight matrix with shape ``(C, D)``.
        bias: Bias vector with shape ``(C,)``.

    Returns:
        Score matrix ``features @ weights.T + bias`` with shape ``(N, C)``.
    """
    features, weights, bias = _as_calculation_arrays(features, weights, bias)
    return features @ weights.T + bias


def predict_linear(
    features: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    """Return the highest-scoring class index for every example."""
    scores = linear_scores(features, weights, bias)
    return np.argmax(scores, axis=1)


def classification_accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Return the fraction of predictions equal to their targets."""
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)
    if predictions.ndim != 1 or targets.ndim != 1:
        raise ValueError("Predictions and targets must be one-dimensional")
    if predictions.shape != targets.shape:
        raise ValueError("Predictions and targets must have the same shape")
    if predictions.size == 0:
        raise ValueError("Predictions and targets must not be empty")
    return float(np.mean(predictions == targets))


def svm_loss_and_gradient(
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    regularization_strength: float = 0.0,
    *,
    delta: float = 1.0,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return multiclass SVM loss and gradients for ``weights`` and ``bias``.

    The data loss is averaged over examples and summed over incorrect classes.
    Under the project convention, regularization adds
    ``regularization_strength * sum(weights**2)`` to the loss and
    ``2 * regularization_strength * weights`` to the weight gradient.
    """
    features, weights, bias = _as_calculation_arrays(features, weights, bias)
    labels = _validate_labels(labels, features.shape[0], weights.shape[0])
    regularization_strength = _validate_regularization(regularization_strength)
    if isinstance(delta, (bool, np.bool_)) or not isinstance(
        delta, (int, float, np.number)
    ):
        raise TypeError("delta must be numeric")
    delta = float(delta)
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError("delta must be finite and positive")

    scores = features @ weights.T + bias
    correct_scores = scores[np.arange(features.shape[0]), labels][:, None]
    margins = np.maximum(0.0, scores - correct_scores + delta)
    margins[np.arange(features.shape[0]), labels] = 0.0

    data_loss = margins.sum() / features.shape[0]
    regularization_loss = regularization_strength * np.sum(weights**2)

    # Every active incorrect margin contributes +1 to its score derivative.
    # The correct score receives -1 for each active margin in the same row.
    score_gradient = (margins > 0).astype(scores.dtype)
    score_gradient[np.arange(features.shape[0]), labels] = -score_gradient.sum(
        axis=1
    )
    score_gradient /= features.shape[0]

    weight_gradient = (
        score_gradient.T @ features + 2 * regularization_strength * weights
    )
    bias_gradient = score_gradient.sum(axis=0)
    return float(data_loss + regularization_loss), weight_gradient, bias_gradient


def softmax_loss_and_gradient(
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    regularization_strength: float = 0.0,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return stable softmax cross-entropy loss and parameter gradients.

    Scores are shifted by their row maximum before exponentiation. The stable
    forward loss uses log-sum-exp directly, while the compact score gradient is
    ``(probabilities - one_hot_targets) / num_examples``.
    """
    features, weights, bias = _as_calculation_arrays(features, weights, bias)
    labels = _validate_labels(labels, features.shape[0], weights.shape[0])
    regularization_strength = _validate_regularization(regularization_strength)

    scores = features @ weights.T + bias
    shifted_scores = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted_scores)
    sum_exp_scores = exp_scores.sum(axis=1, keepdims=True)
    probabilities = exp_scores / sum_exp_scores

    correct_scores = shifted_scores[np.arange(features.shape[0]), labels]
    data_loss = np.mean(np.log(sum_exp_scores[:, 0]) - correct_scores)
    regularization_loss = regularization_strength * np.sum(weights**2)

    score_gradient = probabilities.copy()
    score_gradient[np.arange(features.shape[0]), labels] -= 1.0
    score_gradient /= features.shape[0]
    weight_gradient = (
        score_gradient.T @ features + 2 * regularization_strength * weights
    )
    bias_gradient = score_gradient.sum(axis=0)
    return float(data_loss + regularization_loss), weight_gradient, bias_gradient


def train_linear_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    initial_weights: np.ndarray,
    initial_bias: np.ndarray,
    loss_function: LossFunction,
    *,
    learning_rate: float,
    regularization_strength: float = 0.0,
    batch_size: int = 128,
    num_iterations: int = 100,
    seed: int | None = None,
) -> TrainingResult:
    """Train a linear classifier with vanilla minibatch SGD.

    Starting parameters are copied and never mutated. Sampling is without
    replacement within an iteration; examples may naturally reappear in later
    minibatches.
    """
    features, initial_weights, initial_bias = _as_calculation_arrays(
        features, initial_weights, initial_bias
    )
    labels = _validate_labels(labels, features.shape[0], initial_weights.shape[0])
    regularization_strength = _validate_regularization(regularization_strength)

    if not callable(loss_function):
        raise TypeError("loss_function must be callable")
    if isinstance(learning_rate, (bool, np.bool_)) or not isinstance(
        learning_rate, (int, float, np.number)
    ):
        raise TypeError("learning_rate must be numeric")
    learning_rate = float(learning_rate)
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    for name, value in (("batch_size", batch_size), ("num_iterations", num_iterations)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer)
        ):
            raise TypeError(f"{name} must be an integer")
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if batch_size > features.shape[0]:
        raise ValueError("batch_size cannot exceed the number of examples")

    weights = initial_weights.copy()
    bias = initial_bias.copy()
    generator = np.random.default_rng(seed)
    loss_history = np.empty(num_iterations, dtype=np.float64)

    for iteration in range(num_iterations):
        batch_indices = generator.choice(
            features.shape[0], size=batch_size, replace=False
        )
        loss, weight_gradient, bias_gradient = loss_function(
            features[batch_indices],
            labels[batch_indices],
            weights,
            bias,
            regularization_strength,
        )
        if weight_gradient.shape != weights.shape or bias_gradient.shape != bias.shape:
            raise ValueError("Loss function returned gradients with invalid shapes")
        if not np.isfinite(loss) or not all(
            np.all(np.isfinite(gradient))
            for gradient in (weight_gradient, bias_gradient)
        ):
            raise FloatingPointError("Loss function returned non-finite values")

        weights -= learning_rate * weight_gradient
        bias -= learning_rate * bias_gradient
        loss_history[iteration] = loss

    return TrainingResult(weights=weights, bias=bias, loss_history=loss_history)
