import numpy as np
import pytest

from cs231n_practice.classifiers.linear import (
    classification_accuracy,
    linear_scores,
    predict_linear,
    softmax_loss_and_gradient,
    svm_loss_and_gradient,
    train_linear_classifier,
)


def test_linear_scores_uses_class_major_weights_and_broadcasts_bias() -> None:
    features = np.array([[1.0, 2.0], [-1.0, 3.0]])
    weights = np.array([[2.0, 0.0], [0.0, -1.0], [1.0, 1.0]])
    bias = np.array([0.5, 1.0, -2.0])

    scores = linear_scores(features, weights, bias)

    np.testing.assert_allclose(scores, [[2.5, -1.0, 1.0], [-1.5, -2.0, 0.0]])


def test_prediction_and_accuracy() -> None:
    features = np.array([[1.0, 0.0], [0.0, 2.0], [-1.0, 0.0]])
    weights = np.array([[1.0, 0.0], [0.0, 1.0]])
    bias = np.zeros(2)

    predictions = predict_linear(features, weights, bias)

    np.testing.assert_array_equal(predictions, [0, 1, 1])
    assert classification_accuracy(predictions, np.array([0, 0, 1])) == pytest.approx(
        2 / 3
    )


def test_svm_has_zero_loss_and_gradient_when_all_margins_are_satisfied() -> None:
    features = np.array([[1.0, 0.0]])
    labels = np.array([0])
    weights = np.array([[3.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
    bias = np.zeros(3)

    loss, weight_gradient, bias_gradient = svm_loss_and_gradient(
        features, labels, weights, bias
    )

    assert loss == 0.0
    np.testing.assert_array_equal(weight_gradient, np.zeros_like(weights))
    np.testing.assert_array_equal(bias_gradient, np.zeros_like(bias))


def test_svm_gradient_counts_only_active_incorrect_margins() -> None:
    features = np.array([[2.0, -1.0]])
    labels = np.array([0])
    weights = np.array([[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
    bias = np.zeros(3)

    loss, weight_gradient, bias_gradient = svm_loss_and_gradient(
        features, labels, weights, bias
    )

    assert loss == pytest.approx(1.0)
    np.testing.assert_allclose(bias_gradient, [-1.0, 1.0, 0.0])
    np.testing.assert_allclose(
        weight_gradient,
        [[-2.0, 1.0], [2.0, -1.0], [0.0, 0.0]],
    )


def test_softmax_uniform_scores_have_log_c_loss() -> None:
    features = np.zeros((4, 2))
    labels = np.array([0, 1, 2, 1])
    weights = np.zeros((3, 2))
    bias = np.zeros(3)

    loss, weight_gradient, bias_gradient = softmax_loss_and_gradient(
        features, labels, weights, bias
    )

    assert loss == pytest.approx(np.log(3))
    np.testing.assert_array_equal(weight_gradient, np.zeros_like(weights))
    assert np.isclose(bias_gradient.sum(), 0.0)


def test_softmax_loss_is_stable_for_large_scores() -> None:
    features = np.array([[1_000.0]])
    labels = np.array([2])
    weights = np.array([[1.0], [0.0], [-1.0]])
    bias = np.zeros(3)

    loss, weight_gradient, bias_gradient = softmax_loss_and_gradient(
        features, labels, weights, bias
    )

    assert np.isfinite(loss)
    assert loss == pytest.approx(2_000.0)
    assert np.all(np.isfinite(weight_gradient))
    assert np.all(np.isfinite(bias_gradient))


@pytest.mark.parametrize(
    "loss_function", [svm_loss_and_gradient, softmax_loss_and_gradient]
)
def test_analytic_weight_and_bias_gradients_match_numerical_gradients(
    loss_function,
) -> None:
    generator = np.random.default_rng(7)
    features = generator.normal(size=(4, 3))
    labels = np.array([0, 2, 1, 2])
    weights = generator.normal(scale=0.1, size=(3, 3))
    bias = generator.normal(scale=0.1, size=3)
    regularization_strength = 0.05
    _, analytic_weights, analytic_bias = loss_function(
        features, labels, weights, bias, regularization_strength
    )

    def current_loss() -> float:
        return loss_function(
            features, labels, weights, bias, regularization_strength
        )[0]

    def numerical_gradient(parameter: np.ndarray, step: float = 1e-5) -> np.ndarray:
        gradient = np.zeros_like(parameter)
        for index in np.ndindex(parameter.shape):
            original = parameter[index]
            parameter[index] = original + step
            loss_plus = current_loss()
            parameter[index] = original - step
            loss_minus = current_loss()
            parameter[index] = original
            gradient[index] = (loss_plus - loss_minus) / (2 * step)
        return gradient

    np.testing.assert_allclose(
        analytic_weights, numerical_gradient(weights), rtol=1e-5, atol=1e-7
    )
    np.testing.assert_allclose(
        analytic_bias, numerical_gradient(bias), rtol=1e-5, atol=1e-7
    )


def test_regularization_adds_expected_loss_and_weight_gradient() -> None:
    features = np.array([[1.0, -1.0], [0.5, 2.0]])
    labels = np.array([0, 1])
    weights = np.array([[1.0, -2.0], [0.5, 3.0]])
    bias = np.zeros(2)
    strength = 0.2

    loss_without, gradient_without, bias_without = softmax_loss_and_gradient(
        features, labels, weights, bias, 0.0
    )
    loss_with, gradient_with, bias_with = softmax_loss_and_gradient(
        features, labels, weights, bias, strength
    )

    assert loss_with - loss_without == pytest.approx(strength * np.sum(weights**2))
    np.testing.assert_allclose(
        gradient_with - gradient_without, 2 * strength * weights
    )
    np.testing.assert_allclose(bias_with, bias_without)


def test_training_is_reproducible_and_does_not_mutate_initial_parameters() -> None:
    features = np.array(
        [[2.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-2.0, 0.0]]
    )
    labels = np.array([0, 0, 1, 1])
    initial_weights = np.zeros((2, 2))
    initial_bias = np.zeros(2)
    weights_before = initial_weights.copy()
    bias_before = initial_bias.copy()

    first = train_linear_classifier(
        features,
        labels,
        initial_weights,
        initial_bias,
        softmax_loss_and_gradient,
        learning_rate=0.1,
        batch_size=2,
        num_iterations=20,
        seed=11,
    )
    second = train_linear_classifier(
        features,
        labels,
        initial_weights,
        initial_bias,
        softmax_loss_and_gradient,
        learning_rate=0.1,
        batch_size=2,
        num_iterations=20,
        seed=11,
    )

    np.testing.assert_array_equal(initial_weights, weights_before)
    np.testing.assert_array_equal(initial_bias, bias_before)
    np.testing.assert_allclose(first.weights, second.weights)
    np.testing.assert_allclose(first.bias, second.bias)
    np.testing.assert_allclose(first.loss_history, second.loss_history)
    assert first.loss_history[-5:].mean() < first.loss_history[:5].mean()


def test_loss_functions_reject_invalid_labels() -> None:
    with pytest.raises(ValueError, match="valid class"):
        softmax_loss_and_gradient(
            np.ones((2, 3)),
            np.array([0, 2]),
            np.ones((2, 3)),
            np.zeros(2),
        )


def test_training_rejects_batch_larger_than_dataset() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        train_linear_classifier(
            np.ones((2, 3)),
            np.array([0, 1]),
            np.zeros((2, 3)),
            np.zeros(2),
            softmax_loss_and_gradient,
            learning_rate=0.1,
            batch_size=3,
        )
