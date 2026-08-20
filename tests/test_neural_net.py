import numpy as np

from cs231n_practice.classifiers.neural_net import TwoLayerNet
from cs231n_practice.gradient_check import eval_numerical_gradient


def test_two_layer_net_parameter_and_score_shapes() -> None:
    model = TwoLayerNet(4, 5, 3, weight_scale=0.1, seed=2)

    scores = model.loss(np.ones((6, 4)))

    assert model.parameters["W1"].shape == (4, 5)
    assert model.parameters["b1"].shape == (5,)
    assert model.parameters["W2"].shape == (5, 3)
    assert model.parameters["b2"].shape == (3,)
    assert scores.shape == (6, 3)


def test_two_layer_net_gradients_match_numerical_gradients() -> None:
    model = TwoLayerNet(3, 4, 3, weight_scale=0.1, seed=5)
    features = np.array(
        [[0.2, -0.4, 0.7], [1.1, 0.3, -0.2], [-0.5, 0.8, 0.4]]
    )
    labels = np.array([0, 2, 1])
    strength = 0.05
    _, gradients = model.loss(
        features, labels, regularization_strength=strength
    )

    for name, parameter in model.parameters.items():
        def loss_with_candidate(candidate: np.ndarray) -> float:
            original = model.parameters[name]
            model.parameters[name] = candidate
            try:
                return model.loss(
                    features,
                    labels,
                    regularization_strength=strength,
                )[0]
            finally:
                model.parameters[name] = original

        numerical = eval_numerical_gradient(loss_with_candidate, parameter)
        np.testing.assert_allclose(
            gradients[name], numerical, rtol=1e-6, atol=1e-8
        )


def test_training_is_reproducible_and_reduces_loss() -> None:
    features = np.array(
        [[2.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [-2.0, 0.0]]
    )
    labels = np.array([0, 0, 1, 1])
    first = TwoLayerNet(2, 8, 2, weight_scale=0.1, seed=3)
    second = TwoLayerNet(2, 8, 2, weight_scale=0.1, seed=3)

    first_history = first.train(
        features,
        labels,
        learning_rate=0.2,
        batch_size=4,
        num_iterations=100,
        seed=9,
    )
    second_history = second.train(
        features,
        labels,
        learning_rate=0.2,
        batch_size=4,
        num_iterations=100,
        seed=9,
    )

    np.testing.assert_allclose(first_history, second_history)
    for name in first.parameters:
        np.testing.assert_allclose(first.parameters[name], second.parameters[name])
    assert first_history[-10:].mean() < first_history[:10].mean()
    np.testing.assert_array_equal(first.predict(features), labels)
