import numpy as np
import pytest

from cs231n_practice.classifiers.cnn import SmallConvNet
from cs231n_practice.gradient_check import eval_numerical_gradient


def test_small_conv_net_parameter_and_score_shapes() -> None:
    model = SmallConvNet((3, 8, 8), 4, 3, 6, 3, seed=2)

    scores = model.loss(np.ones((5, 3, 8, 8)))

    assert {name: value.shape for name, value in model.parameters.items()} == {
        "W1": (4, 3, 3, 3),
        "b1": (4,),
        "W2": (64, 6),
        "b2": (6,),
        "W3": (6, 3),
        "b3": (3,),
    }
    assert scores.shape == (5, 3)


def test_small_conv_net_gradients_match_numerical_gradients() -> None:
    generator = np.random.default_rng(7)
    features = generator.normal(size=(2, 2, 4, 4))
    labels = np.array([0, 2])
    model = SmallConvNet((2, 4, 4), 2, 3, 3, 3, seed=8)
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


def test_small_conv_net_batchnorm_gradients_match_numerical_gradients() -> None:
    generator = np.random.default_rng(17)
    features = generator.normal(size=(3, 2, 4, 4))
    labels = np.array([0, 1, 2])
    model = SmallConvNet(
        (2, 4, 4), 2, 3, 3, 3, use_batchnorm=True, seed=18
    )
    _, gradients = model.loss(features, labels)

    assert "gamma2" in model.parameters
    assert "beta2" in model.parameters
    for name, parameter in model.parameters.items():
        def loss_with_candidate(candidate: np.ndarray) -> float:
            original = model.parameters[name]
            model.parameters[name] = candidate
            try:
                return model.loss(features, labels)[0]
            finally:
                model.parameters[name] = original

        numerical = eval_numerical_gradient(loss_with_candidate, parameter)
        np.testing.assert_allclose(
            gradients[name], numerical, rtol=2e-5, atol=1e-7
        )


def test_small_conv_net_batchnorm_and_dropout_modes_are_reproducible() -> None:
    features = np.random.default_rng(21).normal(size=(4, 1, 4, 4))
    labels = np.array([0, 1, 0, 1])
    first = SmallConvNet(
        (1, 4, 4),
        2,
        3,
        4,
        2,
        use_batchnorm=True,
        dropout_keep_probability=0.75,
        seed=22,
    )
    second = SmallConvNet(
        (1, 4, 4),
        2,
        3,
        4,
        2,
        use_batchnorm=True,
        dropout_keep_probability=0.75,
        seed=22,
    )

    first_loss, first_gradients = first.loss(features, labels)
    second_loss, second_gradients = second.loss(features, labels)

    assert first_loss == pytest.approx(second_loss)
    for name in first_gradients:
        np.testing.assert_allclose(first_gradients[name], second_gradients[name])
    assert first.batchnorm_state is not None
    running_mean = np.asarray(first.batchnorm_state["running_mean"]).copy()
    first_scores = first.loss(features)
    second_scores = first.loss(features)
    np.testing.assert_allclose(first_scores, second_scores)
    np.testing.assert_array_equal(
        first.batchnorm_state["running_mean"], running_mean
    )


def test_small_conv_net_training_is_reproducible_and_fits_patterns() -> None:
    features = np.zeros((6, 1, 4, 4), dtype=np.float64)
    features[:3, 0, :, 1:3] = 1.0
    features[3:, 0, 1:3, :] = 1.0
    labels = np.array([0, 0, 0, 1, 1, 1])
    first = SmallConvNet((1, 4, 4), 2, 3, 4, 2, seed=3)
    second = SmallConvNet((1, 4, 4), 2, 3, 4, 2, seed=3)

    first_history = first.train(
        features,
        labels,
        learning_rate=0.1,
        batch_size=6,
        num_iterations=100,
        seed=9,
    )
    second_history = second.train(
        features,
        labels,
        learning_rate=0.1,
        batch_size=6,
        num_iterations=100,
        seed=9,
    )

    np.testing.assert_allclose(first_history, second_history)
    for name in first.parameters:
        np.testing.assert_allclose(first.parameters[name], second.parameters[name])
    assert first_history[-10:].mean() < first_history[:10].mean()
    np.testing.assert_array_equal(first.predict(features), labels)


def test_small_conv_net_rejects_invalid_architecture() -> None:
    with pytest.raises(TypeError, match="input_shape"):
        SmallConvNet([1, 4, 4], 2, 3, 4, 2)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="filter_size must be odd"):
        SmallConvNet((1, 4, 4), 2, 2, 4, 2)
    with pytest.raises(ValueError, match="tile"):
        SmallConvNet((1, 5, 5), 2, 3, 4, 2)
    with pytest.raises(TypeError, match="use_batchnorm"):
        SmallConvNet((1, 4, 4), 2, 3, 4, 2, use_batchnorm=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="dropout_keep_probability"):
        SmallConvNet((1, 4, 4), 2, 3, 4, 2, dropout_keep_probability=0.0)


def test_small_conv_net_rejects_wrong_feature_shape() -> None:
    model = SmallConvNet((3, 8, 8), 4, 3, 6, 3, seed=2)

    with pytest.raises(ValueError, match="features must have shape"):
        model.loss(np.ones((5, 8, 8, 3)))
