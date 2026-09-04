import numpy as np
import pytest

from cs231n_practice.classifiers.transformer import TransformerSequenceClassifier
from cs231n_practice.gradient_check import eval_numerical_gradient


def _small_model(seed: int = 1) -> TransformerSequenceClassifier:
    return TransformerSequenceClassifier(
        vocabulary_size=6,
        sequence_length=3,
        model_dim=4,
        feed_forward_dim=5,
        num_heads=2,
        num_classes=2,
        weight_scale=0.1,
        seed=seed,
    )


def test_transformer_classifier_parameter_score_and_attention_shapes() -> None:
    model = _small_model()
    token_ids = np.array([[1, 2, 0], [1, 3, 4]])
    valid = token_ids != 0

    scores = model.loss(token_ids, valid)
    attention = model.attention_weights(token_ids, valid)

    assert scores.shape == (2, 2)
    assert attention.shape == (2, 2, 3, 3)
    np.testing.assert_array_equal(attention[0, :, :, 2], 0.0)
    assert model.parameters["embedding"].shape == (6, 4)
    assert model.parameters["W1"].shape == (4, 5)
    assert model.parameters["W_classifier"].shape == (4, 2)


def test_transformer_classifier_gradients_match_numerical_gradients() -> None:
    model = _small_model(seed=3)
    token_ids = np.array([[1, 2, 0], [1, 3, 4]])
    valid = token_ids != 0
    labels = np.array([1, 0])
    _, gradients = model.loss(token_ids, valid, labels)

    for name, parameter in model.parameters.items():
        def loss_with_candidate(candidate: np.ndarray) -> float:
            original = model.parameters[name]
            model.parameters[name] = candidate
            try:
                return model.loss(token_ids, valid, labels)[0]
            finally:
                model.parameters[name] = original

        numerical = eval_numerical_gradient(loss_with_candidate, parameter)
        np.testing.assert_allclose(
            gradients[name], numerical, rtol=3e-5, atol=1e-7
        )


def test_transformer_classifier_training_fits_a_tiny_signal_task() -> None:
    token_ids = np.array([
        [1, 2, 0],
        [1, 2, 3],
        [1, 4, 2],
        [1, 3, 0],
        [1, 4, 5],
        [1, 5, 3],
    ])
    valid = token_ids != 0
    labels = np.array([1, 1, 1, 0, 0, 0])
    model = _small_model(seed=5)

    history = model.train(
        token_ids,
        valid,
        labels,
        learning_rate=0.5,
        batch_size=6,
        num_iterations=400,
        seed=7,
    )

    assert history[-20:].mean() < history[:20].mean()
    np.testing.assert_array_equal(model.predict(token_ids, valid), labels)
    assert model.accuracy(token_ids, valid, labels) == 1.0


def test_transformer_classifier_training_is_reproducible() -> None:
    token_ids = np.array([[1, 2, 0], [1, 3, 0]])
    valid = token_ids != 0
    labels = np.array([1, 0])
    first = _small_model(seed=11)
    second = _small_model(seed=11)

    first_history = first.train(
        token_ids, valid, labels,
        learning_rate=0.2, batch_size=2, num_iterations=5, seed=13
    )
    second_history = second.train(
        token_ids, valid, labels,
        learning_rate=0.2, batch_size=2, num_iterations=5, seed=13
    )

    np.testing.assert_allclose(first_history, second_history)
    for name in first.parameters:
        np.testing.assert_allclose(first.parameters[name], second.parameters[name])


def test_transformer_classifier_rejects_invalid_configuration_and_inputs() -> None:
    with pytest.raises(ValueError, match="divisible"):
        TransformerSequenceClassifier(6, 3, 5, 8, 2, 2)

    model = _small_model()
    with pytest.raises(ValueError, match="classification position 0"):
        model.loss(
            np.array([[1, 2, 0]]),
            np.array([[False, True, False]]),
        )
    with pytest.raises(ValueError, match="shape"):
        model.loss(np.array([[1, 2]]), np.array([[True, True]]))
