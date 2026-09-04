"""A small Transformer sequence classifier implemented with NumPy."""

import numpy as np

from cs231n_practice.layers import softmax_loss
from cs231n_practice.positional_encoding import add_positional_encoding
from cs231n_practice.sequence_layers import embedding_backward, embedding_forward
from cs231n_practice.transformer import (
    TransformerEncoderCache,
    transformer_encoder_block_backward,
    transformer_encoder_block_forward,
)


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


def _positive_float(value: float, *, name: str) -> float:
    """Return a finite positive floating-point value."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.number)
    ):
        raise TypeError(f"{name} must be numeric")
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


class TransformerSequenceClassifier:
    """A one-block pre-norm Transformer using position 0 for classification.

    Inputs are fixed-length padded token-ID arrays. The caller places a
    classification token at position 0 and supplies a Boolean validity mask.
    Fixed sinusoidal positions are added to learned token embeddings before the
    encoder. This educational implementation uses vanilla minibatch SGD.
    """

    def __init__(
        self,
        vocabulary_size: int,
        sequence_length: int,
        model_dim: int,
        feed_forward_dim: int,
        num_heads: int,
        num_classes: int,
        *,
        weight_scale: float = 0.08,
        seed: int | None = None,
    ) -> None:
        vocabulary_size = _positive_integer(
            vocabulary_size, name="vocabulary_size"
        )
        sequence_length = _positive_integer(
            sequence_length, name="sequence_length"
        )
        model_dim = _positive_integer(model_dim, name="model_dim")
        feed_forward_dim = _positive_integer(
            feed_forward_dim, name="feed_forward_dim"
        )
        num_heads = _positive_integer(num_heads, name="num_heads")
        num_classes = _positive_integer(num_classes, name="num_classes")
        weight_scale = _positive_float(weight_scale, name="weight_scale")
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")

        self.vocabulary_size = vocabulary_size
        self.sequence_length = sequence_length
        self.model_dim = model_dim
        self.feed_forward_dim = feed_forward_dim
        self.num_heads = num_heads
        self.num_classes = num_classes

        generator = np.random.default_rng(seed)
        self.parameters = {
            "embedding": generator.normal(
                scale=weight_scale,
                size=(vocabulary_size, model_dim),
            ),
            "gamma1": np.ones(model_dim),
            "beta1": np.zeros(model_dim),
            "W_query": generator.normal(
                scale=weight_scale, size=(model_dim, model_dim)
            ),
            "W_key": generator.normal(
                scale=weight_scale, size=(model_dim, model_dim)
            ),
            "W_value": generator.normal(
                scale=weight_scale, size=(model_dim, model_dim)
            ),
            "W_output": generator.normal(
                scale=weight_scale, size=(model_dim, model_dim)
            ),
            "b_query": np.zeros(model_dim),
            "b_key": np.zeros(model_dim),
            "b_value": np.zeros(model_dim),
            "b_output": np.zeros(model_dim),
            "gamma2": np.ones(model_dim),
            "beta2": np.zeros(model_dim),
            "W1": generator.normal(
                scale=weight_scale,
                size=(model_dim, feed_forward_dim),
            ),
            "b1": np.zeros(feed_forward_dim),
            "W2": generator.normal(
                scale=weight_scale,
                size=(feed_forward_dim, model_dim),
            ),
            "b2": np.zeros(model_dim),
            "W_classifier": generator.normal(
                scale=weight_scale,
                size=(model_dim, num_classes),
            ),
            "b_classifier": np.zeros(num_classes),
        }

    def _validate_inputs(
        self,
        token_ids: np.ndarray,
        valid_tokens: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Validate fixed-length token IDs and their Boolean validity mask."""
        token_ids = np.asarray(token_ids)
        valid_tokens = np.asarray(valid_tokens)
        if not np.issubdtype(token_ids.dtype, np.integer) or np.issubdtype(
            token_ids.dtype, np.bool_
        ):
            raise TypeError("token_ids must contain integers")
        if token_ids.ndim != 2 or token_ids.shape[1] != self.sequence_length:
            raise ValueError(
                f"token_ids must have shape (N, {self.sequence_length})"
            )
        if valid_tokens.shape != token_ids.shape:
            raise ValueError("valid_tokens must have the same shape as token_ids")
        if not np.issubdtype(valid_tokens.dtype, np.bool_):
            raise TypeError("valid_tokens must contain Boolean values")
        if np.any(token_ids < 0) or np.any(token_ids >= self.vocabulary_size):
            raise ValueError("token_ids contain an index outside the vocabulary")
        if not np.all(valid_tokens[:, 0]):
            raise ValueError("classification position 0 must be valid")
        return token_ids, valid_tokens

    def _forward(
        self,
        token_ids: np.ndarray,
        valid_tokens: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        tuple[tuple[np.ndarray, tuple[int, int]], TransformerEncoderCache, np.ndarray],
    ]:
        """Return scores, attention weights, and a training cache."""
        token_ids, valid_tokens = self._validate_inputs(token_ids, valid_tokens)
        embeddings, embedding_cache = embedding_forward(
            token_ids, self.parameters["embedding"]
        )
        positioned_embeddings = add_positional_encoding(embeddings)
        padding_mask = valid_tokens[:, None, None, :]
        encoder_output, attention_weights, encoder_cache = (
            transformer_encoder_block_forward(
                positioned_embeddings,
                self.parameters,
                self.num_heads,
                padding_mask,
            )
        )
        cls_representation = encoder_output[:, 0, :]
        scores = (
            cls_representation @ self.parameters["W_classifier"]
            + self.parameters["b_classifier"]
        )
        cache = (embedding_cache, encoder_cache, cls_representation)
        return scores, attention_weights, cache

    def loss(
        self,
        token_ids: np.ndarray,
        valid_tokens: np.ndarray,
        labels: np.ndarray | None = None,
    ) -> np.ndarray | tuple[float, dict[str, np.ndarray]]:
        """Return class scores, or mean loss and all parameter gradients."""
        scores, _, cache = self._forward(token_ids, valid_tokens)
        if labels is None:
            return scores

        data_loss, dscores = softmax_loss(scores, labels)
        embedding_cache, encoder_cache, cls_representation = cache
        gradients: dict[str, np.ndarray] = {
            "W_classifier": cls_representation.T @ dscores,
            "b_classifier": dscores.sum(axis=0),
        }
        dcls = dscores @ self.parameters["W_classifier"].T
        dencoder_output = np.zeros(
            (token_ids.shape[0], self.sequence_length, self.model_dim),
            dtype=dcls.dtype,
        )
        dencoder_output[:, 0, :] = dcls
        dpositioned_embeddings, encoder_gradients = (
            transformer_encoder_block_backward(dencoder_output, encoder_cache)
        )
        gradients.update(encoder_gradients)
        gradients["embedding"] = embedding_backward(
            dpositioned_embeddings, embedding_cache
        )
        return data_loss, {
            name: gradients[name] for name in self.parameters
        }

    def attention_weights(
        self,
        token_ids: np.ndarray,
        valid_tokens: np.ndarray,
    ) -> np.ndarray:
        """Return attention maps with shape ``(N, H, L, L)``."""
        _, attention_weights, _ = self._forward(token_ids, valid_tokens)
        return attention_weights

    def predict(
        self,
        token_ids: np.ndarray,
        valid_tokens: np.ndarray,
    ) -> np.ndarray:
        """Return the highest-scoring class for every sequence."""
        return np.argmax(self.loss(token_ids, valid_tokens), axis=1)

    def accuracy(
        self,
        token_ids: np.ndarray,
        valid_tokens: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """Return the fraction of correctly classified sequences."""
        labels = np.asarray(labels)
        if labels.shape != (token_ids.shape[0],):
            raise ValueError("labels must have shape (N,)")
        return float(np.mean(self.predict(token_ids, valid_tokens) == labels))

    def train(
        self,
        token_ids: np.ndarray,
        valid_tokens: np.ndarray,
        labels: np.ndarray,
        *,
        learning_rate: float,
        batch_size: int = 64,
        num_iterations: int = 100,
        seed: int | None = None,
    ) -> np.ndarray:
        """Update parameters with minibatch SGD and return loss history."""
        token_ids, valid_tokens = self._validate_inputs(token_ids, valid_tokens)
        labels = np.asarray(labels)
        if labels.shape != (token_ids.shape[0],):
            raise ValueError("labels must have shape (N,)")
        learning_rate = _positive_float(learning_rate, name="learning_rate")
        batch_size = _positive_integer(batch_size, name="batch_size")
        num_iterations = _positive_integer(
            num_iterations, name="num_iterations"
        )
        if batch_size > token_ids.shape[0]:
            raise ValueError("batch_size cannot exceed the example count")

        generator = np.random.default_rng(seed)
        history = np.empty(num_iterations, dtype=np.float64)
        for iteration in range(num_iterations):
            indices = generator.choice(
                token_ids.shape[0], size=batch_size, replace=False
            )
            loss, gradients = self.loss(
                token_ids[indices],
                valid_tokens[indices],
                labels[indices],
            )
            for name in self.parameters:
                self.parameters[name] -= learning_rate * gradients[name]
            history[iteration] = loss
        return history

