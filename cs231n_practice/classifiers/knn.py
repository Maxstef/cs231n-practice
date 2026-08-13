"""A NumPy implementation of the k-nearest-neighbor (kNN) classifier.

kNN is a non-parametric, instance-based classifier. ``fit`` does not learn
weights: it stores the training examples. At prediction time, each query is
compared with every stored example, and the most common label among the ``k``
nearest examples becomes the prediction.

This module keeps distance calculation, neighbor selection, voting, and
accuracy separate. The small functions make each step easier to inspect and
test independently.
"""

import numpy as np


def _validate_feature_matrices(
    query_features: np.ndarray,
    stored_features: np.ndarray,
) -> None:
    """Check that two batches use the ``(examples, features)`` convention.

    Pairwise comparison requires both matrices to describe vectors in the same
    feature space, so their second dimensions must match. Their first
    dimensions may differ because the numbers of queries and stored examples
    need not be equal.
    """
    if query_features.ndim != 2 or stored_features.ndim != 2:
        raise ValueError("Feature matrices must be two-dimensional")
    if query_features.shape[1] != stored_features.shape[1]:
        raise ValueError("Feature matrices must have equal feature dimensions")


def squared_l2_distances(
    query_features: np.ndarray,
    stored_features: np.ndarray,
) -> np.ndarray:
    """Return all pairwise squared Euclidean (L2) distances.

    For a query vector ``x`` and stored vector ``z``, the squared distance is

    ``||x - z||² = ||x||² + ||z||² - 2 x·z``.

    Applying this identity to whole matrices avoids Python loops and avoids
    allocating a large ``(num_queries, num_stored, num_features)`` array of
    pairwise differences. The square root is omitted because it is strictly
    increasing and therefore cannot change the ordering of nearest neighbors.

    Args:
        query_features: Array with shape ``(num_queries, num_features)``.
        stored_features: Array with shape ``(num_stored, num_features)``.

    Returns:
        An array with shape ``(num_queries, num_stored)``. Entry ``[i, j]``
        is the squared L2 distance between query ``i`` and stored example
        ``j``.

    Raises:
        ValueError: If either input is not two-dimensional or their feature
            dimensions differ.
    """
    query_features = np.asarray(query_features)
    stored_features = np.asarray(stored_features)
    _validate_feature_matrices(query_features, stored_features)

    # Integer pixels must be converted before subtraction, squaring, and
    # matrix multiplication to avoid overflow. Preserve float64 when supplied.
    calculation_dtype = np.result_type(
        query_features.dtype,
        stored_features.dtype,
        np.float32,
    )
    query_features = query_features.astype(calculation_dtype, copy=False)
    stored_features = stored_features.astype(calculation_dtype, copy=False)

    # Each row is one feature vector. Summing its squared components gives its
    # squared L2 norm. Shapes (Q, 1) and (1, S) are retained deliberately so
    # NumPy can broadcast them across the final (Q, S) distance matrix.
    query_norms = np.sum(query_features**2, axis=1, keepdims=True)  # (Q, 1)
    stored_norms = np.sum(stored_features**2, axis=1, keepdims=True).T  # (1, S)

    # Entry (i, j) is the dot product of query i with stored example j.
    cross_products = query_features @ stored_features.T
    distances = query_norms + stored_norms - 2 * cross_products

    # Algebraically distances cannot be negative, but floating-point
    # roundoff can produce tiny values below zero.
    return np.maximum(distances, 0)


def nearest_neighbor_indices(distances: np.ndarray, k: int) -> np.ndarray:
    """Return indices of the ``k`` nearest stored examples for every query.

    Args:
        distances: Pairwise distance matrix with shape
            ``(num_queries, num_stored)``. Each row describes one query.
        k: Number of neighbors to select.

    Returns:
        Integer array with shape ``(num_queries, k)``. Indices in each row are
        ordered from the smallest distance to the largest selected distance.

    Notes:
        ``argsort`` fully sorts every row. A partial sort can be faster for
        large datasets, but full sorting keeps this educational implementation
        direct and guarantees nearest-to-farthest order.
    """
    distances = np.asarray(distances)
    if distances.ndim != 2:
        raise ValueError("Distances must be two-dimensional")
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)):
        raise TypeError("k must be an integer")
    if not 1 <= k <= distances.shape[1]:
        raise ValueError("k must be between 1 and the number of stored examples")

    return np.argsort(distances, axis=1)[:, :k]


def predict_from_distances(
    distances: np.ndarray,
    stored_labels: np.ndarray,
    k: int,
    *,
    num_classes: int | None = None,
) -> np.ndarray:
    """Predict labels by majority vote among each query's nearest neighbors.

    Vote ties are resolved in favor of the smallest class label because
    ``numpy.argmax`` returns the first maximum. Class numbers are arbitrary
    identifiers, so this is a deterministic convention rather than a
    statistical preference for smaller labels.

    Args:
        distances: Matrix with shape ``(num_queries, num_stored)``.
        stored_labels: Nonnegative integer label for each distance column.
        k: Number of neighbors participating in each vote.
        num_classes: Total number of possible classes. If omitted, it is
            inferred as ``max(stored_labels) + 1``.

    Returns:
        Integer predictions with shape ``(num_queries,)``.
    """
    distances = np.asarray(distances)
    stored_labels = np.asarray(stored_labels)

    if stored_labels.ndim != 1:
        raise ValueError("Stored labels must be one-dimensional")
    if distances.ndim != 2:
        raise ValueError("Distances must be two-dimensional")
    if distances.shape[1] != len(stored_labels):
        raise ValueError("Each distance column must have one stored label")
    if not np.issubdtype(stored_labels.dtype, np.integer):
        raise TypeError("Stored labels must be integers")
    if np.any(stored_labels < 0):
        raise ValueError("Stored labels must be nonnegative")

    if num_classes is None:
        if stored_labels.size == 0:
            raise ValueError("Stored labels must not be empty")
        num_classes = int(stored_labels.max()) + 1
    if isinstance(num_classes, (bool, np.bool_)) or not isinstance(
        num_classes, (int, np.integer)
    ):
        raise TypeError("num_classes must be an integer")
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    if stored_labels.size and stored_labels.max() >= num_classes:
        raise ValueError("Stored label is outside the configured class range")

    neighbor_indices = nearest_neighbor_indices(distances, k)  # (Q, k)
    neighbor_labels = stored_labels[neighbor_indices]  # (Q, k)
    predictions = np.empty(len(distances), dtype=np.int64)

    # bincount converts one query's k labels into per-class vote counts.
    # The loop is over queries, not pixels or training examples; keeping it
    # explicit makes the voting and tie behavior easy to see.
    for query_index, labels in enumerate(neighbor_labels):
        votes = np.bincount(labels, minlength=num_classes)
        predictions[query_index] = np.argmax(votes)

    return predictions


def accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Return the fraction of predictions equal to their targets.

    Both inputs must have the same nonempty shape. The return value lies in
    ``[0.0, 1.0]`` rather than representing a percentage.
    """
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)
    if predictions.shape != targets.shape:
        raise ValueError("Predictions and targets must have the same shape")
    if predictions.size == 0:
        raise ValueError("Predictions and targets must not be empty")
    return float(np.mean(predictions == targets))


class KNearestNeighbor:
    """A classifier that stores training examples and predicts by voting.

    Unlike a linear classifier or neural network, kNN has no learned parameter
    matrix. Its training phase is therefore cheap, while prediction is costly:
    every query must be compared with all stored examples.

    Uses squared L2 distance and majority voting.
    """

    def __init__(self) -> None:
        self._features: np.ndarray | None = None
        self._labels: np.ndarray | None = None
        self._num_classes: int | None = None

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "KNearestNeighbor":
        """Store labeled feature vectors and return this classifier.

        Args:
            features: Training matrix with shape
                ``(num_examples, num_features)``.
            labels: Nonnegative integer labels with shape ``(num_examples,)``.

        Returns:
            This instance, allowing ``classifier.fit(...).predict(...)``.

        Notes:
            Calling this method "fit" follows the conventional estimator API.
            No optimization occurs; the data is simply retained for later
            distance calculations.
        """
        features = np.asarray(features)
        labels = np.asarray(labels)
        if features.ndim != 2:
            raise ValueError("Features must be two-dimensional")
        if labels.ndim != 1:
            raise ValueError("Labels must be one-dimensional")
        if len(features) != len(labels):
            raise ValueError("Each stored example must have one label")
        if len(features) == 0:
            raise ValueError("Training data must not be empty")
        if not np.issubdtype(labels.dtype, np.integer):
            raise TypeError("Labels must be integers")
        if np.any(labels < 0):
            raise ValueError("Labels must be nonnegative")

        self._features = features
        self._labels = labels
        self._num_classes = int(labels.max()) + 1
        return self

    def predict(self, query_features: np.ndarray, k: int = 1) -> np.ndarray:
        """Predict one label for each query feature vector.

        Prediction has two stages: calculate every query-to-training distance,
        then perform majority voting among the ``k`` nearest stored examples.

        Args:
            query_features: Matrix with shape
                ``(num_queries, num_features)``.
            k: Number of stored neighbors participating in each vote.

        Returns:
            Integer labels with shape ``(num_queries,)``.

        Raises:
            RuntimeError: If ``fit`` has not been called.
        """
        if self._features is None or self._labels is None:
            raise RuntimeError("Call fit before predict")

        distances = squared_l2_distances(query_features, self._features)
        return predict_from_distances(
            distances,
            self._labels,
            k,
            num_classes=self._num_classes,
        )
