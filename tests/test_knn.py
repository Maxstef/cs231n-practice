import numpy as np
import pytest

from cs231n_practice.classifiers.knn import (
    KNearestNeighbor,
    accuracy,
    nearest_neighbor_indices,
    predict_from_distances,
    squared_l2_distances,
)


def test_squared_l2_distances_has_expected_values() -> None:
    queries = np.array([[0, 0], [1, 1]], dtype=np.uint8)
    stored = np.array([[0, 0], [3, 4], [1, 2]], dtype=np.uint8)

    distances = squared_l2_distances(queries, stored)

    np.testing.assert_allclose(distances, [[0, 25, 5], [2, 13, 1]])


def test_squared_l2_distances_is_symmetric_when_inputs_are_swapped() -> None:
    queries = np.array([[1.0, 2.0], [-1.0, 3.0]])
    stored = np.array([[0.0, 4.0], [2.0, 1.0], [3.0, 3.0]])

    forward = squared_l2_distances(queries, stored)
    backward = squared_l2_distances(stored, queries)

    np.testing.assert_allclose(forward, backward.T)


def test_squared_l2_distances_supports_empty_queries() -> None:
    distances = squared_l2_distances(np.empty((0, 3)), np.ones((2, 3)))

    assert distances.shape == (0, 2)


def test_squared_l2_distances_rejects_incompatible_shapes() -> None:
    with pytest.raises(ValueError, match="equal feature dimensions"):
        squared_l2_distances(np.ones((2, 3)), np.ones((4, 2)))


def test_nearest_neighbor_indices_orders_each_row() -> None:
    distances = np.array([[9.0, 1.0, 4.0, 2.0], [0.5, 8.0, 3.0, 2.0]])

    nearest = nearest_neighbor_indices(distances, k=3)

    np.testing.assert_array_equal(nearest, [[1, 3, 2], [0, 3, 2]])


@pytest.mark.parametrize("k", [0, -1, 4])
def test_nearest_neighbor_indices_rejects_out_of_range_k(k: int) -> None:
    with pytest.raises(ValueError):
        nearest_neighbor_indices(np.ones((2, 3)), k)


def test_nearest_neighbor_indices_rejects_noninteger_k() -> None:
    with pytest.raises(TypeError, match="integer"):
        nearest_neighbor_indices(np.ones((2, 3)), 1.5)  # type: ignore[arg-type]


def test_predict_from_distances_uses_majority_vote() -> None:
    distances = np.array([[9.0, 1.0, 4.0, 2.0], [0.5, 8.0, 3.0, 2.0]])
    labels = np.array([4, 2, 2, 4])

    predictions = predict_from_distances(distances, labels, k=3, num_classes=5)

    np.testing.assert_array_equal(predictions, [2, 4])


def test_predict_from_distances_breaks_vote_tie_with_smallest_label() -> None:
    distances = np.array([[2.0, 1.0]])
    labels = np.array([4, 2])

    prediction = predict_from_distances(distances, labels, k=2, num_classes=5)

    np.testing.assert_array_equal(prediction, [2])


def test_accuracy_returns_fraction_correct() -> None:
    result = accuracy(np.array([1, 0, 3, 2]), np.array([1, 2, 3, 2]))

    assert result == 0.75


def test_accuracy_rejects_empty_arrays() -> None:
    with pytest.raises(ValueError, match="empty"):
        accuracy(np.array([]), np.array([]))


def test_classifier_fits_and_predicts() -> None:
    features = np.array([[0.0], [1.0], [9.0], [10.0]])
    labels = np.array([0, 0, 1, 1])
    classifier = KNearestNeighbor().fit(features, labels)

    predictions = classifier.predict(np.array([[0.2], [9.8]]), k=3)

    np.testing.assert_array_equal(predictions, [0, 1])


def test_classifier_requires_fit_before_predict() -> None:
    with pytest.raises(RuntimeError, match="fit"):
        KNearestNeighbor().predict(np.array([[1.0]]))
