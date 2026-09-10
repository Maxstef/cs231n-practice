import numpy as np
import pytest

from cs231n_practice.segmentation import (
    segmentation_confusion_matrix,
    segmentation_metrics,
    semantic_cross_entropy,
)


def _example_logits() -> np.ndarray:
    return np.array(
        [
            [
                [[3, 1, 0], [2, 0, 1]],
                [[1, 4, 0], [0, 3, 2]],
                [[0, 2, 5], [1, 1, 0]],
            ]
        ],
        dtype=float,
    )


def test_semantic_cross_entropy_matches_notebook_example() -> None:
    targets = np.array([[[0, 1, 2], [0, -1, 1]]])

    loss, dlogits = semantic_cross_entropy(_example_logits(), targets)

    np.testing.assert_allclose(loss, 0.233658, rtol=1e-5)
    assert dlogits.shape == (1, 3, 2, 3)
    np.testing.assert_array_equal(dlogits[0, :, 1, 1], 0)
    np.testing.assert_allclose(np.sum(dlogits, axis=1), 0, atol=1e-15)


def test_semantic_cross_entropy_is_stable_for_large_logits() -> None:
    logits = np.array([[[[10_000.0]], [[9_999.0]]]])
    targets = np.array([[[0]]])

    loss, dlogits = semantic_cross_entropy(logits, targets)

    np.testing.assert_allclose(loss, np.log1p(np.exp(-1)))
    assert np.all(np.isfinite(dlogits))


def test_semantic_cross_entropy_gradient_matches_numerical_gradient() -> None:
    rng = np.random.default_rng(4)
    logits = rng.normal(size=(2, 3, 2, 2))
    targets = np.array([[[0, 1], [2, -1]], [[1, 0], [2, 1]]])
    _, analytic = semantic_cross_entropy(logits, targets)
    numerical = np.zeros_like(logits)
    step = 1e-5

    for index in np.ndindex(logits.shape):
        old_value = logits[index]
        logits[index] = old_value + step
        positive, _ = semantic_cross_entropy(logits, targets)
        logits[index] = old_value - step
        negative, _ = semantic_cross_entropy(logits, targets)
        logits[index] = old_value
        numerical[index] = (positive - negative) / (2 * step)

    np.testing.assert_allclose(analytic, numerical, rtol=1e-6, atol=1e-8)


def test_semantic_cross_entropy_rejects_invalid_inputs() -> None:
    logits = _example_logits()
    with pytest.raises(ValueError, match="match"):
        semantic_cross_entropy(logits, np.zeros((1, 2, 2), dtype=int))
    with pytest.raises(TypeError, match="integer"):
        semantic_cross_entropy(logits, np.zeros((1, 2, 3)))
    with pytest.raises(ValueError, match="outside"):
        semantic_cross_entropy(logits, np.full((1, 2, 3), 3))
    with pytest.raises(ValueError, match="at least one"):
        semantic_cross_entropy(logits, np.full((1, 2, 3), -1))


def test_confusion_matrix_matches_notebook_example_and_ignores_pixels() -> None:
    targets = np.array([[0, 0, 1, 1], [0, 2, 2, 1], [0, 2, -1, 1]])
    predictions = np.array([[0, 1, 1, 1], [0, 2, 0, 1], [0, 2, 1, 1]])

    confusion = segmentation_confusion_matrix(targets, predictions, 3)

    np.testing.assert_array_equal(confusion, [[3, 1, 0], [0, 4, 0], [1, 0, 2]])


def test_confusion_matrix_handles_no_valid_pixels() -> None:
    confusion = segmentation_confusion_matrix(
        np.full((2, 2), -1), np.full((2, 2), 99), 3
    )

    np.testing.assert_array_equal(confusion, np.zeros((3, 3), dtype=int))


def test_confusion_matrix_rejects_invalid_inputs() -> None:
    targets = np.array([[0, 1]])
    with pytest.raises(ValueError, match="same shape"):
        segmentation_confusion_matrix(targets, np.array([0, 1]), 2)
    with pytest.raises(ValueError, match="predictions"):
        segmentation_confusion_matrix(targets, np.array([[0, 2]]), 2)
    with pytest.raises(TypeError, match="integer"):
        segmentation_confusion_matrix(targets.astype(float), targets, 2)
    with pytest.raises(TypeError, match="number_of_classes"):
        segmentation_confusion_matrix(targets, targets, True)


def test_segmentation_metrics_match_notebook_example() -> None:
    confusion = np.array([[3, 1, 0], [0, 4, 0], [1, 0, 2]])

    pixel_accuracy, class_iou, mean_iou = segmentation_metrics(confusion)

    np.testing.assert_allclose(pixel_accuracy, 9 / 11)
    np.testing.assert_allclose(class_iou, [3 / 5, 4 / 5, 2 / 3])
    np.testing.assert_allclose(mean_iou, 31 / 45)


def test_segmentation_metrics_excludes_absent_classes_from_mean() -> None:
    confusion = np.array([[3, 1, 0], [0, 2, 0], [0, 0, 0]])

    pixel_accuracy, class_iou, mean_iou = segmentation_metrics(confusion)

    np.testing.assert_allclose(pixel_accuracy, 5 / 6)
    np.testing.assert_allclose(class_iou[:2], [3 / 4, 2 / 3])
    assert np.isnan(class_iou[2])
    np.testing.assert_allclose(mean_iou, (3 / 4 + 2 / 3) / 2)


def test_segmentation_metrics_handles_empty_confusion_counts() -> None:
    pixel_accuracy, class_iou, mean_iou = segmentation_metrics(
        np.zeros((2, 2), dtype=int)
    )

    assert np.isnan(pixel_accuracy)
    assert np.all(np.isnan(class_iou))
    assert np.isnan(mean_iou)


def test_segmentation_metrics_rejects_invalid_confusion_matrix() -> None:
    with pytest.raises(ValueError, match="square"):
        segmentation_metrics(np.ones((2, 3)))
    with pytest.raises(ValueError, match="nonnegative"):
        segmentation_metrics(np.array([[1, -1], [0, 1]]))
    with pytest.raises(TypeError, match="numeric"):
        segmentation_metrics(np.array([["1"]]))
