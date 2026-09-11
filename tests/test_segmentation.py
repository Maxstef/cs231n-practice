import numpy as np
import pytest

from cs231n_practice.segmentation import (
    binary_mask_iou,
    compose_instance_map,
    decode_panoptic_ids,
    encode_panoptic_ids,
    panoptic_quality,
    segmentation_confusion_matrix,
    segmentation_metrics,
    select_class_mask_logits,
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


def test_binary_mask_iou_matches_notebook_and_empty_convention() -> None:
    mask_a = np.array([[0, 1, 1, 0], [0, 1, 1, 0]], dtype=bool)
    mask_b = np.array([[0, 0, 1, 0], [0, 1, 1, 1]], dtype=bool)

    np.testing.assert_allclose(binary_mask_iou(mask_a, mask_b), 3 / 5)
    np.testing.assert_allclose(binary_mask_iou(mask_a, mask_a), 1.0)
    assert binary_mask_iou(np.zeros((2, 2), bool), np.zeros((2, 2), bool)) == 0


def test_binary_mask_iou_rejects_non_boolean_and_different_shapes() -> None:
    with pytest.raises(TypeError, match="Boolean"):
        binary_mask_iou(np.ones((2, 2)), np.ones((2, 2)))
    with pytest.raises(ValueError, match="same shape"):
        binary_mask_iou(np.ones((2, 2), bool), np.ones((2, 3), bool))


def test_select_class_mask_logits_selects_one_channel_per_roi() -> None:
    mask_logits = np.arange(2 * 3 * 2 * 2).reshape(2, 3, 2, 2)

    selected = select_class_mask_logits(mask_logits, np.array([2, 0]))

    assert selected.shape == (2, 2, 2)
    np.testing.assert_array_equal(selected[0], mask_logits[0, 2])
    np.testing.assert_array_equal(selected[1], mask_logits[1, 0])


def test_select_class_mask_logits_rejects_invalid_inputs() -> None:
    logits = np.zeros((2, 3, 2, 2))
    with pytest.raises(ValueError, match="shape"):
        select_class_mask_logits(logits, np.array([[0, 1]]))
    with pytest.raises(ValueError, match="outside"):
        select_class_mask_logits(logits, np.array([0, 3]))
    with pytest.raises(TypeError, match="integer"):
        select_class_mask_logits(logits, np.array([0.0, 1.0]))


def test_compose_instance_map_prioritizes_scores_and_preserves_ids() -> None:
    masks = np.array(
        [
            [[0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
            [[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 0]],
            [[0, 0, 0, 0], [0, 0, 1, 0], [1, 1, 1, 0]],
        ],
        dtype=bool,
    )

    composed = compose_instance_map(masks, np.array([0.95, 0.70, 0.85]))

    np.testing.assert_array_equal(
        composed, [[0, 1, 1, 2], [0, 1, 1, 2], [3, 3, 3, 0]]
    )


def test_compose_instance_map_uses_stable_order_for_tied_scores() -> None:
    masks = np.ones((2, 1, 1), dtype=bool)

    composed = compose_instance_map(masks, np.array([0.5, 0.5]))

    np.testing.assert_array_equal(composed, [[1]])


def test_compose_instance_map_handles_no_predictions() -> None:
    composed = compose_instance_map(
        np.empty((0, 2, 3), dtype=bool), np.empty(0)
    )

    np.testing.assert_array_equal(composed, np.zeros((2, 3), dtype=int))


def test_compose_instance_map_rejects_invalid_inputs() -> None:
    masks = np.ones((2, 2, 2), dtype=bool)
    with pytest.raises(ValueError, match="shape"):
        compose_instance_map(masks, np.ones(3))
    with pytest.raises(TypeError, match="Boolean"):
        compose_instance_map(masks.astype(int), np.ones(2))
    with pytest.raises(ValueError, match="finite"):
        compose_instance_map(masks, np.array([0.5, np.nan]))


def test_panoptic_ids_round_trip_notebook_example() -> None:
    semantic_ids = np.array([[3, 1, 1, 2], [4, 1, 1, 2]])
    instance_ids = np.array([[0, 1, 1, 1], [0, 2, 2, 1]])

    panoptic_ids = encode_panoptic_ids(semantic_ids, instance_ids)
    decoded_semantic, decoded_instance = decode_panoptic_ids(panoptic_ids)

    np.testing.assert_array_equal(
        panoptic_ids, [[3000, 1001, 1001, 2001], [4000, 1002, 1002, 2001]]
    )
    np.testing.assert_array_equal(decoded_semantic, semantic_ids)
    np.testing.assert_array_equal(decoded_instance, instance_ids)


def test_panoptic_ids_reject_ambiguous_or_invalid_values() -> None:
    ids = np.array([[0, 1]])
    with pytest.raises(ValueError, match="same shape"):
        encode_panoptic_ids(ids, np.array([0, 1]))
    with pytest.raises(ValueError, match="smaller"):
        encode_panoptic_ids(ids, np.array([[0, 1000]]))
    with pytest.raises(ValueError, match="nonnegative"):
        decode_panoptic_ids(np.array([[-1]]))
    with pytest.raises(TypeError, match="divisor"):
        decode_panoptic_ids(ids, divisor=True)


def test_panoptic_quality_matches_notebook_example() -> None:
    sq, rq, pq = panoptic_quality(np.array([0.8, 0.7, 0.6]), 1, 2)

    np.testing.assert_allclose(sq, 0.7)
    np.testing.assert_allclose(rq, 2 / 3)
    np.testing.assert_allclose(pq, 7 / 15)


def test_panoptic_quality_handles_no_segments_or_no_matches() -> None:
    assert panoptic_quality(np.empty(0), 0, 0) == (0.0, 0.0, 0.0)
    assert panoptic_quality(np.empty(0), 2, 1) == (0.0, 0.0, 0.0)


def test_panoptic_quality_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        panoptic_quality(np.array([1.1]), 0, 0)
    with pytest.raises(ValueError, match="one-dimensional"):
        panoptic_quality(np.array([[0.8]]), 0, 0)
    with pytest.raises(ValueError, match="nonnegative"):
        panoptic_quality(np.array([0.8]), -1, 0)
    with pytest.raises(TypeError, match="integer"):
        panoptic_quality(np.array([0.8]), 1.0, 0)
