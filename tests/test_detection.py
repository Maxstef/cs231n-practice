import numpy as np
import pytest

from cs231n_practice.detection import (
    assign_anchors_to_targets,
    box_area_xyxy,
    box_iou_aligned,
    classwise_non_maximum_suppression,
    clip_boxes_xyxy,
    cxcywh_to_xyxy,
    generate_anchors_at_location,
    interpolated_average_precision,
    match_detections,
    non_maximum_suppression,
    pairwise_box_iou,
    precision_recall_from_matches,
    valid_boxes_xyxy,
    xywh_to_xyxy,
    xyxy_to_cxcywh,
)


def test_box_format_conversions_preserve_shapes_and_round_trip() -> None:
    boxes_xywh = np.array(
        [[[2, 3, 5, 4], [0, 1, 2, 6]]], dtype=np.int64
    )
    boxes_xyxy = xywh_to_xyxy(boxes_xywh)

    assert boxes_xyxy.shape == (1, 2, 4)
    np.testing.assert_array_equal(
        boxes_xyxy, [[[2, 3, 7, 7], [0, 1, 2, 7]]]
    )
    center_boxes = xyxy_to_cxcywh(boxes_xyxy)
    np.testing.assert_allclose(cxcywh_to_xyxy(center_boxes), boxes_xyxy)


def test_clip_boxes_supports_one_box_and_does_not_modify_input() -> None:
    box = np.array([-2, 2, 15, 12])
    original = box.copy()

    clipped = clip_boxes_xyxy(box, image_height=10, image_width=12)

    np.testing.assert_allclose(clipped, [0, 2, 12, 10])
    np.testing.assert_array_equal(box, original)


def test_validity_and_area_handle_degenerate_and_reversed_boxes() -> None:
    boxes = np.array(
        [
            [0, 0, 4, 5],
            [1, 1, 1, 3],
            [4, 0, 2, 3],
            [0, 5, 2, 3],
        ]
    )

    np.testing.assert_array_equal(
        valid_boxes_xyxy(boxes), [True, False, False, False]
    )
    np.testing.assert_array_equal(box_area_xyxy(boxes), [20, 0, 0, 0])


def test_aligned_iou_supports_one_to_many_and_is_symmetric() -> None:
    reference = np.array([0.0, 0.0, 4.0, 4.0])
    comparisons = np.array(
        [
            [0.0, 0.0, 4.0, 4.0],
            [2.0, 0.0, 6.0, 4.0],
            [4.0, 0.0, 8.0, 4.0],
            [5.0, 5.0, 7.0, 7.0],
        ]
    )

    expected = np.array([1.0, 1.0 / 3.0, 0.0, 0.0])
    np.testing.assert_allclose(box_iou_aligned(reference, comparisons), expected)
    np.testing.assert_allclose(box_iou_aligned(comparisons, reference), expected)


def test_aligned_iou_handles_zero_union_without_nan() -> None:
    zero_boxes = np.array([[1.0, 1.0, 1.0, 1.0]])

    iou = box_iou_aligned(zero_boxes, zero_boxes)

    np.testing.assert_array_equal(iou, [0.0])
    assert np.all(np.isfinite(iou))


def test_pairwise_iou_has_one_row_and_column_per_input_box() -> None:
    boxes_a = np.array([[0, 0, 4, 4], [5, 5, 7, 7]], dtype=float)
    boxes_b = np.array(
        [[0, 0, 4, 4], [2, 0, 6, 4], [8, 8, 9, 9]], dtype=float
    )

    result = pairwise_box_iou(boxes_a, boxes_b)

    assert result.shape == (2, 3)
    np.testing.assert_allclose(result[0], [1.0, 1.0 / 3.0, 0.0])
    np.testing.assert_allclose(result[1], [0.0, 0.0, 0.0])


def test_generate_anchors_preserves_area_shape_and_order() -> None:
    anchors = generate_anchors_at_location(
        32, 32, scales=[16, 32], aspect_ratios=[0.5, 1.0, 2.0]
    )

    assert anchors.shape == (6, 4)
    sizes = anchors[:, 2:] - anchors[:, :2]
    np.testing.assert_allclose(
        sizes[:, 0] * sizes[:, 1], [256] * 3 + [1024] * 3
    )
    np.testing.assert_allclose(
        anchors[[1, 4]], [[24, 24, 40, 40], [16, 16, 48, 48]]
    )
    np.testing.assert_allclose(
        (anchors[:, :2] + anchors[:, 2:]) / 2,
        np.tile([32, 32], (len(anchors), 1)),
    )


@pytest.mark.parametrize(
    "center_x,center_y,scales,ratios,error_type",
    [
        (np.nan, 0, [16], [1], ValueError),
        (0, True, [16], [1], TypeError),
        (0, 0, [], [1], ValueError),
        (0, 0, [0], [1], ValueError),
        (0, 0, [16], [-1], ValueError),
        (0, 0, [[16]], [1], ValueError),
    ],
)
def test_generate_anchors_rejects_invalid_parameters(
    center_x: object,
    center_y: object,
    scales: object,
    ratios: object,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        generate_anchors_at_location(  # type: ignore[arg-type]
            center_x, center_y, scales, ratios
        )


def test_assign_anchors_returns_labels_ious_and_target_indices() -> None:
    anchors = np.array(
        [
            [20, 20, 44, 44],
            [18, 18, 46, 46],
            [10, 10, 30, 30],
            [0, 0, 10, 10],
            [15, 15, 40, 40],
        ],
        dtype=float,
    )
    targets = np.array([[20, 20, 44, 44]], dtype=float)

    labels, best_iou, best_target = assign_anchors_to_targets(anchors, targets)

    np.testing.assert_array_equal(labels, [1, 1, 0, 0, -1])
    np.testing.assert_allclose(
        best_iou, [1.0, 576 / 784, 100 / 876, 0.0, 400 / 801]
    )
    np.testing.assert_array_equal(best_target, [0, 0, 0, 0, 0])


def test_assign_anchors_selects_best_of_multiple_targets() -> None:
    anchors = np.array(
        [[0, 0, 4, 4], [10, 10, 14, 14], [20, 20, 22, 22]], dtype=float
    )
    targets = np.array([[10, 10, 14, 14], [0, 0, 4, 4]], dtype=float)

    labels, best_iou, best_target = assign_anchors_to_targets(anchors, targets)

    np.testing.assert_array_equal(labels, [1, 1, 0])
    np.testing.assert_array_equal(best_iou, [1.0, 1.0, 0.0])
    np.testing.assert_array_equal(best_target, [1, 0, 0])


def test_assign_anchors_handles_empty_anchors_and_targets() -> None:
    empty = np.empty((0, 4))
    one_anchor = np.array([[0, 0, 4, 4]], dtype=float)

    labels, best_iou, best_target = assign_anchors_to_targets(
        one_anchor, empty
    )
    np.testing.assert_array_equal(labels, [0])
    np.testing.assert_array_equal(best_iou, [0.0])
    np.testing.assert_array_equal(best_target, [-1])

    labels, best_iou, best_target = assign_anchors_to_targets(
        empty, one_anchor
    )
    assert labels.shape == best_iou.shape == best_target.shape == (0,)


def test_assign_anchors_rejects_invalid_thresholds_and_shapes() -> None:
    anchors = np.array([[0, 0, 4, 4]], dtype=float)
    targets = anchors.copy()

    with pytest.raises(ValueError, match="must not exceed"):
        assign_anchors_to_targets(
            anchors,
            targets,
            positive_iou_threshold=0.4,
            negative_iou_threshold=0.5,
        )
    with pytest.raises(ValueError, match="between 0 and 1"):
        assign_anchors_to_targets(anchors, targets, 1.1, 0.3)
    with pytest.raises(ValueError, match="shape"):
        assign_anchors_to_targets(np.ones(4), targets)


@pytest.mark.parametrize(
    "bad_boxes",
    [
        np.ones((2, 3)),
        np.array([0, 0, np.nan, 1]),
        np.array(["0", "0", "1", "1"]),
        np.array([0 + 1j, 0, 1, 1]),
    ],
)
def test_box_functions_reject_invalid_arrays(bad_boxes: np.ndarray) -> None:
    with pytest.raises((TypeError, ValueError)):
        box_area_xyxy(bad_boxes)


def test_aligned_iou_rejects_nonbroadcastable_leading_dimensions() -> None:
    with pytest.raises(ValueError, match="broadcastable"):
        box_iou_aligned(np.ones((2, 4)), np.ones((3, 4)))


def test_pairwise_iou_requires_two_dimensional_box_collections() -> None:
    with pytest.raises(ValueError, match="shape"):
        pairwise_box_iou(np.ones(4), np.ones((2, 4)))


@pytest.mark.parametrize("height,width", [(0, 10), (10, -1), (np.inf, 10)])
def test_clip_boxes_rejects_invalid_image_size(height: float, width: float) -> None:
    with pytest.raises(ValueError):
        clip_boxes_xyxy(np.ones(4), height, width)


def _nms_candidates() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    boxes = np.array(
        [
            [10.0, 10.0, 55.0, 52.0],
            [12.0, 12.0, 56.0, 50.0],
            [11.0, 11.0, 54.0, 51.0],
            [68.0, 28.0, 110.0, 72.0],
            [70.0, 30.0, 108.0, 70.0],
        ]
    )
    scores = np.array([0.855, 0.765, 0.675, 0.680, 0.585])
    labels = np.array([0, 0, 1, 1, 1])
    return boxes, scores, labels


def test_class_agnostic_nms_suppresses_overlapping_different_classes() -> None:
    boxes, scores, _ = _nms_candidates()

    kept = non_maximum_suppression(boxes, scores, iou_threshold=0.5)

    np.testing.assert_array_equal(kept, [0, 3])


def test_classwise_nms_keeps_overlapping_different_classes() -> None:
    boxes, scores, labels = _nms_candidates()

    kept = classwise_non_maximum_suppression(
        boxes, scores, labels, iou_threshold=0.5
    )

    np.testing.assert_array_equal(kept, [0, 3, 2])


def test_nms_keeps_iou_equal_to_threshold() -> None:
    boxes = np.array([[0, 0, 2, 2], [1, 0, 3, 2]], dtype=float)
    scores = np.array([0.9, 0.8])
    threshold = 1.0 / 3.0

    kept = non_maximum_suppression(boxes, scores, threshold)

    np.testing.assert_array_equal(kept, [0, 1])


def test_nms_uses_original_order_to_break_equal_score_ties() -> None:
    boxes = np.array(
        [[0, 0, 4, 4], [1, 0, 5, 4], [10, 10, 12, 12]], dtype=float
    )
    scores = np.array([0.8, 0.8, 0.7])

    kept = non_maximum_suppression(boxes, scores, iou_threshold=0.5)

    np.testing.assert_array_equal(kept, [0, 2])


def test_nms_handles_empty_candidates() -> None:
    boxes = np.empty((0, 4))
    scores = np.empty(0)
    labels = np.empty(0, dtype=np.int64)

    assert non_maximum_suppression(boxes, scores, 0.5).shape == (0,)
    assert classwise_non_maximum_suppression(
        boxes, scores, labels, 0.5
    ).shape == (0,)


@pytest.mark.parametrize("threshold", [-0.1, 1.1, np.nan, True])
def test_nms_rejects_invalid_iou_threshold(threshold: object) -> None:
    boxes, scores, _ = _nms_candidates()

    with pytest.raises((TypeError, ValueError)):
        non_maximum_suppression(boxes, scores, threshold)  # type: ignore[arg-type]


def test_nms_rejects_mismatched_or_invalid_attributes() -> None:
    boxes, scores, labels = _nms_candidates()
    with pytest.raises(ValueError, match="scores"):
        non_maximum_suppression(boxes, scores[:-1], 0.5)
    with pytest.raises(ValueError, match="labels"):
        classwise_non_maximum_suppression(boxes, scores, labels[:-1], 0.5)
    with pytest.raises(TypeError, match="integers"):
        classwise_non_maximum_suppression(
            boxes, scores, labels.astype(float), 0.5
        )


def _evaluation_example() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    target_boxes = np.array(
        [[10, 10, 50, 50], [65, 15, 105, 55], [20, 58, 55, 78]],
        dtype=float,
    )
    target_labels = np.array([0, 1, 0])
    predicted_boxes = np.array(
        [
            [9, 9, 51, 51],
            [12, 12, 49, 49],
            [66, 16, 104, 54],
            [20, 58, 55, 78],
            [18, 57, 56, 79],
            [75, 60, 105, 78],
            [0, 60, 18, 78],
        ],
        dtype=float,
    )
    predicted_scores = np.array([0.95, 0.90, 0.85, 0.75, 0.70, 0.55, 0.40])
    predicted_labels = np.array([0, 0, 1, 1, 0, 0, 1])
    return (
        predicted_boxes,
        predicted_scores,
        predicted_labels,
        target_boxes,
        target_labels,
    )


def test_match_detections_is_ranked_class_aware_and_one_to_one() -> None:
    inputs = _evaluation_example()

    order, true_positive, matched_target = match_detections(
        *inputs, iou_threshold=0.5
    )

    np.testing.assert_array_equal(order, np.arange(7))
    np.testing.assert_array_equal(
        true_positive, [True, False, True, False, True, False, False]
    )
    np.testing.assert_array_equal(matched_target, [0, -1, 1, -1, 2, -1, -1])


def test_match_detections_returns_decisions_in_score_order() -> None:
    boxes = np.array([[10, 10, 20, 20], [0, 0, 5, 5]], dtype=float)
    scores = np.array([0.2, 0.9])
    labels = np.array([0, 0])
    target_boxes = np.array([[10, 10, 20, 20]], dtype=float)
    target_labels = np.array([0])

    order, true_positive, matched_target = match_detections(
        boxes, scores, labels, target_boxes, target_labels, 0.5
    )

    np.testing.assert_array_equal(order, [1, 0])
    np.testing.assert_array_equal(true_positive, [False, True])
    np.testing.assert_array_equal(matched_target, [-1, 0])


def test_match_detections_handles_empty_predictions_and_targets() -> None:
    empty_boxes = np.empty((0, 4))
    empty_scores = np.empty(0)
    empty_labels = np.empty(0, dtype=np.int64)

    order, true_positive, matched_target = match_detections(
        empty_boxes,
        empty_scores,
        empty_labels,
        empty_boxes,
        empty_labels,
        0.5,
    )

    assert order.shape == true_positive.shape == matched_target.shape == (0,)


def test_precision_recall_and_interpolated_ap_match_notebook_example() -> None:
    true_positive = np.array([True, False, True, False, True, False, False])

    precision, recall = precision_recall_from_matches(true_positive, 3)
    average_precision, envelope_recall, envelope_precision = (
        interpolated_average_precision(precision, recall)
    )

    np.testing.assert_allclose(
        precision, [1, 1 / 2, 2 / 3, 1 / 2, 3 / 5, 1 / 2, 3 / 7]
    )
    np.testing.assert_allclose(
        recall, [1 / 3, 1 / 3, 2 / 3, 2 / 3, 1, 1, 1]
    )
    np.testing.assert_allclose(average_precision, 34 / 45)
    assert envelope_recall.shape == envelope_precision.shape == (9,)


def test_interpolated_ap_does_not_credit_unreachable_recall() -> None:
    true_positive = np.array([True, False, True, False, False, False, False])
    precision, recall = precision_recall_from_matches(true_positive, 3)

    average_precision, _, _ = interpolated_average_precision(precision, recall)

    np.testing.assert_allclose(average_precision, 5 / 9)


def test_interpolated_ap_handles_no_predictions() -> None:
    average_precision, recall, precision = interpolated_average_precision(
        np.empty(0), np.empty(0)
    )

    assert average_precision == 0.0
    np.testing.assert_array_equal(recall, [0.0, 1.0])
    np.testing.assert_array_equal(precision, [0.0, 0.0])


def test_evaluation_helpers_reject_invalid_inputs() -> None:
    boxes, scores, labels, target_boxes, target_labels = _evaluation_example()
    with pytest.raises(ValueError, match="predicted_scores|scores"):
        match_detections(
            boxes, scores[:-1], labels, target_boxes, target_labels, 0.5
        )
    with pytest.raises(TypeError, match="predicted_labels"):
        match_detections(
            boxes, scores, labels.astype(float), target_boxes, target_labels, 0.5
        )
    with pytest.raises(TypeError, match="Boolean"):
        precision_recall_from_matches(np.array([1, 0]), 1)
    with pytest.raises(ValueError, match="positive"):
        precision_recall_from_matches(np.array([True]), 0)
    with pytest.raises(ValueError, match="nondecreasing"):
        interpolated_average_precision([1.0, 0.5], [0.8, 0.4])
    with pytest.raises(ValueError, match="same shape"):
        interpolated_average_precision([1.0], [0.5, 1.0])
