import numpy as np
import pytest

from cs231n_practice.detection import (
    box_area_xyxy,
    box_iou_aligned,
    clip_boxes_xyxy,
    cxcywh_to_xyxy,
    pairwise_box_iou,
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
