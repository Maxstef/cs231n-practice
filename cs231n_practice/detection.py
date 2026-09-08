"""Reusable bounding-box geometry for object-detection practice."""

import numpy as np


def _box_array(
    boxes: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    """Return a finite real array whose final dimension contains four values."""
    array = np.asarray(boxes)
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.complexfloating
    ):
        raise TypeError(f"{name} must contain real numeric values")
    if array.ndim < 1 or array.shape[-1] != 4:
        raise ValueError(f"{name} must have shape (..., 4)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array


def _positive_image_size(value: float, *, name: str) -> float:
    """Return a finite positive image dimension."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.number)
    ):
        raise TypeError(f"{name} must be numeric")
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert ``(..., x, y, width, height)`` boxes to corner format."""
    boxes = _box_array(boxes, name="boxes")
    x, y, width, height = np.moveaxis(boxes, -1, 0)
    return np.stack((x, y, x + width, y + height), axis=-1)


def cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert ``(..., center_x, center_y, width, height)`` to corners."""
    boxes = _box_array(boxes, name="boxes")
    calculation_dtype = np.result_type(boxes.dtype, np.float32)
    boxes = boxes.astype(calculation_dtype, copy=False)
    center_x, center_y, width, height = np.moveaxis(boxes, -1, 0)
    half_width = width / 2
    half_height = height / 2
    return np.stack(
        (
            center_x - half_width,
            center_y - half_height,
            center_x + half_width,
            center_y + half_height,
        ),
        axis=-1,
    )


def xyxy_to_cxcywh(boxes: np.ndarray) -> np.ndarray:
    """Convert ``(..., x1, y1, x2, y2)`` boxes to center/size format."""
    boxes = _box_array(boxes, name="boxes")
    calculation_dtype = np.result_type(boxes.dtype, np.float32)
    boxes = boxes.astype(calculation_dtype, copy=False)
    x1, y1, x2, y2 = np.moveaxis(boxes, -1, 0)
    return np.stack(
        ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1),
        axis=-1,
    )


def clip_boxes_xyxy(
    boxes: np.ndarray,
    image_height: float,
    image_width: float,
) -> np.ndarray:
    """Return corner boxes clipped to continuous image boundaries.

    The input is not modified. Coordinates follow the half-open convention,
    so x boundaries are clipped to ``[0, image_width]`` and y boundaries to
    ``[0, image_height]``.
    """
    boxes = _box_array(boxes, name="boxes")
    image_height = _positive_image_size(image_height, name="image_height")
    image_width = _positive_image_size(image_width, name="image_width")
    calculation_dtype = np.result_type(boxes.dtype, np.float32)
    clipped = boxes.astype(calculation_dtype, copy=True)
    clipped[..., 0::2] = np.clip(clipped[..., 0::2], 0, image_width)
    clipped[..., 1::2] = np.clip(clipped[..., 1::2], 0, image_height)
    return clipped


def valid_boxes_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Return whether every corner box has strictly positive width and height."""
    boxes = _box_array(boxes, name="boxes")
    return (boxes[..., 2] > boxes[..., 0]) & (
        boxes[..., 3] > boxes[..., 1]
    )


def box_area_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Return nonnegative areas for corner-format boxes."""
    boxes = _box_array(boxes, name="boxes")
    width = np.maximum(boxes[..., 2] - boxes[..., 0], 0)
    height = np.maximum(boxes[..., 3] - boxes[..., 1], 0)
    return width * height


def box_iou_aligned(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Return IoU for corresponding or broadcast-aligned corner boxes.

    Leading dimensions must obey NumPy broadcasting. For example, ``(M, 4)``
    against ``(M, 4)`` compares corresponding rows, while ``(4,)`` against
    ``(M, 4)`` compares one box with every row.
    """
    boxes_a = _box_array(boxes_a, name="boxes_a")
    boxes_b = _box_array(boxes_b, name="boxes_b")
    try:
        intersection_top_left = np.maximum(
            boxes_a[..., :2], boxes_b[..., :2]
        )
        intersection_bottom_right = np.minimum(
            boxes_a[..., 2:], boxes_b[..., 2:]
        )
    except ValueError as error:
        raise ValueError(
            "boxes_a and boxes_b leading dimensions are not broadcastable"
        ) from error

    intersection_boxes = np.concatenate(
        (intersection_top_left, intersection_bottom_right), axis=-1
    )
    intersection_area = box_area_xyxy(intersection_boxes)
    union_area = (
        box_area_xyxy(boxes_a)
        + box_area_xyxy(boxes_b)
        - intersection_area
    )
    return np.divide(
        intersection_area,
        union_area,
        out=np.zeros_like(union_area, dtype=float),
        where=union_area > 0,
    )


def pairwise_box_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Return all pairwise IoUs between two ``(M, 4)`` and ``(K, 4)`` arrays."""
    boxes_a = _box_array(boxes_a, name="boxes_a")
    boxes_b = _box_array(boxes_b, name="boxes_b")
    if boxes_a.ndim != 2 or boxes_b.ndim != 2:
        raise ValueError("boxes_a and boxes_b must each have shape (N, 4)")
    return box_iou_aligned(boxes_a[:, None, :], boxes_b[None, :, :])


def _detection_scores(scores: np.ndarray, *, expected_length: int) -> np.ndarray:
    """Return one finite real confidence score per candidate."""
    scores = np.asarray(scores)
    if not np.issubdtype(scores.dtype, np.number) or np.issubdtype(
        scores.dtype, np.complexfloating
    ):
        raise TypeError("scores must contain real numeric values")
    if scores.shape != (expected_length,):
        raise ValueError(f"scores must have shape ({expected_length},)")
    if not np.all(np.isfinite(scores)):
        raise ValueError("scores must contain finite values")
    return scores


def _iou_threshold(value: float) -> float:
    """Return an IoU threshold in the closed interval from zero to one."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.number)
    ):
        raise TypeError("iou_threshold must be numeric")
    value = float(value)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("iou_threshold must be between 0 and 1")
    return value


def non_maximum_suppression(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
) -> np.ndarray:
    """Return indices retained by greedy class-agnostic NMS.

    Candidates are considered from highest to lowest score. After a candidate
    is kept, lower-scoring candidates with IoU strictly greater than
    ``iou_threshold`` are suppressed. An IoU exactly equal to the threshold is
    retained. Returned indices refer to the original inputs and follow
    descending selection order. Equal scores retain their original order.
    """
    boxes = _box_array(boxes, name="boxes")
    if boxes.ndim != 2:
        raise ValueError("boxes must have shape (N, 4)")
    scores = _detection_scores(scores, expected_length=len(boxes))
    iou_threshold = _iou_threshold(iou_threshold)

    # Stable sorting makes equal-score behavior deterministic.
    remaining = np.argsort(-scores, kind="stable")
    kept: list[int] = []
    while remaining.size > 0:
        current = int(remaining[0])
        kept.append(current)
        other_indices = remaining[1:]
        if other_indices.size == 0:
            break
        overlaps = box_iou_aligned(boxes[current], boxes[other_indices])
        remaining = other_indices[overlaps <= iou_threshold]
    return np.asarray(kept, dtype=np.int64)


def classwise_non_maximum_suppression(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    iou_threshold: float,
) -> np.ndarray:
    """Run NMS independently per integer class and return original indices.

    Results from all classes are merged into descending score order. Equal
    scores retain their original input order.
    """
    boxes = _box_array(boxes, name="boxes")
    if boxes.ndim != 2:
        raise ValueError("boxes must have shape (N, 4)")
    scores = _detection_scores(scores, expected_length=len(boxes))
    labels = np.asarray(labels)
    if labels.shape != (len(boxes),):
        raise ValueError(f"labels must have shape ({len(boxes)},)")
    if not np.issubdtype(labels.dtype, np.integer) or np.issubdtype(
        labels.dtype, np.bool_
    ):
        raise TypeError("labels must contain integers")
    iou_threshold = _iou_threshold(iou_threshold)

    kept: list[int] = []
    for class_label in np.unique(labels):
        class_indices = np.flatnonzero(labels == class_label)
        local_keep = non_maximum_suppression(
            boxes[class_indices], scores[class_indices], iou_threshold
        )
        kept.extend(class_indices[local_keep].tolist())

    kept_array = np.asarray(kept, dtype=np.int64)
    if kept_array.size == 0:
        return kept_array
    # Sort original candidate indices first so stable score sorting uses input
    # order as the deterministic tie-breaker across different classes.
    kept_array.sort()
    return kept_array[np.argsort(-scores[kept_array], kind="stable")]


__all__ = [
    "box_area_xyxy",
    "box_iou_aligned",
    "classwise_non_maximum_suppression",
    "clip_boxes_xyxy",
    "cxcywh_to_xyxy",
    "non_maximum_suppression",
    "pairwise_box_iou",
    "valid_boxes_xyxy",
    "xywh_to_xyxy",
    "xyxy_to_cxcywh",
]
