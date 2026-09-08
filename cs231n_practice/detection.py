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


__all__ = [
    "box_area_xyxy",
    "box_iou_aligned",
    "clip_boxes_xyxy",
    "cxcywh_to_xyxy",
    "pairwise_box_iou",
    "valid_boxes_xyxy",
    "xywh_to_xyxy",
    "xyxy_to_cxcywh",
]
