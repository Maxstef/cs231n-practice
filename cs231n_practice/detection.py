"""Reusable geometry, post-processing, and evaluation for object detection."""

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


def _finite_scalar(value: float, *, name: str) -> float:
    """Return a finite real scalar, excluding Boolean values."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.number)
    ):
        raise TypeError(f"{name} must be numeric")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _positive_values(values: np.ndarray, *, name: str) -> np.ndarray:
    """Return a nonempty one-dimensional array of finite positive values."""
    values = np.asarray(values)
    if not np.issubdtype(values.dtype, np.number) or np.issubdtype(
        values.dtype, np.complexfloating
    ):
        raise TypeError(f"{name} must contain real numeric values")
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError(f"{name} must contain finite positive values")
    return values


def generate_anchors_at_location(
    center_x: float,
    center_y: float,
    scales: np.ndarray,
    aspect_ratios: np.ndarray,
) -> np.ndarray:
    """Generate corner-format anchors centered at one spatial location.

    ``scale`` is the square root of anchor area and ``aspect_ratio`` is
    width divided by height. Anchors are returned in scale-major order: every
    aspect ratio for the first scale, then every ratio for the next scale.
    """
    center_x = _finite_scalar(center_x, name="center_x")
    center_y = _finite_scalar(center_y, name="center_y")
    scales = _positive_values(scales, name="scales")
    aspect_ratios = _positive_values(aspect_ratios, name="aspect_ratios")

    scale_grid, ratio_grid = np.meshgrid(
        scales, aspect_ratios, indexing="ij"
    )
    widths = scale_grid * np.sqrt(ratio_grid)
    heights = scale_grid / np.sqrt(ratio_grid)
    anchors = np.stack(
        (
            center_x - widths / 2,
            center_y - heights / 2,
            center_x + widths / 2,
            center_y + heights / 2,
        ),
        axis=-1,
    )
    return anchors.reshape(-1, 4)


def assign_anchors_to_targets(
    anchors: np.ndarray,
    target_boxes: np.ndarray,
    positive_iou_threshold: float = 0.5,
    negative_iou_threshold: float = 0.3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assign positive, negative, or ignored labels using best target IoU.

    Returns integer labels (``1`` positive, ``0`` negative, ``-1`` ignored),
    each anchor's best IoU, and its best target index. When no targets exist,
    every anchor is negative with IoU zero and target index ``-1``.

    This intentionally implements only threshold-based assignment. Production
    detectors may additionally force at least one positive anchor per target.
    """
    anchors = _box_array(anchors, name="anchors")
    target_boxes = _box_array(target_boxes, name="target_boxes")
    if anchors.ndim != 2:
        raise ValueError("anchors must have shape (N, 4)")
    if target_boxes.ndim != 2:
        raise ValueError("target_boxes must have shape (M, 4)")
    positive_iou_threshold = _iou_threshold(positive_iou_threshold)
    negative_iou_threshold = _iou_threshold(negative_iou_threshold)
    if negative_iou_threshold > positive_iou_threshold:
        raise ValueError(
            "negative_iou_threshold must not exceed positive_iou_threshold"
        )

    labels = np.full(len(anchors), -1, dtype=np.int64)
    if len(target_boxes) == 0:
        labels.fill(0)
        return (
            labels,
            np.zeros(len(anchors), dtype=float),
            np.full(len(anchors), -1, dtype=np.int64),
        )

    pairwise_ious = pairwise_box_iou(anchors, target_boxes)
    best_target = np.argmax(pairwise_ious, axis=1).astype(np.int64, copy=False)
    best_iou = pairwise_ious[np.arange(len(anchors)), best_target]
    labels[best_iou < negative_iou_threshold] = 0
    labels[best_iou >= positive_iou_threshold] = 1
    return labels, best_iou, best_target


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


def _detection_labels(
    labels: np.ndarray,
    *,
    expected_length: int,
    name: str,
) -> np.ndarray:
    """Return one integer class label per detection or target."""
    labels = np.asarray(labels)
    if labels.shape != (expected_length,):
        raise ValueError(f"{name} must have shape ({expected_length},)")
    if not np.issubdtype(labels.dtype, np.integer) or np.issubdtype(
        labels.dtype, np.bool_
    ):
        raise TypeError(f"{name} must contain integers")
    return labels


def match_detections(
    predicted_boxes: np.ndarray,
    predicted_scores: np.ndarray,
    predicted_labels: np.ndarray,
    target_boxes: np.ndarray,
    target_labels: np.ndarray,
    iou_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Greedily match one image's predictions to ground-truth objects.

    Predictions are processed from highest to lowest confidence. A prediction
    matches the unmatched, same-class target with greatest IoU when that IoU
    is at least ``iou_threshold``. Each target can be matched only once.

    Returns
    -------
    order:
        Original prediction indices in descending score order.
    true_positive:
        Boolean TP decisions in ranked order, not original input order.
    matched_target:
        Target index for each ranked prediction, or ``-1`` for an FP.
    """
    predicted_boxes = _box_array(predicted_boxes, name="predicted_boxes")
    target_boxes = _box_array(target_boxes, name="target_boxes")
    if predicted_boxes.ndim != 2:
        raise ValueError("predicted_boxes must have shape (N, 4)")
    if target_boxes.ndim != 2:
        raise ValueError("target_boxes must have shape (M, 4)")
    predicted_scores = _detection_scores(
        predicted_scores, expected_length=len(predicted_boxes)
    )
    predicted_labels = _detection_labels(
        predicted_labels,
        expected_length=len(predicted_boxes),
        name="predicted_labels",
    )
    target_labels = _detection_labels(
        target_labels,
        expected_length=len(target_boxes),
        name="target_labels",
    )
    iou_threshold = _iou_threshold(iou_threshold)

    order = np.argsort(-predicted_scores, kind="stable")
    pairwise_ious = pairwise_box_iou(predicted_boxes, target_boxes)
    same_class = predicted_labels[:, None] == target_labels
    target_matched = np.zeros(len(target_boxes), dtype=bool)
    true_positive = np.zeros(len(predicted_boxes), dtype=bool)
    matched_target = np.full(len(predicted_boxes), -1, dtype=np.int64)

    for rank, prediction_index in enumerate(order):
        eligible_targets = np.flatnonzero(
            same_class[prediction_index] & ~target_matched
        )
        if eligible_targets.size == 0:
            continue

        eligible_ious = pairwise_ious[prediction_index, eligible_targets]
        best_target = eligible_targets[np.argmax(eligible_ious)]
        if pairwise_ious[prediction_index, best_target] >= iou_threshold:
            true_positive[rank] = True
            matched_target[rank] = best_target
            target_matched[best_target] = True

    return order, true_positive, matched_target


def precision_recall_from_matches(
    true_positive: np.ndarray,
    number_of_targets: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return precision and recall after every ranked prediction.

    ``true_positive`` must already follow descending confidence order.
    ``number_of_targets`` is the number of ground-truth objects for the class
    and image collection being evaluated.
    """
    true_positive = np.asarray(true_positive)
    if true_positive.ndim != 1:
        raise ValueError("true_positive must be a one-dimensional array")
    if true_positive.dtype != np.bool_:
        raise TypeError("true_positive must contain Boolean values")
    if isinstance(number_of_targets, (bool, np.bool_)) or not isinstance(
        number_of_targets, (int, np.integer)
    ):
        raise TypeError("number_of_targets must be an integer")
    if number_of_targets <= 0:
        raise ValueError("number_of_targets must be positive")

    cumulative_true_positive = np.cumsum(true_positive)
    prediction_count = np.arange(1, len(true_positive) + 1)
    precision = cumulative_true_positive / prediction_count
    recall = cumulative_true_positive / number_of_targets
    return precision, recall


def interpolated_average_precision(
    precision: np.ndarray,
    recall: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return all-point interpolated AP and its precision envelope.

    Recall must be nondecreasing. Sentinel endpoints are included in the
    returned recall and precision arrays so they can be plotted directly.
    """
    precision = np.asarray(precision, dtype=float)
    recall = np.asarray(recall, dtype=float)
    if precision.ndim != 1 or recall.ndim != 1:
        raise ValueError("precision and recall must be one-dimensional")
    if precision.shape != recall.shape:
        raise ValueError("precision and recall must have the same shape")
    if not np.all(np.isfinite(precision)) or not np.all(np.isfinite(recall)):
        raise ValueError("precision and recall must contain finite values")
    if np.any((precision < 0) | (precision > 1)):
        raise ValueError("precision values must be between 0 and 1")
    if np.any((recall < 0) | (recall > 1)):
        raise ValueError("recall values must be between 0 and 1")
    if np.any(np.diff(recall) < 0):
        raise ValueError("recall must be nondecreasing")

    extended_precision = np.r_[0.0, precision, 0.0]
    extended_recall = np.r_[0.0, recall, 1.0]

    # At each recall, retain the best precision available at that recall or
    # any greater recall. This creates a non-increasing precision envelope.
    for index in range(len(extended_precision) - 2, -1, -1):
        extended_precision[index] = max(
            extended_precision[index], extended_precision[index + 1]
        )

    recall_increase = np.diff(extended_recall)
    changing_recall = recall_increase > 0
    # Use the envelope height at the newly reached (right-hand) recall point.
    # The final sentinel therefore contributes zero when recall never reaches 1.
    average_precision = np.sum(
        recall_increase[changing_recall]
        * extended_precision[1:][changing_recall]
    )
    return float(average_precision), extended_recall, extended_precision


__all__ = [
    "assign_anchors_to_targets",
    "box_area_xyxy",
    "box_iou_aligned",
    "classwise_non_maximum_suppression",
    "clip_boxes_xyxy",
    "cxcywh_to_xyxy",
    "generate_anchors_at_location",
    "interpolated_average_precision",
    "match_detections",
    "non_maximum_suppression",
    "pairwise_box_iou",
    "precision_recall_from_matches",
    "valid_boxes_xyxy",
    "xywh_to_xyxy",
    "xyxy_to_cxcywh",
]
