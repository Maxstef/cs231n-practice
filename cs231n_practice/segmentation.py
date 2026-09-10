"""Reusable NumPy losses and metrics for semantic segmentation."""

import numpy as np


def _class_count(value: int) -> int:
    """Return a strictly positive integer number of classes."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError("number_of_classes must be an integer")
    value = int(value)
    if value <= 0:
        raise ValueError("number_of_classes must be positive")
    return value


def _integer_array(value: np.ndarray, *, name: str) -> np.ndarray:
    """Return an array containing integer class IDs."""
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.integer) or np.issubdtype(
        array.dtype, np.bool_
    ):
        raise TypeError(f"{name} must contain integer class IDs")
    return array


def semantic_cross_entropy(
    logits: np.ndarray,
    targets: np.ndarray,
    ignore_index: int = -1,
) -> tuple[float, np.ndarray]:
    """Return mean valid-pixel cross-entropy and its logit gradient.

    Args:
        logits: Class scores with shape ``(N, C, H, W)``.
        targets: Integer class IDs with shape ``(N, H, W)``.
        ignore_index: Target value excluded from the loss and gradient.

    Returns:
        The scalar loss and ``dlogits`` with the same shape as ``logits``.
        Both are averaged over valid pixels, and ignored positions have zero
        gradient.
    """
    logits = np.asarray(logits)
    targets = _integer_array(targets, name="targets")
    if not np.issubdtype(logits.dtype, np.number) or np.issubdtype(
        logits.dtype, np.complexfloating
    ):
        raise TypeError("logits must contain real numeric values")
    if logits.ndim != 4:
        raise ValueError("logits must have shape (N, C, H, W)")
    if logits.shape[1] == 0 or targets.shape != (
        logits.shape[0],
        logits.shape[2],
        logits.shape[3],
    ):
        raise ValueError("targets must match the (N, H, W) logit axes")
    if not np.all(np.isfinite(logits)):
        raise ValueError("logits must contain finite values")
    if isinstance(ignore_index, (bool, np.bool_)) or not isinstance(
        ignore_index, (int, np.integer)
    ):
        raise TypeError("ignore_index must be an integer")

    valid = targets != ignore_index
    if not np.any(valid):
        raise ValueError("at least one target pixel must be valid")

    valid_targets = targets[valid]
    number_of_classes = logits.shape[1]
    if np.any(valid_targets < 0) or np.any(
        valid_targets >= number_of_classes
    ):
        raise ValueError("targets contain a class index outside [0, C)")

    calculation_dtype = np.result_type(logits.dtype, np.float32)
    logits_last = np.moveaxis(
        logits.astype(calculation_dtype, copy=False), 1, -1
    )
    valid_logits = logits_last[valid]
    shifted = valid_logits - np.max(valid_logits, axis=1, keepdims=True)
    exp_shifted = np.exp(shifted)
    normalizers = np.sum(exp_shifted, axis=1, keepdims=True)
    probabilities = exp_shifted / normalizers

    rows = np.arange(valid_targets.size)
    losses = np.log(normalizers[:, 0]) - shifted[rows, valid_targets]
    loss = np.mean(losses)

    dvalid_logits = probabilities
    dvalid_logits[rows, valid_targets] -= 1.0
    dvalid_logits /= valid_targets.size
    dlogits_last = np.zeros(logits_last.shape, dtype=calculation_dtype)
    dlogits_last[valid] = dvalid_logits
    dlogits = np.moveaxis(dlogits_last, -1, 1)
    return float(loss), dlogits


def segmentation_confusion_matrix(
    targets: np.ndarray,
    predictions: np.ndarray,
    number_of_classes: int,
    ignore_index: int = -1,
) -> np.ndarray:
    """Count valid pixels for every ``(target, prediction)`` class pair."""
    targets = _integer_array(targets, name="targets")
    predictions = _integer_array(predictions, name="predictions")
    number_of_classes = _class_count(number_of_classes)
    if targets.shape != predictions.shape:
        raise ValueError("targets and predictions must have the same shape")
    if targets.ndim == 0:
        raise ValueError("targets and predictions must have at least one axis")
    if isinstance(ignore_index, (bool, np.bool_)) or not isinstance(
        ignore_index, (int, np.integer)
    ):
        raise TypeError("ignore_index must be an integer")

    valid = targets != ignore_index
    valid_targets = targets[valid]
    valid_predictions = predictions[valid]
    if np.any(valid_targets < 0) or np.any(
        valid_targets >= number_of_classes
    ):
        raise ValueError("targets contain a class index outside the valid range")
    if np.any(valid_predictions < 0) or np.any(
        valid_predictions >= number_of_classes
    ):
        raise ValueError(
            "predictions contain a class index outside the valid range"
        )

    combined = valid_targets * number_of_classes + valid_predictions
    counts = np.bincount(
        combined, minlength=number_of_classes * number_of_classes
    )
    return counts.reshape(number_of_classes, number_of_classes)


def segmentation_metrics(
    confusion: np.ndarray,
) -> tuple[float, np.ndarray, float]:
    """Return pixel accuracy, per-class IoU, and mean class IoU.

    A class absent from both targets and predictions has an undefined IoU and
    receives ``nan``. Mean IoU averages only the classes with a defined union.
    """
    confusion = np.asarray(confusion)
    if not np.issubdtype(confusion.dtype, np.number) or np.issubdtype(
        confusion.dtype, np.complexfloating
    ):
        raise TypeError("confusion must contain real numeric counts")
    if confusion.ndim != 2 or confusion.shape[0] != confusion.shape[1]:
        raise ValueError("confusion must be a square matrix")
    if confusion.shape[0] == 0:
        raise ValueError("confusion must contain at least one class")
    if not np.all(np.isfinite(confusion)) or np.any(confusion < 0):
        raise ValueError("confusion must contain finite nonnegative counts")

    true_positive = np.diag(confusion)
    target_count = np.sum(confusion, axis=1)
    predicted_count = np.sum(confusion, axis=0)
    total = np.sum(target_count)
    pixel_accuracy = float(true_positive.sum() / total) if total > 0 else np.nan

    union = target_count + predicted_count - true_positive
    class_iou = np.divide(
        true_positive,
        union,
        out=np.full(union.shape, np.nan, dtype=float),
        where=union > 0,
    )
    valid_iou = class_iou[~np.isnan(class_iou)]
    mean_iou = float(np.mean(valid_iou)) if valid_iou.size else np.nan
    return pixel_accuracy, class_iou, mean_iou
