"""Small reusable helpers shared by model-training loops."""

from collections.abc import Mapping

import numpy as np


def clip_gradients_by_global_norm(
    gradients: Mapping[str, np.ndarray],
    max_norm: float,
) -> tuple[dict[str, np.ndarray], float]:
    """Return gradient copies clipped using one shared global-norm scale.

    A common scale preserves the direction of the combined gradient vector.
    The returned norm is the original value before clipping, and the input
    arrays are never mutated.
    """
    if not isinstance(gradients, Mapping) or not gradients:
        raise TypeError("gradients must be a nonempty mapping")
    if isinstance(max_norm, (bool, np.bool_)) or not isinstance(
        max_norm, (int, float, np.number)
    ):
        raise TypeError("max_norm must be numeric")
    max_norm = float(max_norm)
    if not np.isfinite(max_norm) or max_norm <= 0.0:
        raise ValueError("max_norm must be finite and positive")

    checked: dict[str, np.ndarray] = {}
    squared_norm = 0.0
    for name, gradient in gradients.items():
        array = np.asarray(gradient)
        if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
            array.dtype, np.complexfloating
        ):
            raise TypeError("gradient arrays must contain real numbers")
        if array.size == 0 or not np.all(np.isfinite(array)):
            raise ValueError("gradient arrays must be nonempty and finite")
        checked[name] = array
        squared_norm += float(np.sum(array.astype(np.float64) ** 2))

    global_norm = float(np.sqrt(squared_norm))
    scale = 1.0 if global_norm <= max_norm else max_norm / global_norm
    clipped = {name: gradient * scale for name, gradient in checked.items()}
    return clipped, global_norm


def sample_minibatch(
    features: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
    generator: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample matching feature and target rows without replacement."""
    features = np.asarray(features)
    targets = np.asarray(targets)
    if features.ndim == 0 or targets.ndim == 0:
        raise ValueError("features and targets must have an example dimension")
    if features.shape[0] != targets.shape[0]:
        raise ValueError("features and targets must contain equal example counts")
    if isinstance(batch_size, (bool, np.bool_)) or not isinstance(
        batch_size, (int, np.integer)
    ):
        raise TypeError("batch_size must be an integer")
    if batch_size <= 0 or batch_size > features.shape[0]:
        raise ValueError("batch_size must be between 1 and the example count")
    if not isinstance(generator, np.random.Generator):
        raise TypeError("generator must be a numpy.random.Generator")

    indices = generator.choice(features.shape[0], size=batch_size, replace=False)
    return features[indices], targets[indices]
