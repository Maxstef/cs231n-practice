"""Small reusable helpers shared by model-training loops."""

import numpy as np


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
