"""Reusable data and distance helpers for introductory point-cloud work."""

from numbers import Integral, Real

import numpy as np
import torch

from cs231n_practice.geometry import (
    rotation_x,
    rotation_y,
    rotation_z,
    sample_sphere_surface,
)


def _positive_integer(value: int, *, name: str) -> int:
    """Validate a positive integer without accepting Boolean values."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _nonnegative_real(value: float, *, name: str) -> float:
    """Validate a finite nonnegative real number."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be numeric")
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def _generator(value: np.random.Generator | None) -> np.random.Generator:
    """Return a supplied NumPy generator or create a fresh one."""
    if value is None:
        return np.random.default_rng()
    if not isinstance(value, np.random.Generator):
        raise TypeError("generator must be a numpy.random.Generator")
    return value


def _point_set(points: np.ndarray, *, name: str) -> np.ndarray:
    """Validate a finite, nonempty floating-point array shaped ``(P, 3)``."""
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] != 3:
        raise ValueError(f"{name} must have nonempty shape (P, 3)")
    if not np.issubdtype(points.dtype, np.floating):
        raise TypeError(f"{name} must contain floating-point values")
    if not np.isfinite(points).all():
        raise ValueError(f"{name} must contain only finite values")
    return points


def sample_cube_surface(
    number_points: int,
    *,
    generator: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample points uniformly across the six faces of ``[-1, 1]^3``."""
    number_points = _positive_integer(number_points, name="number_points")
    generator = _generator(generator)
    face_ids = generator.integers(0, 6, size=number_points)
    points = generator.uniform(-1.0, 1.0, size=(number_points, 3))
    fixed_axes = face_ids // 2
    fixed_signs = np.where(face_ids % 2 == 0, -1.0, 1.0)
    points[np.arange(number_points), fixed_axes] = fixed_signs
    return points


def sample_cylinder_surface(
    number_points: int,
    *,
    generator: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample a unit-radius cylinder's side and caps in area proportion.

    The cylinder spans ``z = -1`` to ``z = 1``. Its side has two thirds of
    the complete surface area, while each circular cap has one sixth.
    """
    number_points = _positive_integer(number_points, name="number_points")
    generator = _generator(generator)
    points = np.empty((number_points, 3), dtype=np.float64)
    regions = generator.choice(3, size=number_points, p=(2 / 3, 1 / 6, 1 / 6))
    angles = generator.uniform(0.0, 2.0 * np.pi, size=number_points)

    side = regions == 0
    points[side, 0] = np.cos(angles[side])
    points[side, 1] = np.sin(angles[side])
    points[side, 2] = generator.uniform(-1.0, 1.0, size=side.sum())

    for region, height in ((1, -1.0), (2, 1.0)):
        cap = regions == region
        radii = np.sqrt(generator.uniform(0.0, 1.0, size=cap.sum()))
        points[cap, 0] = radii * np.cos(angles[cap])
        points[cap, 1] = radii * np.sin(angles[cap])
        points[cap, 2] = height
    return points


def normalize_point_cloud(points: np.ndarray) -> np.ndarray:
    """Center one point cloud and scale its farthest point to unit radius."""
    points = _point_set(points, name="points")
    centered = points - points.mean(axis=0, keepdims=True)
    maximum_radius = np.linalg.norm(centered, axis=1).max()
    if maximum_radius == 0.0:
        raise ValueError("points must not all occupy the same location")
    return centered / maximum_radius


def random_rotation(
    *, generator: np.random.Generator | None = None
) -> np.ndarray:
    """Return a random 3D rotation composed from three Euler rotations."""
    generator = _generator(generator)
    angles = generator.uniform(0.0, 2.0 * np.pi, size=3)
    return rotation_z(angles[2]) @ rotation_y(angles[1]) @ rotation_x(angles[0])


def sample_shape(
    label: int,
    number_points: int,
    *,
    generator: np.random.Generator | None = None,
    noise_std: float = 0.015,
) -> np.ndarray:
    """Sample a normalized, randomly rotated sphere, cube, or cylinder.

    Labels 0, 1, and 2 select the sphere, cube, and cylinder respectively.
    Independent Gaussian coordinate noise is added before normalization.
    """
    if isinstance(label, bool) or not isinstance(label, Integral):
        raise TypeError("label must be an integer")
    label = int(label)
    number_points = _positive_integer(number_points, name="number_points")
    noise_std = _nonnegative_real(noise_std, name="noise_std")
    generator = _generator(generator)

    if label == 0:
        points = sample_sphere_surface(number_points, generator=generator)
    elif label == 1:
        points = sample_cube_surface(number_points, generator=generator)
    elif label == 2:
        points = sample_cylinder_surface(number_points, generator=generator)
    else:
        raise ValueError("label must be 0 (sphere), 1 (cube), or 2 (cylinder)")

    points = points @ random_rotation(generator=generator).T
    points += generator.normal(scale=noise_std, size=points.shape)
    return normalize_point_cloud(points)


def make_point_cloud_dataset(
    number_examples: int,
    number_points: int,
    *,
    seed: int | None = None,
    noise_std: float = 0.015,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a balanced synthetic sphere, cube, and cylinder dataset.

    Returns:
        Point coordinates shaped ``(N, P, 3)`` with ``float32`` dtype and
        integer class labels shaped ``(N,)``. Class counts differ by at most
        one when ``number_examples`` is not divisible by three.
    """
    number_examples = _positive_integer(number_examples, name="number_examples")
    number_points = _positive_integer(number_points, name="number_points")
    noise_std = _nonnegative_real(noise_std, name="noise_std")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, Integral)):
        raise TypeError("seed must be an integer or None")
    generator = np.random.default_rng(None if seed is None else int(seed))
    labels = np.arange(number_examples) % 3
    generator.shuffle(labels)
    clouds = np.stack(
        [
            sample_shape(
                int(label),
                number_points,
                generator=generator,
                noise_std=noise_std,
            )
            for label in labels
        ]
    )
    return (
        torch.tensor(clouds, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long),
    )


def chamfer_distance(first_points: np.ndarray, second_points: np.ndarray) -> float:
    """Return symmetric mean squared nearest-neighbor distance.

    The first term maps every point in the first set to its closest point in
    the second. The second term reverses the direction, preventing unmatched
    regions in either point set from being ignored.
    """
    first_points = _point_set(first_points, name="first_points")
    second_points = _point_set(second_points, name="second_points")
    differences = first_points[:, None, :] - second_points[None, :, :]
    squared_distances = np.sum(differences**2, axis=-1)
    first_to_second = np.min(squared_distances, axis=1).mean()
    second_to_first = np.min(squared_distances, axis=0).mean()
    return float(first_to_second + second_to_first)
