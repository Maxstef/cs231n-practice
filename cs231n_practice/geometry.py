"""Small NumPy helpers for introductory 3D geometry experiments."""

from numbers import Integral, Real

import numpy as np


def _positive_integer(value: int, *, name: str) -> int:
    """Validate a positive integer without accepting Boolean values."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _positive_real(value: float, *, name: str) -> float:
    """Validate a finite positive real number."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be numeric")
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _points_3d(points: np.ndarray, *, name: str = "points") -> np.ndarray:
    """Return a validated nonempty floating-point array with shape (P, 3)."""
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] != 3:
        raise ValueError(f"{name} must have nonempty shape (P, 3)")
    if not np.issubdtype(points.dtype, np.floating):
        raise TypeError(f"{name} must contain floating-point values")
    if not np.isfinite(points).all():
        raise ValueError(f"{name} must contain only finite values")
    return points


def sample_sphere_surface(
    number_points: int,
    *,
    radius: float = 1.0,
    generator: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample points uniformly from the surface of a sphere at the origin.

    Isotropic Gaussian vectors have no preferred direction. Dividing every
    vector by its Euclidean norm retains only that direction and places it on
    the unit sphere; multiplying by ``radius`` gives the requested sphere.

    Args:
        number_points: Number of surface points to sample.
        radius: Positive sphere radius.
        generator: Optional NumPy random generator. A fresh default generator
            is used when omitted.

    Returns:
        Floating-point coordinates with shape ``(number_points, 3)``.
    """
    number_points = _positive_integer(number_points, name="number_points")
    radius = _positive_real(radius, name="radius")
    if generator is None:
        generator = np.random.default_rng()
    if not isinstance(generator, np.random.Generator):
        raise TypeError("generator must be a numpy.random.Generator")

    directions = generator.standard_normal(size=(number_points, 3))
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    return radius * directions / norms


def rotation_x(angle_radians: float) -> np.ndarray:
    """Return a right-handed rotation matrix around the x axis."""
    angle = _finite_angle(angle_radians)
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cosine, -sine],
            [0.0, sine, cosine],
        ]
    )


def rotation_y(angle_radians: float) -> np.ndarray:
    """Return a right-handed rotation matrix around the y axis."""
    angle = _finite_angle(angle_radians)
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array(
        [
            [cosine, 0.0, sine],
            [0.0, 1.0, 0.0],
            [-sine, 0.0, cosine],
        ]
    )


def rotation_z(angle_radians: float) -> np.ndarray:
    """Return a right-handed rotation matrix around the z axis."""
    angle = _finite_angle(angle_radians)
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def _finite_angle(angle_radians: float) -> float:
    """Validate a finite numeric angle in radians."""
    if isinstance(angle_radians, bool) or not isinstance(angle_radians, Real):
        raise TypeError("angle_radians must be numeric")
    angle = float(angle_radians)
    if not np.isfinite(angle):
        raise ValueError("angle_radians must be finite")
    return angle


def orthographic_projection(points: np.ndarray) -> np.ndarray:
    """Project camera-space points by discarding their depth coordinate.

    Args:
        points: Floating-point camera coordinates with shape ``(P, 3)``.

    Returns:
        Image-plane coordinates ``(x, y)`` with shape ``(P, 2)``.
    """
    points = _points_3d(points)
    return points[:, :2].copy()


def perspective_projection(
    points: np.ndarray,
    *,
    focal_length: float = 1.0,
) -> np.ndarray:
    """Project positive-depth camera points using ``(fx/z, fy/z)``."""
    points = _points_3d(points)
    focal_length = _positive_real(focal_length, name="focal_length")
    depths = points[:, 2:3]
    if np.any(depths <= 0.0):
        raise ValueError("all point depths must be positive")
    return focal_length * points[:, :2] / depths


def voxelize_points(
    points: np.ndarray,
    resolution: int,
    *,
    coordinate_bound: float = 1.0,
) -> np.ndarray:
    """Mark cells touched by points inside ``[-bound, bound]^3``.

    The returned Boolean array has shape ``(R, R, R)``. Points exactly on the
    positive cube boundary are clipped to the final valid voxel index.
    """
    points = _points_3d(points)
    resolution = _positive_integer(resolution, name="resolution")
    coordinate_bound = _positive_real(
        coordinate_bound,
        name="coordinate_bound",
    )
    tolerance = 16 * np.finfo(points.dtype).eps * coordinate_bound
    if np.any(np.abs(points) > coordinate_bound + tolerance):
        raise ValueError("points must lie inside the coordinate bounds")

    scaled = (
        (points + coordinate_bound)
        / (2.0 * coordinate_bound)
        * resolution
    )
    indices = np.floor(scaled).astype(np.int64)
    indices = np.clip(indices, 0, resolution - 1)
    occupancy = np.zeros((resolution,) * 3, dtype=bool)
    occupancy[tuple(indices.T)] = True
    return occupancy


def sphere_signed_distance(
    points: np.ndarray,
    *,
    center: np.ndarray | None = None,
    radius: float = 1.0,
) -> np.ndarray:
    """Return signed distances to a sphere surface.

    Negative values are inside the sphere, zero lies on the surface, and
    positive values are outside.
    """
    points = _points_3d(points)
    radius = _positive_real(radius, name="radius")
    if center is None:
        center_array = np.zeros(3, dtype=points.dtype)
    else:
        center_array = np.asarray(center)
        if center_array.shape != (3,):
            raise ValueError("center must have shape (3,)")
        if not np.issubdtype(center_array.dtype, np.floating):
            raise TypeError("center must contain floating-point values")
        if not np.isfinite(center_array).all():
            raise ValueError("center must contain only finite values")
    return np.linalg.norm(points - center_array, axis=1) - radius
