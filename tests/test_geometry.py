import numpy as np
import pytest

from cs231n_practice.geometry import (
    orthographic_projection,
    perspective_projection,
    rotation_x,
    rotation_y,
    rotation_z,
    sample_sphere_surface,
    sphere_signed_distance,
    voxelize_points,
)


def test_sphere_surface_sampling_has_requested_radius_and_is_reproducible() -> None:
    first = sample_sphere_surface(
        100,
        radius=2.5,
        generator=np.random.default_rng(7),
    )
    second = sample_sphere_surface(
        100,
        radius=2.5,
        generator=np.random.default_rng(7),
    )

    assert first.shape == (100, 3)
    np.testing.assert_allclose(first, second)
    np.testing.assert_allclose(np.linalg.norm(first, axis=1), 2.5)


@pytest.mark.parametrize("rotation", [rotation_x, rotation_y, rotation_z])
def test_rotation_matrices_are_orthogonal_and_preserve_lengths(rotation) -> None:
    matrix = rotation(0.73)
    points = np.random.default_rng(1).normal(size=(12, 3))
    rotated = points @ matrix.T

    np.testing.assert_allclose(matrix.T @ matrix, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(
        np.linalg.norm(rotated, axis=1),
        np.linalg.norm(points, axis=1),
        atol=1e-12,
    )


def test_quarter_turns_follow_the_right_hand_rule() -> None:
    quarter_turn = np.pi / 2

    np.testing.assert_allclose(
        np.array([[0.0, 1.0, 0.0]]) @ rotation_x(quarter_turn).T,
        [[0.0, 0.0, 1.0]],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.array([[0.0, 0.0, 1.0]]) @ rotation_y(quarter_turn).T,
        [[1.0, 0.0, 0.0]],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.array([[1.0, 0.0, 0.0]]) @ rotation_z(quarter_turn).T,
        [[0.0, 1.0, 0.0]],
        atol=1e-12,
    )


def test_orthographic_and_perspective_projection_match_manual_values() -> None:
    points = np.array([[2.0, 4.0, 2.0], [-3.0, 6.0, 3.0]])

    np.testing.assert_allclose(
        orthographic_projection(points),
        [[2.0, 4.0], [-3.0, 6.0]],
    )
    np.testing.assert_allclose(
        perspective_projection(points, focal_length=2.0),
        [[2.0, 4.0], [-2.0, 4.0]],
    )


def test_voxelization_maps_boundaries_and_repeated_points() -> None:
    points = np.array(
        [
            [-1.0, -1.0, -1.0],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
        ]
    )

    occupancy = voxelize_points(points, 4)

    assert occupancy.shape == (4, 4, 4)
    assert occupancy.dtype == np.bool_
    assert occupancy.sum() == 3
    assert occupancy[0, 0, 0]
    assert occupancy[2, 2, 2]
    assert occupancy[3, 3, 3]


def test_sphere_signed_distance_has_correct_sign_and_center() -> None:
    points = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 3.0], [4.0, 2.0, 3.0]])

    distances = sphere_signed_distance(
        points,
        center=np.array([1.0, 2.0, 3.0]),
        radius=2.0,
    )

    np.testing.assert_allclose(distances, [-2.0, 0.0, 1.0])


def test_geometry_helpers_reject_invalid_inputs() -> None:
    with pytest.raises(TypeError, match="integer"):
        sample_sphere_surface(True)
    with pytest.raises(ValueError, match="positive"):
        sample_sphere_surface(2, radius=0.0)
    with pytest.raises(ValueError, match="positive"):
        perspective_projection(np.array([[1.0, 2.0, 0.0]]))
    with pytest.raises(ValueError, match="coordinate bounds"):
        voxelize_points(np.array([[1.1, 0.0, 0.0]]), 8)
    with pytest.raises(ValueError, match="shape"):
        sphere_signed_distance(np.ones((2, 3)), center=np.ones(2))
