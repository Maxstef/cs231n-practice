import numpy as np
import pytest
import torch

from cs231n_practice.point_cloud import (
    chamfer_distance,
    make_point_cloud_dataset,
    normalize_point_cloud,
    random_rotation,
    sample_cube_surface,
    sample_cylinder_surface,
    sample_shape,
)


def test_surface_samplers_return_reproducible_points_on_surfaces() -> None:
    first_cube = sample_cube_surface(100, generator=np.random.default_rng(7))
    second_cube = sample_cube_surface(100, generator=np.random.default_rng(7))
    cylinder = sample_cylinder_surface(200, generator=np.random.default_rng(8))

    np.testing.assert_allclose(first_cube, second_cube)
    assert first_cube.shape == (100, 3)
    assert np.all(np.isclose(np.abs(first_cube), 1.0).any(axis=1))
    assert cylinder.shape == (200, 3)
    on_side = np.isclose(np.linalg.norm(cylinder[:, :2], axis=1), 1.0)
    on_cap = np.isclose(np.abs(cylinder[:, 2]), 1.0)
    assert np.all(on_side | on_cap)


def test_normalization_centers_and_scales_point_cloud() -> None:
    points = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 3.0], [1.0, 6.0, 3.0]])

    normalized = normalize_point_cloud(points)

    np.testing.assert_allclose(normalized.mean(axis=0), 0.0, atol=1e-15)
    np.testing.assert_allclose(np.linalg.norm(normalized, axis=1).max(), 1.0)


def test_random_rotation_is_orthogonal() -> None:
    rotation = random_rotation(generator=np.random.default_rng(9))

    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(rotation), 1.0, atol=1e-12)


def test_shape_dataset_is_balanced_reproducible_and_has_expected_types() -> None:
    first_clouds, first_labels = make_point_cloud_dataset(10, 16, seed=4)
    second_clouds, second_labels = make_point_cloud_dataset(10, 16, seed=4)

    torch.testing.assert_close(first_clouds, second_clouds)
    torch.testing.assert_close(first_labels, second_labels)
    assert first_clouds.shape == (10, 16, 3)
    assert first_clouds.dtype == torch.float32
    assert first_labels.shape == (10,)
    assert first_labels.dtype == torch.long
    assert torch.bincount(first_labels).tolist() == [4, 3, 3]
    torch.testing.assert_close(first_clouds.mean(dim=1), torch.zeros(10, 3))


def test_chamfer_distance_matches_manual_result_and_is_symmetric() -> None:
    first = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    second = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])

    np.testing.assert_allclose(chamfer_distance(first, first[::-1]), 0.0)
    np.testing.assert_allclose(chamfer_distance(first, second), 1.0)
    np.testing.assert_allclose(
        chamfer_distance(first, second), chamfer_distance(second, first)
    )


def test_point_cloud_helpers_reject_invalid_inputs() -> None:
    with pytest.raises(TypeError, match="integer"):
        sample_cube_surface(True)
    with pytest.raises(ValueError, match="label"):
        sample_shape(3, 8)
    with pytest.raises(ValueError, match="same location"):
        normalize_point_cloud(np.ones((4, 3)))
    with pytest.raises(ValueError, match="shape"):
        chamfer_distance(np.ones((3, 2)), np.ones((3, 3)))
    with pytest.raises(TypeError, match="floating-point"):
        chamfer_distance(np.ones((3, 3), dtype=int), np.ones((3, 3)))
