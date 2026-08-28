import numpy as np
import pytest

from cs231n_practice.training import (
    clip_gradients_by_global_norm,
    sample_minibatch,
)


def test_clip_gradients_by_global_norm_uses_one_shared_scale() -> None:
    gradients = {
        "first": np.array([[3.0, 4.0]]),
        "second": np.array([[0.0, 12.0]]),
    }

    clipped, original_norm = clip_gradients_by_global_norm(gradients, 5.0)

    clipped_norm = np.sqrt(sum(np.sum(value**2) for value in clipped.values()))
    np.testing.assert_allclose(original_norm, 13.0)
    np.testing.assert_allclose(clipped_norm, 5.0)
    np.testing.assert_allclose(clipped["first"], gradients["first"] * (5 / 13))
    assert all(clipped[name] is not value for name, value in gradients.items())


def test_clip_gradients_below_threshold_returns_equal_copies() -> None:
    gradients = {"weights": np.array([3.0, 4.0])}

    clipped, original_norm = clip_gradients_by_global_norm(gradients, 6.0)

    np.testing.assert_allclose(original_norm, 5.0)
    np.testing.assert_array_equal(clipped["weights"], gradients["weights"])
    assert clipped["weights"] is not gradients["weights"]


@pytest.mark.parametrize("max_norm", [0.0, -1.0, np.inf])
def test_clip_gradients_rejects_invalid_max_norm(max_norm: float) -> None:
    with pytest.raises(ValueError, match="max_norm"):
        clip_gradients_by_global_norm({"weights": np.ones(2)}, max_norm)


def test_sample_minibatch_preserves_feature_target_pairing() -> None:
    features = np.arange(20).reshape(10, 2)
    targets = np.arange(10)

    batch_features, batch_targets = sample_minibatch(
        features,
        targets,
        5,
        np.random.default_rng(7),
    )

    assert len(np.unique(batch_targets)) == 5
    np.testing.assert_array_equal(batch_features[:, 0] // 2, batch_targets)


def test_sample_minibatch_is_reproducible_for_equal_generators() -> None:
    features = np.arange(12).reshape(6, 2)
    targets = np.arange(6)

    first = sample_minibatch(features, targets, 3, np.random.default_rng(4))
    second = sample_minibatch(features, targets, 3, np.random.default_rng(4))

    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


def test_sample_minibatch_rejects_oversized_batch() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        sample_minibatch(
            np.ones((2, 3)),
            np.zeros(2),
            3,
            np.random.default_rng(1),
        )
