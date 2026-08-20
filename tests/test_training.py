import numpy as np
import pytest

from cs231n_practice.training import sample_minibatch


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
