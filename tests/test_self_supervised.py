import pytest
import torch
from torch import nn

from cs231n_practice.self_supervised import (
    contrastive_similarity_metrics,
    extract_features,
    freeze_encoder,
    info_nce_loss,
    l2_normalize,
    make_contrastive_views,
    make_rotation_batch,
    nt_xent_loss,
    pairwise_cosine_similarity,
    positive_pair_indices,
    simclr_loss,
    train_linear_probe,
)


def test_l2_normalize_normalizes_rows_and_preserves_zero() -> None:
    embeddings = torch.tensor([[3.0, 4.0], [6.0, 8.0], [0.0, 0.0]])

    normalized = l2_normalize(embeddings)

    expected = torch.tensor([[0.6, 0.8], [0.6, 0.8], [0.0, 0.0]])
    torch.testing.assert_close(normalized, expected)
    torch.testing.assert_close(
        normalized.norm(dim=1), torch.tensor([1.0, 1.0, 0.0])
    )


def test_pairwise_cosine_similarity_is_symmetric_and_scale_invariant() -> None:
    embeddings = torch.tensor([[3.0, 4.0], [4.0, -3.0], [-3.0, -4.0]])

    similarities = pairwise_cosine_similarity(embeddings)
    scaled_similarities = pairwise_cosine_similarity(
        embeddings * torch.tensor([[2.0], [5.0], [0.5]])
    )

    expected = torch.tensor([[1.0, 0.0, -1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 1.0]])
    torch.testing.assert_close(similarities, expected)
    torch.testing.assert_close(similarities, similarities.T)
    torch.testing.assert_close(scaled_similarities, similarities)


def test_positive_pair_indices_maps_both_view_directions() -> None:
    torch.testing.assert_close(
        positive_pair_indices(8),
        torch.tensor([4, 5, 6, 7, 0, 1, 2, 3]),
    )


def test_info_nce_matches_explicit_loop_and_nt_xent_name() -> None:
    torch.manual_seed(23)
    embeddings = torch.randn(10, 6)
    temperature = 0.3
    similarities = pairwise_cosine_similarity(embeddings)
    targets = positive_pair_indices(len(embeddings))
    losses = []
    for anchor, positive in enumerate(targets):
        valid = torch.arange(len(embeddings)) != anchor
        scores = similarities[anchor] / temperature
        losses.append(
            -scores[positive] + torch.logsumexp(scores[valid], dim=0)
        )
    expected = torch.stack(losses).mean()

    actual = info_nce_loss(embeddings, temperature)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(nt_xent_loss(embeddings, temperature), actual)


def test_info_nce_collapsed_embeddings_have_uniform_candidate_loss() -> None:
    embeddings = torch.ones(8, 5)

    loss = info_nce_loss(embeddings)

    expected = torch.log(torch.tensor(7.0))
    torch.testing.assert_close(loss, expected)


def test_info_nce_propagates_finite_embedding_gradients() -> None:
    torch.manual_seed(29)
    embeddings = torch.randn(8, 5, requires_grad=True)

    loss = info_nce_loss(embeddings)
    loss.backward()

    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all()
    assert embeddings.grad.abs().sum() > 0


def test_contrastive_helpers_reject_invalid_arguments() -> None:
    with pytest.raises(ValueError, match="shape"):
        l2_normalize(torch.zeros(2, 3, 4))
    with pytest.raises(TypeError, match="floating-point"):
        pairwise_cosine_similarity(torch.ones(4, 3, dtype=torch.int64))
    with pytest.raises(ValueError, match="epsilon"):
        l2_normalize(torch.ones(4, 3), epsilon=0.0)
    with pytest.raises(ValueError, match="at least two"):
        positive_pair_indices(2)
    with pytest.raises(ValueError, match="even"):
        positive_pair_indices(5)
    with pytest.raises(ValueError, match="temperature"):
        info_nce_loss(torch.randn(4, 3), temperature=0.0)
    with pytest.raises(TypeError, match="numeric"):
        info_nce_loss(torch.randn(4, 3), temperature=True)


def test_make_contrastive_views_calls_transform_twice_per_image() -> None:
    images = torch.arange(3 * 2, dtype=torch.float32).reshape(3, 2)
    calls = []

    def transform(image: torch.Tensor) -> torch.Tensor:
        calls.append(len(calls))
        return image + calls[-1]

    view_a, view_b = make_contrastive_views(images, transform)

    assert len(calls) == 6
    torch.testing.assert_close(view_a, images + torch.tensor([[0], [1], [2]]))
    torch.testing.assert_close(view_b, images + torch.tensor([[3], [4], [5]]))


def test_simclr_loss_propagates_through_encoder_and_projector() -> None:
    from cs231n_practice.classifiers.self_supervised import SmallSimCLR

    torch.manual_seed(31)
    model = SmallSimCLR(feature_dim=12, projection_dim=6)
    view_a = torch.randn(4, 3, 8, 8)
    view_b = torch.randn(4, 3, 8, 8)

    loss = simclr_loss(model, view_a, view_b)
    loss.backward()

    assert torch.isfinite(loss)
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_contrastive_similarity_metrics_separates_positive_and_negative_pairs() -> None:
    projections = torch.tensor([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ])

    positive, negative = contrastive_similarity_metrics(projections)

    assert positive == pytest.approx(1.0)
    assert negative == pytest.approx(0.0)


def test_train_linear_probe_learns_separable_fixed_features() -> None:
    train_features = torch.tensor([
        [-2.0, -1.0], [-1.0, -2.0], [-1.5, -1.5],
        [2.0, 1.0], [1.0, 2.0], [1.5, 1.5],
    ])
    train_labels = torch.tensor([0, 0, 0, 1, 1, 1])
    evaluation_features = torch.tensor([[-1.0, -1.0], [1.0, 1.0]])
    evaluation_labels = torch.tensor([0, 1])

    head, history = train_linear_probe(
        train_features,
        train_labels,
        evaluation_features,
        evaluation_labels,
        num_classes=2,
        epochs=20,
        learning_rate=0.1,
        seed=5,
    )

    assert isinstance(head, nn.Linear)
    assert history["train_accuracy"][-1] == pytest.approx(1.0)
    assert history["evaluation_accuracy"][-1] == pytest.approx(1.0)


def test_make_rotation_batch_matches_every_sampled_target() -> None:
    images = torch.arange(8 * 3 * 5 * 5).reshape(8, 3, 5, 5)

    rotated, targets = make_rotation_batch(
        images,
        generator=torch.Generator().manual_seed(7),
    )

    assert rotated.shape == images.shape
    assert rotated.dtype == images.dtype
    assert targets.shape == (8,)
    assert targets.dtype == torch.long
    assert bool(((0 <= targets) & (targets < 4)).all())
    for index, target in enumerate(targets):
        expected = torch.rot90(images[index], int(target), dims=(-2, -1))
        torch.testing.assert_close(rotated[index], expected)


def test_make_rotation_batch_is_reproducible_without_modifying_input() -> None:
    images = torch.randn(12, 3, 6, 6)
    original = images.clone()

    first = make_rotation_batch(images, torch.Generator().manual_seed(11))
    second = make_rotation_batch(images, torch.Generator().manual_seed(11))

    torch.testing.assert_close(first[0], second[0])
    torch.testing.assert_close(first[1], second[1])
    torch.testing.assert_close(images, original)


def test_make_rotation_batch_rejects_invalid_inputs() -> None:
    with pytest.raises(TypeError, match="tensor"):
        make_rotation_batch([[1.0]])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="shape"):
        make_rotation_batch(torch.zeros(3, 8, 8))
    with pytest.raises(ValueError, match="square"):
        make_rotation_batch(torch.zeros(2, 3, 8, 7))
    with pytest.raises(TypeError, match="Generator"):
        make_rotation_batch(torch.zeros(2, 3, 8, 8), generator=3)  # type: ignore[arg-type]


def test_freeze_encoder_disables_gradients_and_evaluation_behavior() -> None:
    encoder = nn.Sequential(nn.Linear(4, 5), nn.Dropout(0.5))
    encoder.train()

    returned = freeze_encoder(encoder)

    assert returned is encoder
    assert not encoder.training
    assert all(not parameter.requires_grad for parameter in encoder.parameters())


def test_freeze_encoder_rejects_non_module() -> None:
    with pytest.raises(TypeError, match="Module"):
        freeze_encoder(lambda x: x)  # type: ignore[arg-type]


def test_extract_features_batches_results_and_detaches_them() -> None:
    encoder = nn.Linear(4, 3, bias=False)
    images = torch.randn(7, 4)
    expected = encoder(images).detach()

    features = extract_features(encoder, images, batch_size=3)

    assert features.shape == (7, 3)
    assert features.device.type == "cpu"
    assert not features.requires_grad
    assert not encoder.training
    torch.testing.assert_close(features, expected)
    assert all(parameter.grad is None for parameter in encoder.parameters())


def test_extract_features_rejects_invalid_arguments() -> None:
    encoder = nn.Linear(4, 3)
    with pytest.raises(TypeError, match="Module"):
        extract_features(lambda x: x, torch.zeros(2, 4))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="tensor"):
        extract_features(encoder, [[1.0, 2.0, 3.0, 4.0]])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="nonempty"):
        extract_features(encoder, torch.zeros(0, 4))
    with pytest.raises(ValueError, match="positive"):
        extract_features(encoder, torch.zeros(2, 4), batch_size=0)
    with pytest.raises(TypeError, match="integer"):
        extract_features(encoder, torch.zeros(2, 4), batch_size=True)
