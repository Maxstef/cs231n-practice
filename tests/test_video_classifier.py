import pytest
import torch

from cs231n_practice.classifiers import (
    Small3DVideoClassifier,
    TwoStreamVideoClassifier,
)
from cs231n_practice.video import NonLocalBlock3D


@pytest.mark.parametrize("use_nonlocal", [False, True])
def test_small_3d_video_classifier_returns_class_scores(
    use_nonlocal: bool,
) -> None:
    model = Small3DVideoClassifier(
        input_channels=2,
        num_classes=5,
        use_nonlocal=use_nonlocal,
    )
    videos = torch.randn(4, 2, 5, 10, 12)

    scores = model(videos)

    assert scores.shape == (4, 5)


def test_small_3d_video_classifier_matches_notebook_parameter_counts() -> None:
    plain = Small3DVideoClassifier()
    nonlocal_model = Small3DVideoClassifier(
        use_nonlocal=True, attention_channels=3
    )

    assert sum(parameter.numel() for parameter in plain.parameters()) == 1_832
    assert (
        sum(parameter.numel() for parameter in nonlocal_model.parameters())
        == 1_920
    )
    assert isinstance(nonlocal_model.context, NonLocalBlock3D)


def test_small_3d_video_classifier_propagates_finite_gradients() -> None:
    model = Small3DVideoClassifier(use_nonlocal=True)
    videos = torch.randn(3, 3, 4, 8, 8, requires_grad=True)

    model(videos).square().mean().backward()

    assert videos.grad is not None and torch.isfinite(videos.grad).all()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_small_3d_video_classifier_rejects_invalid_input() -> None:
    model = Small3DVideoClassifier(input_channels=3)
    with pytest.raises(ValueError, match="shape"):
        model(torch.zeros(2, 3, 8, 8))
    with pytest.raises(ValueError, match="3 channels"):
        model(torch.zeros(2, 2, 4, 8, 8))
    with pytest.raises(TypeError, match="floating-point"):
        model(torch.zeros(2, 3, 4, 8, 8, dtype=torch.int64))


def test_two_stream_video_classifier_returns_class_scores() -> None:
    model = TwoStreamVideoClassifier(
        appearance_channels=3,
        motion_channels=12,
        num_classes=6,
    )
    appearance = torch.randn(5, 3, 10, 12)
    motion = torch.randn(5, 12, 10, 12)

    scores = model(appearance, motion)

    assert scores.shape == (5, 6)


def test_two_stream_classifier_matches_notebook_parameter_count() -> None:
    model = TwoStreamVideoClassifier()

    assert sum(parameter.numel() for parameter in model.parameters()) == 1_770


def test_two_stream_video_classifier_propagates_gradients_to_both_streams() -> None:
    model = TwoStreamVideoClassifier()
    appearance = torch.randn(3, 3, 8, 8, requires_grad=True)
    motion = torch.randn(3, 12, 8, 8, requires_grad=True)

    model(appearance, motion).square().mean().backward()

    assert appearance.grad is not None and torch.isfinite(appearance.grad).all()
    assert motion.grad is not None and torch.isfinite(motion.grad).all()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_two_stream_video_classifier_rejects_mismatched_inputs() -> None:
    model = TwoStreamVideoClassifier()
    with pytest.raises(ValueError, match="batch sizes"):
        model(torch.zeros(2, 3, 8, 8), torch.zeros(3, 12, 8, 8))
    with pytest.raises(ValueError, match="spatial dimensions"):
        model(torch.zeros(2, 3, 8, 8), torch.zeros(2, 12, 7, 8))
    with pytest.raises(ValueError, match="12 channels"):
        model(torch.zeros(2, 3, 8, 8), torch.zeros(2, 8, 8, 8))


@pytest.mark.parametrize(
    ("arguments", "error"),
    [
        ({"input_channels": 0}, ValueError),
        ({"num_classes": True}, TypeError),
        ({"use_nonlocal": 1}, TypeError),
        ({"attention_channels": 0}, ValueError),
    ],
)
def test_small_3d_video_classifier_rejects_invalid_configuration(
    arguments: dict[str, object], error: type[Exception]
) -> None:
    with pytest.raises(error):
        Small3DVideoClassifier(**arguments)  # type: ignore[arg-type]
