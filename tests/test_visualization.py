import matplotlib
import pytest
import torch
from torch import nn

from cs231n_practice.visualization import (
    class_activation_map,
    grad_cam,
    input_gradient_saliency,
    occlusion_sensitivity,
    plot_attribution_overlay,
)

matplotlib.use("Agg")


class TinyCamModel(nn.Module):
    """Small deterministic GAP-linear model used by the attribution tests."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 2, 1), nn.ReLU())
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(2, 2)
        with torch.no_grad():
            self.features[0].weight.copy_(
                torch.tensor(
                    [
                        [[[1.0]], [[0.0]], [[0.0]]],
                        [[[0.0]], [[1.0]], [[0.0]]],
                    ]
                )
            )
            self.features[0].bias.zero_()
            self.classifier.weight.copy_(torch.tensor([[2.0, -1.0], [-1.0, 2.0]]))
            self.classifier.bias.zero_()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.features(image)
        return self.classifier(self.pool(features).flatten(1))


@pytest.fixture
def model_and_image() -> tuple[TinyCamModel, torch.Tensor]:
    model = TinyCamModel()
    image = torch.tensor(
        [[
            [[1.0, 2.0], [3.0, 4.0]],
            [[4.0, 3.0], [2.0, 1.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        ]]
    )
    return model, image


def test_input_gradient_saliency_returns_normalized_spatial_map(
    model_and_image: tuple[TinyCamModel, torch.Tensor],
) -> None:
    model, image = model_and_image
    model.train()

    saliency, class_id = input_gradient_saliency(model, image, class_id=0)

    assert saliency.shape == (2, 2)
    assert class_id == 0
    assert torch.isfinite(saliency).all()
    assert 0 <= saliency.min() <= saliency.max() <= 1
    assert model.training


def test_occlusion_sensitivity_returns_one_drop_per_valid_patch(
    model_and_image: tuple[TinyCamModel, torch.Tensor],
) -> None:
    model, image = model_and_image

    drops, class_id = occlusion_sensitivity(
        model, image, class_id=0, patch_size=1, stride=1
    )

    assert drops.shape == (2, 2)
    assert class_id == 0
    assert torch.isfinite(drops).all()


def test_cam_and_grad_cam_agree_for_gap_linear_model(
    model_and_image: tuple[TinyCamModel, torch.Tensor],
) -> None:
    model, image = model_and_image

    cam, cam_class = class_activation_map(
        model, image, model.features[-1], model.classifier, class_id=0
    )
    gradient_cam, gradient_class = grad_cam(
        model, image, model.features[-1], class_id=0
    )

    assert cam.shape == gradient_cam.shape == (2, 2)
    assert cam_class == gradient_class == 0
    torch.testing.assert_close(cam, gradient_cam)
    assert len(model.features[-1]._forward_hooks) == 0


def test_methods_choose_predicted_class_by_default(
    model_and_image: tuple[TinyCamModel, torch.Tensor],
) -> None:
    model, image = model_and_image
    expected = int(model(image).argmax(dim=1)[0])

    _, saliency_class = input_gradient_saliency(model, image)
    _, occlusion_class = occlusion_sensitivity(
        model, image, patch_size=1, stride=1
    )
    _, cam_class = class_activation_map(
        model, image, model.features[-1], model.classifier
    )
    _, gradcam_class = grad_cam(model, image, model.features[-1])

    assert {saliency_class, occlusion_class, cam_class, gradcam_class} == {expected}


def test_plot_attribution_overlay_returns_two_axes(
    model_and_image: tuple[TinyCamModel, torch.Tensor],
) -> None:
    _, image = model_and_image

    figure, axes = plot_attribution_overlay(
        image,
        torch.tensor([[0.0, 0.5], [0.75, 1.0]]),
        title="CAM",
        class_names=["red", "green"],
        class_id=0,
    )

    assert len(axes) == 2
    assert axes[1].get_title() == "CAM: red"
    matplotlib.pyplot.close(figure)


def test_attribution_methods_reject_invalid_inputs(
    model_and_image: tuple[TinyCamModel, torch.Tensor],
) -> None:
    model, image = model_and_image

    with pytest.raises(ValueError, match="shape"):
        input_gradient_saliency(model, image.repeat(2, 1, 1, 1))
    with pytest.raises(ValueError, match="patch_size"):
        occlusion_sensitivity(model, image, patch_size=3)
    with pytest.raises(TypeError, match="class_id"):
        grad_cam(model, image, model.features[-1], class_id=True)
