"""Small image models for educational self-supervised experiments."""

from numbers import Integral

import torch
from torch import nn

from cs231n_practice.self_supervised import (
    patchify,
    random_mask_patches,
    restore_mask_tokens,
)


def _positive_integer(value: int, name: str) -> int:
    """Validate a positive integer without accepting Boolean values."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


class SmallEncoder(nn.Module):
    """Map small NCHW images to fixed-width feature vectors.

    Two max-pooling stages reduce spatial resolution, while adaptive average
    pooling makes the final representation independent of the input height and
    width. Inputs must be at least 4 by 4 pixels.
    """

    def __init__(self, input_channels: int = 3, feature_dim: int = 64) -> None:
        super().__init__()
        input_channels = _positive_integer(input_channels, "input_channels")
        feature_dim = _positive_integer(feature_dim, "feature_dim")
        self.input_channels = input_channels
        self.feature_dim = feature_dim
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, feature_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return one ``feature_dim``-wide vector for each image."""
        if not isinstance(images, torch.Tensor):
            raise TypeError("images must be a PyTorch tensor")
        if images.ndim != 4 or any(size == 0 for size in images.shape):
            raise ValueError("images must have nonempty shape (N, C, H, W)")
        if images.shape[1] != self.input_channels:
            raise ValueError(f"images must contain {self.input_channels} channels")
        if images.shape[-2] < 4 or images.shape[-1] < 4:
            raise ValueError("image height and width must be at least 4")
        if not images.is_floating_point():
            raise TypeError("images must contain floating-point values")
        return self.features(images).flatten(1)


class RotationPredictionModel(nn.Module):
    """Predict an image's applied quarter turn using a small CNN encoder."""

    def __init__(
        self,
        input_channels: int = 3,
        feature_dim: int = 64,
        num_rotations: int = 4,
    ) -> None:
        super().__init__()
        num_rotations = _positive_integer(num_rotations, "num_rotations")
        self.encoder = SmallEncoder(input_channels, feature_dim)
        self.rotation_head = nn.Linear(self.encoder.feature_dim, num_rotations)
        self.num_rotations = num_rotations

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return one rotation-score vector per input image."""
        return self.rotation_head(self.encoder(images))


class ProjectionHead(nn.Module):
    """Map encoder representations into a contrastive projection space."""

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 64,
        projection_dim: int = 32,
    ) -> None:
        super().__init__()
        input_dim = _positive_integer(input_dim, "input_dim")
        hidden_dim = _positive_integer(hidden_dim, "hidden_dim")
        projection_dim = _positive_integer(projection_dim, "projection_dim")
        self.input_dim = input_dim
        self.projection_dim = projection_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim),
        )

    def forward(self, representations: torch.Tensor) -> torch.Tensor:
        """Return projected rows with shape ``(N, projection_dim)``."""
        if not isinstance(representations, torch.Tensor):
            raise TypeError("representations must be a PyTorch tensor")
        if representations.ndim != 2 or representations.shape[1] != self.input_dim:
            raise ValueError(
                f"representations must have shape (N, {self.input_dim})"
            )
        if representations.shape[0] == 0:
            raise ValueError("representations must contain at least one row")
        if not representations.is_floating_point():
            raise TypeError("representations must contain floating-point values")
        return self.network(representations)


class SmallSimCLR(nn.Module):
    """Return encoder representations and contrastive projections for images."""

    def __init__(
        self,
        input_channels: int = 3,
        feature_dim: int = 64,
        projection_hidden_dim: int = 64,
        projection_dim: int = 32,
    ) -> None:
        super().__init__()
        self.encoder = SmallEncoder(input_channels, feature_dim)
        self.projector = ProjectionHead(
            feature_dim,
            projection_hidden_dim,
            projection_dim,
        )

    def forward(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return reusable representations ``h`` and contrastive projections ``z``."""
        representations = self.encoder(images)
        projections = self.projector(representations)
        return representations, projections


class FineTunedClassifier(nn.Module):
    """Attach a trainable linear classifier to an encoder with ``feature_dim``."""

    def __init__(self, encoder: nn.Module, num_classes: int = 10) -> None:
        super().__init__()
        if not isinstance(encoder, nn.Module):
            raise TypeError("encoder must be a torch.nn.Module")
        if not hasattr(encoder, "feature_dim"):
            raise ValueError("encoder must expose a feature_dim attribute")
        feature_dim = _positive_integer(encoder.feature_dim, "encoder.feature_dim")
        num_classes = _positive_integer(num_classes, "num_classes")
        self.encoder = encoder
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return class scores after updating both encoder and classifier."""
        return self.classifier(self.encoder(images))


class TinyMaskedAutoencoder(nn.Module):
    """Reconstruct masked patches of small square images with a tiny Transformer.

    Only visible patches pass through the encoder. A narrower decoder receives
    projected encoder outputs, learned mask tokens, and positional embeddings,
    then predicts one raw pixel vector for every patch position.
    """

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 8,
        input_channels: int = 3,
        encoder_dim: int = 48,
        decoder_dim: int = 32,
        num_heads: int = 4,
        encoder_layers: int = 2,
        decoder_layers: int = 1,
    ) -> None:
        super().__init__()
        image_size = _positive_integer(image_size, "image_size")
        patch_size = _positive_integer(patch_size, "patch_size")
        input_channels = _positive_integer(input_channels, "input_channels")
        encoder_dim = _positive_integer(encoder_dim, "encoder_dim")
        decoder_dim = _positive_integer(decoder_dim, "decoder_dim")
        num_heads = _positive_integer(num_heads, "num_heads")
        encoder_layers = _positive_integer(encoder_layers, "encoder_layers")
        decoder_layers = _positive_integer(decoder_layers, "decoder_layers")
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if encoder_dim % num_heads != 0:
            raise ValueError("encoder_dim must be divisible by num_heads")
        if decoder_dim % num_heads != 0:
            raise ValueError("decoder_dim must be divisible by num_heads")

        self.image_size = image_size
        self.patch_size = patch_size
        self.input_channels = input_channels
        self.patch_dim = input_channels * patch_size * patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.feature_dim = encoder_dim

        self.patch_embedding = nn.Linear(self.patch_dim, encoder_dim)
        self.encoder_position = nn.Parameter(
            torch.zeros(1, self.num_patches, encoder_dim)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=encoder_dim,
            nhead=num_heads,
            dim_feedforward=2 * encoder_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=encoder_layers,
            enable_nested_tensor=False,
        )

        self.decoder_projection = nn.Linear(encoder_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_position = nn.Parameter(
            torch.zeros(1, self.num_patches, decoder_dim)
        )
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim,
            nhead=num_heads,
            dim_feedforward=2 * decoder_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(
            decoder_layer,
            num_layers=decoder_layers,
            enable_nested_tensor=False,
        )
        self.pixel_prediction = nn.Linear(decoder_dim, self.patch_dim)

        nn.init.normal_(self.encoder_position, std=0.02)
        nn.init.normal_(self.decoder_position, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

    def _validate_images(self, images: torch.Tensor) -> None:
        """Validate an input batch against this model's image configuration."""
        if not isinstance(images, torch.Tensor):
            raise TypeError("images must be a PyTorch tensor")
        expected = (
            self.input_channels,
            self.image_size,
            self.image_size,
        )
        if images.ndim != 4 or images.shape[0] == 0 or images.shape[1:] != expected:
            raise ValueError(
                "images must have nonempty shape "
                f"(N, {self.input_channels}, {self.image_size}, {self.image_size})"
            )
        if not images.is_floating_point():
            raise TypeError("images must contain floating-point values")

    def forward(
        self,
        images: torch.Tensor,
        mask_ratio: float = 0.75,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return predicted patches, target patches, and original-order mask."""
        self._validate_images(images)
        target_patches = patchify(images, self.patch_size)
        visible_patches, mask, ids_keep, ids_restore = random_mask_patches(
            target_patches,
            mask_ratio,
            generator=generator,
        )

        visible_tokens = self.patch_embedding(visible_patches)
        encoder_positions = self.encoder_position.expand(len(images), -1, -1)
        visible_positions = encoder_positions.gather(
            1,
            ids_keep.unsqueeze(-1).expand(-1, -1, self.feature_dim),
        )
        encoded_visible = self.encoder(visible_tokens + visible_positions)

        decoder_visible = self.decoder_projection(encoded_visible)
        complete_tokens = restore_mask_tokens(
            decoder_visible,
            ids_restore,
            self.mask_token,
        )
        decoded = self.decoder(complete_tokens + self.decoder_position)
        predictions = self.pixel_prediction(decoded)
        return predictions, target_patches, mask

    def encode_all(self, images: torch.Tensor) -> torch.Tensor:
        """Encode every patch and mean-pool one representation per image."""
        self._validate_images(images)
        patches = patchify(images, self.patch_size)
        tokens = self.patch_embedding(patches) + self.encoder_position
        return self.encoder(tokens).mean(dim=1)


class MAEEncoderView(nn.Module):
    """Expose a masked autoencoder's unmasked encoder as a standard module."""

    def __init__(self, mae: TinyMaskedAutoencoder) -> None:
        super().__init__()
        if not isinstance(mae, TinyMaskedAutoencoder):
            raise TypeError("mae must be a TinyMaskedAutoencoder")
        self.mae = mae
        self.feature_dim = mae.feature_dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return mean-pooled encoder representations for complete images."""
        return self.mae.encode_all(images)
