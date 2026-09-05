"""A compact PyTorch Vision Transformer for educational experiments."""

import torch
from torch import nn


def _positive_integer(value: int, *, name: str) -> int:
    """Validate and return a positive integer configuration value."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


class TinyVisionTransformer(nn.Module):
    """Classify square RGB images with a small Transformer encoder.

    Non-overlapping patches are projected with a strided convolution. A
    learned classification token is prepended, learned positional embeddings
    are added, and the contextualized classification token is mapped to class
    scores.

    Args:
        image_size: Expected height and width of every input image.
        patch_size: Height and width of each non-overlapping square patch.
        model_dim: Token feature width used by the Transformer.
        num_heads: Number of self-attention heads.
        feed_forward_dim: Hidden width of the position-wise feed-forward layer.
        num_classes: Number of output classes.
        num_layers: Number of Transformer encoder blocks.
        dropout: Dropout probability inside each encoder block.

    Inputs:
        Floating-point images with shape ``(N, 3, image_size, image_size)``.

    Returns:
        Class scores with shape ``(N, num_classes)``.
    """

    def __init__(
        self,
        image_size: int = 16,
        patch_size: int = 4,
        model_dim: int = 32,
        num_heads: int = 4,
        feed_forward_dim: int = 64,
        num_classes: int = 10,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        image_size = _positive_integer(image_size, name="image_size")
        patch_size = _positive_integer(patch_size, name="patch_size")
        model_dim = _positive_integer(model_dim, name="model_dim")
        num_heads = _positive_integer(num_heads, name="num_heads")
        feed_forward_dim = _positive_integer(
            feed_forward_dim, name="feed_forward_dim"
        )
        num_classes = _positive_integer(num_classes, name="num_classes")
        num_layers = _positive_integer(num_layers, name="num_layers")
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if isinstance(dropout, bool) or not isinstance(dropout, (int, float)):
            raise TypeError("dropout must be numeric")
        dropout = float(dropout)
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1")

        self.image_size = image_size
        self.patch_size = patch_size
        self.model_dim = model_dim
        self.num_patches = (image_size // patch_size) ** 2
        self.sequence_length = self.num_patches + 1
        self.num_classes = num_classes

        # Kernel size equals stride, so patches are non-overlapping. The output
        # channels form the learned feature vector for every patch.
        self.patch_projection = nn.Conv2d(
            3,
            model_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.sequence_length, model_dim)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=feed_forward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.final_norm = nn.LayerNorm(model_dim)
        self.classifier = nn.Linear(model_dim, num_classes)

        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.position_embedding, std=0.02)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return class scores for a batch of NCHW RGB images."""
        if not isinstance(images, torch.Tensor):
            raise TypeError("images must be a torch.Tensor")
        expected_tail = (3, self.image_size, self.image_size)
        if images.ndim != 4 or tuple(images.shape[1:]) != expected_tail:
            raise ValueError(f"images must have shape (N, {expected_tail})")
        if not images.is_floating_point():
            raise TypeError("images must contain floating-point values")

        patch_grid = self.patch_projection(images)
        # (N, D, H_p, W_p) -> (N, H_p * W_p, D), in row-major grid order.
        patch_tokens = patch_grid.flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(images.shape[0], -1, -1)
        tokens = torch.cat((cls_tokens, patch_tokens), dim=1)
        encoded = self.encoder(tokens + self.position_embedding)
        return self.classifier(self.final_norm(encoded[:, 0]))
