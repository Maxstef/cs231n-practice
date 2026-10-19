"""Small PyTorch decoders and helpers for conditional image captioning."""

from numbers import Integral
from typing import Iterable

import torch
from torch import nn


def _positive_integer(value: int, *, name: str) -> int:
    """Validate a positive integer without accepting Boolean values."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def causal_attention_mask(
    sequence_length: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Return an additive mask that hides every future key position.

    Allowed entries are zero and forbidden entries above the diagonal are
    negative infinity. Adding this mask before softmax gives forbidden keys
    zero attention probability.
    """
    sequence_length = _positive_integer(
        sequence_length,
        name="sequence_length",
    )
    if dtype is None:
        dtype = torch.get_default_dtype()
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise TypeError("dtype must be a floating-point dtype")
    mask = torch.full(
        (sequence_length, sequence_length),
        -torch.inf,
        device=device,
        dtype=dtype,
    )
    return torch.triu(mask, diagonal=1)


class RecurrentCaptionDecoder(nn.Module):
    """Generate caption-token logits with an image-conditioned GRU.

    The projected image feature initializes the GRU hidden state. Caption
    inputs are then processed from left to right using teacher forcing.

    Shape:
        - Image features: ``(N, image_feature_dim)``
        - Caption inputs: ``(N, T)`` integer token IDs
        - Returned logits: ``(N, T, vocabulary_size)``
    """

    def __init__(
        self,
        image_feature_dim: int,
        vocabulary_size: int,
        model_dim: int = 48,
    ) -> None:
        super().__init__()
        self.image_feature_dim = _positive_integer(
            image_feature_dim,
            name="image_feature_dim",
        )
        self.vocabulary_size = _positive_integer(
            vocabulary_size,
            name="vocabulary_size",
        )
        self.model_dim = _positive_integer(model_dim, name="model_dim")
        self.image_projection = nn.Linear(self.image_feature_dim, self.model_dim)
        self.token_embedding = nn.Embedding(self.vocabulary_size, self.model_dim)
        self.recurrent_layer = nn.GRU(
            self.model_dim,
            self.model_dim,
            batch_first=True,
        )
        self.output_projection = nn.Linear(
            self.model_dim,
            self.vocabulary_size,
        )

    def forward(
        self,
        image_features: torch.Tensor,
        caption_inputs: torch.Tensor,
    ) -> torch.Tensor:
        """Return next-token logits at every caption-input position."""
        _validate_caption_inputs(
            image_features,
            caption_inputs,
            image_feature_dim=self.image_feature_dim,
            vocabulary_size=self.vocabulary_size,
        )
        initial_hidden = self.image_projection(image_features).unsqueeze(0)
        embedded_tokens = self.token_embedding(caption_inputs)
        hidden_states, _ = self.recurrent_layer(
            embedded_tokens,
            initial_hidden,
        )
        return self.output_projection(hidden_states)


class TransformerCaptionDecoder(nn.Module):
    """Generate caption logits with a causal image-token Transformer.

    A projected image feature occupies sequence position zero. Caption tokens
    follow it and may attend to the image and preceding caption positions, but
    a causal mask prevents access to future tokens.

    Shape:
        - Image features: ``(N, image_feature_dim)``
        - Caption inputs: ``(N, T)`` integer token IDs
        - Returned logits: ``(N, T, vocabulary_size)``
    """

    def __init__(
        self,
        image_feature_dim: int,
        vocabulary_size: int,
        model_dim: int = 48,
        num_heads: int = 4,
        feed_forward_dim: int = 96,
        max_caption_length: int = 4,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.image_feature_dim = _positive_integer(
            image_feature_dim,
            name="image_feature_dim",
        )
        self.vocabulary_size = _positive_integer(
            vocabulary_size,
            name="vocabulary_size",
        )
        self.model_dim = _positive_integer(model_dim, name="model_dim")
        self.num_heads = _positive_integer(num_heads, name="num_heads")
        self.feed_forward_dim = _positive_integer(
            feed_forward_dim,
            name="feed_forward_dim",
        )
        self.max_caption_length = _positive_integer(
            max_caption_length,
            name="max_caption_length",
        )
        self.num_layers = _positive_integer(num_layers, name="num_layers")
        if self.model_dim % self.num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        if isinstance(dropout, bool) or not isinstance(dropout, (int, float)):
            raise TypeError("dropout must be numeric")
        self.dropout = float(dropout)
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")

        self.image_projection = nn.Linear(self.image_feature_dim, self.model_dim)
        self.token_embedding = nn.Embedding(self.vocabulary_size, self.model_dim)
        self.position_embedding = nn.Parameter(
            torch.randn(1, self.max_caption_length + 1, self.model_dim) * 0.02
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=self.num_heads,
            dim_feedforward=self.feed_forward_dim,
            dropout=self.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=self.num_layers,
            enable_nested_tensor=False,
        )
        self.output_projection = nn.Linear(
            self.model_dim,
            self.vocabulary_size,
        )

    def forward(
        self,
        image_features: torch.Tensor,
        caption_inputs: torch.Tensor,
    ) -> torch.Tensor:
        """Return causally masked next-token logits."""
        _validate_caption_inputs(
            image_features,
            caption_inputs,
            image_feature_dim=self.image_feature_dim,
            vocabulary_size=self.vocabulary_size,
        )
        number_tokens = caption_inputs.shape[1]
        if number_tokens > self.max_caption_length:
            raise ValueError("caption input exceeds max_caption_length")

        image_token = self.image_projection(image_features).unsqueeze(1)
        caption_tokens = self.token_embedding(caption_inputs)
        tokens = torch.cat((image_token, caption_tokens), dim=1)
        tokens = tokens + self.position_embedding[:, : tokens.shape[1]]
        mask = causal_attention_mask(
            tokens.shape[1],
            device=tokens.device,
            dtype=tokens.dtype,
        )
        encoded = self.transformer(tokens, mask=mask)
        return self.output_projection(encoded[:, 1:])


def _validate_caption_inputs(
    image_features: torch.Tensor,
    caption_inputs: torch.Tensor,
    *,
    image_feature_dim: int,
    vocabulary_size: int,
) -> None:
    """Validate the common decoder input contract."""
    if not isinstance(image_features, torch.Tensor):
        raise TypeError("image_features must be a PyTorch tensor")
    if image_features.ndim != 2 or image_features.shape[1] != image_feature_dim:
        raise ValueError(
            f"image_features must have shape (N, {image_feature_dim})"
        )
    if image_features.shape[0] == 0 or not image_features.is_floating_point():
        raise ValueError("image_features must be nonempty and floating-point")
    if not isinstance(caption_inputs, torch.Tensor):
        raise TypeError("caption_inputs must be a PyTorch tensor")
    if (
        caption_inputs.ndim != 2
        or caption_inputs.shape[0] != image_features.shape[0]
        or caption_inputs.shape[1] == 0
    ):
        raise ValueError("caption_inputs must have nonempty shape (N, T)")
    if caption_inputs.dtype not in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise TypeError("caption_inputs must contain integer token IDs")
    if caption_inputs.device != image_features.device:
        raise ValueError("image_features and caption_inputs must share a device")
    if torch.any(caption_inputs < 0) or torch.any(
        caption_inputs >= vocabulary_size
    ):
        raise ValueError("caption_inputs contain an index outside the vocabulary")


@torch.no_grad()
def generate_captions(
    model: nn.Module,
    image_features: torch.Tensor,
    *,
    start_token_id: int,
    end_token_id: int,
    max_length: int,
    forbidden_token_ids: Iterable[int] = (),
) -> torch.Tensor:
    """Greedily generate image-conditioned token sequences.

    The model must accept ``model(image_features, caption_prefix)`` and return
    logits with shape ``(N, T, V)``. The returned sequence excludes the start
    token. Once a sequence emits the end token, later positions remain fixed
    to that token while unfinished batch elements continue.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not isinstance(image_features, torch.Tensor):
        raise TypeError("image_features must be a PyTorch tensor")
    if image_features.ndim != 2 or image_features.shape[0] == 0:
        raise ValueError("image_features must have nonempty shape (N, D)")
    start_token_id = _positive_or_zero_integer(
        start_token_id,
        name="start_token_id",
    )
    end_token_id = _positive_or_zero_integer(end_token_id, name="end_token_id")
    max_length = _positive_integer(max_length, name="max_length")
    forbidden = tuple(
        _positive_or_zero_integer(value, name="forbidden_token_id")
        for value in forbidden_token_ids
    )

    model.eval()
    generated = torch.full(
        (image_features.shape[0], 1),
        start_token_id,
        dtype=torch.long,
        device=image_features.device,
    )
    finished = torch.zeros(
        image_features.shape[0],
        dtype=torch.bool,
        device=image_features.device,
    )
    for _ in range(max_length):
        next_logits = model(image_features, generated)[:, -1].clone()
        vocabulary_size = next_logits.shape[1]
        special_ids = forbidden + (start_token_id,)
        if any(token_id >= vocabulary_size for token_id in special_ids):
            raise ValueError("a forbidden token ID lies outside the vocabulary")
        next_logits[:, list(set(special_ids))] = -torch.inf
        next_ids = next_logits.argmax(dim=1)
        next_ids = torch.where(
            finished,
            torch.full_like(next_ids, end_token_id),
            next_ids,
        )
        generated = torch.cat((generated, next_ids[:, None]), dim=1)
        finished |= next_ids == end_token_id
        if finished.all():
            break
    return generated[:, 1:]


def caption_accuracies(
    generated_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
) -> tuple[float, float]:
    """Return token accuracy and exact-sequence accuracy.

    Both tensors must already contain the comparable generated/target portion;
    for example, remove an input-only start token from targets beforehand.
    """
    if not isinstance(generated_tokens, torch.Tensor) or not isinstance(
        target_tokens,
        torch.Tensor,
    ):
        raise TypeError("generated_tokens and target_tokens must be tensors")
    if generated_tokens.shape != target_tokens.shape:
        raise ValueError("generated_tokens and target_tokens must have equal shapes")
    if generated_tokens.ndim != 2 or generated_tokens.numel() == 0:
        raise ValueError("caption tokens must have nonempty shape (N, T)")
    matches = generated_tokens == target_tokens
    return (
        matches.float().mean().item(),
        matches.all(dim=1).float().mean().item(),
    )


def _positive_or_zero_integer(value: int, *, name: str) -> int:
    """Validate a nonnegative integer without accepting Boolean values."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value
