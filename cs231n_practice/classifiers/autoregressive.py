"""A small recurrent model for discrete autoregressive generation."""

from numbers import Integral

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


class BinaryImageAutoregressiveModel(nn.Module):
    """Predict each binary image value from its preceding raster context.

    Images are expected to be flattened into raster-ordered binary sequences.
    Teacher-forcing inputs contain three possible token IDs: zero, one, and
    BOS. An embedding maps each ID to a vector, a forward GRU accumulates
    earlier context, and a linear layer returns two logits for the next binary
    target.

    Although the direct input at a position is the previous token, the GRU
    hidden state can summarize every earlier token. Passing the returned hidden
    state back into a later one-step call enables sequential generation.

    Args:
        embedding_dim: Width of the learned embedding for each input token.
        hidden_dim: Width of the recurrent GRU state.

    Shape:
        - Input token_ids: (N, T)
        - Optional hidden state: (1, N, hidden_dim)
        - Output logits: (N, T, 2)
        - Returned hidden state: (1, N, hidden_dim)
    """

    def __init__(self, embedding_dim: int = 16, hidden_dim: int = 64) -> None:
        super().__init__()
        embedding_dim = _positive_integer(embedding_dim, name="embedding_dim")
        hidden_dim = _positive_integer(hidden_dim, name="hidden_dim")
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(3, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, 2)

    def forward(
        self,
        token_ids: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one or more causal recurrent prediction steps.

        Args:
            token_ids: Integer IDs with shape (N, T). Valid values are zero,
                one, and two, where two is conventionally the BOS token.
            hidden: Optional GRU state with shape (1, N, hidden_dim). Omit it
                for the start of a sequence; reuse the returned state when
                generating later tokens one step at a time.

        Returns:
            A pair of binary logits with shape (N, T, 2) and the final hidden
            state with shape (1, N, hidden_dim). The logits are raw scores and
            should be passed to cross-entropy or softmax as appropriate.
        """
        if not isinstance(token_ids, torch.Tensor):
            raise TypeError("token_ids must be a PyTorch tensor")
        if token_ids.ndim != 2 or any(size == 0 for size in token_ids.shape):
            raise ValueError("token_ids must have nonempty shape (N, T)")
        if token_ids.dtype not in {torch.int32, torch.int64}:
            raise TypeError("token_ids must contain integer indexes")
        if torch.any(token_ids < 0) or torch.any(token_ids >= 3):
            raise ValueError("token_ids must contain only 0, 1, or 2")

        embedded = self.embedding(token_ids)
        features, hidden = self.gru(embedded, hidden)
        return self.output(features), hidden
