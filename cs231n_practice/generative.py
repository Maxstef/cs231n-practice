"""Utilities for small autoregressive generative-model experiments.

Sequence targets use shape (N, T) and model logits use shape (N, T, V),
where V is the number of possible next values.
"""

import math
from numbers import Integral, Real

import torch
from torch import nn
import torch.nn.functional as F


_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


def _token_matrix(tokens: torch.Tensor, *, name: str) -> torch.Tensor:
    """Validate and return a nonempty integer tensor with shape (N, T)."""
    if not isinstance(tokens, torch.Tensor):
        raise TypeError(f"{name} must be a PyTorch tensor")
    if tokens.ndim != 2 or any(size == 0 for size in tokens.shape):
        raise ValueError(f"{name} must have nonempty shape (N, T)")
    if tokens.dtype not in _INTEGER_DTYPES:
        raise TypeError(f"{name} must contain integer token IDs")
    return tokens


def _positive_integer(value: int, *, name: str) -> int:
    """Validate a positive integer without accepting Boolean values."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def images_to_sequences(images: torch.Tensor) -> torch.Tensor:
    """Flatten each image into a row-major sequence.

    The batch axis is preserved while the height and width axes are merged.
    Consequently, pixel (h, w) is placed at sequence position h * W + w.
    This is the raster ordering commonly used by pixel autoregressive models.

    Args:
        images: Nonempty image batch with shape (N, H, W).

    Returns:
        A view or reshaped tensor with shape (N, H * W). The result preserves
        the input dtype and device.
    """
    if not isinstance(images, torch.Tensor):
        raise TypeError("images must be a PyTorch tensor")
    if images.ndim != 3 or any(size == 0 for size in images.shape):
        raise ValueError("images must have nonempty shape (N, H, W)")
    return images.reshape(images.shape[0], -1)


def sequences_to_images(
    sequences: torch.Tensor,
    image_size: int,
) -> torch.Tensor:
    """Restore row-major sequences to square images.

    This is the inverse of images_to_sequences when the original images have
    height and width equal to image_size.

    Args:
        sequences: Integer-valued sequences with shape (N, T).
        image_size: Requested image height and width. T must equal
            image_size squared.

    Returns:
        Images with shape (N, image_size, image_size), preserving the input
        dtype and device.
    """
    sequences = _token_matrix(sequences, name="sequences")
    image_size = _positive_integer(image_size, name="image_size")
    if sequences.shape[1] != image_size * image_size:
        raise ValueError("sequence length does not match image_size squared")
    return sequences.reshape(sequences.shape[0], image_size, image_size)


def make_autoregressive_inputs(
    targets: torch.Tensor,
    bos_token: int,
) -> torch.Tensor:
    """Construct teacher-forcing inputs from target sequences.

    For targets [x1, x2, ..., xT], the returned input is
    [BOS, x1, ..., x(T-1)]. At every position, the model therefore receives
    only values that precede the target it must predict.

    Args:
        targets: Integer target IDs with shape (N, T).
        bos_token: Integer ID used to start every sequence.

    Returns:
        Shifted inputs with the same shape, dtype, and device as targets.
    """
    targets = _token_matrix(targets, name="targets")
    if isinstance(bos_token, bool) or not isinstance(bos_token, Integral):
        raise TypeError("bos_token must be an integer")
    beginning = torch.full(
        (targets.shape[0], 1),
        int(bos_token),
        dtype=targets.dtype,
        device=targets.device,
    )
    return torch.cat((beginning, targets[:, :-1]), dim=1)


def autoregressive_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Calculate next-token cross-entropy for an autoregressive sequence.

    Batch and time are flattened before calling PyTorch cross-entropy, while
    the vocabulary axis remains the class axis. Each target selects the log
    probability assigned to the observed token at the corresponding position.

    Args:
        logits: Floating-point class scores with shape (N, T, V).
        targets: Integer target IDs with shape (N, T), each in [0, V).
        reduction: Either "none", "mean", or "sum". With "none", the
            per-position result is restored to shape (N, T).

    Returns:
        A scalar loss for "mean" or "sum", or an (N, T) loss tensor for
        "none". A summed result is the batch sequence NLL; a mean result is
        average NLL per token.
    """
    if not isinstance(logits, torch.Tensor):
        raise TypeError("logits must be a PyTorch tensor")
    if logits.ndim != 3 or any(size == 0 for size in logits.shape):
        raise ValueError("logits must have nonempty shape (N, T, V)")
    if not logits.is_floating_point():
        raise TypeError("logits must contain floating-point values")
    targets = _token_matrix(targets, name="targets")
    if logits.shape[:2] != targets.shape:
        raise ValueError("targets must match the (N, T) logits axes")
    vocabulary_size = logits.shape[-1]
    if vocabulary_size < 2:
        raise ValueError("logits must contain at least two vocabulary classes")
    if torch.any(targets < 0) or torch.any(targets >= vocabulary_size):
        raise ValueError("targets contain an index outside the vocabulary")
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")

    losses = F.cross_entropy(
        logits.reshape(-1, vocabulary_size),
        targets.reshape(-1),
        reduction=reduction,
    )
    if reduction == "none":
        return losses.reshape(targets.shape)
    return losses


def nats_to_bits(value: float) -> float:
    """Convert a finite negative log-likelihood from nats to bits.

    Natural logarithms measure information in nats, while base-2 logarithms
    measure it in bits. The conversion divides by ln(2).

    Args:
        value: A finite numeric value measured using natural logarithms.

    Returns:
        The same information quantity measured in bits.
    """
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("value must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("value must be finite")
    return value / math.log(2.0)


@torch.no_grad()
def evaluate_autoregressive_model(
    model: nn.Module,
    sequences: torch.Tensor,
    *,
    bos_token: int,
    batch_size: int = 256,
    device: torch.device | str | None = None,
) -> tuple[float, float]:
    """Evaluate average autoregressive NLL per sequence position.

    Targets are shifted into teacher-forcing inputs, passed through the model,
    and compared with the original targets. Loss is summed across every batch
    before division by the total number of target positions, so partial final
    batches receive the correct weighting.

    The model must accept model(inputs) and return (logits, hidden), where
    logits has shape (N, T, V).

    Args:
        model: A recurrent autoregressive PyTorch module.
        sequences: Integer targets with shape (N, T).
        bos_token: Token ID prepended to each teacher-forcing input.
        batch_size: Maximum number of sequences evaluated together.
        device: Evaluation device. If omitted, the first model parameter's
            device is used, falling back to the sequence device for a
            parameterless model.

    Returns:
        A pair containing average NLL in nats per token and the equivalent
        bits per token.

    Notes:
        This function switches the model to evaluation mode and disables
        gradient recording. It does not restore the previous training mode.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    sequences = _token_matrix(sequences, name="sequences")
    batch_size = _positive_integer(batch_size, name="batch_size")
    if device is None:
        parameter = next(model.parameters(), None)
        device = parameter.device if parameter is not None else sequences.device
    device = torch.device(device)

    model.eval()
    total_nll = 0.0
    total_tokens = 0
    for start in range(0, sequences.shape[0], batch_size):
        targets = sequences[start : start + batch_size].to(device)
        inputs = make_autoregressive_inputs(targets, bos_token)
        logits, _ = model(inputs)
        total_nll += autoregressive_cross_entropy(
            logits,
            targets,
            reduction="sum",
        ).item()
        total_tokens += targets.numel()

    nll_per_token = total_nll / total_tokens
    return nll_per_token, nats_to_bits(nll_per_token)


@torch.no_grad()
def sample_autoregressive_sequences(
    model: nn.Module,
    number_sequences: int,
    sequence_length: int,
    *,
    bos_token: int,
    temperature: float = 1.0,
    seed: int = 0,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Generate discrete sequences by ancestral sampling.

    Every sequence starts with the same BOS token. At each step, the model
    produces next-token logits, temperature rescales them, and multinomial
    sampling chooses one token independently for each batch item. The sampled
    token becomes the next direct input, while the returned recurrent hidden
    state carries the complete earlier context forward.

    The model must accept model(current_tokens, hidden) and return
    (logits, next_hidden). Since generation processes one new position at a
    time, logits must have shape (N, 1, V).

    Args:
        model: A recurrent autoregressive PyTorch module.
        number_sequences: Number of independent sequences to generate.
        sequence_length: Number of tokens generated after BOS.
        bos_token: Integer token ID used as the initial input.
        temperature: Positive logit scale. Values below one sharpen the
            distribution; values above one increase randomness.
        seed: Seed for the local sampling generator.
        device: Sampling device. If omitted, the first model parameter's
            device is used, falling back to CPU for a parameterless model.

    Returns:
        Sampled token IDs with shape (number_sequences, sequence_length).
        BOS is not included in the returned sequences.

    Notes:
        This function switches the model to evaluation mode, disables gradient
        recording, and uses a local generator so equal seeds reproduce equal
        samples for the same model and device.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    number_sequences = _positive_integer(
        number_sequences,
        name="number_sequences",
    )
    sequence_length = _positive_integer(sequence_length, name="sequence_length")
    if isinstance(temperature, bool) or not isinstance(temperature, Real):
        raise TypeError("temperature must be numeric")
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be positive and finite")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise TypeError("seed must be an integer")
    if isinstance(bos_token, bool) or not isinstance(bos_token, Integral):
        raise TypeError("bos_token must be an integer")
    if device is None:
        parameter = next(model.parameters(), None)
        device = parameter.device if parameter is not None else torch.device("cpu")
    device = torch.device(device)

    model.eval()
    generator = torch.Generator(device=device).manual_seed(int(seed))
    current_tokens = torch.full(
        (number_sequences, 1),
        int(bos_token),
        dtype=torch.long,
        device=device,
    )
    hidden = None
    generated = []

    for _ in range(sequence_length):
        logits, hidden = model(current_tokens, hidden)
        if logits.ndim != 3 or logits.shape[:2] != (number_sequences, 1):
            raise ValueError("model must return logits with shape (N, 1, V)")
        probabilities = F.softmax(logits[:, -1, :] / temperature, dim=-1)
        current_tokens = torch.multinomial(
            probabilities,
            num_samples=1,
            generator=generator,
        )
        generated.append(current_tokens)

    return torch.cat(generated, dim=1)
