"""Utilities for small autoregressive and autoencoder experiments.

Sequence targets use shape (N, T) and model logits use shape (N, T, V),
where V is the number of possible next values.
"""

import math
from numbers import Integral, Real

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


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


def _floating_image_batch(images: torch.Tensor, *, name: str) -> torch.Tensor:
    """Validate grayscale ``(N,H,W)`` or channel-first ``(N,C,H,W)`` images."""
    if not isinstance(images, torch.Tensor):
        raise TypeError(f"{name} must be a PyTorch tensor")
    if images.ndim not in {3, 4} or any(size == 0 for size in images.shape):
        raise ValueError(
            f"{name} must have nonempty shape (N, H, W) or (N, C, H, W)"
        )
    if not images.is_floating_point():
        raise TypeError(f"{name} must contain floating-point values")
    return images


def _model_device(model: nn.Module) -> torch.device:
    """Return the device of the first model parameter."""
    parameter = next(model.parameters(), None)
    if parameter is None:
        raise ValueError("model must contain at least one parameter")
    return parameter.device


def binary_reconstruction_loss(
    reconstruction_logits: torch.Tensor,
    targets: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Calculate stable Bernoulli reconstruction negative log-likelihood.

    The decoder supplies one raw logit per target pixel. PyTorch combines the
    sigmoid transformation and binary cross-entropy in a single numerically
    stable operation.

    Args:
        reconstruction_logits: Floating-point decoder scores with the same
            shape as targets.
        targets: Floating-point target values, normally in the interval [0, 1].
        reduction: Either "none", "mean", or "sum".

    Returns:
        Elementwise losses for "none" or a scalar reduced loss otherwise.
    """
    if not isinstance(reconstruction_logits, torch.Tensor):
        raise TypeError("reconstruction_logits must be a PyTorch tensor")
    if not reconstruction_logits.is_floating_point():
        raise TypeError("reconstruction_logits must be floating-point")
    targets = _floating_image_batch(targets, name="targets")
    if reconstruction_logits.shape != targets.shape:
        raise ValueError("reconstruction_logits and targets must have equal shapes")
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    if torch.any(targets < 0.0) or torch.any(targets > 1.0):
        raise ValueError("targets must lie in the interval [0, 1]")
    return F.binary_cross_entropy_with_logits(
        reconstruction_logits,
        targets,
        reduction=reduction,
    )


def autoencoder_training_step(
    model: nn.Module,
    images: torch.Tensor,
    optimizer: torch.optim.Optimizer,
) -> float:
    """Perform one reconstruction update and return mean pixel BCE.

    Images are moved to the model's device. Gradients flow through both the
    decoder and encoder because they are connected by the latent code.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be a torch optimizer")
    images = _floating_image_batch(images, name="images").to(_model_device(model))
    model.train()
    optimizer.zero_grad()
    reconstruction_logits = model(images)
    loss = binary_reconstruction_loss(reconstruction_logits, images)
    loss.backward()
    optimizer.step()
    return float(loss.item())


def train_autoencoder(
    model: nn.Module,
    images: torch.Tensor,
    *,
    epochs: int = 25,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
) -> list[float]:
    """Train a deterministic autoencoder and return mean pixel BCE per epoch.

    Batch losses are weighted by batch size before epoch averaging, so a
    smaller final batch does not receive the same influence as a full batch.
    All images must have equal spatial dimensions.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    images = _floating_image_batch(images, name="images")
    epochs = _positive_integer(epochs, name="epochs")
    batch_size = _positive_integer(batch_size, name="batch_size")
    if isinstance(learning_rate, bool) or not isinstance(learning_rate, Real):
        raise TypeError("learning_rate must be numeric")
    learning_rate = float(learning_rate)
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive and finite")

    loader = DataLoader(
        TensorDataset(images),
        batch_size=batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    for _ in range(epochs):
        total_loss = 0.0
        total_images = 0
        for (batch,) in loader:
            batch_loss = autoencoder_training_step(model, batch, optimizer)
            total_loss += batch_loss * batch.shape[0]
            total_images += batch.shape[0]
        history.append(total_loss / total_images)
    return history


@torch.no_grad()
def reconstruct_images(
    model: nn.Module,
    images: torch.Tensor,
) -> torch.Tensor:
    """Return sigmoid reconstruction probabilities on the model's device."""
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    images = _floating_image_batch(images, name="images")
    model.eval()
    logits = model(images.to(_model_device(model)))
    return torch.sigmoid(logits)


@torch.no_grad()
def encode_images(
    model: nn.Module,
    images: torch.Tensor,
) -> torch.Tensor:
    """Encode an image batch and return latent codes on the CPU."""
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not hasattr(model, "encode"):
        raise ValueError("model must provide an encode method")
    images = _floating_image_batch(images, name="images")
    model.eval()
    return model.encode(images.to(_model_device(model))).cpu()


@torch.no_grad()
def interpolate_latent_codes(
    model: nn.Module,
    first_image: torch.Tensor,
    second_image: torch.Tensor,
    *,
    number_steps: int = 9,
) -> torch.Tensor:
    """Decode evenly spaced points between two encoded images.

    The first and final results are reconstructions of the supplied endpoint
    images. Intermediate results decode convex combinations of their codes.
    A deterministic encoder may return codes directly; a VAE encoder may
    return ``(mean, log_variance)``, in which case posterior means are used.
    Endpoints may be grayscale ``(H,W)`` or channel-first ``(C,H,W)`` images.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not hasattr(model, "encode") or not hasattr(model, "decode"):
        raise ValueError("model must provide encode and decode methods")
    number_steps = _positive_integer(number_steps, name="number_steps")
    if number_steps < 2:
        raise ValueError("number_steps must be at least two")
    if not isinstance(first_image, torch.Tensor) or not isinstance(
        second_image,
        torch.Tensor,
    ):
        raise TypeError("endpoint images must be PyTorch tensors")
    if first_image.shape != second_image.shape or first_image.ndim not in {2, 3}:
        raise ValueError(
            "endpoint images must have equal shape (H, W) or (C, H, W)"
        )
    if not first_image.is_floating_point() or not second_image.is_floating_point():
        raise TypeError("endpoint images must contain floating-point values")

    device = _model_device(model)
    model.eval()
    endpoints = model.encode(torch.stack((first_image, second_image)).to(device))
    if isinstance(endpoints, tuple):
        endpoints = endpoints[0]
    if not isinstance(endpoints, torch.Tensor) or endpoints.ndim != 2:
        raise ValueError("model.encode must return codes or means with shape (N, D)")
    alpha = torch.linspace(0.0, 1.0, number_steps, device=device).unsqueeze(1)
    latent_codes = (1.0 - alpha) * endpoints[0] + alpha * endpoints[1]
    return torch.sigmoid(model.decode(latent_codes)).cpu()


def spatial_sharpness(images: torch.Tensor) -> torch.Tensor:
    """Return mean horizontal-plus-vertical absolute pixel variation.

    The result is a simple edge-strength proxy rather than a perceptual image
    quality metric. Larger values indicate stronger average local changes.

    Args:
        images: Grayscale ``(N,H,W)`` or channel-first ``(N,C,H,W)`` images.

    Returns:
        A scalar tensor on the input device.
    """
    images = _floating_image_batch(images, name="images")
    if images.shape[-2] < 2 or images.shape[-1] < 2:
        raise ValueError("image height and width must both be at least two")
    vertical = torch.abs(images[..., 1:, :] - images[..., :-1, :]).mean()
    horizontal = torch.abs(images[..., :, 1:] - images[..., :, :-1]).mean()
    return vertical + horizontal


@torch.no_grad()
def decode_latent_grid(
    model: nn.Module,
    observed_codes: torch.Tensor,
    *,
    grid_size: int = 8,
    margin: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode a rectangular grid around observed two-dimensional codes.

    Args:
        model: An autoencoder with a decode method.
        observed_codes: CPU or device codes with shape (N, 2).
        grid_size: Number of positions sampled along each latent axis.
        margin: Nonnegative fraction by which each observed axis range is
            extended on both sides.

    Returns:
        Decoded sigmoid probabilities with shape (grid_size squared, H, W),
        ascending first-axis values, and ascending second-axis values. Decoded
        images are ordered row by row from larger to smaller second-axis value,
        matching an image grid displayed from top to bottom.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not hasattr(model, "decode"):
        raise ValueError("model must provide a decode method")
    if not isinstance(observed_codes, torch.Tensor):
        raise TypeError("observed_codes must be a PyTorch tensor")
    if (
        observed_codes.ndim != 2
        or observed_codes.shape[0] == 0
        or observed_codes.shape[1] != 2
    ):
        raise ValueError("observed_codes must have nonempty shape (N, 2)")
    if not observed_codes.is_floating_point():
        raise TypeError("observed_codes must contain floating-point values")
    grid_size = _positive_integer(grid_size, name="grid_size")
    if grid_size < 2:
        raise ValueError("grid_size must be at least two")
    if isinstance(margin, bool) or not isinstance(margin, Real):
        raise TypeError("margin must be numeric")
    margin = float(margin)
    if not math.isfinite(margin) or margin < 0.0:
        raise ValueError("margin must be nonnegative and finite")

    codes = observed_codes.detach().cpu()
    minimum = codes.min(dim=0).values
    maximum = codes.max(dim=0).values
    span = maximum - minimum
    first_axis = torch.linspace(
        minimum[0] - margin * span[0],
        maximum[0] + margin * span[0],
        grid_size,
    )
    second_axis = torch.linspace(
        minimum[1] - margin * span[1],
        maximum[1] + margin * span[1],
        grid_size,
    )
    grid_second, grid_first = torch.meshgrid(
        second_axis.flip(0),
        first_axis,
        indexing="ij",
    )
    grid_codes = torch.stack(
        (grid_first.reshape(-1), grid_second.reshape(-1)),
        dim=1,
    )
    model.eval()
    decoded = torch.sigmoid(
        model.decode(grid_codes.to(_model_device(model)))
    ).cpu()
    return decoded, first_axis, second_axis


def reparameterize(
    mean: torch.Tensor,
    log_variance: torch.Tensor,
    epsilon: torch.Tensor | None = None,
) -> torch.Tensor:
    """Draw a differentiable sample from a diagonal Gaussian.

    The random variable is rewritten as ``mean + std * epsilon`` so that the
    sampling noise is independent of the learned distribution parameters.
    Gradients can consequently flow from the sampled code into both encoder
    outputs.

    Args:
        mean: Distribution means with shape ``(N, D)``.
        log_variance: Logarithms of the diagonal variances, also ``(N, D)``.
        epsilon: Optional fixed standard-normal noise with matching shape.
            Supplying it is useful for deterministic tests; otherwise fresh
            noise is sampled with the mean's dtype and device.

    Returns:
        Reparameterized latent codes with shape ``(N, D)``.
    """
    mean = _floating_latent_matrix(mean, name="mean")
    log_variance = _floating_latent_matrix(
        log_variance,
        name="log_variance",
    )
    if mean.shape != log_variance.shape:
        raise ValueError("mean and log_variance must have equal shapes")
    if mean.device != log_variance.device or mean.dtype != log_variance.dtype:
        raise ValueError("mean and log_variance must share dtype and device")
    if epsilon is None:
        epsilon = torch.randn_like(mean)
    else:
        epsilon = _floating_latent_matrix(epsilon, name="epsilon")
        if epsilon.shape != mean.shape:
            raise ValueError("epsilon must have the same shape as mean")
        if epsilon.device != mean.device or epsilon.dtype != mean.dtype:
            raise ValueError("epsilon must share the mean's dtype and device")

    standard_deviation = torch.exp(0.5 * log_variance)
    return mean + standard_deviation * epsilon


def gaussian_kl_divergence(
    mean: torch.Tensor,
    log_variance: torch.Tensor,
) -> torch.Tensor:
    """Calculate KL divergence from diagonal Gaussians to ``N(0, I)``.

    Latent-dimension contributions are summed while the batch axis is retained.
    The returned tensor therefore contains one KL value per input example.
    """
    mean = _floating_latent_matrix(mean, name="mean")
    log_variance = _floating_latent_matrix(
        log_variance,
        name="log_variance",
    )
    if mean.shape != log_variance.shape:
        raise ValueError("mean and log_variance must have equal shapes")
    if mean.device != log_variance.device or mean.dtype != log_variance.dtype:
        raise ValueError("mean and log_variance must share dtype and device")
    return 0.5 * torch.sum(
        torch.exp(log_variance) + mean.square() - 1.0 - log_variance,
        dim=1,
    )


def vae_loss(
    reconstruction_logits: torch.Tensor,
    targets: torch.Tensor,
    mean: torch.Tensor,
    log_variance: torch.Tensor,
    *,
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return negative ELBO and its per-image reconstruction and KL terms.

    Pixel Bernoulli NLL is summed within each image and averaged across the
    batch. KL divergence is likewise summed across latent dimensions and then
    averaged across the batch, placing both terms on a per-example scale.
    ``beta=1`` gives the standard VAE objective.
    """
    if not isinstance(reconstruction_logits, torch.Tensor):
        raise TypeError("reconstruction_logits must be a PyTorch tensor")
    if not reconstruction_logits.is_floating_point():
        raise TypeError("reconstruction_logits must be floating-point")
    if not isinstance(targets, torch.Tensor):
        raise TypeError("targets must be a PyTorch tensor")
    if targets.ndim < 2 or any(size == 0 for size in targets.shape):
        raise ValueError("targets must have a nonempty batch and feature dimensions")
    if not targets.is_floating_point():
        raise TypeError("targets must contain floating-point values")
    if torch.any(targets < 0.0) or torch.any(targets > 1.0):
        raise ValueError("targets must lie in the interval [0, 1]")
    if reconstruction_logits.shape != targets.shape:
        raise ValueError("reconstruction_logits and targets must have equal shapes")
    if reconstruction_logits.device != targets.device:
        raise ValueError("reconstruction_logits and targets must share a device")
    if isinstance(beta, bool) or not isinstance(beta, Real):
        raise TypeError("beta must be numeric")
    beta = float(beta)
    if not math.isfinite(beta) or beta < 0.0:
        raise ValueError("beta must be nonnegative and finite")

    per_image_reconstruction = F.binary_cross_entropy_with_logits(
        reconstruction_logits,
        targets,
        reduction="none",
    ).flatten(start_dim=1).sum(dim=1)
    reconstruction_loss = per_image_reconstruction.mean()
    per_image_kl = gaussian_kl_divergence(mean, log_variance)
    if per_image_kl.shape[0] != targets.shape[0]:
        raise ValueError("latent parameters and targets must share a batch size")
    kl_loss = per_image_kl.mean()
    total_loss = reconstruction_loss + beta * kl_loss
    return total_loss, reconstruction_loss, kl_loss


def train_vae(
    model: nn.Module,
    images: torch.Tensor,
    *,
    epochs: int = 30,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    beta: float = 1.0,
) -> dict[str, list[float]]:
    """Train a VAE and return per-epoch negative-ELBO components.

    The model must return ``(logits, mean, log_variance, latent_codes)``.
    Histories are averages per image and correctly weight a partial final
    batch. The sampled latent codes are part of the forward result but are not
    directly needed by the standard VAE loss.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    images = _floating_image_batch(images, name="images")
    epochs = _positive_integer(epochs, name="epochs")
    batch_size = _positive_integer(batch_size, name="batch_size")
    if isinstance(learning_rate, bool) or not isinstance(learning_rate, Real):
        raise TypeError("learning_rate must be numeric")
    learning_rate = float(learning_rate)
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive and finite")
    if isinstance(beta, bool) or not isinstance(beta, Real):
        raise TypeError("beta must be numeric")
    beta = float(beta)
    if not math.isfinite(beta) or beta < 0.0:
        raise ValueError("beta must be nonnegative and finite")

    loader = DataLoader(TensorDataset(images), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = {"total": [], "reconstruction": [], "kl": []}
    device = _model_device(model)

    for _ in range(epochs):
        totals = {name: 0.0 for name in history}
        number_images = 0
        model.train()
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            reconstruction_logits, mean, log_variance, _ = model(batch)
            total_loss, reconstruction_loss, kl_loss = vae_loss(
                reconstruction_logits,
                batch,
                mean,
                log_variance,
                beta=beta,
            )
            total_loss.backward()
            optimizer.step()

            count = batch.shape[0]
            totals["total"] += total_loss.item() * count
            totals["reconstruction"] += reconstruction_loss.item() * count
            totals["kl"] += kl_loss.item() * count
            number_images += count

        for name in history:
            history[name].append(totals[name] / number_images)
    return history


@torch.no_grad()
def reconstruct_with_vae(
    model: nn.Module,
    images: torch.Tensor,
    *,
    sample: bool = False,
) -> torch.Tensor:
    """Reconstruct images from posterior means or posterior samples.

    ``sample=False`` decodes the posterior mean and is deterministic for an
    otherwise deterministic model in evaluation mode. ``sample=True`` draws a
    fresh reparameterized code and allows the posterior variance to influence
    the reconstruction. Results remain on the model's device.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not hasattr(model, "encode") or not hasattr(model, "decode"):
        raise ValueError("model must provide encode and decode methods")
    if not isinstance(sample, bool):
        raise TypeError("sample must be Boolean")
    images = _floating_image_batch(images, name="images")
    model.eval()
    mean, log_variance = model.encode(images.to(_model_device(model)))
    latent_codes = reparameterize(mean, log_variance) if sample else mean
    return torch.sigmoid(model.decode(latent_codes))


@torch.no_grad()
def sample_from_vae(
    model: nn.Module,
    number_images: int,
    *,
    seed: int = 0,
) -> torch.Tensor:
    """Generate images by decoding standard-normal prior samples.

    The encoder is not used. A local random generator makes samples
    reproducible for an equal model, seed, and device. Returned probabilities
    are moved to the CPU for convenient plotting.
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not hasattr(model, "decode") or not hasattr(model, "latent_dim"):
        raise ValueError("model must provide decode and latent_dim")
    number_images = _positive_integer(number_images, name="number_images")
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise TypeError("seed must be an integer")

    device = _model_device(model)
    generator = torch.Generator(device=device).manual_seed(int(seed))
    latent_codes = torch.randn(
        number_images,
        model.latent_dim,
        generator=generator,
        device=device,
    )
    model.eval()
    return torch.sigmoid(model.decode(latent_codes)).cpu()


@torch.no_grad()
def encode_vae_means(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Return approximate-posterior centers on the CPU for visualization."""
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not hasattr(model, "encode"):
        raise ValueError("model must provide an encode method")
    images = _floating_image_batch(images, name="images")
    model.eval()
    mean, _ = model.encode(images.to(_model_device(model)))
    return mean.cpu()


def _floating_latent_matrix(
    values: torch.Tensor,
    *,
    name: str,
) -> torch.Tensor:
    """Validate a nonempty floating-point latent matrix with shape (N, D)."""
    if not isinstance(values, torch.Tensor):
        raise TypeError(f"{name} must be a PyTorch tensor")
    if values.ndim != 2 or any(size == 0 for size in values.shape):
        raise ValueError(f"{name} must have nonempty shape (N, D)")
    if not values.is_floating_point():
        raise TypeError(f"{name} must contain floating-point values")
    return values


def sample_latent(
    number_samples: int,
    latent_dim: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    random_generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw independent standard-normal latent vectors.

    Args:
        number_samples: Number of latent vectors to draw.
        latent_dim: Number of coordinates in each vector.
        device: Device on which to create the vectors.
        dtype: Floating-point dtype. PyTorch's current default is used when
            omitted.
        random_generator: Optional local random generator for reproducibility.

    Returns:
        A tensor with shape ``(number_samples, latent_dim)`` sampled from
        ``N(0, I)``.
    """
    number_samples = _positive_integer(number_samples, name="number_samples")
    latent_dim = _positive_integer(latent_dim, name="latent_dim")
    if dtype is not None and not torch.empty((), dtype=dtype).is_floating_point():
        raise TypeError("dtype must be a floating-point dtype")
    return torch.randn(
        number_samples,
        latent_dim,
        device=device,
        dtype=dtype,
        generator=random_generator,
    )


def gan_discriminator_loss(
    real_logits: torch.Tensor,
    generated_logits: torch.Tensor,
) -> torch.Tensor:
    """Calculate the non-saturating GAN discriminator loss.

    Real examples receive target one and generated examples receive target
    zero. The sigmoid is deliberately not applied by the discriminator because
    ``binary_cross_entropy_with_logits`` combines both operations stably.
    """
    for logits, name in (
        (real_logits, "real_logits"),
        (generated_logits, "generated_logits"),
    ):
        if not isinstance(logits, torch.Tensor):
            raise TypeError(f"{name} must be a PyTorch tensor")
        if logits.numel() == 0:
            raise ValueError(f"{name} must be nonempty")
        if not logits.is_floating_point():
            raise TypeError(f"{name} must contain floating-point values")
    real_loss = F.binary_cross_entropy_with_logits(
        real_logits,
        torch.ones_like(real_logits),
    )
    generated_loss = F.binary_cross_entropy_with_logits(
        generated_logits,
        torch.zeros_like(generated_logits),
    )
    return real_loss + generated_loss


def gan_generator_loss(generated_logits: torch.Tensor) -> torch.Tensor:
    """Calculate the non-saturating GAN generator loss.

    The generator asks the discriminator to classify generated examples as
    real, so their target is one. This form usually provides stronger early
    gradients than directly minimizing ``log(1 - D(G(z)))``.
    """
    if not isinstance(generated_logits, torch.Tensor):
        raise TypeError("generated_logits must be a PyTorch tensor")
    if generated_logits.numel() == 0:
        raise ValueError("generated_logits must be nonempty")
    if not generated_logits.is_floating_point():
        raise TypeError("generated_logits must contain floating-point values")
    return F.binary_cross_entropy_with_logits(
        generated_logits,
        torch.ones_like(generated_logits),
    )


def set_requires_grad(model: nn.Module, requires_grad: bool) -> None:
    """Enable or disable gradient accumulation for every model parameter."""
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if not isinstance(requires_grad, bool):
        raise TypeError("requires_grad must be Boolean")
    for parameter in model.parameters():
        parameter.requires_grad_(requires_grad)


def gan_discriminator_step(
    generator: nn.Module,
    discriminator: nn.Module,
    optimizer: torch.optim.Optimizer,
    real_samples: torch.Tensor,
    latent_vectors: torch.Tensor,
) -> torch.Tensor:
    """Perform one discriminator-only GAN update.

    Generated samples are detached before entering the discriminator. Thus the
    discriminator learns from both real and fake examples while no gradient or
    optimizer update reaches the generator.

    Returns:
        A detached scalar loss tensor.
    """
    if not isinstance(generator, nn.Module) or not isinstance(discriminator, nn.Module):
        raise TypeError("generator and discriminator must be torch.nn modules")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be a torch optimizer")
    if not isinstance(real_samples, torch.Tensor):
        raise TypeError("real_samples must be a PyTorch tensor")
    latent_vectors = _floating_latent_matrix(
        latent_vectors,
        name="latent_vectors",
    )

    discriminator.train()
    optimizer.zero_grad()
    generated_samples = generator(latent_vectors).detach()
    loss = gan_discriminator_loss(
        discriminator(real_samples),
        discriminator(generated_samples),
    )
    loss.backward()
    optimizer.step()
    return loss.detach()


def gan_generator_step(
    generator: nn.Module,
    discriminator: nn.Module,
    optimizer: torch.optim.Optimizer,
    latent_vectors: torch.Tensor,
) -> torch.Tensor:
    """Perform one generator-only update through a frozen discriminator.

    Freezing discriminator parameters avoids calculating their gradients, but
    differentiation still passes through the discriminator operations to the
    generated samples and then into the generator. Every discriminator
    parameter's original ``requires_grad`` state is restored afterward.

    Returns:
        A detached scalar loss tensor.
    """
    if not isinstance(generator, nn.Module) or not isinstance(discriminator, nn.Module):
        raise TypeError("generator and discriminator must be torch.nn modules")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be a torch optimizer")
    latent_vectors = _floating_latent_matrix(
        latent_vectors,
        name="latent_vectors",
    )

    original_gradient_states = [
        parameter.requires_grad for parameter in discriminator.parameters()
    ]
    generator.train()
    optimizer.zero_grad()
    set_requires_grad(discriminator, False)
    try:
        generated_logits = discriminator(generator(latent_vectors))
        loss = gan_generator_loss(generated_logits)
        loss.backward()
        optimizer.step()
    finally:
        for parameter, requires_grad in zip(
            discriminator.parameters(),
            original_gradient_states,
        ):
            parameter.requires_grad_(requires_grad)
    return loss.detach()
