"""Reusable NumPy layers for recurrent neural networks.

Sequence inputs use shape ``(N, T, D)``: ``N`` sequences, ``T`` time steps,
and ``D`` input features per step. Hidden states use ``H`` features. This
module currently contains forward functions only; backward functions will be
added after backpropagation through time is developed and gradient-checked.
"""

import numpy as np

RnnStepCache = tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]
RnnCache = list[RnnStepCache]
MaskedRnnStepCache = tuple[RnnStepCache, np.ndarray]
MaskedRnnCache = list[MaskedRnnStepCache]


def _as_real_array(value: np.ndarray, *, name: str) -> np.ndarray:
    """Return a nonempty, finite NumPy array containing real numbers."""
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.complexfloating
    ):
        raise TypeError(f"{name} must contain real numeric values")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array


def rnn_step_forward(
    x_t: np.ndarray,
    h_previous: np.ndarray,
    weights_x: np.ndarray,
    weights_h: np.ndarray,
    bias: np.ndarray,
) -> tuple[np.ndarray, RnnStepCache]:
    """Compute one vanilla-RNN time step.

    The recurrent update is

    ``h_t = tanh(x_t @ weights_x + h_previous @ weights_h + bias)``.

    Args:
        x_t: Input at one time step with shape ``(N, D)``.
        h_previous: Previous hidden state with shape ``(N, H)``.
        weights_x: Input-to-hidden weights with shape ``(D, H)``.
        weights_h: Hidden-to-hidden weights with shape ``(H, H)``.
        bias: Hidden bias with shape ``(H,)``.

    Returns:
        The next hidden state with shape ``(N, H)`` and a cache for the future
        backward function. The cached next state will allow the tanh derivative
        to be evaluated as ``1 - h_t**2``.
    """
    x_t = _as_real_array(x_t, name="x_t")
    h_previous = _as_real_array(h_previous, name="h_previous")
    weights_x = _as_real_array(weights_x, name="weights_x")
    weights_h = _as_real_array(weights_h, name="weights_h")
    bias = _as_real_array(bias, name="bias")

    if x_t.ndim != 2 or h_previous.ndim != 2:
        raise ValueError("x_t and h_previous must be two-dimensional")
    if weights_x.ndim != 2 or weights_h.ndim != 2:
        raise ValueError("weights_x and weights_h must be two-dimensional")
    if bias.ndim != 1:
        raise ValueError("bias must be one-dimensional")

    num_examples, input_dim = x_t.shape
    hidden_dim = h_previous.shape[1]
    if h_previous.shape[0] != num_examples:
        raise ValueError("x_t and h_previous must have equal batch sizes")
    if weights_x.shape != (input_dim, hidden_dim):
        raise ValueError("weights_x must have shape (D, H)")
    if weights_h.shape != (hidden_dim, hidden_dim):
        raise ValueError("weights_h must have shape (H, H)")
    if bias.shape != (hidden_dim,):
        raise ValueError("bias must have shape (H,)")

    calculation_dtype = np.result_type(
        x_t.dtype,
        h_previous.dtype,
        weights_x.dtype,
        weights_h.dtype,
        bias.dtype,
        np.float32,
    )
    x_t = x_t.astype(calculation_dtype, copy=False)
    h_previous = h_previous.astype(calculation_dtype, copy=False)
    weights_x = weights_x.astype(calculation_dtype, copy=False)
    weights_h = weights_h.astype(calculation_dtype, copy=False)
    bias = bias.astype(calculation_dtype, copy=False)

    pre_activation = x_t @ weights_x + h_previous @ weights_h + bias
    h_next = np.tanh(pre_activation)
    cache = (x_t, h_previous, weights_x, weights_h, h_next)
    return h_next, cache


def rnn_forward(
    x: np.ndarray,
    h0: np.ndarray,
    weights_x: np.ndarray,
    weights_h: np.ndarray,
    bias: np.ndarray,
) -> tuple[np.ndarray, RnnCache]:
    """Run a vanilla RNN forward over a complete sequence batch.

    Args:
        x: Input sequences with shape ``(N, T, D)``.
        h0: Initial hidden state with shape ``(N, H)``.
        weights_x: Input-to-hidden weights with shape ``(D, H)``.
        weights_h: Hidden-to-hidden weights with shape ``(H, H)``.
        bias: Hidden bias with shape ``(H,)``.

    Returns:
        All hidden states with shape ``(N, T, H)`` and an ordered list
        containing one step cache for each of the ``T`` time steps.

    Notes:
        The same parameters are reused at every time step. Processing is
        sequential along the time axis because each state depends on the
        preceding state, while all ``N`` sequences are processed together.
    """
    x = _as_real_array(x, name="x")
    h0 = _as_real_array(h0, name="h0")
    weights_x = _as_real_array(weights_x, name="weights_x")
    weights_h = _as_real_array(weights_h, name="weights_h")
    bias = _as_real_array(bias, name="bias")

    if x.ndim != 3:
        raise ValueError("x must have shape (N, T, D)")
    if h0.ndim != 2:
        raise ValueError("h0 must have shape (N, H)")
    if x.shape[0] != h0.shape[0]:
        raise ValueError("x and h0 must have equal batch sizes")

    num_examples, num_steps, _ = x.shape
    hidden_dim = h0.shape[1]
    calculation_dtype = np.result_type(
        x.dtype,
        h0.dtype,
        weights_x.dtype,
        weights_h.dtype,
        bias.dtype,
        np.float32,
    )
    hidden_states = np.empty(
        (num_examples, num_steps, hidden_dim),
        dtype=calculation_dtype,
    )
    caches: RnnCache = []
    h_previous = h0

    for time_index in range(num_steps):
        h_previous, cache = rnn_step_forward(
            x[:, time_index, :],
            h_previous,
            weights_x,
            weights_h,
            bias,
        )
        hidden_states[:, time_index, :] = h_previous
        caches.append(cache)

    return hidden_states, caches


def rnn_forward_masked(
    x: np.ndarray,
    h0: np.ndarray,
    lengths: np.ndarray,
    weights_x: np.ndarray,
    weights_h: np.ndarray,
    bias: np.ndarray,
) -> tuple[np.ndarray, MaskedRnnCache]:
    """Run a vanilla RNN while preserving state across padded time steps.

    ``lengths[n]`` gives the number of valid steps in sequence ``n``. At time
    ``t``, examples for which ``t >= lengths[n]`` retain their previous hidden
    state instead of accepting the recurrent candidate computed from padding.

    Args:
        x: Padded input sequences with shape ``(N, T, D)``.
        h0: Initial hidden state with shape ``(N, H)``.
        lengths: Integer valid lengths with shape ``(N,)`` and values in
            ``[0, T]``.
        weights_x: Input-to-hidden weights with shape ``(D, H)``.
        weights_h: Hidden-to-hidden weights with shape ``(H, H)``.
        bias: Hidden bias with shape ``(H,)``.

    Returns:
        Hidden states with shape ``(N, T, H)`` and one cache per time step.
        Each cache contains the ordinary recurrent-step cache and its Boolean
        mask with shape ``(N, 1)``. Retaining the mask prepares the forward API
        for a future masked backward pass.
    """
    x = _as_real_array(x, name="x")
    h0 = _as_real_array(h0, name="h0")
    lengths = np.asarray(lengths)
    weights_x = _as_real_array(weights_x, name="weights_x")
    weights_h = _as_real_array(weights_h, name="weights_h")
    bias = _as_real_array(bias, name="bias")

    if x.ndim != 3:
        raise ValueError("x must have shape (N, T, D)")
    if h0.ndim != 2:
        raise ValueError("h0 must have shape (N, H)")
    if x.shape[0] != h0.shape[0]:
        raise ValueError("x and h0 must have equal batch sizes")
    if lengths.ndim != 1 or lengths.shape != (x.shape[0],):
        raise ValueError("lengths must have shape (N,)")
    if not np.issubdtype(lengths.dtype, np.integer):
        raise TypeError("lengths must contain integers")
    if np.any((lengths < 0) | (lengths > x.shape[1])):
        raise ValueError("lengths values must be between 0 and T")

    num_examples, num_steps, _ = x.shape
    hidden_dim = h0.shape[1]
    calculation_dtype = np.result_type(
        x.dtype,
        h0.dtype,
        weights_x.dtype,
        weights_h.dtype,
        bias.dtype,
        np.float32,
    )
    hidden_states = np.empty(
        (num_examples, num_steps, hidden_dim),
        dtype=calculation_dtype,
    )
    caches: MaskedRnnCache = []
    h_previous = h0

    for time_index in range(num_steps):
        candidate, step_cache = rnn_step_forward(
            x[:, time_index, :],
            h_previous,
            weights_x,
            weights_h,
            bias,
        )
        valid = (time_index < lengths)[:, None]
        h_previous = np.where(valid, candidate, h_previous)
        hidden_states[:, time_index, :] = h_previous
        caches.append((step_cache, valid))

    return hidden_states, caches
