"""Reusable NumPy layers for recurrent neural networks.

Sequence inputs use shape ``(N, T, D)``: ``N`` sequences, ``T`` time steps,
and ``D`` input features per step. Hidden states use ``H`` features. This
Forward functions return caches consumed by matching backward functions.
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
        The next hidden state with shape ``(N, H)`` and a cache for the matching
        backward function. The cached next state allows the tanh derivative
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


def rnn_step_backward(
    dh_next: np.ndarray,
    cache: RnnStepCache,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Backpropagate through one vanilla-RNN time step.

    Args:
        dh_next: Total upstream gradient for the next hidden state, with shape
            ``(N, H)``.
        cache: Values returned by :func:`rnn_step_forward`.

    Returns:
        ``(dx_t, dh_previous, dweights_x, dweights_h, dbias)`` with shapes
        ``(N, D)``, ``(N, H)``, ``(D, H)``, ``(H, H)``, and ``(H,)``.
    """
    if not isinstance(cache, tuple) or len(cache) != 5:
        raise TypeError("cache must be the tuple returned by rnn_step_forward")
    x_t = _as_real_array(cache[0], name="cached x_t")
    h_previous = _as_real_array(cache[1], name="cached h_previous")
    weights_x = _as_real_array(cache[2], name="cached weights_x")
    weights_h = _as_real_array(cache[3], name="cached weights_h")
    h_next = _as_real_array(cache[4], name="cached h_next")
    dh_next = _as_real_array(dh_next, name="dh_next")

    if dh_next.shape != h_next.shape or h_next.shape != h_previous.shape:
        raise ValueError("dh_next and cached hidden states must have equal shapes")
    if x_t.ndim != 2 or h_previous.ndim != 2:
        raise ValueError("cached inputs must be two-dimensional")
    if weights_x.shape != (x_t.shape[1], h_next.shape[1]):
        raise ValueError("cached weights_x has an incompatible shape")
    if weights_h.shape != (h_next.shape[1], h_next.shape[1]):
        raise ValueError("cached weights_h has an incompatible shape")

    gradient_dtype = np.result_type(
        dh_next.dtype,
        x_t.dtype,
        h_previous.dtype,
        weights_x.dtype,
        weights_h.dtype,
        h_next.dtype,
        np.float32,
    )
    dh_next = dh_next.astype(gradient_dtype, copy=False)
    x_t = x_t.astype(gradient_dtype, copy=False)
    h_previous = h_previous.astype(gradient_dtype, copy=False)
    weights_x = weights_x.astype(gradient_dtype, copy=False)
    weights_h = weights_h.astype(gradient_dtype, copy=False)
    h_next = h_next.astype(gradient_dtype, copy=False)

    dpre_activation = dh_next * (1.0 - h_next**2)
    dx_t = dpre_activation @ weights_x.T
    dh_previous = dpre_activation @ weights_h.T
    dweights_x = x_t.T @ dpre_activation
    dweights_h = h_previous.T @ dpre_activation
    dbias = dpre_activation.sum(axis=0)
    return dx_t, dh_previous, dweights_x, dweights_h, dbias


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


def rnn_backward(
    dh: np.ndarray,
    caches: RnnCache,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Backpropagate through a complete vanilla-RNN sequence.

    ``dh[:, t, :]`` is the direct loss gradient for hidden state ``t``. During
    reverse iteration, it is added to the gradient carried from later steps.
    Shared-parameter gradients are accumulated across all time steps.
    """
    dh = _as_real_array(dh, name="dh")
    if dh.ndim != 3:
        raise ValueError("dh must have shape (N, T, H)")
    if not isinstance(caches, list) or not caches:
        raise TypeError("caches must be the nonempty list returned by rnn_forward")
    if len(caches) != dh.shape[1]:
        raise ValueError("dh and caches must contain equal time-step counts")

    first_x, first_h, weights_x, weights_h, first_h_next = caches[0]
    num_examples, num_steps, hidden_dim = dh.shape
    if first_x.shape[0] != num_examples or first_h_next.shape != (
        num_examples,
        hidden_dim,
    ):
        raise ValueError("dh shape is incompatible with the recurrent cache")
    input_dim = first_x.shape[1]
    gradient_dtype = np.result_type(
        dh.dtype,
        first_x.dtype,
        first_h.dtype,
        weights_x.dtype,
        weights_h.dtype,
        np.float32,
    )
    dx = np.zeros((num_examples, num_steps, input_dim), dtype=gradient_dtype)
    dh_carry = np.zeros((num_examples, hidden_dim), dtype=gradient_dtype)
    dweights_x = np.zeros_like(weights_x, dtype=gradient_dtype)
    dweights_h = np.zeros_like(weights_h, dtype=gradient_dtype)
    dbias = np.zeros(hidden_dim, dtype=gradient_dtype)

    for time_index in range(num_steps - 1, -1, -1):
        dh_total = dh[:, time_index, :] + dh_carry
        dx_t, dh_carry, dweights_x_t, dweights_h_t, dbias_t = (
            rnn_step_backward(dh_total, caches[time_index])
        )
        dx[:, time_index, :] = dx_t
        dweights_x += dweights_x_t
        dweights_h += dweights_h_t
        dbias += dbias_t

    return dx, dh_carry, dweights_x, dweights_h, dbias


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
        mask with shape ``(N, 1)`` for :func:`rnn_backward_masked`.
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


def rnn_backward_masked(
    dh: np.ndarray,
    caches: MaskedRnnCache,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Backpropagate through a padded sequence using cached validity masks.

    Valid rows route gradients through the recurrent candidate. Padded rows
    bypass the candidate and route gradients directly to the preceding hidden
    state through the identity path selected during the forward pass.
    """
    dh = _as_real_array(dh, name="dh")
    if dh.ndim != 3:
        raise ValueError("dh must have shape (N, T, H)")
    if not isinstance(caches, list) or not caches:
        raise TypeError(
            "caches must be the nonempty list returned by rnn_forward_masked"
        )
    if len(caches) != dh.shape[1]:
        raise ValueError("dh and caches must contain equal time-step counts")

    first_step_cache, first_mask = caches[0]
    first_x, first_h, weights_x, weights_h, first_h_next = first_step_cache
    num_examples, num_steps, hidden_dim = dh.shape
    if first_h_next.shape != (num_examples, hidden_dim):
        raise ValueError("dh shape is incompatible with the recurrent cache")
    if first_mask.shape != (num_examples, 1) or first_mask.dtype != np.bool_:
        raise ValueError("cached masks must be Boolean arrays with shape (N, 1)")

    input_dim = first_x.shape[1]
    gradient_dtype = np.result_type(
        dh.dtype,
        first_x.dtype,
        first_h.dtype,
        weights_x.dtype,
        weights_h.dtype,
        np.float32,
    )
    dx = np.zeros((num_examples, num_steps, input_dim), dtype=gradient_dtype)
    dh_carry = np.zeros((num_examples, hidden_dim), dtype=gradient_dtype)
    dweights_x = np.zeros_like(weights_x, dtype=gradient_dtype)
    dweights_h = np.zeros_like(weights_h, dtype=gradient_dtype)
    dbias = np.zeros(hidden_dim, dtype=gradient_dtype)

    for time_index in range(num_steps - 1, -1, -1):
        step_cache, valid = caches[time_index]
        if valid.shape != (num_examples, 1) or valid.dtype != np.bool_:
            raise ValueError(
                "cached masks must be Boolean arrays with shape (N, 1)"
            )
        dh_total = dh[:, time_index, :] + dh_carry
        dcandidate = dh_total * valid
        dh_identity = dh_total * (~valid)
        dx_t, dh_branch, dweights_x_t, dweights_h_t, dbias_t = (
            rnn_step_backward(dcandidate, step_cache)
        )
        dx[:, time_index, :] = dx_t
        dh_carry = dh_branch + dh_identity
        dweights_x += dweights_x_t
        dweights_h += dweights_h_t
        dbias += dbias_t

    return dx, dh_carry, dweights_x, dweights_h, dbias
