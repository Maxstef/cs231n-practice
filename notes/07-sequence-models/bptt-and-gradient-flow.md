# Backpropagation through time and gradient flow

Backpropagation through time (BPTT) is ordinary reverse-mode backpropagation
applied to an RNN unrolled across its time steps. The name emphasizes that the
hidden state creates a chain of dependencies through time.

## Two gradient sources at a hidden state

Suppose a loss may depend on each hidden state. During the backward pass,
$h_t$ receives:

1. a direct gradient from the loss or output attached at time $t$;
2. a carried gradient through the dependence of $h_{t+1}$ on $h_t$.

Thus the total hidden-state gradient is

$$
dh_t^{total}=dh_t^{direct}+dh_t^{future}.
$$

The name `dh_future` means “gradient arriving from a later time step.” After
backpropagating through step $t$, the resulting gradient with respect to
$h_{t-1}$ becomes the carried gradient for the next backward iteration:

```text
backward at time t:      dh_t -> dh_(t-1)
next loop iteration:     dh_(t-1) is the gradient from the future
```

There is no skipped state; the variable name describes its role relative to the
step currently being processed.

## A loss only at the final step

If only the final state has a loss, then

$$
dh_t^{direct}=0\quad\text{for }t<T.
$$

Earlier steps still receive gradients. The final gradient travels backward
through $h_T,h_{T-1},\ldots,h_1$. Intermediate losses are optional; intermediate
state dependencies are not.

## Shared-parameter accumulation

Because each step uses the same recurrent parameters, their gradients add:

$$
dW_x=\sum_{t=1}^{T}dW_x^{(t)},\qquad
dW_h=\sum_{t=1}^{T}dW_h^{(t)},\qquad
db=\sum_{t=1}^{T}db^{(t)}.
$$

This is the same branch-accumulation rule used elsewhere in computational
graphs: one parameter influenced the loss through many paths.

## Why gradients vanish or explode

Moving a gradient across many recurrent steps repeatedly multiplies it by local
Jacobians. For a vanilla RNN, these include $W_h$ and the derivative of `tanh`.
Schematically,

$$
\frac{\partial h_T}{\partial h_t}
=\prod_{k=t+1}^{T}\frac{\partial h_k}{\partial h_{k-1}}.
$$

If these transformations repeatedly shrink vectors, distant gradients vanish.
If they repeatedly amplify vectors, gradients explode. Since
$0<\tanh'(a)\leq 1$ and is near zero in saturation, vanilla RNNs are especially
susceptible to vanishing gradients.

Vanishing gradients make long-range dependencies hard to learn. Exploding
gradients can create unstable, extremely large parameter updates.

## Gradient clipping

Global-norm clipping controls exploding gradients. Given all parameter
gradients $g_1,\ldots,g_K$, compute one norm:

$$
G=\sqrt{\sum_{k=1}^{K}\sum_i g_{k,i}^2}.
$$

If $G$ exceeds threshold $c$, scale every gradient by the same factor:

$$
g_k\leftarrow g_k\frac{c}{G}.
$$

One shared scale preserves the direction of the complete gradient vector.
Clipping limits explosion; it does not restore a gradient that has already
vanished.

## Truncated BPTT

Full BPTT stores an entire unrolled computation and propagates through all of
it. Truncated BPTT processes shorter windows:

- the final hidden state of one window is carried into the next forward window;
- gradients are propagated only within the current window;
- the carried state is treated as detached at the window boundary.

This reduces memory and computation, but it limits how directly the current
loss can assign credit to events before the truncation boundary. It is a
training approximation, not a different recurrent forward model.

## Practical memory rule

```text
forward:   carry hidden state toward later time steps
backward:  combine direct and future gradients
parameters: accumulate contributions from every unrolled use
```

## Source

- Stanford CS231n Spring 2025, Lecture 7: Recurrent Neural Networks,
  available from the
  [course schedule](https://cs231n.stanford.edu/2025/schedule.html).
