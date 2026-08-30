# Vanilla recurrent neural networks

Recurrent neural networks process ordered data while carrying a state from one
step to the next. Unlike a fixed-size feed-forward network, the same recurrent
operation can be reused for sequences of different lengths.

## Common input-output patterns

Sequence models are often described by how many inputs and outputs they use:

- **one to one:** ordinary fixed-input prediction;
- **one to many:** one input conditions a generated sequence, as in image
  captioning;
- **many to one:** a sequence produces one prediction, as in video or sentiment
  classification;
- **many to many, aligned:** every input step has an output, as in frame-level
  video classification;
- **many to many, unaligned:** an encoder reads one sequence before a decoder
  produces another.

These are not different recurrent cells. They are different ways of connecting
inputs, states, outputs, and losses around a shared recurrence.

## Hidden-state update

For a batch of $N$ sequences, let

$$
x_t\in\mathbb{R}^{N\times D},\qquad
h_{t-1}\in\mathbb{R}^{N\times H}.
$$

A vanilla RNN step computes

$$
a_t=x_tW_x+h_{t-1}W_h+b,
$$

$$
h_t=\tanh(a_t),
$$

where

$$
W_x:(D,H),\qquad W_h:(H,H),\qquad b:(H).
$$

The new state $h_t$ combines the current input with the previous state. It is
therefore both the output of the current recurrent step and the memory passed to
the next step.

The complete sequence has shape

$$
X:(N,T,D)\longrightarrow H:(N,T,H).
$$

## Producing task outputs

The hidden state is an internal representation, not necessarily the final task
output. A shared projection can convert every state into $M$ output scores:

$$
s_t=h_tW_y+b_y,
$$

with $W_y:(H,M)$. Applying this independently at all $T$ positions gives a
temporal score tensor with shape $(N,T,M)$.

A many-to-one model may instead project only the final relevant state. A
many-to-many model usually projects every state and may attach a loss to every
valid output position.

## Parameter sharing through time

The same $W_x$, $W_h$, and $b$ are reused at every time step. Consequently:

- model size does not grow with sequence length;
- the same kind of transition is learned at every position;
- every time step contributes to the gradients of the shared parameters.

The boxes in an unrolled RNN diagram are repeated **uses** of one cell, not
independent layers with separate weights.

## Strengths and limitations

RNNs naturally accept variable-length sequences and, in principle, can carry
information over an arbitrary number of steps. In practice they have two major
limitations:

- steps are sequential, because $h_t$ cannot be computed before $h_{t-1}$;
- long-range learning is difficult because gradients repeatedly pass through
  recurrent transformations.

The second issue motivates gated cells such as LSTMs and GRUs. The first is one
reason attention-based models became attractive for long sequences.

## Shape summary

```text
x_t          (N, D)
h_previous   (N, H)
W_x          (D, H)
W_h          (H, H)
b            (H,)
h_t          (N, H)

complete x   (N, T, D)
complete h   (N, T, H)
```

## Source

- Stanford CS231n Spring 2025, Lecture 7: Recurrent Neural Networks,
  available from the
  [course schedule](https://cs231n.stanford.edu/2025/schedule.html).

