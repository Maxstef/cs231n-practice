# RNN sequence-to-sequence models with attention

Attention became important partly as a solution to a limitation of early RNN
encoder-decoder models.

## The fixed-context encoder-decoder

For an input sequence $x_1,\ldots,x_T$, an encoder RNN calculates hidden
states:

$$
h_t=f(x_t,h_{t-1}).
$$

An early sequence-to-sequence design used the final encoder state as a single
context vector $c$, often also using it to initialize the decoder. The decoder
then generated one output at a time:

$$
s_t=g(y_{t-1},s_{t-1},c).
$$

This requires the complete input sequence to be compressed into one
fixed-width vector. For a long input, details from early or locally important
positions may be difficult to preserve. The problem is not that a fixed vector
can never contain useful information; it is that every decoder step must rely
on the same compressed summary.

## A different context for every decoder step

Attention retains all encoder hidden states $h_1,\ldots,h_T$. At decoder step
$t$, the previous decoder state acts as a query and is compared with every
encoder state:

$$
e_{t,i}=f_{score}(s_{t-1},h_i).
$$

The scalar alignment scores become attention weights:

$$
a_{t,i}=\frac{\exp(e_{t,i})}{\sum_j \exp(e_{t,j})}.
$$

The step-specific context vector is their weighted sum:

$$
c_t=\sum_i a_{t,i}h_i.
$$

The decoder then uses that context:

$$
s_t=g(y_{t-1},s_{t-1},c_t).
$$

At the next decoder step, the new decoder state creates new alignment scores
and therefore a new context vector. One output word can focus on early input
positions while another focuses on later positions.

## Connection to queries, keys, and values

The RNN formulation is an instance of the general attention abstraction:

```text
query:   decoder state
keys:    representations derived from encoder hidden states
values:  encoder hidden states or their projections
output:  context vector for the decoder step
```

It is therefore a form of **cross-attention**: queries come from the decoder,
whereas keys and values come from the encoder.

The alignment function need not be a dot product. Earlier systems often used a
small learned network (additive attention). Scaled dot-product attention later
became especially convenient because many queries can be processed efficiently
with matrix multiplication.

## Attention maps as alignments

For translation, rows of an attention map can represent output words and
columns can represent input words. A bright entry means that an output step
placed a large weight on that input position.

A mostly diagonal pattern often indicates that source and target phrases occur
in a similar order. Off-diagonal blocks can reveal reordered phrases. This is a
useful visualization of a learned soft alignment, but it should not be treated
as a complete explanation of the model: the hidden states, values, recurrent
state, and later computations also affect each prediction.

## Why it helped

RNN attention changes the path from source to output:

```text
fixed context:
all encoder states -> one final summary -> every decoder step

attention:
all encoder states -> a new weighted context for each decoder step
```

This reduces the single-vector bottleneck and gives each decoder step a direct,
differentiable route to all encoder states. The attention weights need no
separate alignment labels: gradients from the sequence prediction loss flow
through the context vector, softmax weights, scores, encoder, and decoder.

## Relation to Transformers

RNN attention still uses sequential recurrence in the encoder and decoder.
Transformers make attention the primary sequence-mixing operation and process
many positions in parallel. The underlying retrieval idea—scores, normalized
weights, and a weighted mixture of values—remains the same.

## Source

- Stanford CS231n Spring 2025, Lecture 8: Attention and Transformers,
  available from the
  [course schedule](https://cs231n.stanford.edu/2025/schedule.html).
