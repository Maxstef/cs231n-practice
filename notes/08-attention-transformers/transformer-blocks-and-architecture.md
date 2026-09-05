# Transformer blocks and architecture

Attention is an operation for retrieving and mixing information. A
**Transformer** is a neural-network architecture built by combining attention
with normalization, residual connections, and position-wise feed-forward
networks.

## From vectors to contextualized vectors

Let a batch of token representations have shape

```text
X: (N, L, D)
```

where $N$ is batch size, $L$ is sequence length, and $D$ is model width. A
Transformer block returns the same shape. Keeping the width unchanged makes it
possible to stack many blocks.

Self-attention is the part that lets different positions exchange information.
Layer normalization and the feed-forward network operate on each token vector
independently; their parameters are shared across positions.

## A pre-normalization encoder block

The implementation in this project uses a **pre-norm** block:

$$
U = X + \mathrm{MHA}(\mathrm{LN}(X)),
$$

$$
Y = U + \mathrm{FFN}(\mathrm{LN}(U)).
$$

Its data flow is:

```text
X
├──────────── residual ────────────┐
└─ LayerNorm -> multi-head attention -> add = U
   U
   ├───────── residual ────────────┐
   └─ LayerNorm -> feed-forward network -> add = Y
```

The original 2017 Transformer used post-norm blocks, in which normalization
followed each residual addition. Pre-norm is common in later models because it
usually makes gradient flow through deep stacks easier. Both variants contain
the same main components, but their ordering matters.

## What each component contributes

### Multi-head self-attention

Self-attention is the only operation in a basic encoder block that directly
mixes information between token positions. It can create a contextualized
representation at every position using information from all allowed keys.

Ignoring biases, masks, softmax, and reshaping, its main matrix-multiplication
stages are:

1. project tokens to queries, keys, and values;
2. calculate query-key similarities;
3. use attention weights to mix values;
4. fuse the concatenated heads with an output projection.

### Position-wise feed-forward network

The feed-forward network, also called an FFN or MLP, applies the same small
network independently at every position:

$$
\mathrm{FFN}(x)=W_2\,\phi(W_1x+b_1)+b_2.
$$

It commonly expands from $D$ to a larger hidden width and projects back to
$D$. A classic choice is roughly $D \rightarrow 4D \rightarrow D$, although
the expansion ratio and activation are design choices.

Attention mixes information **across positions**; the FFN transforms and mixes
features **within each position**.

### Residual connections

Each sublayer learns an update rather than replacing the residual stream:

$$
\text{output}=\text{input}+\text{sublayer update}.
$$

The direct identity path preserves information and supplies a short route for
gradients. The addition requires the input and sublayer output to have the same
shape.

### Layer normalization

Layer normalization stabilizes the features of each token independently. It
does not mix examples or token positions and does not require batch-level
statistics, making it well suited to variable-length sequences.

## Positional information and masks

Plain self-attention is permutation equivariant: permuting the input tokens
permutes the outputs in the same way. Positional encodings or learned position
embeddings are therefore added when order or spatial location matters.

A mask changes which query-key relationships are allowed:

- encoder-style processing commonly permits attention to every valid token;
- autoregressive decoding uses a causal mask to hide future tokens;
- padded sequences use a padding mask to exclude placeholder positions.

Masks change connectivity, not the learned query, key, and value projections.

## Stacking blocks

A Transformer is formed by passing the residual stream through multiple
blocks. The blocks have the same architecture but do **not** normally share
parameters. Early blocks build initial contextual features; later blocks can
transform and combine the representations produced below them.

Long-range dependencies have a short path through self-attention, and all
positions can be processed in parallel during encoder-style training. The
trade-off is that standard full attention constructs an $L \times L$ matrix
per head, so attention time and memory grow quadratically with sequence length.

## Encoder, decoder, and encoder-decoder models

- An **encoder-only** Transformer uses unmasked or padding-masked
  self-attention to understand a complete input.
- A **decoder-only** Transformer uses causal self-attention to predict future
  tokens from a prefix.
- An **encoder-decoder** Transformer has an encoder for the source and a
  decoder containing causal self-attention plus cross-attention to encoder
  representations.

The Transformer block in this project is encoder-style. The sequence and image
classifiers use position 0 as a learned summary token and attach a linear
classifier to its final representation.

## Related practice and implementation

- Notebooks 30–32: attention and multi-head attention
- Notebook 33: positional encoding and masks
- Notebook 34: Transformer encoder block
- Notebook 35: Transformer sequence classifier
- `cs231n_practice/attention.py`
- `cs231n_practice/transformer.py`

## Source

- Stanford CS231n Spring 2025, Lecture 8: Attention and Transformers,
  available from the
  [course schedule](https://cs231n.stanford.edu/2025/schedule.html).
