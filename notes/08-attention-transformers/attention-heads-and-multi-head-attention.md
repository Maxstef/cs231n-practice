# Attention heads and multi-head attention

## What is an attention head?

An **attention head** is one complete attention operation with its own learned
query, key, and value projections:

$$
Q_h=XW_Q^{(h)}, \qquad K_h=XW_K^{(h)}, \qquad V_h=XW_V^{(h)},
$$

$$
A_h=\mathrm{softmax}\left(\frac{Q_hK_h^T}{\sqrt{D_h}}\right),
$$

$$
Y_h=A_hV_h.
$$

For every query position, a head produces one probability distribution over
key positions. It then uses that distribution to mix its value vectors.

## Why not use one large head?

A single head using all model features still produces only **one attention
distribution per query**. All of that head's output features are retrieved
using the same distribution.

Multiple heads produce multiple independent distributions. For the same query,
one head can retrieve mostly from one position while another retrieves from a
different position. This lets the layer represent several relationships or
types of retrieval simultaneously.

Heads do not merely receive fixed slices of the original input. The learned
projection matrices first mix all original input features. The projected
features are then divided among heads, so each head receives a different
learned view of the complete input.

## Multi-head computation and shapes

Let the model width be $D$, the number of heads be $H$, and

$$
D_h=D/H.
$$

In the usual equal-width implementation, $D$ must be divisible by $H$.
For a batch of sequences, the main shapes are:

```text
input X:             (N, L, D)
projected Q, K, V:   (N, L, D)
split Q, K, V:       (N, H, L, D_h)
scores and weights:  (N, H, L_q, L_k)
head outputs:        (N, H, L_q, D_h)
merged heads:        (N, L_q, D)
final output:        (N, L_q, D)
```

Each head runs scaled dot-product attention independently. The head results are
then concatenated and transformed by an output projection:

$$
Y=\mathrm{Concat}(Y_1,\ldots,Y_H)W_O+b_O.
$$

The output projection is not merely a shape adjustment. It learns how to mix
information from the different heads into each final output feature.

## Four conceptual matrix-multiplication stages

Ignoring biases, softmax, masks, and reshapes, multi-head self-attention can be
remembered as four matrix-multiplication stages:

1. project the input to queries, keys, and values;
2. calculate query-key similarities;
3. weight and sum the values;
4. apply the output projection.

The first stage may be implemented as one combined projection and later split
into $Q$, $K$, and $V$, or as three separate projections as in this project.

## Masks across heads

Padding and causal masks normally describe structural restrictions shared by
all heads. The same mask can broadcast over the head dimension, while every
head still calculates different scores and weights among the allowed keys.

Sharing a mask does **not** make the heads identical:

```text
same mask:       same positions are allowed
different head: different Q/K projections -> different scores -> different weights
```

## What might heads learn?

After training, different heads may emphasize local context, repeated objects,
long-range dependencies, particular syntactic relations, or other useful
patterns. These roles are not assigned in advance, and heads need not have a
simple human-readable specialization.

Attention maps should therefore be read cautiously. They show weight patterns
inside individual heads, but the values, output projection, other heads,
residual connections, and later layers all influence the final prediction.

## Computational trade-off

Self-attention connects each query directly to every key, making long-range
information paths short and allowing all positions to be processed in
parallel. Standard full attention nevertheless constructs an $L \times L$
score matrix, giving quadratic time and memory growth with sequence length.
Multi-head attention changes the representation capacity but does not remove
that quadratic sequence-length cost.

## Related implementation

The project implementation is in `cs231n_practice/attention.py`:

- `scaled_dot_product_attention_forward` and its backward pass;
- `split_heads` and `merge_heads`;
- `multi_head_attention_forward` and its backward pass.

