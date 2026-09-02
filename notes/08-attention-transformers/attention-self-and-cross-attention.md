# Attention, self-attention, and cross-attention

Attention is a differentiable retrieval operation. A **query** asks for
information, **keys** describe what stored items contain, and **values** carry
the information that will be retrieved.

## The general attention operation

Suppose there are $L_q$ queries and $L_k$ stored key-value pairs:

```text
Q: queries   (L_q, D_k)
K: keys      (L_k, D_k)
V: values    (L_k, D_v)
```

Scaled dot-product attention performs three steps:

$$
S = \frac{QK^T}{\sqrt{D_k}},
$$

$$
A = \mathrm{softmax}_{keys}(S),
$$

$$
Y = AV.
$$

The score $S_{ij}$ measures how well query $i$ matches key $j$. Softmax is
applied across the keys, so every query row of $A$ sums to one. The result for
query $i$ is

$$
Y_i = \sum_j A_{ij}V_j.
$$

Thus, each output is a weighted mixture of the value vectors. Keys determine
**where to look**; values determine **what information is returned**.

The division by $\sqrt{D_k}$ controls the typical magnitude of dot products.
Without it, increasing the feature dimension tends to produce larger scores,
which can make softmax excessively sharp and its gradients small.

## Attention versus self-attention

**Attention** is the general operation and does not require queries and stored
items to come from the same source.

In **self-attention**, queries, keys, and values are all learned projections of
the same input sequence $X$:

$$
Q=XW_Q, \qquad K=XW_K, \qquad V=XW_V.
$$

Every input position produces one query and can retrieve information from the
other positions. The output has one contextualized vector per input position.
For example, a token representation can be updated using information from all
tokens in the same sentence.

Calling it *self*-attention does not mean a token attends only to itself. It
means that the sequence supplying the queries is also the sequence supplying
the keys and values.

## Cross-attention

In **cross-attention**, queries come from one source while keys and values come
from another:

```text
queries:      target or decoder representations
keys/values:  source or encoder representations
```

The query length and key/value length may differ. This is useful in
encoder-decoder models: each decoder position asks which encoder positions are
relevant to producing its next output.

The relationship can be summarized as:

| Kind | Queries come from | Keys and values come from |
| --- | --- | --- |
| Self-attention | sequence $X$ | the same sequence $X$ |
| Cross-attention | sequence $X_q$ | another sequence $X_{kv}$ |

Both use the same score, softmax, and weighted-value operations.

## Masks

A mask removes forbidden query-key pairs before softmax. Their scores are
treated as negative infinity, making their final attention weights zero.

- A **padding mask** prevents attention to positions that contain padding.
- A **causal mask** prevents a decoder position from looking at future tokens.

A causal mask is required for autoregressive training because the model should
not use a later target token to predict an earlier one. Every query must retain
at least one allowed key; a completely masked row has no valid probability
distribution.

## Self-attention does not know order by itself

If the input positions are permuted, unmasked self-attention produces the same
outputs with the same permutation. In compact form,

$$
f(PX)=P f(X),
$$

where $P$ is a permutation. This is **permutation equivariance**. Self-attention
can compare vector contents, but it does not independently know which vector
was first or last. Sequence models therefore add positional information to the
input representations.

## A compact mental model

```text
query:   What am I looking for?
key:     What does this stored position advertise?
score:   How well do query and key match?
weight:  How much should this position contribute?
value:   What information will it contribute?
output:  Weighted mixture of the contributed information
```

## Related practice

- Notebook 30: attention as weighted retrieval
- Notebook 31: scaled dot-product attention and masks
- Notebook 32: multi-head attention
- Notebook 33: positional encodings

## Source

- Stanford CS231n Spring 2025, Lecture 8: Attention and Transformers,
  available from the
  [course schedule](https://cs231n.stanford.edu/2025/schedule.html).
