# Contrastive learning, SimCLR, and InfoNCE

Contrastive learning trains an encoder so that related views are close in
representation space and unrelated examples are farther apart.

For a reference example $x$:

- $x^+$ is a **positive**, usually another augmented view of the same source;
- $x^-$ is a **negative**, usually a view originating from another source.

The desired relationship is

$$
s(f(x),f(x^+)) > s(f(x),f(x^-)),
$$

where $f$ is the encoder and $s$ is a similarity function.

## SimCLR pipeline

SimCLR creates two independently augmented views of every image:

```text
image x -> augmentation t  -> encoder f -> representation h -> projector g -> z
        -> augmentation t' -> encoder f -> representation h' -> projector g -> z'
```

The same encoder and projection head process both branches. Random crops,
resizing, flips, color distortion, and blur define which changes the learned
representation should ignore. If an augmentation removes class-relevant
information, forcing invariance to it can be harmful.

The projection head separates two roles:

- $h$ is retained as the downstream representation;
- $z=g(h)$ is optimized directly by the contrastive loss.

This lets the projection space discard augmentation-specific details without
requiring the reusable representation to discard all of them.

## Normalized similarity and temperature

L2 normalization produces unit-length vectors:

$$
\hat{z}_i=\frac{z_i}{\lVert z_i\rVert_2}.
$$

Their dot product is then cosine similarity:

$$
s_{ij}=\hat{z}_i^T\hat{z}_j.
$$

Temperature $\tau$ scales the logits before softmax. A smaller value makes the
distribution sharper and emphasizes similarity differences; a larger value
makes it softer.

## InfoNCE / NT-Xent

For anchor $i$ and its positive partner $p(i)$, one directional loss is

$$
\ell_i=-\log
\frac{\exp(s_{i,p(i)}/\tau)}
{\sum_{k\ne i}\exp(s_{ik}/\tau)}.
$$

The anchor itself is excluded from the denominator. SimCLR averages this loss
over all $2N$ views, so every view acts as an anchor and the two directions of
each positive pair are both trained.

For views ordered as `[view_a; view_b]`, the positive mapping is

$$
p(i)=
\begin{cases}
i+N, & 0\le i<N,\\
i-N, & N\le i<2N.
\end{cases}
$$

The $2N\times2N$ similarity matrix contains self-similarities on its diagonal,
positive pairs at the cross-view positions, and candidate negatives elsewhere.

## Important limitations

- More negatives can improve the discrimination signal but increase memory
  use and the chance of **false negatives**: different images with the same
  semantic class treated as unrelated.
- If every representation becomes identical, the system has **collapsed** and
  no longer distinguishes inputs. Negative competition in InfoNCE resists this
  trivial solution.
- Strong contrastive results depend heavily on augmentation design, batch
  construction, temperature, encoder capacity, and training duration.

## Sources

- Stanford CS231n Spring 2025, Lecture 12: Self-Supervised Learning, available
  from the [course schedule](https://cs231n.stanford.edu/2025/schedule.html).
- Chen et al., [A Simple Framework for Contrastive Learning of Visual
  Representations](https://arxiv.org/abs/2002.05709).
