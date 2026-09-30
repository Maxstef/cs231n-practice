# Vector normalization and cosine similarity

Feature vectors produced by a neural network can differ in both **direction**
and **magnitude**. Sometimes we want to compare their patterns while ignoring
their lengths. L2 normalization and cosine similarity provide this comparison.

## L2 norm

For a vector

$$
h = [h_1, h_2, \ldots, h_D],
$$

its L2 norm, or Euclidean length, is

$$
\lVert h \rVert_2 = \sqrt{h_1^2 + h_2^2 + \cdots + h_D^2}.
$$

For example,

$$
h=[3,4], \qquad \lVert h \rVert_2=5.
$$

## L2 normalization

To normalize a vector, divide every component by its length:

$$
\hat{h}=\frac{h}{\lVert h \rVert_2}.
$$

Therefore,

$$
[3,4] \longrightarrow [0.6,0.8].
$$

The normalized vector has length 1. Its direction is unchanged, but its original
magnitude is discarded. Vectors such as $[3,4]$ and $[6,8]$ therefore normalize
to the same unit vector.

## Cosine similarity

Cosine similarity compares the angle between two vectors:

$$
\mathrm{cosine}(a,b)=\frac{a^T b}{\lVert a \rVert_2\lVert b \rVert_2}.
$$

If both vectors have already been normalized, the denominator is 1, so cosine
similarity becomes an ordinary dot product:

$$
\mathrm{cosine}(a,b)=\hat{a}^T\hat{b}.
$$

Its interpretation is:

- similarity near $1$: the vectors point in nearly the same direction;
- similarity near $0$: the vectors are approximately perpendicular;
- similarity near $-1$: the vectors point in opposite directions.

Cosine similarity is unchanged if a vector is multiplied by a positive scalar.
It therefore emphasizes the relative pattern of feature values rather than the
overall activation size.

## Why it is useful

Cosine similarity is commonly used for:

- retrieving nearest neighbors in an embedding space;
- comparing image, text, or audio representations;
- contrastive-learning objectives;
- matching a query representation against stored representations.

For normalized vectors, cosine similarity and Euclidean distance produce the
same neighbor ordering because

$$
\lVert \hat{a}-\hat{b} \rVert_2^2=2-2\hat{a}^T\hat{b}.
$$

Higher cosine similarity therefore means smaller Euclidean distance on the unit
sphere.

## PyTorch and NumPy

For a PyTorch feature matrix with shape `(N, D)`, normalize each row along the
feature dimension:

```python
normalized = torch.nn.functional.normalize(features, p=2, dim=1)
similarities = normalized @ normalized.T
```

The result has shape `(N, N)` and contains every pairwise cosine similarity.
PyTorch can also compare corresponding vectors directly:

```python
similarities = torch.nn.functional.cosine_similarity(a, b, dim=1)
```

The equivalent NumPy normalization is:

```python
norms = np.linalg.norm(features, axis=1, keepdims=True)
normalized = features / np.maximum(norms, 1e-12)
```

`keepdims=True` retains a `(N, 1)` result so division broadcasts across all
features in each row.

## Important cautions

- A zero vector has no direction, so its exact cosine similarity is undefined.
  Implementations use a small epsilon or clamp to avoid division by zero.
- Normalization deliberately removes magnitude. Do not use it when vector
  length contains information that the task needs.
- High cosine similarity means similar representations, not necessarily the
  same semantic class. The meaning depends on what trained or produced the
  features.

## Official references

- [PyTorch `normalize`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.normalize.html)
- [PyTorch `cosine_similarity`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cosine_similarity.html)
- [NumPy `linalg.norm`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html)
