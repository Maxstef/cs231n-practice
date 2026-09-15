# Indexing and scatter-add

Integer-array indexing **gathers** selected values. Repeated indices are fine
when reading: `table[token_ids]` looks up one embedding vector per token. When
writing gradients back, repeated IDs must have their contributions **added**.

```python
import numpy as np

token_ids = np.array([1, 1, 3])
updates = np.array([2.0, 5.0, 7.0])
gradient = np.zeros(4)
np.add.at(gradient, token_ids, updates)
# gradient is [0, 7, 0, 7]
```

The first two updates both belong to token `1`, so its gradient is `2 + 5`.
Do **not** assume `gradient[token_ids] += updates` handles repeated indices:
advanced indexing uses a temporary buffer and may apply an update only once.
An explicit loop is equivalent and often easier to understand:

```python
for token_id, update in zip(token_ids, updates):
    gradient[token_id] += update
```

For 2D class scores, `scores[np.arange(N), targets]` gathers one target-class
score from each row. Here the two integer index arrays work **together**: they
form `(row, class)` pairs, not a rectangular slice.

Official references: [NumPy indexing](https://numpy.org/doc/stable/user/basics.indexing.html)
and [NumPy `ufunc.at`](https://numpy.org/doc/stable/reference/generated/numpy.ufunc.at.html).
