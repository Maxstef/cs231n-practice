# Reshape, flatten, ravel, and view

Use **axis reordering** to change *which dimension means what*; use **reshape**
to split or merge dimensions while keeping the elements in their current logical
order. The number of elements must stay the same.

```python
import numpy as np

x = np.arange(24).reshape(2, 3, 4)  # N, T, D
x.reshape(6, 4).shape               # (N*T, D): (6, 4)
```

This is how we flatten examples and time steps for a temporal affine layer.
Reshaping does **not** move the time axis to another position. If axes need
reordering, use `transpose`/`moveaxis` (or PyTorch `permute`) *first*.

| Operation | Result | Copy rule |
| --- | --- | --- |
| `np.reshape(x, shape)` / `x.reshape(shape)` | Any compatible shape | View if possible; otherwise may copy |
| `np.ravel(x)` | One-dimensional | View if possible; otherwise may copy |
| `x.flatten()` in NumPy | One-dimensional | Always copies |
| `x.reshape(shape)` in PyTorch | Compatible shape | View if possible; otherwise may copy |
| `x.view(shape)` in PyTorch | Compatible shape | Requires compatible strides; does not silently copy |

`-1` asks the library to infer one dimension: `x.reshape(-1, 4)` gives `(6, 4)`.
PyTorch `Tensor.flatten(start_dim, end_dim)` is convenient when only a selected
range of dimensions should be merged. A permuted tensor may be non-contiguous;
`reshape` can still work by copying, while `view` may reject it.

Official references: [NumPy `reshape`](https://numpy.org/doc/stable/reference/generated/numpy.reshape.html),
[NumPy `ravel`](https://numpy.org/doc/stable/reference/generated/numpy.ravel.html),
[NumPy `flatten`](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flatten.html),
[PyTorch `reshape`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.reshape.html),
and [PyTorch `view`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.view.html).
