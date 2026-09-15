# Reductions: axis, dim, and keepdims

A reduction combines values **along selected axes**. In NumPy the argument is
`axis`; in PyTorch it is commonly `dim`. By default, the reduced axis disappears.

```python
import numpy as np

scores = np.arange(12).reshape(3, 4)  # examples, classes
scores.sum(axis=1).shape             # (3,): one sum per example
scores.sum(axis=0).shape             # (4,): one sum per class
scores.sum(axis=1, keepdims=True).shape  # (3, 1)
```

`keepdims=True` retains the reduced axis at size `1`, usually so the result
broadcasts back against the original array. For stable softmax, we subtract
`scores.max(axis=1, keepdims=True)` from each class score in its own row.

Multiple axes can be reduced together: for feature maps shaped `(N,C,H,W)`,
`features.mean(axis=(2, 3))` has shape `(N,C)` because height and width vanish.
In Grad-CAM, the corresponding PyTorch call is
`activation_gradients.mean(dim=(0, 2, 3))`, leaving `(C,)` channel weights.

`argmax(axis=1)` returns the **index** of the maximum class per example, not
the maximum value. `max(axis=1)` returns the values. When unsure, name the axes
and ask: *which axes should remain in the answer?*

Official references: [NumPy `sum`](https://numpy.org/doc/stable/reference/generated/numpy.sum.html),
[NumPy `argmax`](https://numpy.org/doc/stable/reference/generated/numpy.argmax.html),
and [PyTorch `mean`](https://docs.pytorch.org/docs/stable/generated/torch.mean.html).
