# Broadcasting and new axes

Broadcasting lets arrays of different shapes participate in elementwise
operations. Compare shapes from the **right**: dimensions match if they are
equal or one is `1`. A missing leading dimension behaves like `1`.

```python
import numpy as np

features = np.ones((16, 12, 12))  # channels, height, width
weights = np.arange(16)          # one weight per channel

weights[:, None, None].shape    # (16, 1, 1)
(features * weights[:, None, None]).shape  # (16, 12, 12)
```

`None` is the same as `np.newaxis`: it **inserts a length-one axis**. Each weight
can now apply across every spatial position in its own channel. In PyTorch,
`tensor[:, None, None]` works similarly; `unsqueeze` is another way to insert
an axis.

Another project example is pairwise box comparison:

```python
boxes_a[:, None, :].shape  # (A, 1, 4)
boxes_b[None, :, :].shape  # (1, B, 4)
# An elementwise operation broadcasts to (A, B, 4).
```

Broadcasting avoids manually repeating the small input, but the **result** can
still be large. Always write the expected output shape before constructing a
pairwise `(A, B, ...)` array.

Official references: [NumPy broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html)
and [NumPy new-axis indexing](https://numpy.org/doc/stable/user/basics.indexing.html#dimensional-indexing-tools).
