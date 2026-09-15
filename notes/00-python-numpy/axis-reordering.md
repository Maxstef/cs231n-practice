# Reordering array and tensor axes

An **axis** is one dimension of an array. Reordering axes changes how we *index*
the same values; it does not change the values or the number of elements. First
write down what each axis means, then write the desired output order.

For example, an RGB image may be stored as `C, H, W` with shape `(3, 4, 5)`.
Plotting usually expects `H, W, C`, with shape `(4, 5, 3)`.

| Operation | Library | What you specify | Good choice when |
| --- | --- | --- | --- |
| `np.transpose(a, axes)` or `a.transpose(axes)` | NumPy | The **complete output order** of all input axes | Several axes need rearranging, such as `N,H,W,C` → `N,C,H,W` |
| `np.moveaxis(a, source, destination)` | NumPy | Which axis to move and where to put it; the other axes keep their relative order | One named axis needs to move, such as channels from last to first |
| `x.permute(*dims)` | PyTorch | The **complete output order** of all input dimensions | Several tensor dimensions need rearranging; closest counterpart to NumPy's `transpose(..., axes)` |
| `x.transpose(dim0, dim1)` | PyTorch | Two dimensions to **swap** | Only two tensor dimensions need exchanging |

Here, an output order such as `(1, 2, 0)` means: output axis 0 comes from input
axis 1, output axis 1 comes from input axis 2, and output axis 2 comes from input
axis 0. It is **not** a list of destination positions for each input axis.

```python
import numpy as np
import torch

image_np = np.zeros((3, 4, 5))         # C, H, W
image_torch = torch.zeros((3, 4, 5))   # C, H, W

np.transpose(image_np, (1, 2, 0)).shape  # (4, 5, 3): H, W, C
np.moveaxis(image_np, 0, -1).shape       # (4, 5, 3): H, W, C
image_torch.permute(1, 2, 0).shape      # (4, 5, 3): H, W, C
```

The two NumPy calls above give the same axis order. `moveaxis` reads as “move
channels from axis 0 to the end”; `transpose` reads as “make the entire output
order H, W, C.” Use whichever more clearly expresses your intent.

For a batched image array, `N,H,W,C` → `N,C,H,W` is:

```python
np.transpose(images, (0, 3, 1, 2))  # explicitly list all output axes
np.moveaxis(images, -1, 1)          # move just C; N, H, W stay in order
```

One important naming trap: `numpy_array.transpose(1, 2, 0)` reorders **all three**
axes, whereas `torch_tensor.transpose(1, 2)` swaps **only axes 1 and 2**. In our
video notebook, `videos.permute(0, 2, 1, 3, 4)` changes `N,T,C,H,W` into
`N,C,T,H,W`. PyTorch `transpose(1, 2)` would also work there because only T and
C exchange positions.

Remember: **NumPy `transpose` / PyTorch `permute` = specify the whole order;
NumPy `moveaxis` = move selected axis; PyTorch `transpose` = swap two axes.**
NumPy's `a.T` (or `np.transpose(a)` without `axes`) reverses *all* axes; for an
n-dimensional image tensor, prefer an explicit order so the result is obvious.

These operations normally return views or share storage rather than copying all
values. Some later operations may require contiguous memory or create a copy;
axis reordering itself is not the same thing as `reshape`, which changes the
shape without explicitly assigning a new meaning/order to each axis.

Official references: [NumPy `transpose`](https://numpy.org/doc/stable/reference/generated/numpy.transpose.html),
[NumPy `moveaxis`](https://numpy.org/doc/stable/reference/generated/numpy.moveaxis.html),
[PyTorch `Tensor.permute`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.permute.html),
and [PyTorch `Tensor.transpose`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.transpose.html).
