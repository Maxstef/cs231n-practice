# PyTorch shape operations and `gather`

These operations may all appear near shape-changing code, but they answer
different questions:

| Operation | Main purpose | Changes element count? |
| --- | --- | --- |
| `unsqueeze(dim)` | Insert one size-1 axis | No |
| `squeeze(dim)` | Remove a size-1 axis | No |
| `flatten(start_dim, end_dim)` | Merge consecutive axes | No |
| `unflatten(dim, sizes)` | Split one axis into several axes | No |
| `expand(...)` | Broadcast size-1 axes without copying data | Logically yes, physically no |
| `gather(input, dim, index)` | Select values using an index tensor | Output size comes from `index` |

## `unsqueeze` and `squeeze`

`unsqueeze` inserts an axis whose size is `1`:

```python
import torch

x = torch.tensor([10, 20, 30])  # shape (3,)

x.unsqueeze(0)  # shape (1, 3): one row
x.unsqueeze(1)  # shape (3, 1): one column
```

No values are added. Only the way the existing values are described changes.
A size-1 axis is especially useful for broadcasting.

`squeeze` is the inverse operation, but it can remove only axes of size `1`:

```python
x.unsqueeze(1).squeeze(1)  # shape (3,)
```

Prefer specifying the axis. Calling `squeeze()` without an axis removes *all*
size-1 axes, which can accidentally remove a batch axis when the batch size is
one.

## `flatten` and `unflatten`

`flatten` merges consecutive axes. Unlike `squeeze`, it can merge axes of any
size:

```python
x = torch.arange(24).reshape(2, 3, 4)  # shape (2, 3, 4)
y = x.flatten(start_dim=1)              # shape (2, 12)
```

`unflatten` splits one axis when the requested sizes multiply to the original
axis size:

```python
restored = y.unflatten(1, (3, 4))  # shape (2, 3, 4)
```

Therefore:

- `unsqueeze` inserts a size-1 axis;
- `unflatten` splits one axis into axes that may be larger than one;
- `squeeze` removes a size-1 axis;
- `flatten` merges a range of axes.

For example, `(2, 12)` can be unflattened to `(2, 3, 4)`, but `unsqueeze`
could produce only shapes such as `(2, 1, 12)` or `(2, 12, 1)`.

## `expand`

`expand` makes a size-1 axis behave as though it contained repeated values:

```python
x = torch.tensor([[10], [20]])  # shape (2, 1)
y = x.expand(2, 3)
# tensor([[10, 10, 10],
#         [20, 20, 20]])         shape (2, 3)
```

It normally creates a **view**, not three stored copies of each value. The
expanded entries share underlying storage, so avoid in-place writes to an
expanded tensor. Only axes whose current size is `1` can be expanded; `-1`
means “keep this axis unchanged.”

Use `repeat` instead when actual repeated storage is required.

## `gather`

`gather` selects values along one chosen axis. The `index` tensor says which
input position to read at every output position.

```python
values = torch.tensor([
    [10, 11, 12, 13],
    [20, 21, 22, 23],
])
indices = torch.tensor([
    [3, 1],
    [0, 2],
])

selected = values.gather(dim=1, index=indices)
# tensor([[13, 11],
#         [20, 22]])
```

Here `dim=1` means “choose columns.” Each batch row uses its own indices:

```text
row 0 selects columns 3 and 1
row 1 selects columns 0 and 2
```

The result has the same shape as `indices`, here `(2, 2)`. `input` and
`index` must have the same number of dimensions. On axes other than the gather
axis, the coordinates are kept rather than selected by the index values.

## Combining `unsqueeze`, `expand`, and `gather`

Suppose a batch contains `L` vectors, each with `D` features, and we want to
select `K` vectors independently for every batch item:

```python
vectors.shape   == (N, L, D)
ids_keep.shape  == (N, K)
```

For each of the `N` batch items, we want to select `K` of its `L` vectors,
including all `D` feature values inside each selected vector.

First, insert a feature axis:

```python
ids_keep.unsqueeze(-1).shape == (N, K, 1)
```

Then broadcast every vector index across all `D` feature positions:

```python
indices = ids_keep.unsqueeze(-1).expand(-1, -1, D)
indices.shape == (N, K, D)
```

Finally, select along the vector axis (`dim=1`):

```python
selected_vectors = vectors.gather(1, indices)
selected_vectors.shape == (N, K, D)
```

The expanded indices repeat only the *selection instruction*. They say, for
example, “take vector 7 for each of its `D` features.” `gather` then reads the
actual feature values from `vectors`.

Official references: [PyTorch `unsqueeze`](https://docs.pytorch.org/docs/stable/generated/torch.unsqueeze.html),
[`squeeze`](https://docs.pytorch.org/docs/stable/generated/torch.squeeze.html),
[`flatten`](https://docs.pytorch.org/docs/stable/generated/torch.flatten.html),
[`unflatten`](https://docs.pytorch.org/docs/stable/generated/torch.unflatten.html),
[`expand`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.expand.html),
and [`gather`](https://docs.pytorch.org/docs/stable/generated/torch.gather.html).
