# Convolution layers

A convolution layer preserves the spatial organization of an image while learning
local feature detectors. Instead of connecting every output to every input value,
it applies the same small filter at many spatial positions.

## Tensor shapes

This project uses channels-first tensors:

$$
X:(N,C,H,W).
$$

$N$ is the batch size, $C$ the number of input channels, and $H,W$ the spatial
dimensions. A bank of $F$ filters has shape

$$
W_f:(F,C,HH,WW),
$$

and the bias has shape

$$
b:(F).
$$

Every filter spans all $C$ input channels. It produces one output feature map, so
$F$ filters produce $F$ output channels:

$$
Y:(N,F,H_{out},W_{out}).
$$

The number of filters, rather than the number of input channels, determines output
depth.

## One output value

For output position $(n,f,i,j)$, stride $S$, and an already padded input, the
filter window starts at

$$
h_{start}=iS,
\qquad
w_{start}=jS.
$$

One response is

$$
Y_{n,f,i,j}
=
\sum_{c,u,v}
X_{pad}[n,c,h_{start}+u,w_{start}+v]W_f[f,c,u,v]+b_f.
$$

Here $c$ indexes channels, while $u,v$ index rows and columns inside the filter.
NumPy can sum the complete local product without explicit loops over these three
indices:

```python
response = np.sum(patch * weights[filter_index]) + bias[filter_index]
```

Deep-learning libraries conventionally call this operation convolution even
though the filter is not flipped. Mathematically, the implemented operation is
cross-correlation. Learned filters make either convention equally expressive.

## Output dimensions

With padding $P$ and stride $S$,

$$
H_{out}
=
1+\left\lfloor\frac{H+2P-HH}{S}\right\rfloor,
$$

$$
W_{out}
=
1+\left\lfloor\frac{W+2P-WW}{S}\right\rfloor.
$$

The project implementation requires exact tiling, so the travel distances must be
divisible by the stride. For an odd square filter of size $K$, stride 1, and

$$
P=\frac{K-1}{2},
$$

the layer preserves height and width. This is commonly called **same padding**.

Stride greater than one evaluates fewer positions and downsamples spatially. A
$1\times1$ convolution does not combine neighboring locations, but it still learns
combinations across input channels.

## Local connectivity and parameter sharing

One output depends only on a local input region. This local connectivity encodes
the useful image assumption that nearby values are strongly related.

The same filter and bias are reused at every position and for every example. This
parameter sharing allows one pattern detector to respond wherever its pattern
appears.

The number of learnable parameters is

$$
F(C\cdot HH\cdot WW+1).
$$

The `+1` is one bias per filter. Input height and width do not appear because
spatial positions reuse parameters.

## Backward pass

For one patch, filter, bias, and upstream scalar $g$,

$$
dP=gW,
\qquad
dW=gP,
\qquad
db=g.
$$

The complete layer must accumulate these local contributions:

- overlapping windows make one input affect several outputs, so `dx` uses `+=`;
- a filter is shared across examples and positions, so `dweights` uses `+=`;
- one bias is shared across every response of its filter, so `dbias` sums over
  examples and spatial positions.

For upstream gradient $dY:(N,F,H_{out},W_{out})$,

$$
db_f=\sum_{n,i,j}dY_{n,f,i,j}.
$$

Padding is retained while accumulating `dx_padded`, then its spatial border is
removed so the returned `dx` has the original input shape.

## Connection to this project

- Notebooks 16–18 derive shapes and implement forward/backward passes.
- `cs231n_practice/cnn_layers.py` contains their reusable implementations.
- Numerical gradient checks verify `dx`, `dweights`, and `dbias` independently.

## Key takeaway

A convolution layer learns local filters, shares them across space, and preserves a
grid of feature responses. Its efficiency and spatial inductive bias make it much
better suited to images than immediately flattening all pixels.
