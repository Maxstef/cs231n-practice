# Dilation and the parameters that affect receptive fields

A unit's **receptive field** is the region of the original input that can
influence it. Kernel size determines how many values a filter samples, while
**dilation** determines the spacing between those sampled values.

## What dilation does

For a one-dimensional kernel with three weights:

```text
dilation 1:  x x x          samples positions 0, 1, 2
dilation 2:  x . x . x      samples positions 0, 2, 4
dilation 3:  x . . x . . x  samples positions 0, 3, 6
```

The dots are positions inside the covered interval that this filter does not
sample. The kernel still has three weights in every case, but its **effective
kernel size** is

$$
K_{effective}=1+(K-1)D,
$$

where $K$ is the number of kernel weights and $D$ is dilation. A $K=3$ filter
therefore covers 3, 5, or 7 positions for dilation 1, 2, or 3.

The purpose is to collect wider context without increasing the number of filter
parameters or immediately reducing feature-map resolution. The tradeoff is
sparse sampling: a large dilation may miss fine local structure and repeated
dilated layers can produce uneven, grid-like coverage.

## Receptive field and jump across layers

Track two quantities before each layer:

- $R_{old}$: receptive-field size in original input positions;
- $J_{old}$: distance, in original input positions, between adjacent features.

For kernel size $K$, dilation $D$, and stride $S$:

$$
R_{new}=R_{old}+(K-1)D J_{old},
$$

$$
J_{new}=J_{old}S.
$$

Dilation increases how widely one kernel samples. It does not directly change
the jump. Stride changes the jump, which makes the receptive field of later
layers grow faster.

For example, start with $(R,J)=(1,1)$:

```text
3-wide convolution, dilation 2, stride 1: (R, J) = (5, 1)
3-wide convolution, dilation 1, stride 2: (R, J) = (7, 2)
3-wide convolution, dilation 1, stride 1: (R, J) = (11, 2)
```

The final layer sees 11 original positions. Its adjacent outputs are centered
two original positions apart.

## What else affects receptive field?

| Choice | Effect |
| --- | --- |
| Larger kernel | Directly covers more positions, but uses more parameters |
| Larger dilation | Widens coverage without adding kernel weights, but samples sparsely |
| Larger stride | Increases jump and accelerates later receptive-field growth; reduces resolution |
| Pooling | Also has a kernel and stride, so it changes receptive field and often jump |
| More layers | Composes local regions into progressively wider context |
| `1×1` convolution | Mixes channels but does not enlarge spatial receptive field when dilation and stride are 1 |
| Padding | Controls output size and boundary alignment; does not itself enlarge the nominal receptive-field formula |

Padding can make an output depend partly on artificial padded values near a
boundary. Thus the **nominal** receptive field may be larger than the number of
real input pixels contributing to an edge output.

## Output size with dilation

Dilation also changes the effective filter size in the convolution output-shape
formula:

$$
D_{out}=\left\lfloor
\frac{D_{in}+2P-Dilation\,(K-1)-1}{S}+1
\right\rfloor.
$$

Increasing dilation without adding padding therefore usually reduces output
size. To preserve size with an odd kernel and stride 1, padding is often chosen
to compensate for the effective kernel size.

## Images and videos

For 2D convolution, kernel size, stride, padding, and dilation can differ across
height and width. For 3D video convolution they can differ across time, height,
and width:

```python
torch.nn.Conv3d(
    in_channels=8,
    out_channels=16,
    kernel_size=(3, 3, 3),
    dilation=(2, 1, 1),
)
```

This filter has effective temporal coverage 5 and spatial coverage $3\times3$.
It samples three frames from a five-frame interval while keeping ordinary
spatial sampling. This can expand temporal context, but it may skip motion that
occurs only in the unsampled frames.

## Key takeaway

Kernel size says **how many filter values exist**; dilation says **how far apart
they are**; stride says **how far the whole filter moves**. Across layers, track
both receptive field and jump to understand coverage in the original input.

Official references: [PyTorch `Conv2d`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
and [PyTorch `Conv3d`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv3d.html).
