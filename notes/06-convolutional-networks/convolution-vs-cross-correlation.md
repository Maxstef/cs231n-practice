# Convolution versus cross-correlation

In mathematics and signal processing, **convolution** reverses a filter before
sliding it across an input. **Cross-correlation** slides the filter exactly as
stored. Deep-learning libraries normally compute cross-correlation but call the
layer a convolution by convention.

For an input patch $x=(1,2,4)$ and stored filter $w=(1,0,-1)$:

```text
Cross-correlation                    Mathematical convolution

input:          1   2   4            input:           1   2   4
stored filter:  1   0  -1            reversed filter: -1   0   1
products:       1   0  -4            products:        -1   0   4
sum:           -3                    sum:               3
```

For a length-$K$ filter at one valid output position, the distinction can be
written as

$$
\text{cross-correlation:}\qquad y_i=\sum_{u=0}^{K-1}x_{i+u}w_u,
$$

$$
\text{convolution:}\qquad y_i=\sum_{u=0}^{K-1}x_{i+u}w_{K-1-u}.
$$

The second formula uses the filter in reversed order. NumPy makes the difference
visible directly:

```python
import numpy as np

x = np.array([1, 2, 4])
w = np.array([1, 0, -1])

np.correlate(x, w, mode="valid")  # array([-3])
np.convolve(x, w, mode="valid")   # array([3])
```

For 2D convolution, mathematical convolution flips both spatial filter axes. For
3D video convolution, it flips the temporal axis and both spatial axes. Input
and output channel axes are not sliding kernel-position axes and are not flipped.

PyTorch `Conv1d`, `Conv2d`, and `Conv3d` use the cross-correlation form. Our
naïve implementations do the same when they calculate:

```python
output_value = np.sum(input_patch * stored_filter) + bias
```

There is no filter reversal before the multiplication.

## Why does deep learning still call it convolution?

CNN filters are learned. If a cross-correlation layer needs the behavior of the
stored filter $(1,0,-1)$ under mathematical convolution, it can learn the
reversed weights $(-1,0,1)$ instead. Every convolution filter therefore has a
corresponding reversed cross-correlation filter that produces the same outputs.
The two conventions have equal expressive power for learned filters.

The distinction still matters when interpreting or manually setting weights:
the orientation and sign of an edge or temporal-change response depend on the
stored order. It also matters when comparing a CNN implementation with a
mathematical or signal-processing formula. In this project, unless stated
otherwise, **convolution layer means the deep-learning convention:
cross-correlation without flipping the stored filter**.

Official references: [NumPy `convolve`](https://numpy.org/doc/stable/reference/generated/numpy.convolve.html),
[NumPy `correlate`](https://numpy.org/doc/stable/reference/generated/numpy.correlate.html),
[PyTorch `Conv2d`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html),
and [PyTorch `Conv3d`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv3d.html).
