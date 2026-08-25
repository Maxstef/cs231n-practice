# Pooling and spatial downsampling

Pooling replaces a local spatial neighborhood with a fixed summary. It reduces
height and width while processing each channel independently.

## Shapes and parameters

For input

$$
X:(N,C,H,W),
$$

a pooling window of size $(PH,PW)$ and stride $S$ produces

$$
Y:(N,C,H_{out},W_{out}),
$$

where

$$
H_{out}=1+\left\lfloor\frac{H-PH}{S}\right\rfloor,
$$

$$
W_{out}=1+\left\lfloor\frac{W-PW}{S}\right\rfloor.
$$

Pooling retains $C$ channels because it neither mixes channels nor creates new
feature detectors. Standard max and average pooling have no learned weights or
biases.

A common configuration is a $2\times2$ window with stride 2, which halves each
spatial dimension when the input is compatible.

## Max pooling

Max pooling preserves the strongest activation in each window:

$$
Y_{n,c,i,j}=\max_{u,v}X_{n,c,iS+u,jS+v}.
$$

It discards the other values and their precise arrangement. This can make a feature
less sensitive to small spatial shifts, but aggressive pooling can remove small
objects, boundaries, or precise spatial relationships.

### Backward pass

With a unique maximum, changing a non-maximum input slightly does not change the
pooled output. The upstream gradient is therefore routed only to the winning
position. All other entries in that window receive zero from this path.

Overlapping pooling windows require accumulation: one input may be the maximum in
several windows and must receive every contribution.

If multiple entries tie for the maximum, max pooling is not differentiable there.
The project uses `np.argmax`, which selects the first maximum in flattened
row-major order and routes the complete upstream gradient to it. Numerical checks
use unique maxima to avoid this boundary.

## Average pooling

Average pooling returns the mean of each window. It preserves average activation
level but smooths strong peaks. Its backward pass distributes the upstream gradient
equally among all entries in the window.

Average pooling remains useful for global spatial reduction. Global average
pooling summarizes each complete feature map into one number per channel.

## Stride and coverage

The relationship between stride and window size determines coverage:

- $S$ smaller than the window creates overlapping windows;
- $S$ equal to the window creates touching, non-overlapping windows;
- $S$ larger than the window creates gaps, so some inputs are skipped.

Skipped inputs do not affect the output and receive zero gradient. This is valid
but usually discards information too aggressively.

Exact boundary alignment and complete coverage are different conditions. A final
window can end exactly at the input boundary even when gaps exist between windows.

## Pooling versus strided convolution

Both can reduce spatial resolution, but they do different work:

- pooling applies a fixed maximum or average independently to each channel;
- strided convolution learns weighted local combinations and can change the number
  of channels.

Modern architectures often prefer strided convolution because the downsampling
operation itself can be learned. Pooling remains valuable for understanding fixed
spatial summarization and appears in many established architectures.

## Connection to this project

- Notebook 19 implements and gradient-checks max pooling.
- `cs231n_practice/cnn_layers.py` contains reusable pooling forward/backward passes.
- Tests include overlap, tied maxima, and stride-created gaps.

## Key takeaway

Pooling trades spatial precision for smaller feature maps and broader context in
later layers. Its backward behavior follows the summary rule: max pooling routes
gradient to a winner, while average pooling distributes it.
