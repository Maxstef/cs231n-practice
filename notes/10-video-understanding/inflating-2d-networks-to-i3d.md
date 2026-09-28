# Inflating 2D networks into I3D

**I3D** means **Inflated 3D ConvNet**. Its central idea is to reuse a successful
2D image architecture and its pretrained weights for video instead of designing
and training a 3D network entirely from scratch.

## Inflating the architecture

A spatial operation is given a temporal dimension:

```text
2D convolution:  (K_h, K_w)
3D convolution:  (K_t, K_h, K_w)

2D pooling:      (K_h, K_w)
3D pooling:      (K_t, K_h, K_w)
```

The extra temporal extent lets the network combine evidence across neighboring
frames. It also increases computation and makes temporal stride and padding new
architectural choices.

## Inflating pretrained weights

Suppose a pretrained 2D kernel is $W_{2D}$. To initialize a 3D kernel with
temporal width $K_t$, copy the kernel into every temporal slice and divide by
$K_t$:

$$
W_{3D}[:,:,t,:,:]=\frac{W_{2D}}{K_t},
\qquad t=0,\ldots,K_t-1.
$$

Why divide? Consider a constant video in which all $K_t$ input frames are the
same image $X$. The initial 3D response is

$$
\sum_{t=1}^{K_t}X * \frac{W_{2D}}{K_t}
=X * W_{2D}.
$$

Thus the inflated layer initially behaves like the original 2D layer on a
static video. Without division, its response would be multiplied by $K_t$.

## What happens during training?

The copied temporal slices start identically, but they are separate parameters.
Video training can make them different so that the filter becomes sensitive to
temporal change. Inflation therefore supplies a useful initialization; it does
not restrict the model to static image processing.

I3D is often paired with two streams: one processes RGB clips and another
processes optical flow. Their predictions or features are then fused.

## Related practice

- Notebook 46: weight inflation and the constant-video check
- `inflate_conv2d_weight` in `cs231n_practice/video.py`
- [3D convolution and video CNNs](3d-convolution-and-video-cnns.md)

## Source

- Stanford CS231n Spring 2025, Lecture 10: Video Understanding, available from
  the [course schedule](https://cs231n.stanford.edu/2025/schedule.html).
