# 3D convolution and video CNNs

A 2D convolution processes height and width. A 3D convolution extends its
kernel across time as well:

```text
input:   (N, C_in,  T,     H,     W)
weights: (C_out, C_in, K_t, K_h, K_w)
output:  (N, C_out, T_out, H_out, W_out)
```

One output value is a weighted sum over input channels and a local
$K_t \times K_h \times K_w$ space-time neighborhood. A collection of
`C_out` learned kernels is called a **filter bank**; “bank” simply means a set
of filters producing different output channels.

## Why convolve through time?

A temporal kernel can respond to local changes, such as an edge moving between
nearby frames. Stacking layers grows both spatial and temporal receptive fields,
allowing later features to describe longer motion patterns.

This costs more than applying a 2D CNN to individual frames. A 3D kernel has an
extra temporal dimension, and its intermediate feature maps also retain time.
Temporal stride or pooling reduces this cost but may discard brief events.

## C3D and I3D

**C3D** applies 3D convolution throughout a CNN, commonly using small
$3 \times 3 \times 3$ kernels. It is conceptually similar to extending a
deep image CNN into time, but training is computationally expensive.

**I3D** initializes a 3D network from a pretrained 2D network by copying each
2D kernel across the temporal dimension. Dividing the copies by $K_t$ preserves
the response for a video made of repeated identical frames:

$$
W_{3D}[:,:,t,:,:]=\frac{W_{2D}}{K_t}.
$$

This reuses useful image features while allowing subsequent training to adapt
them to motion. Inflation is an initialization strategy, not a guarantee that
the final temporal slices remain identical.

## Key takeaway

3D CNNs learn local appearance and motion jointly. They offer a direct video
extension of CNNs, with greater temporal modeling capacity and greater compute
and data requirements.

## Related practice

- Notebook 46: naïve 3D convolution, receptive fields, and inflation
- Notebook 49: small 3D video classifier experiment
- `cs231n_practice/video.py`
- `cs231n_practice/classifiers/video.py`

## Source

- Stanford CS231n Spring 2025, Lecture 10: Video Understanding, available from
  the [course schedule](https://cs231n.stanford.edu/2025/schedule.html).
