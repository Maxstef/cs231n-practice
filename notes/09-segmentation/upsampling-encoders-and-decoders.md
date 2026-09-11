# Upsampling, encoders, and decoders

Segmentation networks need two properties that pull in opposite directions:

- broad semantic context to recognize what occupies a region;
- precise spatial detail to place object boundaries at the correct pixels.

An encoder obtains context by reducing spatial resolution. A decoder restores
resolution so the model can make dense predictions.

```text
image -> high resolution -> medium resolution -> deep low resolution
                                                   |
mask  <- high resolution <- medium resolution <- decoder
```

Downsampling can use pooling or strided convolution. Upsampling has several
possible meanings and should not be thought of as perfectly undoing the
encoder: information discarded during downsampling cannot generally be
reconstructed from the coarse tensor alone.

## Fixed nearest-neighbor upsampling

Nearest-neighbor upsampling copies each input value into a larger block. For
scale two,

```text
1 2        1 1 2 2
3 4   ->   1 1 2 2
           3 3 4 4
           3 3 4 4
```

It has no learned parameters. A convolution after resizing can learn how to
mix and refine the repeated features. This `resize -> convolution` pattern is
simple and avoids some uneven-overlap artifacts associated with transposed
convolutions.

## Zero insertion and max unpooling

A bed-of-nails operation inserts zeros between existing values:

```text
1 2        1 0 2 0
3 4   ->   0 0 0 0
           3 0 4 0
           0 0 0 0
```

By itself this is sparse; a following convolution spreads information into
the gaps.

Max unpooling additionally remembers where each maximum came from during max
pooling. The pooled values are placed back at those recorded positions, while
other positions receive zero. It restores positional hints, but it does not
restore the non-maximum values that pooling discarded.

## Transposed convolution

A transposed convolution is a learned linear upsampling operation. Each input
value scales a learned kernel, and that weighted kernel is placed into an
output region. Contributions are summed where regions overlap.

In one dimension, input $[a,b]$ and kernel $[x,y,z]$ with stride one produce

$$
[ax,\; ay+bx,\; az+by,\; bz].
$$

The stride controls how far apart adjacent input contributions are placed in
the output. Kernel size, stride, padding, and output padding together determine
the output size. Despite its name, a transposed convolution is not the inverse
of a convolution. It corresponds to the transpose of the matrix representing
the related convolutional linear operation.

Uneven kernel overlap can create checkerboard-like artifacts. Common
alternatives include carefully chosen kernel/stride combinations or explicit
interpolation followed by ordinary convolution.

## Skip connections and U-Net

Decoder upsampling cannot invent fine details lost by the encoder. U-Net-style
skip connections pass earlier high-resolution encoder features directly to a
decoder stage at the corresponding resolution.

```text
encoder high-resolution features --------+
                                          v
deep semantic features -> upsample -> concatenate -> convolution -> mask
```

The deep path contributes recognition and context: *this region contains a
cat*. The skip path contributes localization and boundaries: *an edge passes
through these pixels*. Concatenation increases the channel count; subsequent
convolutions learn how to combine the two sources. Addition is another design
option when channel shapes are compatible.

## Keep these distinctions clear

| Mechanism | Learned? | Main information supplied |
| --- | --- | --- |
| Nearest/bilinear resize | No | A larger continuous grid |
| Max unpooling | No | Recorded locations of pooled maxima |
| Transposed convolution | Yes | Learned spatial spreading and mixing |
| Skip connection | Indirectly through surrounding layers | Earlier high-resolution features |

These mechanisms are compatible rather than mutually exclusive. A network may
use interpolation for resizing, convolutions for refinement, and skip
connections for detail recovery.

## Review questions

1. Why is upsampling not a true reversal of downsampling?
2. What information does max unpooling retain and what has already been lost?
3. How does a transposed convolution construct its output?
4. Why can overlapping transposed-convolution kernels cause artifacts?
5. What complementary information travels through the deep and skip paths?

## Related practice

- Notebook 42: FCNs, encoder-decoders, skip connections, and resizing

## Source

- Stanford CS231n Spring 2025, Lecture 9: Object Detection, Image
  Segmentation, Visualizing and Understanding, available from the
  [course schedule](https://cs231n.stanford.edu/2025/schedule.html).
