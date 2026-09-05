# Vision Transformers

A Vision Transformer (ViT) applies a Transformer encoder to an image by first
turning the image into a sequence of patch tokens. The central change is the
input representation; the Transformer blocks themselves remain close to those
used for other modalities.

## From image to patch sequence

For images with shape $(N,C,H,W)$ and non-overlapping square patches of width
$P$, the patch grid has

$$
H_p=H/P, \qquad W_p=W/P,
$$

and the number of patches is

$$
L_p=H_pW_p=\frac{HW}{P^2}.
$$

Every patch contains $CP^2$ values. After flattening, a learned linear
projection maps it to the model width $D$:

$$
x_{patch}=\mathrm{flatten}(patch)W_{patch}+b_{patch}.
$$

The shapes are therefore:

```text
images:             (N, C, H, W)
flattened patches:  (N, L_p, C * P * P)
projected tokens:   (N, L_p, D)
```

A convolution with kernel size $P$ and stride $P$ computes the same kind of
learned non-overlapping patch projection efficiently.

## Position and image structure

Flattening and projecting a patch does not discard its pixel values, but the
token tensor alone does not explicitly describe where each patch came from.
Self-attention is also insensitive to sequence order without additional
position information.

ViT therefore adds learned or fixed positional representations to the patch
tokens. They allow the model to distinguish top from bottom, left from right,
and nearby from distant patches. A one-dimensional token order can represent a
two-dimensional grid because each sequence index consistently corresponds to
one grid location.

For image classification, no causal mask is used: every patch may exchange
information with every other patch.

## Producing one image-level prediction

Two common ways to turn the final patch representations into one image vector
are:

1. prepend a learned `[CLS]` token and classify its final representation;
2. average-pool the final patch representations and classify the result.

With `[CLS]`, the full sequence length is

$$
L=L_p+1.
$$

Its query can retrieve information from patch tokens through every
self-attention layer. The final classifier maps the contextualized `[CLS]`
vector from $D$ features to $C_{classes}$ scores.

```text
image
-> non-overlapping patches
-> learned patch tokens + positions
-> prepend [CLS]
-> stack of Transformer encoder blocks
-> final [CLS] representation
-> linear class scores
```

## Patch-size trade-off

Smaller patches preserve finer spatial detail but create longer sequences. For
a fixed image size, halving $P$ produces four times as many patch tokens.
Because full attention is quadratic in sequence length, its score matrices
grow by approximately sixteen times when the sequence is already much longer
than the single `[CLS]` token.

For the $16 \times 16$ images used in Notebook 37:

| Patch size | Patch grid | Patch tokens | With `[CLS]` | Entries per attention matrix |
| --- | --- | ---: | ---: | ---: |
| `4 x 4` | `4 x 4` | 16 | 17 | `17² = 289` |
| `2 x 2` | `8 x 8` | 64 | 65 | `65² = 4225` |

The smaller patches improved the observed CIFAR-10 result, plausibly because
they retained more fine-grained structure. They also made each attention
matrix about 14.6 times larger. This is an experiment-specific observation,
not a universal rule that smaller patches are always better.

## Why CNNs can be stronger with limited data

CNNs encode strong image-specific inductive biases:

- local neighborhoods interact first;
- the same filters are reused across spatial locations;
- deeper layers gradually enlarge their receptive fields;
- translation-related patterns are handled naturally.

A basic ViT has weaker built-in locality and must learn more of this structure
from examples. This flexibility can be powerful with large datasets,
augmentation, regularization, or pretraining, but a small CNN can generalize
better in a limited-data experiment.

## Common implementation mistakes

- mixing NCHW and NHWC image layouts;
- using a patch size that does not divide the image dimensions;
- flattening the grid in an unintended spatial order;
- forgetting the learned patch projection;
- omitting positional information;
- forgetting to expand `[CLS]` across the batch;
- confusing number of tokens $L$ with attention entries $L^2$;
- interpreting attention maps from an untrained model as meaningful evidence.

## Related practice and implementation

- Notebook 33: positional encodings
- Notebook 34: Transformer encoder blocks
- Notebook 36: ViT patchification and shape intuition
- Notebook 37: controlled ViT CIFAR-10 experiment
- `cs231n_practice/classifiers/vision_transformer.py`

## Source

- Stanford CS231n Spring 2025, Lecture 8: Attention and Transformers,
  available from the
  [course schedule](https://cs231n.stanford.edu/2025/schedule.html).
