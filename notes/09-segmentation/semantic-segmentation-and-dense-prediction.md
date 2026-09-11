# Semantic segmentation and dense prediction

Semantic segmentation assigns a class to every image pixel. For a batch of
images with shape $(N,3,H,W)$ and $C$ semantic classes, a segmentation model
typically produces logits with shape $(N,C,H,W)$. The prediction is

$$
\hat y_{n,h,w}=\mathop{\mathrm{argmax}}_c S_{n,c,h,w}.
$$

The target has shape $(N,H,W)$ because it stores only one correct class index
per pixel. Separate objects of the same class share the same semantic label;
semantic segmentation does not assign instance identities.

## Why independent sliding windows are wasteful

One possible strategy is to crop a patch around every pixel and classify its
center. Nearby patches overlap heavily, so the CNN repeatedly calculates
almost the same features. It also turns one image into a large collection of
separate forward passes.

A convolutional network already applies shared filters at all spatial
locations. Computing one feature map for the complete image reuses that work:

```text
many overlapping crops -> repeated CNN computation
complete image          -> one shared spatial feature map
```

This is the same efficiency principle that made shared detector backbones
preferable to running a CNN independently on every candidate box.

## Fully convolutional prediction

A classification network eventually collapses spatial features into one
vector. A Fully Convolutional Network (FCN) instead preserves a spatial grid
and replaces the fixed fully connected classifier with convolutional
prediction layers.

```text
image:       (3, H, W)
features:    (D, H', W')
class scores:(C, H', W')
prediction:  (H', W')
```

A $1\times1$ convolution is a common final layer. At every location it maps a
$D$-dimensional feature vector to $C$ class logits using the same learned
weights. This makes a dense classifier rather than one image-level
classifier.

Keeping every intermediate feature at full image resolution would be
expensive and would limit the receptive field. Practical FCNs therefore often
downsample to learn efficient, contextual features and later upsample the
scores or features.

## Per-pixel loss

Softmax cross-entropy can be applied independently at every valid pixel and
then averaged:

$$
L=-\frac{1}{P}\sum_{(n,h,w)\in\mathcal V}
\log p_{n,y_{n,h,w},h,w},
$$

where $\mathcal V$ is the set of valid pixels and $P=|\mathcal V|$. An ignore
index excludes uncertain, padded, or unlabeled pixels from both the loss and
its gradient.

Class imbalance matters because large background regions may contribute far
more pixels than small objects. Pixel accuracy can therefore look strong even
when minority classes are poorly segmented. Per-class IoU and mean IoU expose
this problem more clearly.

For class $c$,

$$
\mathrm{IoU}_c=\frac{TP_c}{TP_c+FP_c+FN_c}.
$$

## Discrete masks versus continuous scores

Ground-truth mask values are class IDs, not continuous intensities. Resizing a
target mask with bilinear interpolation can create meaningless fractional
labels. Target masks normally use nearest-neighbor interpolation. Logits and
feature maps are continuous and may use bilinear interpolation before
`argmax`.

## Review questions

1. Why is a patch classifier at every pixel computationally redundant?
2. What turns an image classifier into a fully convolutional dense predictor?
3. Why do logits contain a class axis while target masks do not?
4. Why can pixel accuracy hide failures on small classes?
5. Why do targets and logits use different resizing rules?

## Related practice

- Notebook 42: semantic segmentation
- `cs231n_practice/segmentation.py`

## Source

- Stanford CS231n Spring 2025, Lecture 9: Object Detection, Image
  Segmentation, Visualizing and Understanding, available from the
  [course schedule](https://cs231n.stanford.edu/2025/schedule.html).
