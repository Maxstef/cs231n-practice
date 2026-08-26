# Image data augmentation

Data augmentation creates varied training inputs by applying transformations
that should preserve the target label. It exposes the model to plausible
variations without requiring a separately stored image for every variation.

For an original training pair $(x,y)$, sample a random transformation $T$ and
train on

$$
(T(x), y).
$$

The unchanged label assumes that $T$ does not alter the semantic class.

## Why augmentation helps

A model should often recognize an object despite modest changes in position,
scale, orientation, illumination, or color. Augmentation teaches these desired
invariances and reduces reliance on accidental details of the training images.
It therefore acts as a form of regularization.

The transformed samples are not fully independent new observations: they still
come from the same originals. Augmentation cannot replace genuinely diverse
data, but it can make much better use of available data.

## Common image transformations

- random crops, translations, and resizing;
- horizontal flips when left and right do not change the label;
- small rotations or affine transformations;
- brightness, contrast, saturation, or color changes;
- random erasing or masking.

Transformations must match the task. Horizontal flipping is often sensible for
natural-object classification but may be wrong for text, traffic signs,
medical laterality, or classes whose direction matters. Large crops can remove
the object entirely and introduce incorrect labels.

## Training pipeline

Augmentation is usually sampled on the fly:

1. Load an image and label.
2. Randomly transform the image.
3. Normalize it using the training-data statistics.
4. Feed it to the model and compute the loss with the original label.

Because new random choices can be made on each epoch, the model may see a
different version of the same source image many times. Validation and test
pipelines should normally be deterministic so experiments remain comparable.

## Random crops and scales

A common large-image pipeline first resizes the image and then samples a
fixed-size crop. Randomizing the resize scale and crop position teaches some
robustness to object scale and location while still producing a fixed input
shape for batching.

For small images such as CIFAR-10, a typical alternative is to pad the image by
a few pixels, take a random crop back to the original size, and optionally flip
it horizontally.

## Test-time augmentation

Test-time augmentation evaluates several deterministic versions of one input,
such as a center crop, corner crops, and their flips, and combines their
predictions. This can improve robustness but increases inference cost roughly
in proportion to the number of views.

It should be treated as part of the evaluation protocol and reported clearly.
A simple baseline should first use one deterministic resize or crop.

## Related notes

- [Regularization](../03-linear-classifiers/regularization.md)
- [Reading training and validation curves](../04-optimization/reading-learning-curves.md)

## Further reading

- [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)

