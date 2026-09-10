# The object-detection problem

Object detection answers two questions for every visible object:

1. **What is it?** — predict an object class.
2. **Where is it?** — predict its spatial extent, usually a bounding box.

This combines classification with localization. Unlike ordinary image
classification, the number of outputs is not fixed by the batch shape: one
image may contain no objects, another one object, and another dozens.

## Related vision tasks

| Task | Output |
| --- | --- |
| Image classification | One class label for the complete image |
| Single-object localization | One class label and one bounding box |
| Object detection | A variable-size set of classes, boxes, and confidence scores |
| Semantic segmentation | One semantic class per pixel |
| Instance segmentation | One class and pixel mask per object instance |

Detection and instance segmentation distinguish separate objects. Semantic
segmentation labels pixels by class but does not distinguish two instances of
the same class.

## Single-object localization as multitask learning

For an image known to contain one main object, a shared visual representation
can feed two prediction heads:

```text
image -> feature extractor -> class scores
                         +-> box coordinates
```

The classification head is trained with a classification loss such as
softmax cross-entropy. The localization head is trained with a regression
loss. A simplified total objective is

$$
L = L_{classification} + \lambda_{box} L_{box}.
$$

The coefficient $\lambda_{box}$ balances errors measured on different scales.
Improving one task should not make its numerical loss dominate the other task
merely because of units or magnitude.

Modern detectors may use L1-style coordinate losses, smooth L1, and
overlap-based losses rather than only squared error. The durable idea is that
classification and localization are learned jointly.

## Box representations

Two common representations are:

```text
corner form:       (x1, y1, x2, y2)
center-size form:  (cx, cy, width, height)
```

Coordinates may be measured in pixels or normalized by image dimensions. A
format is not meaningful without its coordinate convention: corner inclusion,
image bounds, axis order, and normalization must all be consistent.

## Why multiple objects are harder

A fixed vector for one box does not naturally describe a variable number of
objects. Detectors solve this by producing a fixed, usually large collection
of **candidate predictions**, then converting it into a variable-size result:

```text
fixed dense or query outputs
-> decode boxes and scores
-> remove low-confidence candidates and duplicates
-> variable number of final detections
```

Candidate mechanisms include anchors at feature-map locations, anchor-free
points, region proposals, and learned object queries. They differ in structure
but all give the model enough potential outputs to represent multiple objects.

## Training and inference are different views

During training, candidate predictions must be assigned to ground-truth
objects so classification and box targets can be constructed. During
inference, ground truth is unavailable, so confidence filtering and often
non-maximum suppression select the final detections.

This distinction explains why a detector can produce thousands of internal
candidates but only a few visible results.

## Review questions

1. Why is object detection not ordinary image classification?
2. Why does a localization model need more than a classification loss?
3. Why is a detector's internal output often much larger than its final output?
4. What information must accompany a box format to make it unambiguous?

## Related practice

- Notebook 38: bounding-box formats and IoU
- Notebook 39: detection outputs and post-processing
- Notebook 40: detection evaluation

## Source

- Stanford CS231n Spring 2025, Lecture 9: Object Detection, Image
  Segmentation, Visualizing and Understanding, available from the
  [course schedule](https://cs231n.stanford.edu/2025/schedule.html).
