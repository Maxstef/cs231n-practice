# Instance and panoptic segmentation

Semantic segmentation assigns categories to pixels but merges separate
objects of the same class. Instance segmentation adds object identity.
Panoptic segmentation combines instance-aware objects with complete semantic
coverage of the scene.

| Task | Semantic class per pixel | Separate same-class objects | Covers every pixel |
| --- | --- | --- | --- |
| Semantic segmentation | Yes | No | Yes |
| Instance segmentation | For detected objects | Yes | Not necessarily |
| Panoptic segmentation | Yes | Yes for thing classes | Yes |

## Instance masks and mask IoU

An instance prediction is naturally represented as a set of masks, classes,
and confidence scores. Two cats have the same class ID but different instance
IDs.

Mask IoU compares their actual foreground pixels:

$$
\mathrm{IoU}(A,B)=\frac{|A\cap B|}{|A\cup B|}.
$$

This differs from box IoU. Two masks can have identical enclosing rectangles
and box IoU one while disagreeing strongly about the object shape inside those
rectangles. For instance matching, two empty masks are assigned IoU zero
rather than treated as a perfect object match.

## Mask R-CNN

Mask R-CNN extends Faster R-CNN with a mask branch:

```text
image -> backbone -> RPN proposals -> RoI Align -> class head
                                             +-> box head
                                             +-> mask head
```

For each proposal, the heads answer three different questions:

- **class head:** what object, or background, is in this RoI?
- **box head:** how should the proposal coordinates be refined?
- **mask head:** which positions inside the RoI belong to this instance?

RoI Align samples without coarse integer boundary rounding. Precise alignment
is especially important for a mask because a small feature displacement can
move an output boundary by several image pixels.

The mask head commonly predicts logits with shape $(R,C,M,M)$, where $R$ is
the number of RoIs, $C$ is the number of foreground classes, and $M\times M$
is the local mask resolution. Only one class channel is used per RoI. The
ground-truth class selects it during training; the predicted class selects it
during inference.

Each selected mask pixel uses sigmoid and binary cross-entropy. It asks
whether that pixel is foreground for one specific object; it does not need to
compete with other semantic classes through softmax. The small mask is resized
to the detected box and placed on the image canvas.

## Overlapping instance predictions

Separate predicted masks may overlap. One simple composition rule lets
higher-confidence masks claim pixels first and prevents later masks from
overwriting them. This is useful for visualization, but production systems may
combine mask probabilities, scores, and class-specific policies.

## Thing and stuff classes

**Things** are countable objects with instance identity: people, dogs, cars,
and chairs. **Stuff** describes amorphous scene regions such as sky, grass,
road, or water. A panoptic result assigns every pixel to exactly one
non-overlapping segment:

- thing segments have semantic class and instance identity;
- stuff segments have semantic class but normally no countable instance.

A compact representation can combine the IDs:

$$
\text{segment ID}=\text{class ID}\times D+\text{instance ID},
$$

where every instance ID is smaller than divisor $D$. Integer division recovers
the class, and remainder recovers the instance. Dataset conventions vary, so
this encoding should not be assumed universally.

## Panoptic Quality

After one-to-one matching of same-class predicted and ground-truth segments,

$$
PQ=\frac{\sum_{(p,g)\in TP}\mathrm{IoU}(p,g)}
{|TP|+0.5|FP|+0.5|FN|}.
$$

It separates into segmentation and recognition components:

$$
SQ=\frac{\sum_{(p,g)\in TP}\mathrm{IoU}(p,g)}{|TP|},
$$

$$
RQ=\frac{|TP|}{|TP|+0.5|FP|+0.5|FN|},
\qquad PQ=SQ\times RQ.
$$

High SQ means matched shapes are accurate. High RQ means few objects are
missed and few extra segments are predicted. A model can therefore have high
SQ but modest PQ when its good masks cover only some required objects.

## Review questions

1. What is lost when an instance map is converted into a semantic map?
2. Why is mask IoU stricter about shape than box IoU?
3. Why does Mask R-CNN predict a small mask in RoI coordinates?
4. Why does the mask head use sigmoid rather than class softmax?
5. What distinguishes things from stuff?
6. What different errors are isolated by SQ and RQ?

## Related practice

- Notebook 43: instance and panoptic segmentation
- `cs231n_practice/segmentation.py`

## Source

- Stanford CS231n Spring 2025, Lecture 9: Object Detection, Image
  Segmentation, Visualizing and Understanding, available from the
  [course schedule](https://cs231n.stanford.edu/2025/schedule.html).
