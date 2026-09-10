# Two-stage and one-stage object detectors

Object detectors can be organized by what happens between dense image features
and final class-labelled boxes. The most useful distinction is whether a
separate learned head re-examines a selected set of regions.

## Why naive crop classification is expensive

A direct strategy is to crop many image windows at different positions,
scales, and aspect ratios, then classify every crop as an object class or
background. This creates two problems:

- the number of possible windows is enormous;
- overlapping crops repeatedly compute nearly identical visual features.

Region proposals reduce the search space by suggesting boxes likely to contain
objects, but early systems still repeated the CNN computation for each region.

## R-CNN to Faster R-CNN

### R-CNN

Original R-CNN used an external proposal method such as Selective Search. Each
proposed image region was cropped, resized, and passed independently through a
CNN. Separate components then classified the region and refined its box.

```text
image -> external proposals -> crop each region -> CNN for each crop
      -> class prediction + box correction
```

This was accurate for its time but slow because an image required many CNN
forward passes.

### Fast R-CNN

Fast R-CNN shares the expensive convolutional work:

```text
image -> CNN once -> shared feature map
                  + proposals -> fixed-size RoI features
                  -> class prediction + box correction
```

RoI Pooling converts differently sized feature regions to a fixed spatial
shape. RoI Align improves spatial precision by sampling fractional coordinates
instead of relying on coarse boundary quantization.

The proposal method is still external: it is not learned jointly through the
detector's backpropagation.

### Faster R-CNN

Faster R-CNN adds a **Region Proposal Network (RPN)** on the shared feature map.
The RPN predicts whether dense candidate regions contain objects and predicts
rough box corrections. Selected proposals then enter the second-stage RoI
head for final class prediction and further box refinement.

```text
image -> backbone features -> RPN objectness + rough boxes
                           -> proposal selection
                           -> RoI head -> final classes + refined boxes
```

The RPN and RoI head share backbone features and can be trained as parts of one
detection system.

## Anchors and box offsets

An anchor is a predefined reference box attached to a feature-map location.
Multiple scales and aspect ratios give a location several starting shapes.
The anchor itself is not the final prediction: the network predicts offsets
that shift and resize it.

For anchor center and size $(c_x^a,c_y^a,w^a,h^a)$ and predicted offsets
$(t_x,t_y,t_w,t_h)$, a common decoding convention is

$$
c_x=c_x^a+t_xw^a,
$$

$$
c_y=c_y^a+t_yh^a,
$$

$$
w=w^a e^{t_w}, \qquad h=h^a e^{t_h}.
$$

The center offsets are relative to anchor size. The exponential makes width
and height positive and represents multiplicative size changes: zero preserves
size, $\log 2$ doubles it, and $\log 0.5$ halves it.

Training commonly labels anchors using their best IoU with ground truth:

- high IoU: positive object candidate;
- low IoU: negative background candidate;
- intermediate IoU: ignored as ambiguous.

Positive anchors receive box-regression targets. Real assignment policies may
also force at least one positive candidate for each ground-truth object.

## One-stage detectors

One-stage detectors such as SSD, YOLO, and RetinaNet do not send a selected
proposal list through a separate RoI classification head. A dense head directly
predicts final class scores and boxes across feature-map locations.

```text
image -> backbone / feature pyramid -> dense class + box predictions
      -> confidence filtering -> NMS -> final detections
```

"One stage" does not mean one neural-network layer. It refers to the absence of
a separate learned proposal-to-RoI classification stage.

## Candidate flow comparison

| Property | Two-stage detector | One-stage detector |
| --- | --- | --- |
| Dense first output | Objectness and proposal boxes | Final class and box candidates |
| Region selection | Before a learned RoI head | No separate RoI head |
| Region features | Explicit fixed-size RoI features | Dense feature-map features |
| Typical tendency | More region-level refinement | Simpler, often faster pipeline |

This is an architectural tendency, not a universal speed or accuracy ranking.

## Dense imbalance and focal loss

Dense detectors evaluate many candidates, most of which are easy background.
Their combined classification loss can overwhelm the small number of positive
objects. RetinaNet introduced focal loss to reduce the influence of already
easy examples, allowing training to focus on hard negatives and foreground.

## Anchor-free detection

Dense prediction does not require anchors. An anchor-free detector may treat a
feature location as a reference point and predict distances to the left, top,
right, and bottom sides of a box. Anchor-based versus anchor-free and
one-stage versus two-stage are separate design choices.

## Review questions

1. What repeated computation made original R-CNN slow?
2. What computation did Fast R-CNN share?
3. What did the RPN replace in Faster R-CNN?
4. How can predicted offsets move and resize an anchor?
5. What makes a detector one-stage rather than two-stage?
6. Why is background imbalance severe in dense detection?

## Related practice

- Notebook 41: object-detection architectures
- `cs231n_practice/detection.py`

## Source

- Stanford CS231n Spring 2025, Lecture 9: Object Detection, Image
  Segmentation, Visualizing and Understanding, available from the
  [course schedule](https://cs231n.stanford.edu/2025/schedule.html).
