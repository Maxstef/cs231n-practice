# Detection post-processing and evaluation

A detector usually emits many more candidates than there are real objects.
Post-processing converts these dense candidates into a smaller result, while
evaluation determines whether the retained boxes are correct.

## Confidence and class scores

Candidate outputs may separate:

- **objectness:** confidence that a candidate contains some object;
- **class probability:** confidence in a particular category given the visual
  evidence;
- **box coordinates or offsets:** the candidate's location and extent.

Exact score conventions vary by architecture. When objectness and conditional
class probability are separate, a class-specific confidence can be formed by
multiplying them. One should inspect the model API rather than assume all
detectors use the same score definition.

## Intersection over Union

For boxes $A$ and $B$,

$$
\mathrm{IoU}(A,B)=\frac{|A\cap B|}{|A\cup B|}.
$$

IoU is zero for non-overlapping boxes and one for identical nondegenerate
boxes. It is used for several distinct decisions:

- assigning training candidates to ground truth;
- suppressing duplicate predictions;
- deciding whether a prediction is correct during evaluation.

These uses may have different thresholds and must not be conflated.

## Non-maximum suppression

Greedy NMS processes candidates from highest to lowest confidence:

1. keep the highest-scoring remaining box;
2. remove lower-scoring boxes whose IoU with it exceeds a threshold;
3. repeat until no candidates remain.

Class-wise NMS runs this process independently for each class. Class-agnostic
NMS can suppress highly overlapping boxes even when their predicted classes
differ. The appropriate choice depends on the detector and task.

The NMS IoU threshold has a trade-off:

- too low may remove nearby distinct objects;
- too high may leave duplicate detections.

## Matching predictions for evaluation

For a fixed class and IoU threshold, predictions are ranked by descending
confidence. Each prediction may match an unmatched ground-truth object in the
same image when their IoU is sufficient. Each ground truth can be matched only
once.

```text
first valid prediction for an object -> true positive
duplicate prediction                 -> false positive
wrong class or insufficient IoU      -> false positive
ground truth never matched           -> false negative
```

One-to-one matching prevents a detector from gaining extra credit by reporting
the same object repeatedly.

## Precision and recall over the ranking

After considering the first $k$ predictions,

$$
\mathrm{precision}_k=\frac{\mathrm{TP}_{1:k}}{k},
$$

$$
\mathrm{recall}_k=\frac{\mathrm{TP}_{1:k}}{N_{gt}}.
$$

A true positive increases cumulative TP and recall. A false positive leaves
recall unchanged and reduces precision. Confidence scores determine the order
in which these events occur, which is why changing scores can change AP even
when boxes and labels stay fixed.

## Average Precision

Average Precision summarizes the precision-recall curve. All-point
interpolation first forms a precision envelope:

$$
p_{envelope}(r)=\max_{\tilde r\geq r}p(\tilde r).
$$

It then accumulates rectangle areas only where recall increases:

$$
\mathrm{AP}=\sum_k(r_k-r_{k-1})p_{envelope,k}.
$$

An important edge case occurs when the detector never reaches recall one. The
unreachable recall interval must have precision zero; it must not receive the
last observed precision value.

Late false positives may contribute no new area after recall has already
reached one. Therefore AP equal to one means all required true positives were
ranked before harmful false positives, not necessarily that no false-positive
prediction exists.

## AP and mAP

AP is normally computed independently for each class. Mean Average Precision
averages those class AP values. It is not the same as placing every class in
one shared ranking and calculating one pooled AP.

Metric names must state their convention. For example, AP at IoU 0.50 uses a
single localization threshold. The COCO headline metric additionally averages
over IoU thresholds from 0.50 through 0.95, along with its benchmark-specific
policies.

## Keep these thresholds separate

| Threshold | Purpose |
| --- | --- |
| Confidence threshold | Remove weak predictions |
| NMS IoU threshold | Decide when predictions are duplicates |
| Evaluation IoU threshold | Decide whether a prediction matches ground truth |

## Review questions

1. Why can the same IoU function appear in training, NMS, and evaluation?
2. Why must each ground-truth object be matched only once?
3. How does a false positive change precision and recall?
4. Why can AP change when only confidence scores change?
5. Why is per-class mAP not a pooled multiclass AP?

## Related practice

- Notebook 38: box geometry and IoU
- Notebook 39: confidence filtering and NMS
- Notebook 40: precision, recall, AP, and mAP
- `cs231n_practice/detection.py`

## Source

- Stanford CS231n Spring 2025, Lecture 9: Object Detection, Image
  Segmentation, Visualizing and Understanding, available from the
  [course schedule](https://cs231n.stanford.edu/2025/schedule.html).
