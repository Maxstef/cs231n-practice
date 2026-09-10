# DETR and object detection as set prediction

DETR reorganizes object detection around transformers and direct set
prediction. Instead of starting with dense anchors or an external proposal
algorithm, it produces a fixed-size set of candidate object predictions.

## High-level pipeline

The original DETR combines a CNN feature extractor with a transformer:

```text
image
-> CNN visual feature map
-> flatten to feature tokens + positional information
-> transformer encoder
-> transformer decoder + learned object queries
-> one class and one box per query
```

Calling DETR a transformer detector does not imply that its original backbone
was transformer-only. CNN backbones, transformer backbones, and hybrid designs
can all participate in modern detection systems.

## Object queries

The decoder receives a fixed number of learned **object queries**. A query is
not initially a specific image crop or predefined anchor box. Through decoder
attention, it retrieves information from encoded image features and develops a
representation used by prediction heads.

Each final query output predicts:

- a class, including a special **no object** category;
- a bounding box.

If the model has more queries than visible objects, the unused queries should
predict no object. This converts a variable number of real objects into a fixed
number of model output slots.

## Why matching is necessary

Ground-truth objects are an unordered set. If an image contains two birds,
there is no natural rule saying which must be assigned to query 3 and which to
query 17. Training therefore searches for a one-to-one assignment between
ground-truth objects and predicted queries.

The matching cost can combine classification disagreement and box mismatch.
After the assignment:

- matched queries receive class and box losses;
- unmatched queries receive the no-object classification target.

This is **bipartite matching**: each selected prediction matches at most one
target, and each target matches at most one prediction.

## Why set prediction reduces duplicates

Traditional dense detectors often produce several high-confidence boxes around
one object and rely on NMS to remove duplicates. DETR includes one-to-one
assignment in training. Only one query receives credit for each ground-truth
object; competing duplicate queries are trained toward no object.

The original DETR formulation can therefore avoid:

- predefined anchor boxes;
- external region proposals;
- RoI Pooling or RoI Align;
- traditional greedy NMS.

It still predicts and regresses box coordinates. "No regression of anchor box
transforms" means its boxes are not decoded as corrections to predefined
anchors, not that box regression disappears.

## DETR compared with conventional detection

| Conventional anchor-based detector | Original DETR |
| --- | --- |
| Many spatial anchors | Fixed learned object queries |
| Box offsets relative to anchors | Direct normalized box prediction |
| Local candidate assignment | Global one-to-one assignment |
| Duplicate removal commonly uses NMS | Set loss discourages duplicates |
| CNN/FPN heads common | Transformer encoder-decoder head |

These columns describe the original conceptual contrast, not every later
variant. Transformer-based, one-stage, anchor-free, and NMS-free are different
properties and should not be treated as synonyms.

## Connection to attention

Encoder self-attention lets spatial image features exchange context. Decoder
queries use cross-attention to retrieve relevant encoded image information.
Different queries can specialize toward different objects because their
representations and attention patterns evolve separately.

Positional information remains necessary: without it, visual tokens do not
explicitly communicate where their features came from, yet box prediction is
inherently spatial.

## Trade-offs to remember

DETR simplifies the conceptual post-processing pipeline, but that does not make
training or optimization automatically simple. Its set-matching objective,
large no-object population, attention computation, spatial resolution, and
small-object behavior all require careful design. Later variants modify these
parts while retaining the broad set-prediction idea.

## Review questions

1. What does an object query represent before and after decoder attention?
2. Why can query indices not be assigned permanently to object classes?
3. Why is bipartite matching needed during training?
4. How does one-to-one training discourage duplicate detections?
5. Which conventional components can original DETR remove?
6. Why does DETR still need positional information?

## Related practice

- Notebook 30: attention intuition
- Notebook 31: scaled dot-product attention
- Notebook 32: multi-head attention
- Notebook 41: conventional and transformer detection architectures

## Source

- Stanford CS231n Spring 2025, Lecture 9: Object Detection, Image
  Segmentation, Visualizing and Understanding, available from the
  [course schedule](https://cs231n.stanford.edu/2025/schedule.html).
