# Self-supervised learning and evaluation

Supervised learning obtains a training target from a human-provided label.
**Self-supervised learning (SSL)** constructs its target from the data itself.
It can therefore use large unlabeled datasets to learn an encoder before a
smaller labeled dataset is available.

```text
unlabeled data -> self-supervised objective -> pretrained encoder
                                                   ↓
labeled data   -> downstream objective      -> target-task model
```

The objective is still optimized with ordinary supervised machinery such as a
loss and backpropagation. “Self-supervised” describes where the target comes
from, not a different optimization algorithm.

## Pretext and downstream tasks

A **pretext task** automatically creates a learning problem whose main purpose
is to produce useful features. Classical vision examples include:

- predicting an applied image rotation;
- predicting the relative location of two patches;
- recovering a shuffled patch arrangement;
- colorizing a grayscale image;
- reconstructing missing image regions.

A **downstream task** is the application we ultimately care about, such as
classification, detection, or segmentation. Pretext-task accuracy alone is
not sufficient: an encoder can learn shortcuts that solve the pretext task
without capturing transferable visual meaning.

For rotation prediction, for example, the image provides the label because we
know which rotation we applied. The intended pressure is to recognize normal
object orientation, but borders, interpolation artifacts, or dataset biases
may provide easier shortcuts.

## Evaluating a learned representation

Several protocols answer different questions:

| Protocol | Encoder updated? | What it measures |
| --- | --- | --- |
| Frozen linear probe | No | How linearly accessible target information already is |
| Fine-tuning | Yes | How useful the initialization is after task-specific adaptation |
| Nearest neighbors | No | Whether similar examples are nearby in feature space |
| Clustering | No | Whether feature groups align with meaningful structure |

A frozen linear probe trains only a linear classifier on encoder features. It
is deliberately limited: if it performs well, the encoder has already made
the classes relatively easy to separate. Fine-tuning can achieve higher task
performance, but it is harder to tell how much came from pretraining versus
subsequent supervised adaptation.

Evaluation should use the same labeled split, classifier capacity, training
budget, and preprocessing when comparing encoders. Small experiments should
also be repeated with multiple seeds before treating a modest difference as a
general conclusion.

## Main idea

The pretext loss is a means, not the final goal. A lower pretext loss does not
necessarily imply a stronger downstream representation because the two tasks
reward different information.

## Sources

- Stanford CS231n Spring 2025, Lecture 12: Self-Supervised Learning, available
  from the [course schedule](https://cs231n.stanford.edu/2025/schedule.html).
- Gidaris et al., [Unsupervised Representation Learning by Predicting Image
  Rotations](https://arxiv.org/abs/1803.07728).

