# Computer Vision and Deep Learning

## What is computer vision?

Computer vision studies how computational systems can extract useful meaning
from visual data. Images are numerical arrays, but the desired outputs usually
describe higher-level concepts: object identities, locations, shapes, motion,
depth, relationships, or actions.

The field overlaps several areas:

- **machine learning**, because models learn patterns from examples;
- **artificial intelligence**, because visual perception supports reasoning and
  action;
- **mathematics**, especially linear algebra, calculus, probability, and
  optimization;
- **computer science**, which supplies algorithms, data structures, and
  efficient systems;
- **neuroscience and psychology**, which offer perspectives on biological and
  human perception;
- **robotics**, where visual predictions must guide physical actions.

These areas overlap rather than forming a simple hierarchy. Computer vision is
not identical to deep learning: deep neural networks are currently a powerful
family of methods for vision, but the field also includes geometry, image
processing, optimization, probabilistic modeling, and classical feature-based
methods.

## A compressed historical perspective

Modern vision systems emerged from several interacting lines of work:

1. Research on biological vision motivated computational models of receptive
   fields and hierarchical processing.
2. Early neural models introduced trainable units and layered representations.
3. Classical computer vision developed explicit algorithms for edges, local
   features, segmentation, geometry, and recognition.
4. Backpropagation made multilayer networks trainable through gradient-based
   optimization.
5. Larger labeled datasets, faster hardware, and improved training methods made
   deep convolutional networks practical at scale.
6. AlexNet's 2012 ImageNet result became a prominent demonstration that these
   ingredients could dramatically improve large-scale image classification.

The important lesson is not that progress came from one isolated algorithm.
Representations, objectives, optimization, datasets, compute, and evaluation
benchmarks developed together.

## Why image classification is a useful starting point

Image classification asks a deliberately simple question:

> Given an image, which label from a fixed set best describes it?

This task lets us study the complete learning pipeline without immediately
handling object locations or dense pixel predictions. It introduces:

- numerical image representations;
- training, validation, and test data;
- score functions and decision rules;
- loss functions;
- optimization;
- generalization and error analysis.

The same foundations recur in detection, segmentation, captioning, and other
vision tasks.

## Review questions

1. Why is computer vision broader than deep learning?
2. Which ingredients besides model architecture enabled modern deep learning?
3. Why is classification a convenient setting for learning the ML pipeline?

## Source

- Stanford CS231n Spring 2025, Lecture 1: Introduction, available from the
  [course schedule](https://cs231n.stanford.edu/2025/schedule.html).
