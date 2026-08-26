# Notes

Concise conceptual references accompanying the executable notebooks. These
notes are written in original language from course study and experimentation;
they are not copies of Stanford lecture slides.

## Contents

### Course context

- [Computer vision and deep learning](01-course-context/computer-vision-and-deep-learning.md)

### Image classification

- [k-Nearest Neighbors](02-image-classification/knn.md)
- [Data splitting and model selection](02-image-classification/model-selection.md)

### Linear classifiers

- [Scores and geometry](03-linear-classifiers/scores-and-geometry.md)
- [Multiclass SVM loss](03-linear-classifiers/multiclass-svm-loss.md)
- [Regularization](03-linear-classifiers/regularization.md)
- [Softmax, cross-entropy, and log loss](03-linear-classifiers/softmax-cross-entropy-log-loss.md)
- [Stable softmax cross-entropy from logits](03-linear-classifiers/stable-softmax-cross-entropy.md)
- [Sigmoid, logistic, and softmax](03-linear-classifiers/sigmoid-logistic-softmax.md)

### Optimization

- [Gradient descent and stochastic gradient descent](04-optimization/gradient-descent-vs-sgd.md)
- [SGD optimizers](04-optimization/sgd-optimizers.md)
- [Hyperparameter tuning and learning rates](04-optimization/hyperparameter-tuning-and-learning-rates.md)
- [Reading training and validation curves](04-optimization/reading-learning-curves.md)

### Neural networks and backpropagation

- [Computational graphs](05-neural-networks/computational-graphs.md)
- [Backpropagation with vectors and matrices](05-neural-networks/backpropagation.md)
- [Activation functions](05-neural-networks/activation-functions.md)
- [Two-layer neural networks](05-neural-networks/two-layer-neural-network.md)
- [Weight initialization](05-neural-networks/weight-initialization.md)
- [Normalization layers](05-neural-networks/normalization.md)
- [Dropout](05-neural-networks/dropout.md)

### Convolutional neural networks

- [Convolution layers](06-convolutional-networks/convolution-layers.md)
- [Pooling and spatial downsampling](06-convolutional-networks/pooling-and-downsampling.md)
- [Receptive fields and small CNNs](06-convolutional-networks/receptive-fields-and-small-cnns.md)
- [From early CNNs to deeper architectures](06-convolutional-networks/cnn-architectures.md)
- [Residual connections](06-convolutional-networks/residual-connections.md)
- [Image data augmentation](06-convolutional-networks/data-augmentation.md)
- [Transfer learning with CNNs](06-convolutional-networks/transfer-learning.md)

## Conventions

- Equations and diagrams should be recreated rather than copied from slides.
- Each note should distinguish established facts from experiment observations.
- Reusable implementation details belong in `cs231n_practice/`.
- Longer executable derivations and visualizations belong in `notebooks/`.
- Private lecture screenshots used for study stay in the ignored
  `_references/` directory and must not be committed.

## Primary course source

- [Stanford CS231n Spring 2025 schedule and materials](https://cs231n.stanford.edu/2025/schedule.html)
