# Data Splitting and Model Selection

## Parameters and hyperparameters

**Parameters** are learned from training data. Examples include a linear
classifier's weights and a neural network's biases.

**Hyperparameters** configure the learning procedure or model. Examples include
the number of neighbors in kNN, regularization strength, and learning rate.
They are chosen through experiments rather than learned directly by the model's
optimization rule.

## Why training performance is insufficient

Selecting hyperparameters by training accuracy rewards memorization. For kNN,
`k=1` can classify every stored training example correctly because each example
is its own nearest neighbor. This says little about performance on new images.

## Why the test set cannot guide choices

If test results influence hyperparameters, preprocessing, architecture, or
other decisions, information from the test set leaks into development. The
reported test result is then no longer an unbiased final estimate.

Repeatedly checking test performance can gradually turn the test set into an
unofficial validation set even if its examples never enter training.

## Train, validation, and test roles

```text
training data   -> learn parameters or store examples
validation data -> select hyperparameters and compare alternatives
test data       -> evaluate the finalized procedure once
```

The validation and test sets should represent the target data distribution but
must remain separate from training. Any preprocessing statistics learned from
data must also respect this separation.

## Held-out validation

A single held-out split is simple and computationally efficient:

```text
available labeled training data
├── training subset
└── validation subset

official test data -> untouched until final evaluation
```

Its weakness is sensitivity to the particular validation examples selected.
With small datasets, one split may produce a noisy estimate.

## k-fold cross-validation

In k-fold cross-validation, the available development data is divided into
`k` folds. Each fold becomes validation data once while the other folds provide
training data. The validation results are averaged:

```text
run 1: [validation][ training ][ training ][ training ][ training ]
run 2: [ training ][validation][ training ][ training ][ training ]
run 3: [ training ][ training ][validation][ training ][ training ]
run 4: [ training ][ training ][ training ][validation][ training ]
run 5: [ training ][ training ][ training ][ training ][validation]
```

Cross-validation uses limited data more thoroughly and exposes variability
between folds, but it multiplies training cost. It is common for smaller
datasets and relatively inexpensive models, and less common for large deep
networks.

The final test set stays outside all folds.

## Our current practice

The kNN notebooks use a seeded held-out validation subset to select `k`, then
evaluate the selected value on a separate CIFAR-10 test subset. K-fold
cross-validation remains planned for the linear-classifier stage.

## Review questions

1. Why is training accuracy unsuitable for selecting `k`?
2. How does test-set reuse bias the final estimate?
3. What does k-fold cross-validation average over?
4. Why might cross-validation be avoided for a very expensive neural network?

## Source

- Stanford CS231n Spring 2025, Lecture 2: Image Classification with Linear
  Classifiers, available from the
  [course schedule](https://cs231n.stanford.edu/2025/schedule.html).
