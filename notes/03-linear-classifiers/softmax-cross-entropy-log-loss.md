# Softmax, Cross-Entropy, and Log Loss

These terms are often used together—and sometimes used as shorthand for one
another—but they name different pieces of the classification pipeline.

## Softmax is a function

Given $C$ class scores, also called **logits**, softmax produces a probability
distribution:

$$
p_j=\frac{e^{s_j}}{\sum_{k=1}^{C}e^{s_k}}.
$$

Softmax is not itself a loss. It maps one vector of unrestricted scores to
positive values that sum to one. Increasing one class probability necessarily
reduces the total probability available to the other classes.

For numerical stability, subtract the largest score first:

$$
p_j=\frac{e^{s_j-\max(s)}}{\sum_k e^{s_k-\max(s)}}.
$$

This changes intermediate values but not the resulting probabilities.

## Cross-entropy compares distributions

For a target distribution $q$ and predicted distribution $p$, cross-entropy is

$$
H(q,p)=-\sum_{j=1}^{C}q_j\log p_j.
$$

In ordinary single-label classification, the target is one-hot: $q_y=1$ for
the correct class and all other entries are zero. Cross-entropy then reduces to

$$
L=-\log p_y.
$$

Cross-entropy is the loss; softmax is the function that produced $p$ from
multiclass logits.

With soft targets, such as label smoothing or knowledge distillation, several
$q_j$ values can be nonzero. The full sum then matters and cannot be replaced
by selecting only one correct-class probability.

## Log loss and negative log-likelihood

**Log loss** describes the negative logarithm of the probability assigned to
the observed outcome. For one multiclass example it is

$$
-\log p_y,
$$

which is exactly one-hot cross-entropy. The mean over a dataset is also the
negative log-likelihood (NLL) of a categorical probabilistic model.

For binary classification, with target $y\in\{0,1\}$ and predicted probability
$p=P(y=1)$, log loss is

$$
L=-\left[y\log p+(1-y)\log(1-p)\right].
$$

This is binary cross-entropy. Thus, in standard classification contexts,
*log loss*, *cross-entropy loss*, and *negative log-likelihood* often refer to
the same numerical objective. Their broader meanings differ:

- cross-entropy is a general quantity between two distributions;
- log loss emphasizes the probability assigned to the observed label;
- negative log-likelihood emphasizes fitting a probabilistic model;
- softmax is a score-to-probability transformation, not a loss.

## What does “softmax loss” mean?

People commonly say **softmax loss** to mean the combined operation

$$
\text{logits}\rightarrow\text{softmax probabilities}
\rightarrow\text{cross-entropy loss}.
$$

The phrase is convenient but imprecise. Libraries usually combine the two
operations directly from logits for stability. Computing `softmax`, allowing a
tiny probability to underflow to zero, and then computing `log(0)` can produce
an infinite loss. A stable implementation uses log-sum-exp:

$$
L_i=-s_{y_i}^{shifted}
+\log\left(\sum_j e^{s_j^{shifted}}\right).
$$

## Terminology map

| Term | What it is | Typical formula |
|---|---|---|
| Logit or score | Unnormalized model output | $s_j$ |
| Softmax | Multiclass probability function | $e^{s_j}/\sum_k e^{s_k}$ |
| Cross-entropy | Distribution-comparison loss | $-\sum_j q_j\log p_j$ |
| Multiclass log loss | One-hot cross-entropy | $-\log p_y$ |
| Binary log loss | Binary cross-entropy | $-[y\log p+(1-y)\log(1-p)]$ |
| Negative log-likelihood | Likelihood viewpoint on the same objective | $-\log P(y\mid x)$ |
| “Softmax loss” | Shorthand for softmax plus cross-entropy | Usually computed jointly from logits |

## Related project material

- `notebooks/07_softmax_loss.ipynb`

## Review questions

1. Why is softmax not itself a loss function?
2. When does cross-entropy reduce to $-\log p_y$?
3. In standard single-label classification, why are log loss and cross-entropy
   numerically identical?
4. When must the full target distribution $q$ be retained?
5. Why do libraries combine softmax and cross-entropy from logits?
6. What does the informal phrase “softmax loss” usually mean?

## Source

- Stanford CS231n Spring 2025, Lecture 2: Image Classification with Linear
  Classifiers, available from the
  [course schedule](https://cs231n.stanford.edu/2025/schedule.html).
