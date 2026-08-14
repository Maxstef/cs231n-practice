# Multiclass SVM Loss

## Why scores need a loss

A linear classifier produces one compatibility score per class, but training
requires a scalar objective that says how undesirable those scores are. The
multiclass SVM loss requires the correct-class score to exceed every incorrect
score by at least a margin $\Delta$.

For example $i$, correct label $y_i$, and incorrect class $j$:

$$
s_{y_i}\ge s_j+\Delta.
$$

A violation contributes hinge loss:

$$
L_{i,j}=\max(0,s_j-s_{y_i}+\Delta).
$$

The complete example loss excludes the correct class:

$$
L_i=\sum_{j\ne y_i}\max(0,s_j-s_{y_i}+\Delta).
$$

Correct classification only requires $s_{y_i}>s_j$. Zero SVM loss is stronger:
every correct score must lead every incorrect score by at least $\Delta$.

## Hinge behavior

The hinge function has two regions:

- If an incorrect class violates the margin, its positive loss measures the
  size of the violation.
- If the margin is satisfied, its contribution is zero. Increasing the correct
  score further does not reduce that contribution below zero.

This means many different score vectors—and many different parameter
matrices—can have the same zero data loss.

## Batch and vectorized loss

For scores $S\in\mathbb{R}^{N\times C}$ and labels
$y\in\{0,\ldots,C-1\}^N$:

1. Select the correct score from every row, producing shape `(N,)`.
2. Reshape to `(N, 1)` so it broadcasts across the class dimension.
3. Compute all raw margins with shape `(N, C)`.
4. Apply the hinge and set each correct-class entry to zero.
5. Sum across incorrect classes and average across examples.

The correct-class entry must be removed because otherwise it equals

$$
\max(0,s_y-s_y+\Delta)=\Delta.
$$

One-example implementations sometimes sum every entry and subtract one
$\Delta$. Explicitly zeroing the correct entry is equivalent and usually more
readable.

## Useful invariants

These properties are valuable implementation checks:

- Adding the same scalar to every class score for an example changes no score
  differences and therefore changes no loss.
- Increasing only the correct-class score cannot increase loss.
- If all $C$ scores are equal and $\Delta=1$, loss is exactly $C-1$.
- Reordering examples does not change their mean loss.

Near-zero random weights produce nearly equal scores, so a ten-class classifier
should initially have data loss near $10-1=9$. Our CIFAR-10 experiment produced
approximately this value.

## Data loss and regularization

Mean data loss measures margin violations on labeled examples:

$$
L_{data}=\frac{1}{N}\sum_{i=1}^{N}L_i.
$$

L2 regularization penalizes weight magnitude:

$$
L_{reg}=\lambda\sum_{c,d}W_{c,d}^2.
$$

The total objective is

$$
L=L_{data}+L_{reg}.
$$

This project does not include a factor of $1/2$ in the regularization term, so
its later gradient is $2\lambda W$. Biases are left unregularized because they
shift scores but do not control sensitivity to input features.

Zero data loss does not imply zero total loss, unique weights, or good
generalization. Regularization expresses a preference among parameter matrices
that explain the training data similarly.

## Related project material

- `notebooks/05_multiclass_svm_loss.ipynb`

## Review questions

1. How is satisfying the margin stronger than correct classification?
2. Why must the correct-class margin be removed?
3. Why does adding a row-wise constant leave the loss unchanged?
4. Why is loss summed over incorrect classes but averaged over examples?
5. Why should near-zero scores give loss near $C-1$?
6. What different roles do data loss and regularization loss play?

## Source

- Stanford CS231n Spring 2025, Lecture 2: Image Classification with Linear
  Classifiers, available from the
  [course schedule](https://cs231n.stanford.edu/2025/schedule.html).
