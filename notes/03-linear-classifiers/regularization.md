# Regularization

## Why regularize a model?

Training minimizes loss on observed examples, but the real goal is to perform
well on unseen data. A sufficiently flexible model may fit peculiarities or
noise in the training set instead of learning a pattern that generalizes.

Regularization adds a preference for some parameter values over others:

$$
L_{total}=L_{data}+\lambda R(W),
$$

where

- $L_{data}$ measures fit to the training examples;
- $R(W)$ measures an undesirable property of the parameters;
- $\lambda\geq0$ controls the strength of that preference.

A larger $\lambda$ accepts more training error in exchange for a stronger
preference for simpler parameters. It is a hyperparameter and should be chosen
using validation data, not test data.

Regularization does not guarantee good generalization. It changes the
objective so that, among models with similar data loss, training prefers ones
that satisfy the chosen prior preference.

## L2 regularization

L2 regularization penalizes squared weights:

$$
R(W)=\sum_{c,d}W_{c,d}^2.
$$

Under this project's convention,

$$
L_{reg}=\lambda\sum W^2,
\qquad
\frac{\partial L_{reg}}{\partial W}=2\lambda W.
$$

Some sources instead write $\frac{1}{2}\lambda\sum W^2$, whose gradient is
$\lambda W$. These conventions are equivalent after rescaling $\lambda$, but
the implemented loss and gradient must use matching factors.

L2 continuously pushes every weight toward zero. Large weights receive a
larger penalty and gradient, so the model is encouraged to distribute its
influence rather than rely heavily on a few features. L2 usually produces many
small, nonzero weights rather than exact zeros.

When used with gradient descent, the L2 part of an update is

$$
W \leftarrow W-\eta(2\lambda W)
  =(1-2\eta\lambda)W,
$$

which explains the related term *weight decay*. The equivalence between L2
regularization and decoupled weight decay depends on the optimizer; they should
not be treated as universally identical.

## L1 regularization

L1 regularization penalizes absolute weights:

$$
R(W)=\sum_{c,d}|W_{c,d}|.
$$

Away from zero, its derivative is the sign of each weight:

$$
\frac{\partial |w|}{\partial w}=
\begin{cases}
-1 & w<0,\\
+1 & w>0.
\end{cases}
$$

At zero, $|w|$ has a corner and uses a subgradient rather than one unique
derivative. L1 applies a constant-magnitude pull toward zero and can produce
exactly zero weights. It is therefore associated with sparse models and a form
of feature selection.

## L1 versus L2

| Property | L1 | L2 |
|---|---|---|
| Penalty | $\lambda\sum|W|$ | $\lambda\sum W^2$ |
| Effect | Can set weights exactly to zero | Usually shrinks weights without making them exactly zero |
| Typical structure | Sparse, relies on fewer features | Distributed, uses many small weights |
| At zero | Not differentiable; use a subgradient | Smooth and differentiable |
| Sensitivity to large weights | Linear penalty | Quadratically increasing penalty |

L1 and L2 can also be combined, often called **elastic-net regularization**.
Neither is automatically superior: the useful choice depends on the model,
data, optimizer, and desired parameter structure.

## What is usually regularized?

For a linear classifier, weights $W$ are normally regularized because they
control sensitivity to input features. Biases $b$ are often left unregularized:
they shift class scores but do not multiply input features. This is a
convention rather than a mathematical requirement.

Regularization is broader than explicit L1 and L2 penalties. Data augmentation,
early stopping, dropout, and even properties of an optimizer can also have
regularizing effects. They work through different mechanisms and should not be
collapsed into the same formula.

## Keep these ideas separate

- **Data loss:** how poorly the model fits labeled training examples.
- **Regularization loss:** the explicit parameter preference.
- **Total loss:** the quantity optimized during training.
- **Validation loss:** an estimate used to choose hyperparameters such as
  $\lambda$.
- **Test loss:** a final estimate; it must not be used to tune $\lambda$.

## Related project material

- `notebooks/05_multiclass_svm_loss.ipynb`
- `notebooks/06_svm_gradient_checking.ipynb`

## Review questions

1. Why can minimizing training data loss alone lead to poor test performance?
2. What role does $\lambda$ play?
3. Why does L1 tend to produce more exact zeros than L2?
4. Under this project's convention, what are the L2 loss and gradient?
5. Why are biases often excluded from regularization?
6. Why should regularization strength be selected with validation data?

## Source

- Stanford CS231n Spring 2025, Lecture 3: Regularization and Optimization,
  available from the
  [course schedule](https://cs231n.stanford.edu/2025/schedule.html).
