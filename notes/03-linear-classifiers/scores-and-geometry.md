# Linear Classifiers: Scores and Geometry

This note summarizes the score-function and geometric viewpoints practiced in
`notebooks/04_linear_scores_and_geometry.ipynb`.

## Score function

A linear classifier maps an input feature vector $x\in\mathbb{R}^D$ to one
score for each of $C$ classes:

$$
s = Wx+b,
$$

where

- $W\in\mathbb{R}^{C\times D}$ contains one weight vector per class;
- $x\in\mathbb{R}^{D}$ is one flattened image;
- $b\in\mathbb{R}^{C}$ contains one bias per class;
- $s\in\mathbb{R}^{C}$ contains the class scores.

The predicted class is the index of the largest score. Scores are relative
compatibility values, not probabilities.

## Algebraic viewpoint

For class $c$,

$$
s_c=w_c^Tx+b_c.
$$

Each class score is a weighted sum of input features plus a bias. With a batch
$X\in\mathbb{R}^{N\times D}$, it is often convenient to store examples as
rows and compute

$$
S=XW^T+b,
$$

where $S\in\mathbb{R}^{N\times C}$. NumPy broadcasts $b$ across the batch.

### Bias trick

The bias can be absorbed into the weight matrix by appending a constant feature:

$$
\tilde{x}=\begin{bmatrix}x\\1\end{bmatrix},
\qquad
\tilde{W}=\begin{bmatrix}W&b\end{bmatrix}.
$$

Then $\tilde{W}\tilde{x}=Wx+b$. This is algebraically convenient, although
keeping weights and biases separate is often clearer in an implementation.

## Geometric viewpoint

For two classes $a$ and $b$, the decision boundary occurs when their scores
are equal:

$$
w_a^Tx+b_a=w_b^Tx+b_b.
$$

Rearranging gives

$$
(w_a-w_b)^Tx+(b_a-b_b)=0,
$$

which is a hyperplane in feature space. A linear classifier divides the feature
space using linear boundaries. It cannot directly represent arbitrary curved
or disconnected class regions in the original feature space.

In two dimensions, if $w_a-w_b=[u,v]$ and $b_a-b_b=d$, the equality is

$$
ux_1+vx_2+d=0.
$$

When $v\ne0$, it can be drawn by solving

$$
x_2=-\frac{ux_1+d}{v}.
$$

This equality line is only a *candidate* pairwise boundary. A segment is visible
in the final multiclass decision map only where no third class has a larger
score.

## Visual-template viewpoint

If input features are raw pixels, one row of $W$ can be reshaped back into an
image. It acts like a learned class template: pixels aligned with the template
increase the class score, while pixels with opposing weights decrease it.

A single template per class tends to average over multiple appearances. This
helps explain why linear classifiers may learn blurry combinations of colors,
poses, and backgrounds rather than clean object prototypes.

## From scores to probabilities

The softmax function will later transform scores (also called logits) into
nonnegative values that sum to one:

$$
p_c=\frac{e^{s_c}}{\sum_j e^{s_j}}.
$$

The loss for a correct class $y$ is the negative log-probability

$$
L=-\log p_y.
$$

The project studies both multiclass SVM loss and softmax cross-entropy as ways
to turn scores into an optimization objective. See the related terminology
notes for the distinction between the softmax function and cross-entropy loss.

## Related project material

- `notebooks/04_linear_scores_and_geometry.ipynb`
- `notes/03-linear-classifiers/softmax-cross-entropy-log-loss.md`
- `notes/03-linear-classifiers/sigmoid-logistic-softmax.md`

## Review questions

1. What does each row of $W$ represent algebraically and visually?
2. What role does the bias play?
3. Why is a raw score not automatically a probability?
4. What shapes result from scoring one example and a batch?
5. What kinds of decision boundaries can a linear classifier represent?

## Source

- Stanford CS231n Spring 2025, Lecture 2: Image Classification with Linear
  Classifiers, available from the
  [course schedule](https://cs231n.stanford.edu/2025/schedule.html).
