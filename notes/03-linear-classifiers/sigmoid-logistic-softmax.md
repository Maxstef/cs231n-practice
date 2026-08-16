# Sigmoid, Logistic, and Softmax

## Sigmoid and logistic

In machine learning, **the sigmoid function** usually means the **logistic
function**:

$$
\sigma(z)=\frac{1}{1+e^{-z}}.
$$

So *sigmoid function* and *logistic function* normally refer to the same
formula. Strictly speaking, *sigmoid* can describe any S-shaped function, while
*logistic* names this particular one.

The logistic function maps one real-valued logit to a number between zero and
one. In binary classification, that number is commonly interpreted as

$$
p=P(y=1\mid x),
\qquad
1-p=P(y=0\mid x).
$$

**Logistic regression** is a model, not merely another name for the function.
It computes a linear logit $z=w^Tx+b$, applies the logistic sigmoid, and is
normally trained with binary cross-entropy, also called binary log loss.

## Softmax

Softmax takes a vector of $C$ logits and produces $C$ probabilities:

$$
p_j=\frac{e^{s_j}}{\sum_{k=1}^{C}e^{s_k}}.
$$

The probabilities sum to one and compete with one another. Softmax is therefore
suited to **single-label multiclass classification**, where exactly one class
is the target—for example, choosing one CIFAR-10 label for an image.

A linear model followed by softmax is often called **multinomial logistic
regression** or **softmax regression**. Despite the word *regression*, it is a
classification model; the historical name refers to modeling log-odds.

## Binary, multiclass, and multilabel

| Task | Output transformation | Usual loss | Interpretation |
|---|---|---|---|
| Binary classification | One sigmoid | Binary cross-entropy | Probability of the positive class |
| Single-label multiclass | One softmax over $C$ logits | Categorical cross-entropy | Competing probabilities sum to one |
| Multilabel classification | $C$ independent sigmoids | Sum/mean of binary cross-entropies | Each label can independently be present |

The distinction between multiclass and multilabel is essential. An image
classified as exactly one of `cat`, `dog`, or `car` uses softmax. An image that
may simultaneously contain `person`, `bicycle`, and `tree` uses one sigmoid per
label because several answers may be true at once.

Independent sigmoids do not force their outputs to sum to one. This is correct
for multilabel tasks, not a normalization error.

## Connection between sigmoid and two-class softmax

For two softmax logits $s_0$ and $s_1$, the probability of class 1 is

$$
p_1=\frac{e^{s_1}}{e^{s_0}+e^{s_1}}
=\frac{1}{1+e^{-(s_1-s_0)}}
=\sigma(s_1-s_0).
$$

Thus, a two-class softmax depends only on the logit difference and is equivalent
to applying a sigmoid to that difference. The parameterizations differ—two
logits versus one—but the represented probabilities can be the same.

## Keep these terms separate

- **Sigmoid/logistic function:** maps one logit to a value in `(0, 1)`.
- **Logistic regression:** a binary linear classification model using sigmoid.
- **Softmax:** maps several logits to competing probabilities summing to one.
- **Softmax regression:** a multiclass linear classification model using
  softmax.
- **Binary cross-entropy/log loss:** the usual loss paired with sigmoid.
- **Categorical cross-entropy/log loss:** the usual loss paired with softmax.

The output function and loss are separate choices, even though common APIs may
combine them into one stable operation.

## Related project material

- `notebooks/04_linear_scores_and_geometry.ipynb`
- `notebooks/07_softmax_loss.ipynb`

## Review questions

1. In typical ML terminology, how are sigmoid and logistic related?
2. Why is logistic regression still a classification method?
3. Why does softmax fit single-label multiclass problems?
4. Why do multilabel problems use independent sigmoids rather than softmax?
5. How is two-class softmax related to sigmoid?
6. Which losses are normally paired with sigmoid and softmax?

## Source

- Stanford CS231n Spring 2025, Lecture 2: Image Classification with Linear
  Classifiers, available from the
  [course schedule](https://cs231n.stanford.edu/2025/schedule.html).
