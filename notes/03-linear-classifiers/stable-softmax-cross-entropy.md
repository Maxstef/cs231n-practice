# Stable Softmax Cross-Entropy from Logits

## Starting point

For one example, softmax converts class scores—or **logits**—into
probabilities:

$$
p_j=\frac{e^{s_j}}{\sum_k e^{s_k}}.
$$

If $y$ is the correct class, cross-entropy is the negative logarithm of its
probability:

$$
L=-\log p_y.
$$

Substitute the softmax expression for $p_y$:

$$
L=-\log\left(\frac{e^{s_y}}{\sum_k e^{s_k}}\right).
$$

Use the logarithm rules

$$
\log\left(\frac{a}{b}\right)=\log a-\log b
\qquad\text{and}\qquad
\log(e^x)=x.
$$

Then

$$
\begin{aligned}
L
&=-\left[\log(e^{s_y})-\log\left(\sum_k e^{s_k}\right)\right]\\
&=-s_y+\log\left(\sum_k e^{s_k}\right).
\end{aligned}
$$

This is the **log-sum-exp form** of softmax cross-entropy:

$$
\boxed{L=\log\left(\sum_k e^{s_k}\right)-s_y}.
$$

## Why shift the scores?

Directly calculating $e^{s_k}$ can overflow when a score is large. Let

$$
m=\max_k s_k
$$

and define shifted scores

$$
\tilde{s}_k=s_k-m.
$$

The largest shifted score is zero, so all exponentials satisfy

$$
0<e^{\tilde{s}_k}\leq1.
$$

This keeps intermediate values representable. The loss does not change:

$$
\begin{aligned}
L
&=\log\left(\sum_k e^{s_k}\right)-s_y\\
&=\log\left(\sum_k e^{\tilde{s}_k+m}\right)-(\tilde{s}_y+m)\\
&=\log\left(e^m\sum_k e^{\tilde{s}_k}\right)-\tilde{s}_y-m\\
&=m+\log\left(\sum_k e^{\tilde{s}_k}\right)-\tilde{s}_y-m\\
&=\log\left(\sum_k e^{\tilde{s}_k}\right)-\tilde{s}_y.
\end{aligned}
$$

The added and subtracted $m$ cancel. Therefore, the stable formula is

$$
\boxed{L=\log\left(\sum_k e^{\tilde{s}_k}\right)-\tilde{s}_y}.
$$

## NumPy implementation for one example

```python
# Make the largest shifted score zero.
shifted_scores = scores - np.max(scores)

# These exponentials are now at most one.
exp_scores = np.exp(shifted_scores)

# Compute log-sum-exp minus the shifted correct-class score.
loss = (
    np.log(np.sum(exp_scores))
    - shifted_scores[correct_label]
)
```

We compute the loss directly from logits rather than using
`-np.log(probabilities[correct_label])`. A very small probability may underflow
to exactly zero, making `log(0)` infinite even when the mathematical loss is
finite.

## Vectorized batch version

For scores with shape `(N, C)`, shift each example independently:

```python
shifted_scores = scores - np.max(scores, axis=1, keepdims=True)
exp_scores = np.exp(shifted_scores)
sum_exp_scores = np.sum(exp_scores, axis=1)
correct_scores = shifted_scores[np.arange(scores.shape[0]), labels]

example_losses = np.log(sum_exp_scores) - correct_scores
batch_loss = np.mean(example_losses)
```

The maximum and sum operate across classes (`axis=1`). The final mean operates
across examples.

## Connection to the gradient

The same expression is also convenient to differentiate:

$$
L=-s_y+\log\left(\sum_k e^{s_k}\right).
$$

Its derivative with respect to score $s_j$ is

$$
\frac{\partial L}{\partial s_j}
=p_j-\mathbb{1}[j=y].
$$

Thus, stable forward loss and the compact gradient $dS=P-Y$ come from the same
algebraic form.

## Remember

```text
softmax probability + negative logarithm
                    ↓ algebra
       log-sum-exp − correct score
                    ↓ shift by maximum
      stable log-sum-exp − shifted correct score
```

## Related project material

- `notebooks/07_softmax_loss.ipynb`
- `notebooks/08_softmax_gradient_checking.ipynb`
- `notebooks/10_svm_vs_softmax_training.ipynb`

## Review questions

1. Which two logarithm rules produce the log-sum-exp expression?
2. Why is every shifted exponential at most one?
3. Where does the maximum-score shift cancel from the loss?
4. Why can calculating probability first and logarithm second be unstable?
5. Along which axis should a batch of scores be shifted and normalized?
