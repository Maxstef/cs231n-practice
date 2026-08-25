# Normalization layers

Normalization transforms activations to have a controlled center and scale. A
generic normalization step is

$$
\hat{x}=\frac{x-\mu}{\sqrt{\sigma^2+\varepsilon}},
\qquad
y=\gamma\hat{x}+\beta.
$$

The small constant $\varepsilon$ prevents division by zero. The learned
parameters $\gamma$ and $\beta$ then scale and shift the normalized values. This
means normalization does not force every layer to keep zero mean and unit
variance: the model can learn a more useful scale and offset.

## The central question: which axes provide the statistics?

For image activations with shape $(N,C,H,W)$:

- **Batch normalization** computes statistics for each channel using the batch
  and spatial axes $(N,H,W)$. Different examples therefore influence one
  another during training.
- **Layer normalization** computes statistics independently for each example,
  across all of that example's feature axes $(C,H,W)$.
- **Instance normalization** computes statistics independently for every
  example and channel, using only $(H,W)$.
- **Group normalization** splits the channels into groups and, for each example,
  computes statistics across the channels in one group and their spatial axes.

One way to remember the distinction is to ask which values are put into the
same mean and variance calculation. The normalization equation is largely the
same; the reduction axes change.

## Batch normalization in training

For a simple feature matrix $X$ with shape $(N,D)$, batch normalization computes
one mean and variance per feature:

$$
\mu_d=\frac{1}{N}\sum_{n=1}^{N}X_{n,d},
$$

$$
\sigma_d^2=\frac{1}{N}\sum_{n=1}^{N}(X_{n,d}-\mu_d)^2.
$$

It also maintains exponential moving averages for later inference. With
momentum $m$:

$$
\mu_{run}\leftarrow m\mu_{run}+(1-m)\mu_{batch},
$$

and the running variance is updated analogously. These running values are state,
not parameters learned by gradient descent.

## Training and inference differ

During training, batch normalization uses the current minibatch statistics and
updates its running statistics. During inference, it uses the stored running
statistics. A prediction is therefore stable and does not depend on which other
examples happen to share its inference batch.

Layer, instance, and group normalization do not aggregate statistics across
different examples. They can use the current example's statistics in both
training and inference and do not require batch-level running averages.

## Choosing a normalization method

Batch normalization is a strong option when batches are sufficiently large and
representative. With very small or variable batches, its statistics may be noisy.
Layer and group normalization are independent of batch size; group normalization
is especially natural for convolutional feature maps. Instance normalization
removes per-instance channel statistics more aggressively and is common in tasks
where image appearance or style should be normalized.

Normalization may make optimization easier, but it does not replace nonlinear
activations or regularization. It also introduces implementation details such as
the reduction axes, numerical stability, learned scale and shift, and—in batch
normalization—training versus inference state.

## Related project material

- `notebooks/21_normalization_and_dropout.ipynb`

## Review questions

1. What changes between batch, layer, instance, and group normalization?
2. Why are $\gamma$ and $\beta$ useful after normalization?
3. Why does batch normalization need separate training and inference behavior?
4. Why can batch normalization be inconvenient with very small batches?

## Source

- Stanford CS231n Spring 2025 lecture material on training neural networks,
  available from the [course schedule](https://cs231n.stanford.edu/2025/schedule.html).

