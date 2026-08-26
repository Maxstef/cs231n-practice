# Hyperparameter tuning and learning rates

## Parameters versus hyperparameters

**Parameters** are learned from training data, such as the weights and biases
of a classifier. **Hyperparameters** are choices that control the model or its
training, for example:

- learning rate and learning-rate schedule;
- regularization strength;
- batch size and number of epochs;
- optimizer and momentum coefficients;
- model size or architecture.

We train parameters on the training set, choose hyperparameters using validation
performance, and use the test set only for the final unbiased evaluation.

## A practical tuning process

1. Make a small configuration overfit a tiny dataset. This checks that the
   model, loss, gradients, and update code can learn at all.
2. Search broadly with short runs to find plausible ranges.
3. Narrow the ranges and run the promising configurations for longer.
4. Compare validation metrics, not only training loss.
5. Retrain or evaluate the selected configuration, then inspect the test set
   once at the end.

Positive scale parameters such as learning rate and regularization strength
should usually be searched on a logarithmic scale. Testing $10^{-4}$,
$10^{-3}$, and $10^{-2}$ is more informative than testing $0.001$, $0.002$,
and $0.003$ when the correct order of magnitude is unknown. Random search is
often useful when several hyperparameters are being varied because it explores
more distinct values of each one than a small grid.

### Random search versus grid search

A grid evaluates every combination of a fixed list of values. This is easy to
reproduce and visualize, but it spends the same number of trials on every
dimension. If only one of two hyperparameters strongly affects performance, a
$k \times k$ grid tests only $k$ distinct values of the important parameter,
despite costing $k^2$ runs.

Random search samples combinations independently. With the same $k^2$-run
budget, it can test up to $k^2$ distinct values along the important dimension.
It is therefore a strong default when the important hyperparameters are not
known in advance.

Sample positive scale parameters log-uniformly. For example, sample
$u \sim \mathrm{Uniform}(-5,-2)$ and set

$$
\alpha = 10^u
$$

to explore learning rates between $10^{-5}$ and $10^{-2}$ without concentrating
most samples near the upper end. After a broad search, narrow the promising
range and search again.

Grid search can still be useful for a very small number of discrete choices or
for a local, interpretable sweep. Neither method compensates for an unfair
validation protocol or too little training per trial.

For a fair comparison, keep the data split and training budget fixed. Record
the random seed and consider repeated runs when noise could change the ranking.
Cross-validation provides a more stable estimate on small datasets, at greater
computational cost.

## The learning rate

Every discussed optimizer has a base learning rate $\alpha$. It is often the
first hyperparameter to tune because it directly scales parameter updates.

- **Too small:** loss falls smoothly but progress is very slow.
- **Too large:** loss oscillates, plateaus at a poor value, or diverges.
- **Reasonable:** loss decreases quickly without unstable growth.

A fast initial loss decrease does not by itself identify the best learning
rate. The relevant comparison is performance after a fair training budget,
especially on validation data.

## Why decay the learning rate?

Larger steps are useful early when the parameters are far from a good region.
Smaller steps later allow finer adjustments and reduce oscillation around a
minimum. A schedule makes the learning rate a function of step or epoch $t$.

### Step decay

Reduce the rate by a fixed factor at chosen milestones, for example

$$
\alpha_t \leftarrow 0.1\alpha_t
$$

after specified epochs. The milestones and factor are themselves
hyperparameters.

### Cosine decay

Over a planned run of $T$ epochs, decay smoothly from $\alpha_0$ toward zero:

$$
\alpha_t = \frac{1}{2}\alpha_0
\left(1+\cos\left(\frac{\pi t}{T}\right)\right).
$$

### Linear decay

Decrease at a constant rate over the run:

$$
\alpha_t = \alpha_0\left(1-\frac{t}{T}\right).
$$

### Inverse-square-root decay

Decrease rapidly at first and more slowly later:

$$
\alpha_t = \frac{\alpha_0}{\sqrt{t}}.
$$

In practice the formula is defined carefully near $t=0$, for example by
starting at $t=1$ or adding an offset.

## Tuning the rate and schedule together

- First find an initial learning rate that makes stable progress.
- Add decay when a fixed rate stops improving or continues to oscillate.
- Compare schedules over the same number of epochs; cosine and linear schedules
  depend explicitly on the planned duration $T$.
- Retune the base rate when changing optimizer or batch size.
- Treat adaptive optimizers as having per-parameter scaling, not as eliminating
  the need for a base learning rate or schedule.

The model-selection workflow is practiced in
[`11_linear_classifier_model_selection.ipynb`](../../notebooks/11_linear_classifier_model_selection.ipynb).

## Further reading

- [Random Search for Hyper-Parameter Optimization](https://www.jmlr.org/papers/v13/bergstra12a.html)
