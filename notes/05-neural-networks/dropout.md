# Dropout

Dropout is a stochastic regularization method. During training, it independently
removes activations with some probability. The model cannot assume that any one
hidden feature will always be present, which discourages fragile co-adaptation
and encourages information to be represented through multiple paths.

Let $p$ be the probability of **keeping** an activation, and sample

$$
M_i\sim\operatorname{Bernoulli}(p).
$$

## Inverted dropout

The convention used in this project scales retained activations during training:

$$
Y_i=X_i\frac{M_i}{p}.
$$

Thus a mask value is either $0$ or $1/p$. The tensor shape does not change;
dropped positions become zero. Since $\mathbb{E}[M_i]=p$,

$$
\mathbb{E}[Y_i]
=X_i\frac{\mathbb{E}[M_i]}{p}
=X_i.
$$

Inference therefore becomes the identity operation:

$$
Y_i=X_i.
$$

All units are active, no random mask is sampled, and no additional scaling is
needed.

## An older equivalent convention

Some lecture material shows unscaled activations during training and multiplies
them by $p$ during inference. That convention and inverted dropout preserve the
same expected scale but place the correction at different times:

- standard form: mask during training, multiply by $p$ during inference;
- inverted form: mask and divide by $p$ during training, do nothing during
  inference.

The two conventions must not be mixed. Dividing by $p$ during training and then
multiplying by $p$ at inference would scale twice.

## Backward pass

The backward pass must reuse the exact scaled mask sampled in the corresponding
forward pass:

$$
dX=dY\frac{M}{p}.
$$

A dropped activation had no influence on the forward output, so its gradient is
zero. Sampling a new backward mask would describe a different computational
graph and produce an incorrect gradient.

## Practical interpretation

Each training pass evaluates a randomly thinned subnetwork. This is sometimes
viewed as training and approximately combining many related subnetworks that
share parameters. The useful concrete intuition is simpler: features must remain
useful even when some collaborating features are temporarily unavailable.

Dropout adds noise, so training loss can become less smooth and optimization may
take longer. The keep probability is a hyperparameter: a smaller $p$ means
stronger regularization. Dropout should be enabled only in training mode.

Dropout and normalization are not interchangeable. Normalization controls
activation statistics; dropout explicitly removes activations. Whether both are
useful depends on the architecture and experiment.

## Related project material

- `notebooks/21_normalization_and_dropout.ipynb`
- `notes/03-linear-classifiers/regularization.md`

## Review questions

1. Why does inverted dropout divide retained activations by $p$?
2. Why is dropout disabled during inference?
3. Why must backward reuse the forward mask?
4. What goes wrong if the standard and inverted conventions are mixed?

## Source

- Stanford CS231n Spring 2025 lecture material on regularization, available from
  the [course schedule](https://cs231n.stanford.edu/2025/schedule.html).

