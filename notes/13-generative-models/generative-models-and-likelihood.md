# Generative models and likelihood

A discriminative classifier models labels given data:

$$
p(y\mid x).
$$

An unconditional generative model instead models the data distribution:

$$
p(x),
$$

while a conditional generative model learns possible outputs given some
condition:

$$
p(x\mid y).
$$

Here $y$ need not be a class label. It could be text, an image, the first video
frame, or any other conditioning information.

## Why model a distribution?

Many prediction problems have more than one valid answer. A text prompt can
describe many plausible images, and one video frame can have several plausible
futures. Predicting only one average answer can hide this ambiguity. A
conditional distribution represents multiple possibilities and lets us sample
different outputs.

Generative models can be used to:

- generate new samples;
- represent ambiguity and uncertainty;
- detect unusual examples using density or a related score;
- learn features without class labels;
- fill in, transform, or conditionally produce data.

A model can be generative even when generation is not its main practical use.
The defining idea is modeling the data distribution or a conditional data
distribution.

## Probability density

For continuous data, a density $p(x)$ assigns nonnegative relative likelihood
and satisfies

$$
\int p(x)\,dx=1.
$$

A density value is not itself the probability of one exact continuous point.
Probabilities come from integrating density over regions. Different possible
values compete for a total probability mass of one, which makes high-dimensional
image density modeling difficult.

## Maximum likelihood estimation

Maximum likelihood chooses parameters that make the observed training data
probable. In practice, this is usually implemented by minimizing average
negative log-likelihood:

$$
\mathcal{L}_{NLL}=-\frac{1}{N}\sum_{i=1}^{N}\log p_\theta(x^{(i)}).
$$

See [Maximum likelihood estimation](maximum-likelihood-estimation.md) for a
symbol-by-symbol derivation, the log transformation, and a small Bernoulli
example.

## A useful taxonomy

| Family | Can evaluate density? | Can sample? | Main idea |
| --- | --- | --- | --- |
| Autoregressive | Exactly, through conditional factors | Yes, sequentially | Predict the next value from previous values |
| Variational autoencoder | Uses a tractable likelihood lower bound | Yes, from a latent prior | Decode samples from a structured latent space |
| GAN | No explicit normalized density | Yes, directly | Train a generator against a discriminator |
| Diffusion | Likelihood treatment varies; generation is iterative | Yes, iteratively | Reverse a gradual noising process |

“Explicit” does not always mean that exact likelihood computation is easy. A
VAE defines a probabilistic density, but its latent-variable marginal is
generally intractable, so it optimizes a lower bound.

## Related conditional examples

- An image-captioning model learns a distribution over captions conditioned on
  an image.
- A language model learns the next token conditioned on earlier tokens.
- A text-to-image model learns images conditioned on text.
- A video-prediction model learns future frames conditioned on observed frames.

Thus the captioning models from
[Notebook 29 — Image captioning with RNNs](../../notebooks/29_image_captioning_with_rnn.ipynb)
are conditional generative models even though their outputs are discrete text
sequences.

## Source

- Stanford CS231n Spring 2025, Lecture 13: Generative Models 1, available from
  the [course schedule](https://cs231n.stanford.edu/2025/schedule.html).

