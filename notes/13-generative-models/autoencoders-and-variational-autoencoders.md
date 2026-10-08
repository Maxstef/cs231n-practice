# Autoencoders and variational autoencoders

An ordinary autoencoder compresses an input and reconstructs it:

```text
x -> encoder -> latent code z -> decoder -> reconstruction x_hat
```

With squared reconstruction error, it minimizes

$$
\mathcal{L}_{reconstruction}=\lVert x-\hat{x}\rVert_2^2.
$$

The latent bottleneck encourages the encoder to retain useful information, and
the encoder can later provide features for a downstream task.

## Why an ordinary autoencoder is not automatically generative

Reconstructing training inputs does not tell us how latent codes are
distributed. The encoder might place examples in an irregular collection of
isolated regions. Sampling an arbitrary new $z$ could then land somewhere the
decoder never saw during training and produce meaningless output.

```text
reconstruction: x -> known encoder output z -> decoder       easy to define
generation:     sample a new z from where? -> decoder        not defined
```

A small bottleneck alone does not solve this. We need a known distribution
from which new latent values can be sampled.

## Probabilistic latent-variable model

A variational autoencoder assumes a prior such as

$$
p(z)=\mathcal{N}(0,I)
$$

and a decoder likelihood

$$
p_\theta(x\mid z).
$$

The generative story is:

```text
sample z ~ p(z) -> sample or predict x ~ p_theta(x | z)
```

The data likelihood marginalizes the unobserved latent code:

$$
p_\theta(x)=\int p_\theta(x\mid z)p(z)\,dz.
$$

That integral is generally intractable. The true posterior
$p_\theta(z\mid x)$ is also difficult to compute, so an encoder defines an
approximate posterior:

$$
q_\phi(z\mid x)=\mathcal{N}
\left(\mu_\phi(x),\mathrm{diag}(\sigma_\phi^2(x))\right).
$$

Unlike a deterministic autoencoder, the VAE encoder outputs the parameters of
a distribution rather than one fixed code.

## Encoder and decoder meanings

| Component | Input | Output | Interpretation |
| --- | --- | --- | --- |
| Encoder `q(z given x)` | Data `x` | Mean and standard deviation | Approximate distribution over likely latent causes |
| Decoder `p(x given z)` | Latent sample `z` | Data-distribution parameters | Likelihood of observations given a latent cause |

For a fixed-variance Gaussian decoder, maximizing
$\log p_\theta(x\mid z)$ is equivalent, up to constants and scale, to
minimizing squared reconstruction error. Other data types may require other
likelihoods, such as Bernoulli or categorical distributions.

## Sampling after training

To reconstruct an existing example, sample from $q_\phi(z\mid x)$ and decode.
To generate a new example, the encoder is not required:

```text
z ~ N(0, I) -> decoder -> new sample parameters
```

The decoder is useful for generation because training encourages encoded
examples to occupy a latent space compatible with the known prior.

## Latent-space caution

A factorized Gaussian prior encourages a simple and smooth latent space, but it
does not guarantee that individual coordinates correspond cleanly to human
concepts such as pose, color, or identity. Disentanglement depends on the data,
objective, architecture, and additional assumptions.

## Sources

- Stanford CS231n Spring 2025, Lecture 13: Generative Models 1, available from
  the [course schedule](https://cs231n.stanford.edu/2025/schedule.html).
- Kingma and Welling, [Auto-Encoding Variational
  Bayes](https://arxiv.org/abs/1312.6114).
