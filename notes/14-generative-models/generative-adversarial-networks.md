# Generative adversarial networks

A generative adversarial network (GAN) learns to turn simple random noise into
data-like samples. It uses two neural networks with opposing jobs:

- the **generator** $G$ maps a random latent vector $z$ to a synthetic sample;
- the **discriminator** $D$ estimates whether a sample looks real rather than
  generated.

```text
z from a simple distribution -> generator G -> generated image
                                             \
real image ----------------------------------> discriminator D -> real or fake
```

The generator does not receive a target image that its output must reproduce.
Instead, the discriminator supplies a learned training signal: it identifies
properties that still make generated images distinguishable from real ones.

## What is an implicit generative model?

A GAN defines a procedure for sampling:

$$
z\sim p(z),
\qquad
\hat{x}=G(z).
$$

This induces a generator distribution $p_G(x)$. We can draw samples from it,
but a standard GAN does not provide a convenient normalized formula for
$p_G(x)$. This is why it is called an **implicit-density** model.

The goal is not to reproduce individual training images. It is to make the
overall generated distribution resemble the data distribution:

$$
p_G(x)\approx p_{data}(x).
$$

## The adversarial game

The original objective is

$$
\min_G\max_D
\left(
\mathbb{E}_{x\sim p_{data}}[\log D(x)]+\mathbb{E}_{z\sim p(z)}[\log(1-D(G(z)))]
\right).
$$

Read it in two parts:

- with $G$ fixed, $D$ tries to output values near one for real data and near
  zero for generated data;
- with $D$ fixed, $G$ changes its samples so that $D(G(z))$ becomes large.

Training therefore alternates between discriminator and generator updates.
This is a game between two changing networks, not ordinary minimization of one
fixed loss. Consequently, a lower loss value alone is not a reliable measure
of sample quality.

## Why the generator loss is often changed

Early in training, generated images are easy to reject, so $D(G(z))$ may be
close to zero. The original minimax generator objective can then give a weak
gradient. In practice, the generator is commonly trained with the
**non-saturating** loss

$$
\mathcal{L}_G=-\mathbb{E}_{z\sim p(z)}[\log D(G(z))].
$$

It has the same intuitive goal—make generated samples look real—but usually
provides a stronger learning signal early in training.

## Architectures and latent space

- **DCGAN** uses convolutional networks to generate and judge images.
- **StyleGAN** transforms the latent code into style controls that influence
  multiple generator layers, giving more control over coarse and fine visual
  properties.
- Interpolating between latent vectors can produce smooth transitions between
  generated images. This suggests that the generator has learned a structured
  latent space, although individual dimensions need not have simple meanings.

## Strengths and difficulties

GANs can produce sharp, high-quality images and sample in one generator pass.
However, training can be unstable because neither player sees a stationary
objective. A generator may also suffer **mode collapse**, producing only a
limited variety of samples that successfully fool the discriminator.

The main idea to remember is:

> A GAN learns a generator through feedback from a learned real-versus-fake
> critic, rather than through an explicit likelihood or reconstruction target.

## Sources

- Stanford CS231n Spring 2025, Lecture 14: Generative Models 2, available from
  the [course schedule](https://cs231n.stanford.edu/2025/schedule.html).
- Goodfellow et al., [Generative Adversarial Nets](https://arxiv.org/abs/1406.2661).
- Radford, Metz, and Chintala, [Unsupervised Representation Learning with Deep
  Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434).
- Karras, Laine, and Aila, [A Style-Based Generator Architecture for Generative
  Adversarial Networks](https://arxiv.org/abs/1812.04948).
