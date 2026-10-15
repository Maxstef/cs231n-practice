# Conditional generation and classifier-free guidance

An unconditional model learns to generate a plausible sample $x$. A
conditional model also receives information $y$ describing what should be
generated:

$$
p(x)
\quad\longrightarrow\quad
p(x\mid y).
$$

The condition can be a class label, text embedding, image, segmentation mask,
audio signal, or another modality.

## Conditional rectified flow

The unconditional velocity model receives the noisy sample and time:

$$
v=f_\theta(x_t,t).
$$

For conditional generation, it additionally receives $y$:

$$
v=f_\theta(x_t,y,t).
$$

The model then learns directions associated with the requested subset of the
data—for example, dog images rather than arbitrary images.

## Why classifier-free guidance trains with a null condition

Classifier-free guidance (CFG) trains one model to work in two modes. During
training, the real condition is randomly replaced by a special null condition
$\varnothing$:

```text
sometimes: model(noisy_sample, real_condition, time)
sometimes: model(noisy_sample, null_condition, time)
```

The same network therefore learns:

- an unconditional prediction $v_{uncond}$ using the null condition;
- a conditional prediction $v_{cond}$ using the requested condition.

It is called *classifier-free* because it does not need a separate classifier
to steer generation.

## Guidance during sampling

The difference

$$
v_{cond}-v_{uncond}
$$

represents the change caused by the condition. Sampling can amplify it:

$$
v_{guided}=v_{uncond}+s\left(v_{cond}-v_{uncond}\right).
$$

Here $s$ is the **extra guidance strength** used by the lecture. This is
equivalent to
$v_{guided}=(1+s)v_{cond}-s v_{uncond}$ under the lecture's convention.

- $s=0$ uses the ordinary conditional prediction;
- larger $s$ emphasizes the condition more strongly;
- excessively large $s$ can reduce diversity or introduce artifacts.

Thus guidance is not a new model. It combines two predictions from the same
model to trade diversity for stronger condition adherence.

## Text conditioning

In text-to-image generation, a pretrained text encoder converts a prompt into
a sequence of embeddings. The image generator can read those embeddings using
cross-attention or joint attention. The model is not converting one text
vector directly into all image pixels; it repeatedly uses the text
representation while refining noisy image or latent tokens.

## Source

- Stanford CS231n Spring 2025, Lecture 14: Generative Models 2, available from
  the [course schedule](https://cs231n.stanford.edu/2025/schedule.html).
- Ho and Salimans, [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
