# Diffusion intuition and rectified flow

The central diffusion idea is easier to understand as a transformation:

```text
data <---------------- many small learned steps ---------------- noise
```

Noise is easy to sample. Realistic data is not. A model learns how to move a
noisy point a little closer to the data distribution, and generation repeats
that update many times.

Different papers describe this idea using noise prediction, denoising, score
matching, stochastic differential equations, or flow matching. The notation
varies, but the shared intuition is **learn a direction from noise toward
data**. This note uses rectified flow because it makes that direction concrete.

## Constructing a training example

Choose independently:

- a real example $x\sim p_{data}$;
- Gaussian noise $z\sim p_{noise}$ with the same shape as $x$;
- a time or noise level $t$ between zero and one.

Mix the real example and noise:

$$
x_t=(1-t)x+tz.
$$

The endpoints make the formula intuitive:

- $t=0$ gives $x_t=x$, a clean example;
- $t=1$ gives $x_t=z$, pure noise;
- an intermediate $t$ gives a partly corrupted example.

The straight-line velocity from the data point to the noise point is

$$
v=z-x.
$$

A time-conditioned neural network learns to predict that velocity:

$$
\mathcal{L}=\left\lVert f_\theta(x_t,t)-(z-x)\right\rVert_2^2.
$$

The model needs $t$ because the same-looking input may require different
interpretations at different noise levels.

## Why prediction can be ambiguous

Many different pairs $(x,z)$ can produce a similar intermediate point $x_t$.
The model cannot know which exact pair created it. With squared-error training,
the best prediction averages the plausible target directions. Across many
examples, these local average directions form a useful vector field between
the noise and data distributions.

## Sampling reverses the direction

Training defined $v=z-x$, which points from data toward noise as $t$ increases.
Generation starts at the other end, near $t=1$, and moves backward toward
$t=0$. A simple numerical update is

$$
x\leftarrow x-\Delta t\,f_\theta(x,t).
$$

Repeated small updates approximately follow the learned field from random
noise to a data-like sample. More steps usually give a more accurate path but
require more model evaluations.

## Noise-level sampling

If $t$ is sampled uniformly, all noise levels appear equally often. The middle
of the path can be especially ambiguous, so practical systems may sample some
regions more frequently using a **noise schedule**. This changes which parts
of the denoising problem receive the most training attention.

This is an optimization choice, not a change to the basic goal.

## What to remember

1. Create noisy points between real data and simple noise.
2. Train one time-conditioned model to predict how those points move.
3. Start from fresh noise and repeatedly follow the learned field backward.

General diffusion formalisms change the mixing path and prediction target.
Understanding those variants is useful later, but it is not necessary for
grasping the core generation process.

## Sources

- Stanford CS231n Spring 2025, Lecture 14: Generative Models 2, available from
  the [course schedule](https://cs231n.stanford.edu/2025/schedule.html).
- Liu, Gong, and Liu, [Flow Straight and Fast: Learning to Generate and
  Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003).
- Lipman et al., [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747).
- Dieleman, [Perspectives on diffusion](https://sander.ai/2023/07/20/perspectives.html).
