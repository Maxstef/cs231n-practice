# Probability, likelihood, and tractability

Probability and likelihood can use the same mathematical expression, but they
answer different questions depending on what is treated as fixed.

Suppose a model assigns data $x$ a distribution $p_\theta(x)$, controlled by
parameters $\theta$.

## Probability

For fixed model parameters, probability asks:

> If this model is fixed, how probable are different possible observations?

Here $\theta$ is fixed and $x$ varies:

$$
p_\theta(x).
$$

For a discrete distribution, probabilities across every possible $x$ sum to
one. For continuous data, $p_\theta(x)$ is a probability density; probability
is assigned to a region by integrating that density.

## Likelihood

Once data $x$ has been observed, likelihood asks:

> Which parameter values would make this fixed observation plausible?

Now $x$ is fixed and $\theta$ varies:

$$
L(\theta;x)=p_\theta(x).
$$

The numerical expression is the same, but its interpretation has changed.
Likelihood is a function of the parameters and is not a probability
distribution over $\theta$. In particular, it does not generally sum or
integrate to one across parameter values.

### Small Bernoulli example

For one binary observation $x=1$:

$$
p_\theta(x=1)=\theta.
$$

- Probability interpretation: for a fixed $\theta=0.8$, the probability of
  observing one is $0.8$.
- Likelihood interpretation: after observing one, values of $\theta$ closer
  to one have greater likelihood.

Maximum likelihood estimation chooses parameters that maximize the likelihood
of all observed training examples. In practice, neural networks usually
minimize the equivalent negative log-likelihood.

## Use in generative models

A generative model learns a distribution over data, such as $p_\theta(x)$ or
a conditional distribution $p_\theta(x\mid c)$. During training, the observed
data is fixed and we adjust $\theta$, so we speak about maximizing
**likelihood**. After training, $\theta$ is fixed and the model assigns
**probabilities** or densities to possible observations and can often sample
from them.

```text
training:    fixed observations, change theta -> likelihood view
evaluation:  fixed theta, score observations  -> probability view
generation:  fixed theta, sample observations -> probability view
```

## What does exact likelihood mean?

An **exact tractable likelihood** means that the model's log-likelihood for an
observation can be calculated directly, apart from ordinary finite-precision
numerical error.

For an autoregressive model:

$$
\log p_\theta(x)
=\sum_{t=1}^{T}\log p_\theta(x_t\mid x_{<t}).
$$

Every conditional distribution is normalized, and the observed conditional
probabilities can be evaluated directly. No latent-variable integral or
sampling estimate is required.

“Exact” does not mean the model has learned the true real-world distribution.
It means we can exactly evaluate the probability or density defined by this
particular model.

## When exact likelihood is unavailable

There is no single category called “non-exact likelihood.” Several different
situations are possible:

| Situation | Meaning | Example |
| --- | --- | --- |
| Exact and tractable | Model likelihood can be evaluated directly | Autoregressive models, normalizing flows |
| Defined but intractable | Model defines a likelihood, but exact evaluation requires an impractical sum or integral | Most latent-variable models |
| Optimized through a bound | A tractable lower bound substitutes for the intractable log-likelihood | Variational autoencoders using the ELBO |
| Estimated | Sampling or another numerical procedure approximates the likelihood | Importance-sampling estimates for latent-variable models |
| Implicit | The model can generate samples but does not provide a normalized evaluable density | Standard GANs |

For a VAE, the marginal likelihood is

$$
p_\theta(x)=\int p_\theta(x\mid z)p(z)\,dz.
$$

The model defines this likelihood, but the integral is usually intractable.
Training therefore maximizes the evidence lower bound (ELBO), not the exact
log-likelihood itself. This is different from saying that the VAE has no
likelihood.

## Comparisons require care

Likelihood values are comparable only when the data representation and
measurement conventions match. For image models, preprocessing, quantization,
continuous versus discrete modeling, and whether results are reported per
pixel or per channel can all change the numerical value.

High likelihood also does not automatically imply the most realistic-looking
samples. Likelihood measures probability assignment under the model, whereas
human perceptual quality is a different criterion.

## Related notes and practice

- [Generative models and likelihood](generative-models-and-likelihood.md)
- [Maximum likelihood estimation](maximum-likelihood-estimation.md)
- [Autoregressive generative models](autoregressive-generative-models.md)
- [VAE ELBO and reparameterization](vae-elbo-and-reparameterization.md)
- [Nats and bits per pixel](nats-and-bits-per-pixel.md)
- [Notebook 54 — Autoregressive models and exact likelihood](../../notebooks/54_autoregressive_models_and_exact_likelihood.ipynb)
