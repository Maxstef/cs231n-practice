# VAE ELBO and reparameterization

The VAE would ideally maximize

$$
\log p_\theta(x)=\log\int p_\theta(x,z)\,dz,
$$

but directly integrating over every possible latent $z$ is usually
intractable. The encoder distribution $q_\phi(z\mid x)$ lets us derive a
tractable lower bound.

## Deriving the ELBO

Multiply and divide inside the integral by $q_\phi(z\mid x)$:

$$
\log p_\theta(x)
=\log\int q_\phi(z\mid x)
\frac{p_\theta(x,z)}{q_\phi(z\mid x)}\,dz.
$$

This is the logarithm of an expectation. Because logarithm is concave,
Jensen's inequality gives

$$
\log p_\theta(x)
\ge
\mathbb{E}_{z\sim q_\phi(z\mid x)}
\left[
\log\frac{p_\theta(x,z)}{q_\phi(z\mid x)}
\right].
$$

Using $p_\theta(x,z)=p_\theta(x\mid z)p(z)$ gives the **evidence lower bound**:

$$
\mathrm{ELBO}(x)=
\mathbb{E}_{z\sim q_\phi(z\mid x)}[\log p_\theta(x\mid z)]
-D_{KL}\left(q_\phi(z\mid x)\,\|\,p(z)\right).
$$

“Evidence” means the observed data likelihood $p_\theta(x)$. It is a lower
bound because

$$
\log p_\theta(x)
=\mathrm{ELBO}(x)
+D_{KL}\left(q_\phi(z\mid x)\,\|\,p_\theta(z\mid x)\right),
$$

and KL divergence is nonnegative.

## Meaning of the two terms

The reconstruction term

$$
\mathbb{E}_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)]
$$

rewards latent samples that let the decoder explain the input. The prior term

$$
D_{KL}\left(q_\phi(z\mid x)\,\|\,p(z)\right)
$$

penalizes an encoded distribution that moves too far from the known prior.

In minimization form, the usual VAE loss is

$$
\mathcal{L}_{VAE}
=\mathcal{L}_{reconstruction}+\mathcal{L}_{KL}
=-\mathrm{ELBO}.
$$

The terms compete. Perfectly separate deterministic codes help reconstruction
but do not resemble the shared prior. Matching the prior too strongly can make
$z$ uninformative and cause the decoder to ignore it, a failure called
**posterior collapse**.

## Gaussian KL term

For

$$
q_\phi(z\mid x)=\mathcal{N}(\mu,\mathrm{diag}(\sigma^2)),
\qquad p(z)=\mathcal{N}(0,I),
$$

the KL term has a closed form:

$$
D_{KL}(q\|p)
=\frac{1}{2}\sum_j
\left(\mu_j^2+\sigma_j^2-\log\sigma_j^2-1\right).
$$

Implementations commonly predict `log_variance` rather than variance directly:

```python
variance = log_variance.exp()
std = (0.5 * log_variance).exp()
```

This keeps the variance positive without constraining an unconstrained network
output.

## Reparameterization trick

Sampling $z\sim q_\phi(z\mid x)$ directly appears to interrupt differentiation
because the random sampling operation depends on encoder parameters. Rewrite
the sample as

$$
\epsilon\sim\mathcal{N}(0,I),
\qquad
z=\mu+\sigma\odot\epsilon.
$$

The randomness is now isolated in $\epsilon$, which does not depend on
$\mu$ or $\sigma$. The remaining operations are differentiable, so
reconstruction gradients can flow through $z$ into the encoder.

```text
x -> encoder -> mu, log_variance
                    ↓
epsilon ~ N(0,I) -> z = mu + std * epsilon
                    ↓
                 decoder -> reconstruction distribution
```

During training, one or a few Monte Carlo samples usually approximate the
expectation in the reconstruction term. During generation, sample $z$ directly
from the prior and run only the decoder.

## Sources

- Stanford CS231n Spring 2025, Lecture 13: Generative Models 1, available from
  the [course schedule](https://cs231n.stanford.edu/2025/schedule.html).
- Kingma and Welling, [Auto-Encoding Variational
  Bayes](https://arxiv.org/abs/1312.6114).

