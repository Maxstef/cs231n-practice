# Nats and bits per pixel

Negative log-likelihood measures how much probability a model assigns to
observed data. Its unit depends only on the base of the logarithm:

- natural logarithm, `ln`: **nats**;
- base-2 logarithm, `log2`: **bits**.

For an event with probability $p$, let $L_e$ be NLL measured with the natural
logarithm and $L_2$ be NLL measured with the base-2 logarithm:

$$
L_e=-\ln p,
\qquad
L_2=-\log_2 p.
$$

Since $\log_2 p=\ln p/\ln 2$, conversion is straightforward:

$$
L_2=\frac{L_e}{\ln 2}.
$$

For example, an observed outcome with probability $0.25$ has
$-\ln(0.25)\approx1.386$ nats, or exactly $-\log_2(0.25)=2$ bits.

## Why normalize per pixel?

The NLL of an autoregressive image is the sum of its pixel NLLs:

$$
L_e(x)
=-\sum_{t=1}^{T}\ln p(x_t\mid x_{<t}).
$$

Longer images naturally have larger total NLL. Dividing by the number of
modeled pixel values $T$ gives a length-independent average:

Nats per pixel:

$$
\frac{L_e(x)}{T}.
$$

Bits per pixel:

$$
\frac{L_e(x)}{T\ln 2}
=\frac{L_2(x)}{T}.
$$

For RGB models, authors may instead report **bits per dimension**, where every
channel value counts as one dimension. Always check whether “pixel” means one
whole RGB pixel or one individual channel value.

## Interpretation

Lower is better: it means the model assigned greater probability to the
observed data.

A uniform predictor for one binary value assigns probability $0.5$ to either
outcome:

$$
-\log_2(0.5)=1.
$$

The unit on the right-hand side is one bit. It therefore obtains $1$ bit per
binary pixel. A result below $1$ means the
model exploits predictable structure better than this uniform baseline.

Bits also have an information-coding interpretation: an ideal code based on
the model would need approximately $-\log_2 p(x)$ bits to encode an outcome
$x$. Bits per pixel is therefore the model's average idealized coding cost per
pixel.

## Important limitation

NLL and bits per pixel measure probability assignment, not human perceptual
quality. Two models can rank differently by likelihood and by how realistic
their samples look. Comparisons are meaningful only when data preprocessing,
quantization, and the counted dimensions are defined consistently.

## Related practice

- [Notebook 54 — Autoregressive models and exact likelihood](../../notebooks/54_autoregressive_models_and_exact_likelihood.ipynb)
- [Autoregressive generative models](autoregressive-generative-models.md)
- [Maximum likelihood estimation](maximum-likelihood-estimation.md)
