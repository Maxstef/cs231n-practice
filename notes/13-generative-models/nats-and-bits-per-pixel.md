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

The total NLL of an image is the sum of the NLL contributions from all its
pixels. Equivalently, it is the negative logarithm of the probability assigned
to the complete image:

$$
E(x)=-\ln p(x)
$$

Longer images naturally have larger total NLL. Dividing by the number of
modeled pixel values $T$ gives a length-independent average. Here $E(x)$ is
the total image NLL in nats and $B(x)$ is the same quantity in bits.

Nats per pixel:

$$
E(x) / T
$$

Bits per pixel:

$$
B(x) / T = E(x) / (T \ln 2)
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

## Connection to data storage

This is the same binary information unit used in storage: eight bits make one
byte. A good probability model can support compression because predictable
values require fewer bits, while surprising values require more bits.

Suppose an $8$ by $8$ binary image contains $64$ pixels:

- storing every pixel directly requires $64$ bits, or $8$ bytes;
- at $0.328$ bits per pixel, the ideal average cost is
  $64 \times 0.328 \approx 21$ bits, or about $2.6$ bytes.

The fractional byte is an average theoretical cost across data, not a claim
that one ordinary file occupies part of a byte. An entropy coder such as
arithmetic coding or range coding can turn the model's predicted probabilities
into an actual bitstream whose average length approaches this ideal.

Do not confuse likelihood-based bits per pixel with raw **bit depth**:

| Quantity | Meaning |
| --- | --- |
| 8 bits per channel | Number of raw bits used to represent one channel value |
| 24 bits per RGB pixel | Three raw 8-bit channel values |
| Model bits per pixel | Average ideal coding cost based on predicted probabilities |

Actual compressed files also include finite-precision effects, metadata, file
headers, and coding overhead. The decoder must also already possess the same
probability model; including the trained model's parameters would add further
storage cost.

## Important limitation

NLL and bits per pixel measure probability assignment, not human perceptual
quality. Two models can rank differently by likelihood and by how realistic
their samples look. Comparisons are meaningful only when data preprocessing,
quantization, and the counted dimensions are defined consistently.

## Related practice

- [Notebook 54 — Autoregressive models and exact likelihood](../../notebooks/54_autoregressive_models_and_exact_likelihood.ipynb)
- [Autoregressive generative models](autoregressive-generative-models.md)
- [Maximum likelihood estimation](maximum-likelihood-estimation.md)
