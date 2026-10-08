# Autoregressive generative models

An autoregressive model turns one complicated joint distribution into a
product of conditional distributions using the probability chain rule.

For a sequence $x=(x_1,\ldots,x_T)$:

$$
p(x)=p(x_1)\prod_{t=2}^{T}p(x_t\mid x_1,\ldots,x_{t-1})
=\prod_{t=1}^{T}p(x_t\mid x_{<t}).
$$

Each factor asks a simpler question: given the values already observed, what
should come next?

## Training and generation

During training, the complete target sequence is available. A causal RNN or
masked Transformer predicts every next-token distribution, and negative log
likelihood compares those distributions with the known next values.

```text
training:   known sequence -> predict all next-value targets
generation: sample x1 -> sample x2 | x1 -> sample x3 | x1,x2 -> ...
```

Generation is sequential because a sampled value becomes context for the next
step. Training a masked Transformer can evaluate positions in parallel;
ordinary ancestral sampling cannot.

## Images as sequences

An image may be serialized in row-major order and treated as a sequence of
channel or pixel values. For 8-bit values, each conditional distribution can
be a 256-class softmax.

PixelRNN carries context through recurrent state. PixelCNN uses masked
convolutions so an output position can depend only on earlier pixels in the
chosen ordering. A causal Transformer can play the same role with an attention
mask.

The model assigns an exact tractable likelihood because every conditional
factor is normalized. However, pixel-by-pixel generation is slow: a
$1024\times1024$ RGB image contains more than three million channel values.
Modern systems can shorten the sequence by modeling learned image tokens or
tiles rather than individual subpixels.

## Ordering is both useful and artificial

The chain rule is valid for any ordering, but an image has no uniquely correct
one-dimensional order. A raster scan makes nearby left/up context available
before right/down context and therefore introduces an architectural bias.

Text has a natural sequential order, which makes autoregressive modeling a
particularly direct fit. The same factorization underlies:

- language modeling;
- recurrent or Transformer image captioning;
- image token generation;
- some audio and video generation systems.

## Strengths and limitations

**Strengths**

- exact log-likelihood for a chosen factorization;
- stable maximum-likelihood training;
- clear conditional generation procedure;
- works with both discrete and continuous outputs.

**Limitations**

- sampling is sequential and can be slow;
- errors become part of the context during generation;
- the chosen ordering affects efficiency and inductive bias;
- exact pixel modeling creates extremely long sequences.

## Sources

- Stanford CS231n Spring 2025, Lecture 13: Generative Models 1, available from
  the [course schedule](https://cs231n.stanford.edu/2025/schedule.html).
- van den Oord et al., [Pixel Recurrent Neural
  Networks](https://arxiv.org/abs/1601.06759).

