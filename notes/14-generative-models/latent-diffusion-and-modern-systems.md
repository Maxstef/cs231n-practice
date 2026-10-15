# Latent diffusion and modern generative systems

Running a generative model directly on every high-resolution pixel is
expensive. A latent diffusion model first compresses images into smaller
spatial representations and performs the iterative generative process there.

## Three-part pipeline

```text
image -> encoder -> compact latent

random latent -> repeated denoising or flow steps -> clean latent -> decoder -> image
                         condition: time, text, class, ...
```

The components have different jobs:

| Component | Purpose |
| --- | --- |
| Encoder | Compress an image while preserving visually useful information |
| Generative model | Transform random latent noise into a structured latent |
| Decoder | Convert the final latent representation back into pixels |

The encoder and decoder are usually trained first. They are then frozen while
the diffusion or flow model learns in latent space. At generation time, the
encoder is unnecessary because sampling begins from a random latent; only the
generative model and decoder are needed.

## Why work in latent space?

If an encoder reduces each spatial dimension by a factor $D$, the generative
model processes roughly $D^2$ fewer spatial positions. This can greatly reduce
memory and computation. The trade-off is that the final output cannot contain
information the encoder-decoder pair failed to preserve.

The image autoencoder may combine ideas from earlier model families:

- a VAE-style prior regularizer encourages a manageable latent space;
- an adversarial discriminator can encourage sharper decoded images;
- diffusion or flow performs the actual iterative generation of new latents.

These parts cooperate; calling the complete pipeline “a diffusion model” can
hide that it is a system made from several learned modules.

## Diffusion Transformers

A **Diffusion Transformer (DiT)** uses Transformer blocks as the network that
processes noisy latent tokens. Its core operations are familiar from Vision
Transformers: patchify the latent grid, add position information, apply
self-attention and feed-forward blocks, then project tokens back to the desired
latent-shaped output.

The important extra question is how to inject conditioning:

- the timestep can modulate normalization using learned scale and shift;
- text or image embeddings can be read through cross-attention;
- conditioning tokens can also join the latent tokens in joint attention.

A DiT does not replace the complete latent-diffusion pipeline. It replaces the
neural-network backbone that predicts the denoising or flow update.

## Images and video use the same high-level idea

For video, the latent representation also has a time axis. The Transformer
must model spatial appearance and temporal consistency, which greatly
increases the number of tokens and the computational cost. The high-level
pipeline remains the same: text and time condition an iterative latent
generator, and a decoder reconstructs the output frames.

## Why sampling can be slow

Generation may evaluate the same large model tens of times. **Diffusion
distillation** trains a faster student sampler to approximate several teacher
steps with fewer steps, sometimes only one or a few. Faster sampling may come
with quality, diversity, or training-complexity trade-offs.

## Scope note

Diffusion can also be described through score functions, stochastic
differential equations, deep latent-variable models, or generalized noise
parameterizations. These views are mathematically connected. For an initial
understanding, the useful invariant is simpler:

> learn a time-dependent update field, then follow it iteratively from an easy
> noise distribution toward the data distribution.

## Sources

- Stanford CS231n Spring 2025, Lecture 14: Generative Models 2, available from
  the [course schedule](https://cs231n.stanford.edu/2025/schedule.html).
- Rombach et al., [High-Resolution Image Synthesis with Latent Diffusion
  Models](https://arxiv.org/abs/2112.10752).
- Peebles and Xie, [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748).
