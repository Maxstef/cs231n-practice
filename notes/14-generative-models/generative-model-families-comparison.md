# Generative-model families: a compact comparison

Generative-model families differ mainly in what they learn and how they turn
randomness into a sample. The boundaries are useful, but modern systems often
combine several families.

## Core comparison

| Family | Basic generation process | Likelihood treatment | Characteristic trade-off |
| --- | --- | --- | --- |
| Autoregressive | Generate one value or token after another | Exact factorized likelihood | Simple objective, but sequential sampling |
| Variational autoencoder (VAE) | Sample a latent code and decode it | Optimizes an evidence lower bound | Organized latent space, often softer reconstructions |
| Generative adversarial network (GAN) | Map noise through a generator trained against a discriminator | No explicit tractable likelihood | Fast one-pass sampling, but difficult adversarial training |
| Diffusion or rectified flow | Iteratively transform noise into data | Depends on the formulation | Stable supervised-style objective, but repeated sampling steps |
| Conditional captioning | Generate tokens conditioned on image features | Exact autoregressive caption likelihood | Quality and diversity depend on conditioning and decoding |

“Exact likelihood” does not mean that training or sampling is automatically
easy. It means that the model assigns a likelihood that can be evaluated
directly through its factorization. See
[Probability, likelihood, and tractability](../13-generative-models/probability-likelihood-and-tractability.md)
for the distinction between exact, approximate, and implicit likelihoods.

## Choosing by priority

- Choose an **autoregressive model** when exact likelihood and ordered output
  generation are central.
- Choose a **VAE** when a regular, interpretable, or easily sampled latent space
  matters.
- Choose a **GAN** when one-pass generation is valuable and adversarial
  training is acceptable.
- Choose **diffusion or flow** when iterative generation and strong sample
  quality are worth the additional sampling cost.
- Use **conditional generation** when outputs must respond to text, labels,
  images, or another input. Conditioning is not a separate underlying family:
  autoregressive, VAE, GAN, diffusion, and flow models can all be conditional.

## The families can be combined

A modern image system may use:

1. an autoencoder to compress images into a latent space;
2. a Transformer to predict iterative diffusion or flow updates;
3. text embeddings and classifier-free guidance for conditioning;
4. an adversarial loss to improve the image decoder.

Model-family names therefore describe important components or objectives, not
necessarily an entire standalone product.

## Connections to the practice notebooks

- [Notebook 54 — Autoregressive models and exact likelihood](../../notebooks/54_autoregressive_models_and_exact_likelihood.ipynb)
- [Notebook 55 — Autoencoders and latent spaces](../../notebooks/55_autoencoders_and_latent_spaces.ipynb)
- [Notebook 56 — Variational autoencoders and ELBO](../../notebooks/56_variational_autoencoders_and_elbo.ipynb)
- [Notebook 57 — VAE image experiment](../../notebooks/57_vae_image_experiment_and_latent_space_analysis.ipynb)
- [Notebook 58 — GAN intuition and adversarial training](../../notebooks/58_gan_intuition_and_adversarial_training.ipynb)
- [Notebook 59 — Diffusion intuition and rectified flow](../../notebooks/59_diffusion_intuition_and_rectified_flow.ipynb)
- [Notebook 60 — Conditional guidance and diffusion systems](../../notebooks/60_conditional_guidance_and_diffusion_systems.ipynb)
- [Notebook 61 — Small image-generation comparison](../../notebooks/61_small_image_generation_comparison.ipynb)
- [Notebook 62 — Recurrent and Transformer captioning comparison](../../notebooks/62_recurrent_and_transformer_captioning_comparison.ipynb)

## Related notes

- [Generative models and likelihood](../13-generative-models/generative-models-and-likelihood.md)
- [Autoregressive generative models](../13-generative-models/autoregressive-generative-models.md)
- [VAEs, ELBO, and reparameterization](../13-generative-models/vae-elbo-and-reparameterization.md)
- [Generative adversarial networks](generative-adversarial-networks.md)
- [Diffusion and rectified flow](diffusion-and-rectified-flow.md)
- [Conditional generation and guidance](conditional-generation-and-guidance.md)
- [Latent diffusion and modern generative systems](latent-diffusion-and-modern-systems.md)
