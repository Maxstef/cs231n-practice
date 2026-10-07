# Masked image modeling and MAE

Masked image modeling hides part of an image and trains a model to predict the
missing content. The original image supplies the targets, so no class labels
are required.

The **Masked Autoencoder (MAE)** design is asymmetric: a large encoder sees
only visible patches, while a smaller decoder reconstructs the complete patch
sequence.

## Patch and mask flow

For $32\times32$ RGB images and $8\times8$ patches:

```text
image                    (N, 3, 32, 32)
patchify                 (N, 16, 192)
keep 25% visible         (N,  4, 192)
patch embedding          (N,  4, encoder_dim)
Transformer encoder      (N,  4, encoder_dim)
decoder-width projection (N,  4, decoder_dim)
restore with mask tokens (N, 16, decoder_dim)
Transformer decoder      (N, 16, decoder_dim)
pixel prediction         (N, 16, 192)
```

Random masking records two useful index tensors:

- `ids_keep` selects the original patch positions sent to the encoder;
- `ids_restore` is the inverse shuffle that restores visible and mask tokens
  to original spatial order.

## Encoder and decoder roles

The encoder embeds visible pixel patches, adds their positions, and uses
self-attention to build contextual representations. Removing masked positions
shortens its sequence. Since full attention grows quadratically with sequence
length, encoding 25% of the patches requires far less attention work than
encoding all patches.

The encoder-to-decoder projection changes feature width, not token count. Mask
tokens are then inserted to recover the full sequence length. They are learned
placeholders rather than pixel predictions. Decoder positional embeddings tell
the model which spatial location every visible or masked token represents.

The decoder uses bidirectional self-attention across the complete sequence and
a final linear layer predicts the raw pixels of each patch. It is primarily a
pretraining component; downstream tasks normally reuse the encoder rather than
the reconstruction decoder.

## Masked-only reconstruction loss

With prediction $\hat{X}$, target $X$, binary mask $M$, and $D$ pixel values per
patch, the loss is

$$
L=\frac{\sum_{n,l,d}M_{nl}(\hat{X}_{nld}-X_{nld})^2}
{D\sum_{n,l}M_{nl}}.
$$

Only masked positions contribute. Visible patches are already provided to the
model, so rewarding their reconstruction would encourage copying rather than
inference from context.

## Mask-ratio trade-off

A higher mask ratio:

- leaves less evidence and makes reconstruction harder;
- shortens the expensive encoder sequence;
- may encourage greater use of global context;
- can become too difficult if too little useful content remains.

Lower pixel loss does not automatically imply better features. Pixel loss
rewards colors, textures, and local statistics, while classification needs
semantic separation. A frozen linear probe should therefore evaluate the
encoder separately.

For downstream classification without a `[CLS]` token, one simple choice is to
encode all unmasked patches and mean-pool their final representations into one
vector per image.

## Sources

- Stanford CS231n Spring 2025, Lecture 12: Self-Supervised Learning, available
  from the [course schedule](https://cs231n.stanford.edu/2025/schedule.html).
- He et al., [Masked Autoencoders Are Scalable Vision
  Learners](https://arxiv.org/abs/2111.06377).

