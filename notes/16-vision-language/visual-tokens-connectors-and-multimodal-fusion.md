# Visual tokens, connectors, and multimodal fusion

A CLIP-style dual encoder often reduces an entire image and an entire text to
one vector each. This is efficient for retrieval and zero-shot classification,
but a generative vision-language model usually needs local visual information:
which object is where, which region contains text, and which patch a question
refers to.

The common solution is to preserve an image as a sequence of visual tokens and
let text interact with those tokens.

## From an image to visual tokens

A Vision Transformer divides an image into non-overlapping patches. For patch
width $P$, an image batch shaped

$$
(N,C,H,W)
$$

becomes a patch sequence shaped

$$
(N,L_v,CP^2),
\qquad
L_v=\frac{H}{P}\frac{W}{P}.
$$

Each flattened patch is projected to the vision encoder's feature width. After
the Transformer layers, the model has one contextual representation per patch.

A classification model may use only a CLS or pooled token as its final global
representation. A VLM can instead retain the patch tokens from a late vision
layer. They expose spatially organized local information that would be harder
for a language model to recover from one pooled vector.

The final layer is not automatically the best interface. Features from a
penultimate or other late layer may preserve useful local detail that the final
classification objective compresses. The best layer is an empirical design
choice, not a universal rule.

## What a connector does

The vision encoder and language model are usually pretrained separately and
may use different feature widths:

$$
V\in\mathbb{R}^{N\times L_v\times D_v},
\qquad
T\in\mathbb{R}^{N\times L_t\times D_{LM}}.
$$

A connector maps visual features into the width expected by the language
model:

$$
V' = f_{\text{connector}}(V),
\qquad
V'\in\mathbb{R}^{N\times L'_v\times D_{LM}}.
$$

The connector may be:

- one linear projection;
- a small multilayer perceptron;
- a resampler that converts many patch tokens into fewer learned tokens;
- a more specialized module trained to align vision and language features.

It can change feature width, token count, or both. It does not itself guarantee
visual understanding; it creates a compatible interface through which the
language model can consume visual information.

## Position and modality are different signals

Token representations may receive several additions:

- a **positional embedding** says where a token occurs in its patch grid or
  text sequence;
- a **modality embedding** says whether the token came from the image or text.

Two tokens can occupy similar sequence positions while belonging to different
modalities. Conversely, two visual tokens share a modality but correspond to
different image locations. The signals therefore answer different questions.

## Three broad fusion patterns

| Pattern | Interaction | Main use |
| --- | --- | --- |
| Global dual encoder | Compare one image vector with one text vector | Retrieval and zero-shot classification |
| Joint self-attention | Concatenate visual and text tokens into one sequence | Direct all-to-all multimodal interaction |
| Cross-attention | Tokens from one modality query tokens from another | Selective reading of visual information |

Real systems can combine these patterns. A vision encoder may first perform
visual self-attention, a connector may project or resample the result, and a
language model may then use joint or cross-attention.

## Joint self-attention

Visual and text tokens can be concatenated:

$$
X=[v_1,\ldots,v_{L_v},t_1,\ldots,t_{L_t}].
$$

Self-attention over this sequence has a score matrix shaped

$$
(N,H_{\text{heads}},L_v+L_t,L_v+L_t).
$$

It contains four kinds of interaction:

- visual queries attending to visual keys;
- visual queries attending to text keys;
- text queries attending to visual keys;
- text queries attending to text keys.

Padding masks must prevent padded text positions from being used as keys. A
causal language model also needs a causal mask so generated text cannot inspect
future text tokens.

## Text-to-image cross-attention

Cross-attention can keep the two streams separate:

$$
Q=T W_Q,
\qquad
K=V' W_K,
\qquad
U=V' W_V.
$$

Its attention weights have shape

$$
(N,H_{\text{heads}},L_t,L'_v).
$$

Here $H_{\text{heads}}$ is the number of attention heads, not image height.
Each row corresponds to one text query, and each column corresponds to one
visual key. Query and key sequence lengths need not match; only their projected
per-head feature widths must match for the dot product.

The output for every text position is a weighted combination of visual value
vectors. This gives the language stream a way to read different image regions
for different words or questions.

Attention weights should still be interpreted cautiously. They describe where
one attention operation reads, but value projections, residual paths, other
heads, and later layers determine how that information affects the output.

## LLaVA: projected visual tokens in an LLM

The basic LLaVA recipe illustrates a simple connector-based design:

1. start with a pretrained vision encoder and pretrained language model;
2. extract visual tokens from the vision encoder;
3. train a projection that maps them into the language model's input width;
4. present projected visual tokens together with instruction text;
5. train the system to generate the desired textual response.

The projected visual tokens behave like additional context tokens for the
autoregressive language model. Early alignment training teaches the connector
how visual features correspond to language space; instruction tuning then
teaches useful conversational and question-answering behavior. Exact freezing
and fine-tuning choices differ among LLaVA versions.

## Flamingo: resampling and gated cross-attention

Flamingo uses a different integration strategy:

1. a pretrained vision encoder produces many visual features;
2. a Perceiver Resampler converts them to a fixed, smaller set of visual
   tokens;
3. gated cross-attention layers are inserted between frozen language-model
   blocks;
4. text tokens query visual tokens while ordinary causal language modeling
   continues to predict the next text token.

The gates allow newly inserted visual pathways to begin with a limited effect
on the pretrained language model and become useful during multimodal training.

Flamingo accepts interleaved images and text. Its masking associates text with
the appropriate preceding images rather than allowing every text position to
read every image indiscriminately. This supports multimodal in-context
learning: demonstrations consisting of images and text can appear in the
prompt before a new query.

## Main intuition

> A connector makes visual features compatible with a language model; fusion
> determines how and where language can use those features.

Global embeddings are efficient when one similarity score is enough. Visual
tokens and multimodal attention are more suitable when the output depends on
local details, grounding, relations, or generated language.

## Related practice and notes

- [Notebook 68 — Visual tokens and multimodal fusion](../../notebooks/68_visual_tokens_and_multimodal_fusion.ipynb)
- [Foundation models for vision and language](foundation-models-for-vision-and-language.md)
- [CLIP and contrastive image–text learning](clip-and-contrastive-image-text-learning.md)
- [Molmo and PixMo](molmo-and-pixmo.md)
- [Attention, self-attention, and cross-attention](../08-attention-transformers/attention-self-and-cross-attention.md)

## Sources

- Stanford CS231n Spring 2025, Lecture 16: Vision and Language, available from
  the [course schedule](https://cs231n.stanford.edu/2025/schedule.html).
- Liu et al., [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485).
- Alayrac et al., [Flamingo: a Visual Language Model for Few-Shot
  Learning](https://arxiv.org/abs/2204.14198).

