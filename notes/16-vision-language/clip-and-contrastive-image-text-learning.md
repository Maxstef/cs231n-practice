# CLIP and contrastive image–text learning

CLIP trains an image encoder and a text encoder to represent matching images
and texts near one another in a shared embedding space.

For a batch of $N$ paired examples, the encoders produce

$$
I\in\mathbb{R}^{N\times D},
\qquad
T\in\mathbb{R}^{N\times D}.
$$

The architectures may differ, but both outputs need width $D$ so their vectors
can be compared.

## Normalize, compare, and classify

Each embedding is normalized to unit length:

$$
\hat v_i=\frac{v_i}{\lVert v_i\rVert_2}.
$$

The complete cosine-similarity matrix is

$$
S=\hat I\hat T^T,
\qquad
S_{ij}=\hat I_i^T\hat T_j.
$$

Row $i$ compares image $i$ with all texts. Column $j$ compares text $j$ with
all images. If paired examples use matching indices, positive pairs occupy the
diagonal.

For image-to-text training, each row is treated as an $N$-class classification
problem:

$$
L_{I\rightarrow T}
=\frac{1}{N}\sum_i
-\log
\frac{\exp(S_{ii}/\tau)}
{\sum_j\exp(S_{ij}/\tau)}.
$$

Transposing the logits gives text-to-image training. CLIP averages both:

$$
L_{CLIP}=\frac{1}{2}
\left(L_{I\rightarrow T}+L_{T\rightarrow I}\right).
$$

The similarity matrix need not be numerically symmetric because image and text
vectors come from different encoders. Even for a square matrix, row-wise and
column-wise softmax use different competitor sets.

## Temperature and learned logit scale

Temperature controls softmax sharpness. Dividing by $\tau$ is equivalent to
multiplying by logit scale $s$:

$$
\frac{S}{\tau}=sS,
\qquad
s=\frac{1}{\tau}.
$$

CLIP-style models commonly learn one global logit scale. It can be stored as a
logarithm and exponentiated so it remains positive. A maximum constraint keeps
the model from producing arbitrarily sharp, overconfident logits.

Temperature changes probabilities and gradient scale, but multiplying all
scores by the same positive value does not change their ranking.

## Retrieval

The same matrix supports both retrieval directions:

- image-to-text retrieval sorts a row;
- text-to-image retrieval sorts a column.

Candidate embeddings can be computed in advance, so inference mainly requires
encoding the query and performing efficient similarity search. This is one
reason dual encoders scale well to large retrieval collections.

## Relation to SimCLR

SimCLR aligns two augmented views using one shared image encoder. CLIP aligns
two different modalities using separate image and text encoders:

```text
SimCLR: image view A <-> image view B
CLIP:   image        <-> natural-language text
```

Both use normalized embeddings, temperature-scaled similarities, positive
pairs, and in-batch negatives. CLIP gains a flexible language interface, which
enables open-vocabulary retrieval and classification.

## Alignment is not generation

CLIP scores compatibility between complete image and text representations. It
does not predict a caption token by token. CoCa combines contrastive alignment
with an autoregressive captioning objective, illustrating that retrieval and
generation are related but distinct capabilities.

## Related practice and notes

- [Notebook 66 — CLIP embeddings and contrastive alignment](../../notebooks/66_clip_embeddings_and_contrastive_alignment.ipynb)
- [Contrastive learning, SimCLR, and InfoNCE](../12-self-supervised/contrastive-learning-simclr-and-infonce.md)
- [Zero-shot classification and prompt ensembling](zero-shot-classification-and-prompt-ensembling.md)

## Sources

- Stanford CS231n Spring 2025, Lecture 16: Vision and Language, available from
  the [course schedule](https://cs231n.stanford.edu/2025/schedule.html).
- Radford et al., [Learning Transferable Visual Models From Natural Language
  Supervision](https://arxiv.org/abs/2103.00020).
- Yu et al., [CoCa: Contrastive Captioners are Image-Text Foundation
  Models](https://arxiv.org/abs/2205.01917).

