# CLIP limitations and data quality

CLIP-style models are efficient and flexible, but a shared embedding space does
not guarantee complete visual understanding. Their behavior reflects the
contrastive objective, batch construction, and pretraining data.

## Dependence on in-batch negatives

For one positive image–text pair, the other batch items act as negatives.
Larger batches supply more comparisons and are more likely to include hard,
fine-grained alternatives. This strengthens the learning signal but requires
more memory and still cannot cover every important distinction.

The objective also creates **false negatives**. An image may have several valid
descriptions, but only its recorded partner is designated positive. Another
semantically correct caption in the batch can be treated as a negative.

## Compositionality

A model may recognize individual objects and words yet struggle with their
relations. These phrases contain similar components but describe different
scenes:

```text
a mug in some grass
some grass in a mug
```

Bag-of-concepts behavior can also confuse subject–object order, spatial
relations, counts, and attribute binding. Carefully constructed benchmarks
such as Winoground, CREPE, and ARO probe these weaknesses rather than relying
only on ordinary category recognition.

- **Winoground** presents two images and two captions made from the same words
  in different orders. A model must match both captions correctly, so word
  presence alone is insufficient.
- **CREPE** tests systematicity—recombining familiar concepts in unfamiliar
  ways—and productivity—understanding descriptions as their compositional
  complexity grows. It includes hard negative captions with changed objects,
  relations, swaps, or negations.
- **ARO** stands for Attribution, Relation, and Order. Its tests alter
  attribute–object bindings, relations between objects, or word order to check
  whether a model understands structure rather than treating text as a bag of
  words.

## Image-level captions are incomplete supervision

Typical captions mention only some visible content. They may omit:

- exact object counts;
- precise positions and relations;
- small or background objects;
- text present in the image;
- colors, styles, and subtle details;
- which words refer to which regions.

The model can therefore learn broad semantic alignment without reliable local
grounding. Dense descriptions, pointing annotations, boxes, masks, and
question-based annotation can provide stronger supervision.

## Data quality versus quantity

Scaling the number of image–text pairs is useful only when the data provide
relevant, diverse, and trustworthy learning signals. Important concerns
include:

- incorrect or weakly related captions;
- duplicated examples and benchmark contamination;
- demographic and cultural biases;
- missing rare or safety-critical cases;
- licensing, privacy, and provenance;
- insufficient coverage of compositional contrasts.

The lecture’s [Molmo and PixMo](molmo-and-pixmo.md) examples emphasize
intentional data collection: dense captions, spoken descriptions converted to
text, pointing, counting, and questions designed to elicit visual details.
Smaller but more informative data can sometimes be more valuable than a much
larger weakly curated corpus.

## Distribution robustness is not universal understanding

Natural-language supervision can improve transfer to sketches, unusual image
styles, and new label vocabularies. Strong performance on such shifts is
valuable, but it should not be confused with universal robustness. New domains,
relations, and concepts can still fall outside the learned representation.

## Main intuition

> Open-vocabulary recognition expands the interface of a classifier; it does
> not remove the need to examine training data, prompts, grounding, and failure
> cases.

## Related practice and notes

- [Notebook 66 — CLIP embeddings and contrastive alignment](../../notebooks/66_clip_embeddings_and_contrastive_alignment.ipynb)
- [Notebook 67 — Zero-shot classification and prompt design](../../notebooks/67_zero_shot_classification_and_prompt_design.ipynb)
- [CLIP and contrastive image–text learning](clip-and-contrastive-image-text-learning.md)
- [Zero-shot classification and prompt ensembling](zero-shot-classification-and-prompt-ensembling.md)

## Sources

- Stanford CS231n Spring 2025, Lecture 16: Vision and Language, available from
  the [course schedule](https://cs231n.stanford.edu/2025/schedule.html).
- Radford et al., [Learning Transferable Visual Models From Natural Language
  Supervision](https://arxiv.org/abs/2103.00020).
- Thrush et al., [Winoground: Probing Vision and Language Models for
  Visio-Linguistic Compositionality](https://arxiv.org/abs/2204.03162).
- Ma et al., [CREPE: Can Vision-Language Foundation Models Reason
  Compositionally?](https://arxiv.org/abs/2212.07796).
- Yuksekgonul et al., [When and why vision-language models behave like
  bags-of-words, and what to do about it?](https://arxiv.org/abs/2210.01936).

