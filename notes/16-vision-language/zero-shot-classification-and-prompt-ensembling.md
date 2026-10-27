# Zero-shot classification and prompt ensembling

A CLIP-style model can turn a class vocabulary into a classifier without
training a new classification head. Each candidate class is written as text,
encoded, and compared with the image embedding.

```text
class names -> text encoder -> class prototype vectors
image       -> image encoder -> image vector
                              -> choose the most similar prototype
```

This is called **zero-shot classification** because no labeled examples from
the downstream classification dataset are used to fit a classifier.

## Text embeddings become classifier weights

For class descriptions $c_1,\ldots,c_K$, let their normalized text embeddings
be $t_1,\ldots,t_K$. For normalized image embedding $v$, the class scores are

$$
s_k=v^Tt_k.
$$

The largest score selects the predicted class. This resembles a linear
classifier whose weight vectors are produced by the text encoder rather than
learned from downstream labeled images.

## Why prompt wording matters

CLIP was trained on natural image–text pairs, not isolated label names. The
bare word “dog” may therefore create a less appropriate representation than a
phrase such as “a photo of a dog.” Other datasets may benefit from prompts
such as “a sketch of a dog” or “a satellite photo of a city.”

Prompt choice supplies context but can also introduce bias. One template may
favor particular visual styles or interpretations.

## Prompt ensembling

Instead of relying on one phrase, encode several templates for each class:

```text
a photo of a dog
a close-up photo of a dog
a drawing of a dog
```

A common procedure is:

1. encode every phrase;
2. normalize each phrase embedding;
3. average the embeddings for one class;
4. normalize the averaged class prototype again;
5. compare images with the resulting prototypes.

The average reduces sensitivity to one wording and represents the class across
several contexts. Renormalization is needed because an average of unit vectors
does not generally have unit length.

## Customized prompts

Generic templates do not describe fine-grained distinctions. A language model
can generate richer descriptions such as appearance, shape, context, and
diagnostic attributes for each candidate class. CuPL is an example of using
generated class descriptions to improve zero-shot classification.

This is a form of model chaining:

```text
class name -> language model description -> text encoder -> class prototype
```

The language model does not inspect the query image in this pipeline. It
enriches the class definitions used by the vision–language model.

## What “open vocabulary” does and does not mean

The classifier is not restricted to one label list fixed during downstream
training. New categories can be supplied as text at inference time. However,
performance still depends on whether pretraining taught the relevant visual
and linguistic concepts. A model cannot reliably classify every phrase merely
because it can encode the words.

## Related practice and notes

- [Notebook 66 — CLIP embeddings and contrastive alignment](../../notebooks/66_clip_embeddings_and_contrastive_alignment.ipynb)
- [CLIP and contrastive image–text learning](clip-and-contrastive-image-text-learning.md)
- [CLIP limitations and data quality](clip-limitations-and-data-quality.md)

## Sources

- Stanford CS231n Spring 2025, Lecture 16: Vision and Language, available from
  the [course schedule](https://cs231n.stanford.edu/2025/schedule.html).
- Radford et al., [Learning Transferable Visual Models From Natural Language
  Supervision](https://arxiv.org/abs/2103.00020).
- Pratt et al., [What does a platypus look like? Generating customized prompts
  for zero-shot image classification](https://arxiv.org/abs/2209.03320).

