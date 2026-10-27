# Knowledge distillation

Knowledge distillation trains a **student** model to reproduce useful behavior
from a **teacher** model. The teacher is usually larger, more expensive, or an
ensemble; the student is usually smaller and cheaper to run.

```text
input -> teacher -> teaching signal
   \                         |
    -> student --------------+-> distillation loss
```

The student does not copy the teacher’s parameter values. It learns from
signals produced by the teacher.

## Hard labels versus soft targets

A hard class label says only which class is correct:

```text
cat -> [0, 1, 0, 0]
```

A teacher’s probability distribution contains additional structure:

```text
[dog: 0.12, cat: 0.75, fox: 0.10, car: 0.03]
```

The smaller probabilities reveal which alternatives the teacher considers
similar. This information is often called **dark knowledge**. A higher
softmax temperature can soften the teacher distribution so these relationships
are easier for the student to observe.

A typical objective combines two signals:

$$
L_{student}
=\alpha L_{teacher}
+(1-\alpha)L_{labels}.
$$

- $L_{teacher}$ encourages the student to match softened teacher outputs.
- $L_{labels}$ keeps the student connected to ground-truth supervision.
- $\alpha$ controls the trade-off.

The exact loss varies by task and model family.

## Other forms of distillation

- **Response distillation:** imitate output probabilities, logits, generated
  answers, captions, or reasoning traces.
- **Feature distillation:** make intermediate student representations resemble
  teacher features.
- **Relation distillation:** preserve similarities or relationships among
  examples, tokens, or regions.
- **Synthetic-data distillation:** ask a teacher to generate training examples
  or annotations, then train the student on that dataset.

For language and vision-language models, the last form is common because the
teacher can generate detailed answers and captions even when its logits or
internal features are unavailable through an API.

## Why distill a model?

A successful student may offer:

- lower inference latency;
- less memory use;
- lower serving cost and energy use;
- deployment on smaller devices;
- a single deployable model that approximates an ensemble.

The student normally loses some teacher capability because it has less
capacity and sees only selected teacher behavior. It may also inherit the
teacher’s factual errors, biases, and blind spots.

## Distillation is not the same as related techniques

| Technique | Main operation |
| --- | --- |
| Distillation | Train a student to imitate a teacher signal |
| Fine-tuning | Adapt an existing model using task or domain data |
| Quantization | Store or compute weights and activations at lower precision |
| Pruning | Remove parameters, connections, or structures |

These techniques can be combined. For example, a distilled student can later
be quantized.

## Vision-language openness nuance

An open-weight student trained mainly on outputs from a closed teacher is still
useful, but the process is not fully independent or reproducible without
teacher access. The student may encode capabilities and limitations inherited
from an unavailable model. This is why the distinction among open weights,
open code, and open training data matters in the
[Molmo and PixMo](molmo-and-pixmo.md) discussion.

## Main intuition

> Distillation transfers behavior, not weights: a smaller student learns from
> the richer signals produced by a stronger teacher.

## Sources

- Stanford CS231n Spring 2025, Lecture 16: Vision and Language, available from
  the [course schedule](https://cs231n.stanford.edu/2025/schedule.html).
- Hinton, Vinyals, and Dean, [Distilling the Knowledge in a Neural
  Network](https://arxiv.org/abs/1503.02531).
- Deitke et al., [Molmo and PixMo: Open Weights and Open Data for
  State-of-the-Art Vision-Language Models](https://openaccess.thecvf.com/content/CVPR2025/papers/Deitke_Molmo_and_PixMo_Open_Weights_and_Open_Data_for_State-of-the-Art_CVPR_2025_paper.pdf).

