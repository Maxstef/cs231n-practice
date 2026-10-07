# MoCo, CPC, and DINO

Self-supervised methods differ mainly in how they define related examples,
construct targets, and avoid uninformative representations.

| Method | Learning signal | Important mechanism |
| --- | --- | --- |
| SimCLR | Match two augmented image views against in-batch negatives | Large batches and projection head |
| MoCo | Match queries to keys while contrasting against queued keys | Momentum key encoder and negative queue |
| CPC | Predict representations of future sequence regions | Autoregressive context and contrastive prediction |
| DINO | Match a student's view predictions to a teacher's targets | EMA teacher, stop-gradient, centering, sharpening |
| DINOv2 | Match views at both image and patch levels at large scale | DINO + masked-patch iBOT objectives, curated data, and scalable training |

## Momentum Contrast (MoCo)

SimCLR obtains negatives from its current minibatch, tying the number of
negatives to batch size. MoCo maintains a queue of key representations from
recent minibatches. A slowly changing momentum encoder produces those keys:

$$
\theta_k \leftarrow m\theta_k+(1-m)\theta_q,
$$

where $\theta_q$ are query-encoder parameters and $m$ is close to one. The key
encoder is not updated by ordinary backpropagation through the queued keys.
Slow updates keep queued representations more consistent while allowing many
negatives without an equally large current batch.

## Contrastive Predictive Coding (CPC)

CPC applies contrastive learning to ordered data. An encoder maps observations
to latent vectors, an autoregressive model summarizes the past, and the
objective identifies future latent vectors among negatives.

```text
past observations -> latent sequence -> context
                                      -> predict future latent positions
```

The prediction target is a representation rather than raw future pixels or
audio samples. CPC therefore emphasizes information useful for distinguishing
the actual future from alternatives and applies naturally to audio, video, and
other sequences.

## DINO

DINO is a teacher–student self-distillation method. The student processes one
augmented view and learns to match probability targets produced by a teacher
from another view. Gradients update the student only; the teacher follows an
exponential moving average of student parameters.

Without special treatment, teacher and student could agree on the same
constant output for every image. DINO combines several asymmetries to resist
collapse:

- stop-gradient prevents direct optimization through teacher targets;
- the momentum teacher changes more slowly than the student;
- centering prevents one output dimension from dominating;
- sharpening makes teacher targets more informative and selective;
- multiple image crops expose the networks to different views.

Unlike InfoNCE, DINO does not rely on ordinary negative pairs. Unlike MAE, its
target is a learned teacher distribution rather than missing pixels.

## DINOv2

DINOv2 aims to turn the DINO teacher–student idea into a source of
**general-purpose visual features** that transfer across image distributions
and both image-level and dense tasks. It is better understood as a scaled and
strengthened training recipe than as one isolated new loss.

Its objective combines two kinds of teacher targets:

1. **Image-level DINO objective:** student and teacher class-token outputs from
   different crops should agree.
2. **Patch-level iBOT objective:** some student patches are masked, and the
   student predicts the teacher's representations for the corresponding
   visible patches.

The second objective resembles masked image modeling, but the target is a
teacher feature distribution rather than raw pixels:

```text
MAE:     masked patch -> predict original pixel values
DINOv2:  masked patch -> predict teacher patch representation
```

DINOv2 also adds KoLeo feature-spreading regularization and uses
Sinkhorn–Knopp centering to balance teacher outputs. These mechanisms
complement the momentum teacher and help prevent representations from
collapsing or concentrating in a small part of feature space.

### What made DINOv2 different in practice

- **Data curation:** an automated retrieval and deduplication pipeline created
  a diverse 142-million-image training set; self-supervised training used no
  class or text targets.
- **Scale:** the largest teacher was a roughly one-billion-parameter Vision
  Transformer, with smaller models distilled from it.
- **Image and patch features:** class-token features support image-level tasks,
  while patch tokens retain spatial information useful for correspondence,
  segmentation, and depth estimation.
- **Training engineering:** memory-efficient attention, sequence packing,
  sharded training, and a short high-resolution phase made large-scale
  pretraining practical.

The conceptual progression is therefore:

```text
DINO   -> learn view-invariant image representations from a momentum teacher
DINOv2 -> combine image- and patch-level self-distillation, then scale the
          data, model, curation, and training system
```

DINOv2 remains label-free in its training objective, but its quality does not
come from the objective alone. Dataset diversity, curation, model capacity,
and compute are central parts of the method.

## Sources

- Stanford CS231n Spring 2025, Lecture 12: Self-Supervised Learning, available
  from the [course schedule](https://cs231n.stanford.edu/2025/schedule.html).
- He et al., [Momentum Contrast for Unsupervised Visual Representation
  Learning](https://arxiv.org/abs/1911.05722).
- van den Oord et al., [Representation Learning with Contrastive Predictive
  Coding](https://arxiv.org/abs/1807.03748).
- Caron et al., [Emerging Properties in Self-Supervised Vision
  Transformers](https://arxiv.org/abs/2104.14294).
- Oquab et al., [DINOv2: Learning Robust Visual Features without
  Supervision](https://arxiv.org/abs/2304.07193).
