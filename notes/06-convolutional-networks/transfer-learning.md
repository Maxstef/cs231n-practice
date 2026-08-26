# Transfer learning with CNNs

Transfer learning starts from a model trained on a source task and reuses its
learned representation for a target task. It is especially useful when the
target dataset is too small to train a large CNN reliably from scratch.

## Why learned features can transfer

Earlier CNN layers commonly detect relatively general visual patterns such as
edges, colors, and textures. Deeper layers combine them into increasingly
task-specific patterns. A model trained on a large, varied dataset can therefore
provide a useful feature extractor for another visual task.

This “general early, specific late” description is a useful intuition rather
than a strict rule. How well features transfer depends on the source and target
domains, architecture, and pretraining objective.

## Replace the classifier

A pretrained classifier produces scores for its source classes. For a new task
with $C$ classes:

1. Keep the pretrained feature-producing layers.
2. Remove the original classification layer.
3. Add a new randomly initialized layer that produces $C$ scores.
4. Train the new head, optionally followed by fine-tuning some or all of the
   backbone.

## Feature extraction versus fine-tuning

### Frozen feature extractor

Freeze the pretrained backbone and train only the new classifier. This is fast,
requires less memory for gradients, and reduces overfitting risk on a small
target dataset.

### Fine-tuning

Initialize from pretrained weights, then continue updating some or all layers
on the target data. Fine-tuning lets features adapt to the target task but needs
more computation and can overfit or destroy useful pretrained features if the
learning rate is too large.

A common workflow is to train the new head first, then unfreeze later layers
and fine-tune them with a smaller learning rate.

## Choosing an approach

Two useful considerations are target-data size and source/target similarity:

- **Little data and similar domains:** begin with a frozen backbone and a new
  linear classifier.
- **More data and similar domains:** fine-tune more or all layers.
- **Little data and very different domains:** transferred features may be a poor
  match; try a more suitable pretrained model or obtain more data.
- **More data and different domains:** compare fine-tuning with training from
  scratch rather than assuming either must win.

These are starting heuristics, not guarantees. Validation experiments should
decide the final strategy.

## Practical details

- Match the pretrained model's expected input size, channel order, and
  normalization statistics.
- Use a smaller learning rate for pretrained parameters than for the new head.
- When freezing a backbone, decide deliberately whether normalization layers
  should also keep fixed running statistics.
- Save the best validation checkpoint rather than only the final epoch.
- Never choose the strategy using test-set performance.

## Further reading

- [DeCAF: A Deep Convolutional Activation Feature for Generic Visual Recognition](https://arxiv.org/abs/1310.1531)
- [CNN Features Off-the-Shelf](https://arxiv.org/abs/1403.6382)

