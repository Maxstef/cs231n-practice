# Reading training and validation curves

Training curves are diagnostic tools. They help distinguish optimization
problems from generalization problems, but they should be interpreted together
with the absolute metric values and the training setup.

## What the two curves measure

- **Training performance** measures the data used to update the parameters.
- **Validation performance** estimates behavior on unseen examples and is used
  for model selection.

Training accuracy is normally at least as high as validation accuracy because
the optimizer directly minimizes training loss. The difference between them is
often called the **generalization gap**.

## Large and growing gap

If training performance keeps improving while validation performance stops
improving or becomes worse, the model is overfitting. It is fitting details of
the training set that do not transfer well to unseen examples.

Possible responses include:

- stronger regularization or dropout;
- more training data or suitable data augmentation;
- a smaller model;
- early stopping at the best validation checkpoint;
- checking for train/validation distribution differences.

The best response is not always simply “more regularization.” A data bug,
duplicate examples, label noise, or a mismatched validation distribution can
produce a misleading gap.

## Small gap with poor performance

If training and validation performance are both poor and close together, the
model is underfitting or has not yet been optimized successfully. Possible
causes include insufficient training, a poor learning rate, excessive
regularization, or insufficient model capacity.

Possible responses include:

- train longer if the curves are still improving;
- improve the optimizer or learning-rate schedule;
- reduce excessive regularization;
- use a model with greater capacity.

A small gap alone does **not** prove underfitting. If both curves have strong
performance, a small gap is desirable.

## Other common patterns

- **Loss does not fall:** suspect the implementation, data preprocessing,
  initialization, or learning rate.
- **Loss oscillates or explodes:** the learning rate may be too large, although
  unstable data or gradients can also cause this.
- **Training improves very slowly:** the learning rate may be too small, or the
  model may be difficult to optimize.
- **Validation is briefly better than training:** this can happen because
  dropout and augmentation make training examples harder, while evaluation uses
  deterministic inference behavior.

## Compare curves fairly

Use the same evaluation definition for both splits. For example, measure both
accuracies with dropout disabled and normalization layers in evaluation mode.
If training accuracy is measured on randomly augmented inputs but validation
accuracy on clean inputs, their gap mixes generalization with different input
difficulty.

Select checkpoints and hyperparameters with validation data. Keep the test set
untouched until the final evaluation.

## Related notes

- [Regularization](../03-linear-classifiers/regularization.md)
- [Hyperparameter tuning and learning rates](hyperparameter-tuning-and-learning-rates.md)
- [Dropout](../05-neural-networks/dropout.md)

