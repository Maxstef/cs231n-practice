# Video representations, sampling, and fusion

A video adds a temporal axis to an image. A common batch layout is
`(N, T, C, H, W)`: batch, time, channels, height, and width. PyTorch
`Conv3d` instead expects `(N, C, T, H, W)`, so changing the axis order is often
necessary.

## Sampling clips

Long videos are usually processed as shorter clips. For a clip containing $T$
sampled frames with temporal stride $s$, the covered source-frame interval is

$$
1+(T-1)s.
$$

A larger stride covers more time but skips more intermediate motion. During
training, random starting positions provide temporal augmentation. During
evaluation, several uniformly spaced clips can be averaged for more stable
predictions.

## Ways to combine frames

| Method | Basic idea | Main limitation |
| --- | --- | --- |
| Single frame | Classify one frame | Cannot observe motion |
| Temporal mean | Average frames or frame features | Discards temporal order |
| Early fusion | Combine frames before or near the first layers | Must learn appearance and motion together |
| Late fusion | Extract each frame's features, then combine them | Low-level motion may already be lost |
| Score fusion | Average predictions from clips or streams | Only combines final decisions |

Concatenating frame features preserves their order, whereas averaging produces
the same result after any permutation of the frames. Pooling is compact and can
handle varying clip lengths, but it cannot distinguish actions that differ only
by direction or ordering.

## Key takeaway

Video representation is a trade-off between temporal detail, computation, and
invariance. Before choosing a model, ask whether the label depends mostly on
appearance, short motion, or the order of events over a longer interval.

## Related practice

- Notebook 45: video tensors, sampling, and fusion
- Notebook 49: comparison of several video classifiers
- `cs231n_practice/video.py`

## Source

- Stanford CS231n Spring 2025, Lecture 10: Video Understanding, available from
  the [course schedule](https://cs231n.stanford.edu/2025/schedule.html).
