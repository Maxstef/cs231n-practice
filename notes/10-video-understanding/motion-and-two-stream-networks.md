# Motion and two-stream networks

Appearance and motion provide complementary evidence. A single frame may show
*what* is present, while changes across frames show *what it is doing*.

## Simple and explicit motion representations

A frame difference is

$$
D_t=I_{t+1}-I_t.
$$

It is easy to compute and highlights change, but it mixes object motion,
camera motion, lighting changes, and noise.

**Optical flow** estimates a two-dimensional displacement $(u,v)$ for image
locations between frames. It describes the direction and magnitude of apparent
motion more explicitly. Optical flow is an estimated representation, however,
not ground truth; occlusion, blur, and textureless regions make it difficult.

## Two-stream networks

A two-stream model separates the inputs:

```text
RGB frames --------> spatial stream  ---\
                                         +--> fused prediction
differences/flow --> temporal stream ---/
```

The spatial stream learns objects, scenes, and pose. The temporal stream learns
movement patterns. Fusion may combine feature vectors or final class scores.
The streams are useful precisely because they receive different evidence; two
identical streams would offer much less benefit.

The lecture results show that combining spatial and temporal streams can
outperform either one alone. This is an empirical result, not a rule that every
dataset requires optical flow. Modern end-to-end models can also learn motion
features directly from RGB clips.

## Practice observation

In our synthetic experiment, temporal differences aligned closely with the
motion-based labels, so the two-stream model had a particularly helpful input.
That result should not be generalized to real videos, where backgrounds,
camera motion, and ambiguous actions make the problem harder.

## Related practice

- Notebook 47: frame differences, motion, and two-stream models
- Notebook 49: comparative video-classification experiment
- `cs231n_practice/video.py`
- `cs231n_practice/classifiers/video.py`

## Source

- Stanford CS231n Spring 2025, Lecture 10: Video Understanding, available from
  the [course schedule](https://cs231n.stanford.edu/2025/schedule.html).
