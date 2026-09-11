# Visualizing and understanding vision models

Interpretability methods ask what representations a model learned and what
input evidence affected a particular prediction. Different visualizations
answer different questions; a bright region should not automatically be read
as a causal explanation or an object segmentation mask.

## First-layer filters

An RGB convolutional filter has shape $(3,K_h,K_w)$ and can be displayed as a
small color image after normalization. First layers often learn interpretable
color contrasts, oriented edges, and frequency patterns because they operate
directly on image pixels.

Deeper filters cannot usually be displayed directly as RGB images: their input
channels refer to learned features rather than red, green, and blue. Feature
maps, activation maximization, or attribution methods are more appropriate at
those layers.

## Feature-map activations

For an activation tensor $(C,H,W)$, displaying individual channels shows where
particular learned features respond. Early layers tend to retain detailed
spatial patterns; deeper layers tend to be more semantic and spatially coarse.

An activation indicates that a feature responded at a location. It does not by
itself establish that the feature caused the final class decision.

Vision Transformer patch-token features can also be reshaped back to their
patch grid for visualization. The resulting grid is coarse at patch
resolution and must not be confused with pixel-level segmentation.

## Input-gradient saliency

For class score $S_c(x)$ and input image $x$, ordinary saliency computes

$$
G=\frac{\partial S_c}{\partial x}.
$$

The magnitude $|G|$ estimates how sensitive the class score is to a small
change in each input value. An image-like heatmap can take the maximum absolute
gradient across RGB channels:

$$
M_{h,w}=\max_k |G_{k,h,w}|.
$$

Saliency is local: it describes derivatives near the current input. It can be
noisy, can change under small perturbations, and does not mean that changing a
highlighted pixel in an arbitrary way will necessarily improve the score.
Using the unnormalized class logit avoids gradient interactions introduced by
softmax probabilities.

## Class Activation Mapping

CAM applies to architectures where the last convolutional features
$A_{k,h,w}$ are globally averaged and passed directly to a linear class layer.
For class weights $W_{c,k}$,

$$
M^c_{h,w}=\sum_k W_{c,k}A_{k,h,w}.
$$

The map shows which spatial positions contribute through class-weighted
channels. CAM is efficient and class-specific, but its direct form depends on
this global-average-pooling architecture and normally uses the last
convolutional layer.

## Grad-CAM

Grad-CAM generalizes the channel-weighting idea. Choose a convolutional layer
with activations $A_{k,h,w}$, differentiate class score $S_c$ with respect to
them, and average each channel's gradients spatially:

$$
\alpha_k^c=\frac{1}{HW}\sum_{h,w}
\frac{\partial S_c}{\partial A_{k,h,w}}.
$$

Then form

$$
M^c_{h,w}=\mathrm{ReLU}\left(\sum_k\alpha_k^c A_{k,h,w}\right).
$$

The gradient-derived $\alpha_k^c$ estimates how important activation channel
$k$ is for class $c$. ReLU retains evidence that supports the class. The map
is class-specific and can be computed at different convolutional layers:
earlier layers provide finer spatial resolution, while later layers generally
provide stronger semantic abstraction.

## Do not confuse interpretation with segmentation

| Output | Meaning |
| --- | --- |
| Segmentation mask | A trained pixel-level class or instance prediction |
| Saliency map | Local sensitivity of a score to input pixels |
| CAM / Grad-CAM | Coarse class-related evidence in a feature layer |
| Feature activation | Where one learned channel responds |

A heatmap may emphasize only the most discriminative part of an object, use
context outside the object, or omit pixels unnecessary for classification.
Interpretation methods are diagnostic tools, not guaranteed explanations of
the full decision process.

## Review questions

1. Why are first-layer CNN filters easier to display than deep filters?
2. What does a large input-gradient magnitude mean locally?
3. Which architectural restriction does direct CAM impose?
4. Where do Grad-CAM channel weights come from?
5. What trade-off appears when choosing an early or late Grad-CAM layer?
6. Why is a Grad-CAM heatmap not an object segmentation mask?

## Related practice

- Notebook 44: visualizing and understanding networks
- Notebook 36: Vision Transformer intuition

## Source

- Stanford CS231n Spring 2025, Lecture 9: Object Detection, Image
  Segmentation, Visualizing and Understanding, available from the
  [course schedule](https://cs231n.stanford.edu/2025/schedule.html).
