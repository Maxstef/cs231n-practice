# Neural fields, NeRF, and Gaussian splatting

Neural rendering connects a 3D scene representation with the 2D images seen by
cameras. If the rendering process is differentiable, disagreement between a
rendered image and a real photograph can train the underlying 3D
representation.

## Learned implicit fields

Instead of storing one value in every voxel, a neural network can answer a
query at a continuous coordinate:

$$
(x,y,z) \longmapsto \text{occupancy, distance, density, color, or features}.
$$

The network parameters compactly define the field. Resolution is no longer
fixed by a voxel grid, although evaluating the field at many positions may be
slow.

An occupancy network predicts whether a location lies inside an object. A
signed-distance network predicts distance to the surface, with the sign
indicating inside or outside. The surface is obtained from a level set such as
$f(x,y,z)=0$.

## NeRF intuition

A Neural Radiance Field represents a scene using a coordinate-based network.
Conceptually, it answers:

```text
3D position + viewing direction -> density + view-dependent color
```

For each output pixel, the camera casts a ray through the scene. The model is
queried at multiple positions along the ray. Volume rendering combines the
predicted colors according to their densities and visibility.

The important visibility intuition is:

- a dense point can contribute color;
- dense material earlier on the ray blocks contributions behind it;
- empty regions contribute almost nothing.

A simplified rendered color is a weighted sum:

$$
C(r)\approx\sum_i w_i c_i,
$$

where $c_i$ is the color at sample $i$. The weight $w_i$ depends both on that
sample's opacity and on how much light has survived all earlier samples.

Because this weighted composition is differentiable, errors in rendered pixels
can update the neural field. Training from multiple known camera views
encourages one 3D field to explain all images consistently.

NeRF does not normally store an explicit triangle surface. It densely
parameterizes a continuous scene field and may require many network queries
per ray, making rendering expensive.

## Gaussian splatting intuition

3D Gaussian splatting uses an explicit collection of soft 3D blobs. Each
Gaussian has properties such as:

- a 3D position;
- size, orientation, and opacity;
- color or view-dependent appearance.

The Gaussians are projected into the image and blended in depth order. Unlike a
dense neural field, computation is concentrated around the stored primitives.
This often enables much faster rendering.

The high-level contrast is:

| NeRF-style field | 3D Gaussian splatting |
| --- | --- |
| Continuous implicit neural function | Explicit collection of soft primitives |
| Query many samples along each ray | Project and blend relevant Gaussians |
| Compact network representation | Potentially many stored Gaussians |
| Rendering can be slow | Rendering is often real-time or interactive |

Neither representation is universally better. Quality, training time, memory,
editing, and rendering speed can favor different choices.

## Relation to earlier topics

This is another example of learning through a differentiable computation
graph:

```text
scene parameters -> renderer -> predicted image -> image loss
       ^                                             |
       +---------------- backpropagation ------------+
```

The renderer connects 2D supervision to 3D parameters, much as a decoder
connects a latent representation to reconstructed pixels.

## Main intuition

> Neural rendering learns a 3D scene by requiring it to reproduce observations
> from multiple viewpoints.

NeRF represents the scene as a dense queryable field. Gaussian splatting
represents it as sparse, explicit primitives optimized for efficient
projection and blending.

## Source

- Stanford CS231n Spring 2025, Lecture 15: 3D Vision, available from the
  [course schedule](https://cs231n.stanford.edu/2025/schedule.html).
