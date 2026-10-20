# Learning from 3D data

Neural architectures for 3D data are strongly shaped by the input
representation. A regular voxel grid, an unordered point set, and several 2D
views of one object require different invariances and computational choices.

## Common datasets and tasks

3D datasets may contain synthetic object models, videos around individual
objects, or scans of complete rooms. Examples mentioned in the lecture include
ShapeNet, Objaverse, CO3D, and ScanNet.

They support different supervision:

- object classification and retrieval;
- object-part segmentation;
- semantic or instance segmentation of scenes;
- reconstruction and novel-view synthesis;
- conditional or unconditional 3D generation.

Synthetic objects offer scale and clean geometry. Real captures contain
lighting, missing observations, sensor noise, clutter, and uncertain camera
poses.

## Multi-view CNNs

One way to avoid processing 3D data directly is to render an object from
several camera viewpoints:

```text
3D object -> several 2D images -> shared CNN -> combine view features -> prediction
```

This reuses strong 2D CNNs and pretrained image features. Combining the views
must preserve useful evidence while becoming insensitive to their ordering.
The representation can still miss geometry that is invisible from the chosen
views.

## Voxel CNNs and octrees

A voxel CNN applies 3D convolutions to a regular volume. This is conceptually
close to image CNNs, but dense memory and computation grow approximately with
the cube of spatial resolution.

An **octree** recursively subdivides only relevant parts of 3D space. Large
empty regions remain coarse while regions near surfaces receive smaller cells.
This can provide much higher effective resolution than a dense voxel grid for
the same memory, at the cost of irregular data structures and operations.

## PointNet intuition

A point cloud is a set, so changing point order must not change the result.
PointNet follows a simple pattern:

```text
each point -> shared feature network -> symmetric aggregation -> global feature
```

In compact notation,

$$
F(\{x_i\})=\gamma\left(\max_i h(x_i)\right).
$$

- $h$ applies the same learned transformation to every point;
- coordinate-wise maximum is unchanged by point permutation;
- $\gamma$ converts the aggregated feature into the desired output.

The max operation can discover whether a useful geometric feature occurs
somewhere in the set. Permutation invariance is built in. Robustness to
different sampling densities is related but not automatically guaranteed.

## Local point-cloud models

Global aggregation alone can lose local structure. A point cloud can instead
be treated as a graph:

- points become nodes;
- spatial neighbors define edges;
- messages combine information from nearby points.

This captures local surfaces and parts, but the graph must be constructed and
may change as the representation changes.

## Comparing point sets

Two generated point clouds rarely contain corresponding points in the same
order. A direct point-by-point loss is therefore inappropriate.

**Chamfer distance** finds a nearest neighbor in the other set for every point,
in both directions. It is relatively simple but can match several points to the
same neighbor.

**Earth Mover's Distance** searches for a one-to-one transport between equally
sized sets. It captures global matching more directly but is more expensive.

Both compare sampled point sets rather than guaranteeing that the underlying
continuous surfaces are identical.

## Main intuition

> The architecture should respect the structure and invariances of the chosen
> representation.

Images have a regular grid, point clouds are unordered sets, meshes and local
point neighborhoods resemble graphs, and sparse volumes benefit from spatial
hierarchies such as octrees.

## Source

- Stanford CS231n Spring 2025, Lecture 15: 3D Vision, available from the
  [course schedule](https://cs231n.stanford.edu/2025/schedule.html).
