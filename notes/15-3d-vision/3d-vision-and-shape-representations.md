# 3D vision and shape representations

3D vision asks models to reason about objects and scenes in space rather than
only about their 2D projections. Typical tasks include shape classification,
part and scene segmentation, reconstruction from images, novel-view
rendering, editing, and generation.

The first important lesson is that there is no single natural 3D equivalent
of an image tensor. The same chair can be stored as points, triangles, occupied
grid cells, or a function over continuous coordinates. The best representation
depends on what we want to compute.

## Explicit and implicit descriptions

An **explicit representation** directly lists samples or surface elements:

- **Point cloud:** a set of 3D coordinates, sometimes with color or a surface
  normal at each point.
- **Polygon mesh:** vertices connected into faces, usually triangles.
- **Parametric surface:** a function maps parameters such as $(u,v)$ directly
  to a point $(x,y,z)$ on the surface.

Explicit surfaces are usually easy to sample and render. For example, a
parametric surface gives a surface point simply by choosing $(u,v)$. But a
question such as “is this arbitrary 3D point inside the object?” may be hard.

An **implicit representation** defines a function over space and identifies
the surface through a condition such as

$$
f(x,y,z)=0.
$$

The function may instead represent occupancy or signed distance. Inside/outside
queries can then be easy: inspect the function value at the point. Extracting
an explicit surface for rendering may require an additional algorithm.

This gives the central trade-off:

| Representation | Easy operations | Common difficulties |
| --- | --- | --- |
| Point cloud | Store scanner samples; process sparse observations | No connectivity or guaranteed surface |
| Mesh | Render surfaces; preserve local connectivity | Irregular topology; editing connectivity |
| Parametric surface | Sample surface points directly | Complex topology may need many patches |
| Voxel grid | Apply ordinary 3D convolutions; test occupancy | Memory grows cubically with resolution |
| Implicit field | Continuous queries; inside/outside and smooth surfaces | Surface extraction and rendering can be expensive |

## Point clouds

A point cloud is an unordered set

$$
\{(x_i,y_i,z_i)\}_{i=1}^{N}.
$$

It is simple, flexible, and often produced directly by depth sensors or 3D
scanners. Points with oriented surface normals are sometimes called
**surfels**.

A point cloud does not say which points are connected. Sparse regions may
contain holes, and changing the sampling density changes the input even when
the underlying surface is unchanged.

## Polygon meshes

A mesh represents the boundary using vertices and faces. Connectivity makes
neighborhoods and surfaces explicit. Common operations include:

- **subdivision:** add vertices and interpolate to obtain a smoother or
  higher-resolution mesh;
- **simplification:** use fewer faces while preserving overall shape;
- **regularization:** redistribute elements to improve mesh quality.

The object has not necessarily changed when its mesh resolution changes. The
representation has changed.

## Voxels and level sets

A voxel grid is the 3D analogue of a pixel grid. Each cell stores occupancy or
another feature. It is regular and convenient for convolutions, but doubling
resolution along each axis produces roughly eight times as many cells.

A level set stores values on a grid and treats an interpolated zero crossing
as the surface. It supports smooth changes, splitting, and merging more
naturally than a fixed mesh topology.

## Parametric and implicit are different axes

Two distinctions are easy to mix up:

- **explicit versus implicit:** is geometry listed directly, or defined by a
  field/constraint?
- **parametric versus non-parametric:** is it described by a fixed function
  with parameters, or stored as many individual elements?

A mesh and a point cloud are explicit and non-parametric. A spline surface is
explicit and parametric. A voxel occupancy grid is implicit and
non-parametric. An algebraic surface or a learned coordinate network is
implicit and parametric.

## Main intuition

> A 3D representation is an interface: it makes some questions cheap and
> other questions difficult.

Before choosing a model, ask whether the task primarily needs surface
sampling, rendering, inside/outside queries, local connectivity, memory
efficiency, editing, or semantic structure.

## Source

- Stanford CS231n Spring 2025, Lecture 15: 3D Vision, available from the
  [course schedule](https://cs231n.stanford.edu/2025/schedule.html).
