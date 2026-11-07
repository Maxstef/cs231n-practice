# Camera geometry basics

A camera converts points in a 3D scene into positions on a 2D image. A useful simplified pipeline is:

```text
world coordinates -> camera coordinates -> normalized image plane -> pixels
```

## Coordinate frames and extrinsics

A 3D point first has to be expressed relative to the camera. Rotation $R$ and translation $t$ perform this world-to-camera transformation:

$$
X_{camera}=R X_{world}+t.
$$

These are the camera **extrinsic parameters**. They describe the camera pose relative to the world. Coordinate-axis conventions differ between libraries, so the meanings of forward, up, and the signs of axes must always be checked.

## Orthographic and perspective projection

For a camera-space point $(x,y,z)$, orthographic projection ignores depth:

$$
(u,v)=(x,y).
$$

Perspective projection divides by depth:

$$
u=f\frac{x}{z},
\qquad
v=f\frac{y}{z},
$$

where $f$ is focal length and $z>0$ is depth. This division is why distant objects appear smaller. For example, with $f=2$:

$$
(1,1,2)\mapsto(1,1),
\qquad
(1,1,4)\mapsto(0.5,0.5).
$$

## Intrinsic parameters

Normalized image coordinates must be converted to pixel coordinates. The camera **intrinsic matrix** is commonly written as

$$
K=
\begin{bmatrix}
f_x & 0 & c_x \\
0 & f_y & c_y \\
0 & 0 & 1
\end{bmatrix}.
$$

- $f_x,f_y$ are focal lengths measured in pixel units;
- $(c_x,c_y)$ is the principal point, usually near the image center.

Real camera models may also include skew and lens distortion.

## One compact projection equation

Using homogeneous coordinates, the complete mapping is often summarized as

$$
\tilde{x}_{pixel}\sim K[R\mid t]\tilde{X}_{world}.
$$

The symbol $\sim$ means equality up to a nonzero scale. After multiplying, divide the first two coordinates by the third to obtain ordinary pixel coordinates. Homogeneous coordinates make rotation, translation, and projection expressible in one matrix pipeline.

## Calibration, pose, and visibility

- **Camera calibration** estimates intrinsic parameters and often lens distortion.
- **Pose estimation** determines the extrinsic relationship between camera and world.
- **Projection** says where a point would appear, but not whether it is visible. Occlusion, clipping, and surface ordering require additional reasoning or rendering.

Multi-view geometry uses corresponding observations from several calibrated or partially calibrated cameras to infer camera motion and 3D structure. That broader topic includes epipolar geometry, triangulation, and structure from motion, which are outside this project's current practice scope.

## Practice connection

[Notebook 63 — 3D representations and geometric intuition](../../notebooks/63_3d_representations_and_geometric_intuition.ipynb) compares orthographic and perspective projection while assuming the points are already in camera coordinates. It intentionally omits calibration, extrinsics, distortion, and visibility.

## Source

- Stanford CS231n Spring 2025, Lecture 15: 3D Vision, available from the [course schedule](https://cs231n.stanford.edu/2025/schedule.html).
