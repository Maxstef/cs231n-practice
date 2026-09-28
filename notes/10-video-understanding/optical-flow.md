# Measuring motion with optical flow

**Optical flow** is a dense displacement field between two frames. For every
image location $(x,y)$, it estimates a horizontal and vertical displacement:

$$
F(x,y)=(u(x,y),v(x,y)).
$$

Here, $u$ says how far the visual point appears to move horizontally and $v$
says how far it appears to move vertically. The result has two values per
location, unlike an RGB frame's three color values.

## Brightness-constancy intuition

A basic assumption is that a visible point keeps approximately the same
appearance while moving:

$$
I_t(x,y)\approx I_{t+1}(x+u, y+v).
$$

The algorithm searches for a displacement that makes corresponding regions
agree. In practice, flow estimation also needs spatial smoothness or learned
priors because one pixel alone does not provide enough information to determine
two displacement values reliably.

## Reading a flow field

- direction is represented by the pair $(u,v)$;
- magnitude is $\sqrt{u^2+v^2}$;
- zero flow means no estimated image-plane displacement;
- separate visualizations may show horizontal flow $u$ and vertical flow $v$;
- color-wheel visualizations encode direction as color and magnitude as
  saturation or brightness.

Optical flow describes **apparent image motion**, not necessarily physical
object motion. A moving camera can produce flow over the entire frame, and
motion directly toward the camera may appear as expansion rather than a simple
translation.

## Optical flow versus frame differences

| Frame difference | Optical flow |
| --- | --- |
| Measures pixel-value change | Estimates displacement |
| Simple and inexpensive | More computationally demanding |
| Does not directly encode direction | Explicitly represents direction and magnitude |
| Sensitive to lighting and camera changes | Also difficult under blur, occlusion, and ambiguous texture |

Classical optical flow is computed before the recognition model. Learned flow
estimators and modern video networks can instead learn motion representations
from data. Optical flow is therefore one useful motion input, not the only way
to understand motion.

## Related practice

- Notebook 47: frame differences and optical-flow intuition
- [Motion and two-stream networks](motion-and-two-stream-networks.md)

## Source

- Stanford CS231n Spring 2025, Lecture 10: Video Understanding, available from
  the [course schedule](https://cs231n.stanford.edu/2025/schedule.html).
