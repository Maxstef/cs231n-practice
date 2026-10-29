# Robot perception: embodied, active, and situated

In ordinary image classification, a model receives an image and returns a prediction. A robot instead perceives while acting inside a changing physical environment. The lecture summarizes this distinction by describing robot perception as **embodied, active, and environmentally situated**.

## Embodied

The sensing system belongs to a physical body. Camera position, joint limits, reachable space, collisions, and the robot's previous motion all affect what it can observe and do. The same visual prediction can therefore have different meanings for two robots with different bodies or capabilities.

## Active

A robot can act to improve its future observations. It may move closer, change its viewpoint, turn an object, or touch it to reduce uncertainty. Perception is therefore part of a loop:

$$
\text{observe} \rightarrow \text{decide} \rightarrow \text{act} \rightarrow \text{observe again}.
$$

This differs from passively processing a fixed collection of images.

## Environmentally situated

What matters depends on the current task and environment. A mug's category may be enough for image recognition, but a robot may also need its pose, whether it is full, whether its handle is reachable, and what obstacles surround it. Time, context, and consequences are part of the perception problem.

## State is not always observation

The complete physical **state** might contain object poses, velocities, friction, and hidden geometry. The robot's **observation** may contain only camera pixels, depth measurements, proprioception, or language instructions. When the observation is incomplete, useful decisions may require memory or repeated sensing.

## Consequences for learning systems

Robot perception commonly needs:

- temporal and multimodal information rather than one isolated image;
- calibrated geometry and coordinate frames;
- estimates of uncertainty;
- robustness to viewpoint, lighting, and environment changes;
- low latency and safe fallback behavior;
- evaluation through closed-loop task success.

[Notebook 69](../../notebooks/69_agent_environment_and_tabular_rl.ipynb) begins with a fully observed grid world so that the agent–environment loop is explicit. Later practice can replace the simple grid state with partial visual observations and learned representations.

## Resources

- [CS231N Spring 2025 schedule](https://cs231n.stanford.edu/2025/schedule.html)
- [Agent–environment interaction and reinforcement learning](agent-environment-mdp-and-reinforcement-learning.md)
