# Robotic foundation models and cross-embodiment data

A robotic foundation model aims to support many tasks, environments, and robot bodies rather than one narrowly trained controller. A common interface is a goal-conditioned policy:

$$
a_t \sim \pi(a_t\mid o_t,g).
$$

The observation may contain camera images and proprioception, while the goal may be a language instruction or another representation of the desired outcome.

## Why cross-embodiment data matters

Robot data is fragmented across arms, grippers, cameras, action spaces, and laboratories. **Cross-embodiment training** combines experience from different robot bodies so that shared visual and semantic structure can transfer between them.

This requires some way to reconcile embodiment-specific inputs and actions. The model may share a common backbone while using robot-specific action representations, adapters, or metadata. More varied data can improve generalization, but incompatible sensors and controls make naive dataset merging difficult.

## Pretraining and post-training

- Broad pretraining provides reusable visual, language, and behavioral representations.
- Robot demonstrations connect those representations to physical actions.
- Specialized post-training adapts the general model to difficult tasks, new hardware, or higher precision requirements.

Systems such as RT-X, OpenVLA, and $\pi_0$ illustrate this direction. They are often called **vision-language-action models** or **large behavior models**, although terminology is still evolving.

Related note: [Visual goal-conditioned policies and VLA intuition](visual-goal-conditioned-policies-and-vla.md).

