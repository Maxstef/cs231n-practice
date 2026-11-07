# Explicit, implicit, and diffusion policies

A policy must convert an observation and possibly a goal into an action. Several model families differ mainly in how that action is represented and selected.

## Explicit policy

An explicit policy directly predicts an action or an action distribution:

$$
\hat{a}=f_\theta(o,g).
$$

It is simple and fast, but a single regression output may average several valid actions and produce an action that matches none of them well.

## Implicit policy

An implicit policy learns an energy or compatibility score for an observation–action pair. At inference time it searches for an action with low energy:

$$
\hat{a}=\arg\min_a E_\theta(o,g,a).
$$

This can represent several distinct valid actions, but selecting an action requires an optimization or sampling procedure.

## Diffusion policy

A diffusion policy begins with noisy action samples or action sequences and iteratively denoises them while conditioning on the observation and goal. This is useful when robot behavior is multimodal: several different trajectories may all complete the task.

The trade-off is that iterative generation is slower than one direct forward pass. Diffusion changes how actions are generated; it does not remove the need for representative demonstrations, closed-loop evaluation, or safety mechanisms.

Related note: [Inverse reinforcement learning and implicit behavior cloning](inverse-reinforcement-learning-and-implicit-behavior-cloning.md).

