# Ecological embodied AI and evaluation

Many robot-learning demonstrations use short tasks, carefully selected objects, fixed camera views, and uncluttered environments. These setups isolate technical questions, but success may not transfer to everyday life.

## What ecological complexity adds

Real environments are:

- visually cluttered and physically diverse;
- dynamic and uncertain;
- interactive and social;
- composed of long-horizon, multitask activities;
- shaped by individual human preferences and needs.

A household instruction such as “clean the table” may require finding objects, choosing an order of operations, remembering earlier actions, handling deformable materials, and recovering from unexpected outcomes.

## Open instructions and structured reasoning

Language and vision models can help robots interpret open-ended instructions and unfamiliar objects. Systems may translate language into intermediate constraints, affordance maps, value maps, or planning code rather than predicting low-level motion in one step.

Such structure can improve compositional generalization, but generated constraints can still be wrong. Physical execution therefore needs perception updates, safety checks, and closed-loop replanning.

## Ecological benchmarks and simulation

Benchmarks such as BEHAVIOR aim to evaluate many everyday activities across diverse interactive environments. Simulation offers:

- faster and safer training;
- repeatable experiments;
- standardized tasks and metrics;
- access to privileged state for diagnosis.

Realistic physics for fluids, deformable objects, transparency, contact, and human interaction remains difficult. Strong simulated results do not remove the need to study the sim-to-real gap.

Long-horizon performance also depends on memory. Action primitives may simplify control, while memory prevents the agent from repeatedly searching, undoing progress, or forgetting earlier subgoals. Evaluation should state which privileged information, action abstractions, and memory mechanisms are available.

## Human-centered task selection

A benchmark should not contain tasks only because they are convenient to simulate. Human-centered design begins by asking which activities people actually want help with, then builds diverse environments and metrics around those needs.

Related notes:

- [Robot perception: embodied, active, and situated](../17-robot-learning/robot-perception-embodied-active-and-situated.md)
- [Robot-learning evaluation and sim-to-real](../17-robot-learning/robot-learning-evaluation-and-sim-to-real.md)
- [Visual goal-conditioned policies and VLA intuition](../17-robot-learning/visual-goal-conditioned-policies-and-vla.md)
