# Visual goal-conditioned policies and VLA intuition

A policy does not always receive a single fixed task. A **goal-conditioned policy** also receives a description of the desired outcome:

$$
a_t \sim \pi(a_t \mid o_t, g).
$$

Here, $o_t$ is the current observation and $g$ is the goal. The same observation may require different actions for different goals, so the goal is part of the policy input rather than merely part of the evaluation.

## Ways to represent a goal

- A goal image shows what the desired visual state should look like.
- Coordinates or another structured vector describe a known state directly.
- A language instruction such as “go to the top-right corner” describes the task symbolically.

A visual goal and a language instruction can carry similar task information in different representations. The model must learn how that information should affect its action.

## Vision-language-action models

A **vision-language-action (VLA)** model combines:

1. vision, describing the current scene;
2. language, describing the requested task;
3. actions, controlling the robot.

A simplified pipeline is:

$$
\text{image} \rightarrow \text{visual features},
\qquad
\text{instruction} \rightarrow \text{text features},
\qquad
(\text{visual features},\text{text features}) \rightarrow \text{action}.
$$

Large systems may begin with pretrained vision-language representations and then learn robot actions from demonstrations. The notebook uses a tiny learned instruction embedding instead; this preserves the conditioning idea without pretending to reproduce a large VLA system.

## Why closed-loop evaluation matters

Action accuracy on a held-out dataset is useful, but a robot executes predictions repeatedly. One mistake changes the next observation and may move the policy into an unfamiliar state. A policy can therefore have good one-step accuracy but poor task success.

Evaluate both:

- **offline action accuracy** on labeled state-action pairs;
- **closed-loop success**, where the policy must repeatedly act until it reaches the goal or the time limit.

Failed rollouts are also useful diagnostics. A policy may repeatedly choose an action blocked by an obstacle, remain in one cell, or enter a loop even when most of its individual predictions are correct.

## Practice connection

[Notebook 72 — Visual goal-conditioned policies and VLA intuition](../../notebooks/72_visual_goal_conditioned_policies_and_vla_intuition.ipynb) compares two small CNN policies:

- one receives obstacle, agent, and goal image channels;
- the other receives the scene without the goal channel plus a learned instruction embedding;
- both predict one of four discrete moves and are evaluated with one-step accuracy and closed-loop rollouts.

Using a different obstacle layout for each goal makes navigation more varied, but it also creates a shortcut: the layout itself can reveal the requested task. A cleaner conditioning test would reuse layouts across goals or randomize layouts independently of the goal.

## Scope and limitations

The grid-world exercise illustrates conditional control, not realistic robotics. Real VLA systems must also handle continuous actions, diverse cameras and embodiments, imperfect language, long tasks, safety constraints, and distribution shift.

Further reading:

- [RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control](https://robotics-transformer2.github.io/)
- [OpenVLA: An Open-Source Vision-Language-Action Model](https://openvla.github.io/)

