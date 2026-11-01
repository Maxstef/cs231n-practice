# Inverse reinforcement learning and implicit behavior cloning

Behavioral cloning directly learns to reproduce expert actions. Inverse reinforcement learning and implicit behavior cloning use demonstrations differently and address different limitations of direct prediction.

## Inverse reinforcement learning

Ordinary reinforcement learning starts with a reward and searches for a policy that maximizes it:

$$
\text{reward} \longrightarrow \text{policy}.
$$

Inverse reinforcement learning (IRL) begins with expert demonstrations and tries to infer a reward function that could explain the demonstrated behavior:

$$
\text{expert behavior} \longrightarrow \text{inferred reward} \longrightarrow \text{policy}.
$$

For example, a robot may demonstrate routes that reach a destination, avoid obstacles, and remain short. Rather than copying each turn, IRL attempts to recover an objective that assigns high value to those properties. A policy can then optimize the inferred objective in situations not present in the demonstrations.

This introduces important difficulties:

- many different reward functions can explain the same demonstrations;
- the inferred reward reflects what is observable in the data, not necessarily the expert's complete intention;
- after learning a reward, the system usually still needs reinforcement learning or planning to obtain a policy;
- optimization and environment interaction can be expensive.

IRL is therefore not simply behavioral cloning with another loss. It learns an objective first and behavior second.

## Implicit behavior cloning

An explicit behavioral-cloning policy directly predicts an action:

$$
a=\pi(o).
$$

Here $o$ is an observation and $a$ is an action. Direct regression can struggle when several very different actions are valid. If demonstrations avoid an obstacle sometimes on the left and sometimes on the right, averaging the two action targets could point toward the obstacle.

Implicit behavior cloning can instead learn an energy or compatibility function:

$$
E(o,a).
$$

Low energy means that an action is compatible with the observation. Action selection searches for a low-energy candidate:

$$
a^*=\arg\min_a E(o,a).
$$

The model can assign low energy to multiple distinct valid actions without averaging them into one intermediate action. This can be useful for continuous, multimodal robot controls.

The trade-off is inference cost. An explicit policy can produce an action in one forward pass, while an implicit policy commonly needs sampling, search, or iterative optimization over candidate actions.

## Relationship to behavioral cloning and DAgger

- **Behavioral cloning** directly predicts expert actions from expert-visited observations.
- **DAgger** collects expert action labels for states visited by the learner and then continues direct action prediction.
- **Inverse reinforcement learning** infers an objective that may explain the expert's behavior and then optimizes a policy for it.
- **Implicit behavior cloning** still imitates expert actions, but represents their compatibility through an energy function rather than one direct prediction.

IRL and implicit behavior cloning are left conceptual in this project. A meaningful implementation would add either a reward-learning and policy-optimization loop or continuous-action energy-based inference, substantially expanding the scope beyond the small tabular experiment in [Notebook 70](../../notebooks/70_imitation_learning_and_covariate_shift.ipynb).

## Resources

- [Algorithms for Inverse Reinforcement Learning](https://ai.stanford.edu/~ang/papers/icml00-irl.pdf)
- [Implicit Behavioral Cloning](https://arxiv.org/abs/2109.00137)
