# Reinforcement learning versus supervised learning

Supervised learning trains on fixed input–target pairs. Reinforcement learning instead learns by acting in an environment and observing later rewards and states. The agent therefore influences the data it will receive next.

## Four important differences

- **Stochasticity:** the same action may produce different transitions or rewards.
- **Credit assignment:** a reward received later may depend on several earlier actions, so it is unclear which action deserves credit.
- **No direct world gradient:** a physical environment is not normally a differentiable computation graph through which we can backpropagate.
- **Nonstationary data:** as the policy changes, it visits different states and changes its own training distribution.

The last point resembles the covariate-shift problem in imitation learning, but RL additionally has to discover useful behavior from rewards rather than expert action labels.

## Practical consequence

Robot RL is often much less sample-efficient than ordinary supervised learning. Real interactions are slow, costly, and potentially unsafe, which motivates simulation, demonstrations, learned models, and offline datasets.

Related notes:

- [Agent–environment interaction and reinforcement learning](agent-environment-mdp-and-reinforcement-learning.md)
- [Imitation learning, behavioral cloning, and DAgger](imitation-learning-behavioral-cloning-and-dagger.md)

