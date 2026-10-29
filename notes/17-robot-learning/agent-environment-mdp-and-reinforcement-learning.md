# Agent–environment interaction and reinforcement learning

Reinforcement learning studies an **agent** that repeatedly observes an environment, chooses an action, and receives a reward. Unlike supervised learning, the correct action is usually not supplied as a label. The agent must discover useful behavior through interaction.

## The interaction loop

At time step $t$:

1. the environment is in state $s_t$;
2. the agent receives an observation $o_t$;
3. its policy chooses action $a_t$;
4. the environment produces reward $r_{t+1}$ and moves to $s_{t+1}$.

A **policy** $\pi(a\mid s)$ describes how actions are selected. A deterministic policy returns one action, while a stochastic policy gives a probability distribution over actions.

The **state** is the information needed to describe the environment for future decisions. The **observation** is what the agent can actually measure. They are equal in a fully observed grid world, but a real robot may only see camera images that reveal part of the physical state.

## Markov decision process

A Markov decision process (MDP) is commonly described by:

- a state space $\mathcal{S}$;
- an action space $\mathcal{A}$;
- transition dynamics $P(s'\mid s,a)$;
- a reward function $R(s,a,s')$;
- a discount factor $\gamma$.

The Markov assumption says that, given the current state and action, the next-state distribution does not require the complete earlier history. If the observation does not contain enough information for this assumption, the problem is partially observed.

## Return and delayed consequences

An action can matter many steps later. The discounted return from time $t$ is

$$
G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \cdots.
$$

Here $0\leq\gamma\leq1$. A smaller $\gamma$ emphasizes immediate rewards; a value near one preserves longer-term consequences. Determining which earlier actions deserve credit for a later reward is the **credit-assignment problem**.

## Values and Q-learning

The action-value function $Q(s,a)$ estimates the expected return after taking action $a$ in state $s$ and then continuing according to a policy. Tabular Q-learning uses the target

$$
y = r + \gamma\max_{a'}Q(s',a')
$$

for a non-terminal transition, followed by

$$
Q(s,a) \leftarrow Q(s,a) + \alpha\bigl(y-Q(s,a)\bigr).
$$

The term $y-Q(s,a)$ is the temporal-difference error. For a terminal transition there is no future state to value, so the target is only $r$.

## Exploration and evaluation

Always choosing the current best action can prevent discovery of better behavior. An $\epsilon$-greedy policy chooses a random action with probability $\epsilon$ and a greedy action otherwise.

Interaction data is policy-dependent: when the policy changes, the states the agent visits also change. This is one reason reinforcement learning differs from training on a fixed supervised dataset. Results should therefore be evaluated as closed-loop task success over multiple episodes and random seeds, not only by inspecting a training loss.

## Practical robot-learning difficulties

- Real interaction can be slow and expensive.
- Exploratory actions may be unsafe.
- Sparse rewards provide little learning signal.
- A poorly specified reward may encourage unintended behavior.
- Simulation may not reproduce every property of the real world.

The small grid world in [Notebook 69](../../notebooks/69_agent_environment_and_tabular_rl.ipynb) makes the interaction loop, returns, exploration, and Q-learning update visible without requiring a robotics simulator.

## Resources

- [CS231N Spring 2025 schedule](https://cs231n.stanford.edu/2025/schedule.html)
- [Reinforcement Learning: An Introduction, second edition](http://incompleteideas.net/book/the-book-2nd.html)
