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

The Markov assumption says that the current state contains all information needed to predict the next-state distribution. Once $s_t$ and $a_t$ are known, the earlier history adds no further predictive information:

$$
P(s_{t+1}\mid s_t,a_t,s_{t-1},a_{t-1},\ldots)=P(s_{t+1}\mid s_t,a_t).
$$

For example, if movement in a fixed grid depends only on the current position and selected action, the agent does not need to remember its earlier route. But suppose a closed door can be crossed only after collecting a key. A state containing only $(row,column)$ is then incomplete: the same position and action can have different outcomes depending on the earlier key collection. Including `has_key` in the state, $(row,column,has\_key)$, restores the required information.

The underlying physical state can be Markov even when the agent cannot observe it completely. A robot's current camera image may not reveal an object hidden behind it, while earlier images could show where the object went. This is **partial observability**: the current observation is not a complete state, so the agent may need observation history, memory, multiple sensors, or a belief about hidden state.

## Return and delayed consequences

An action can matter many steps later. The discounted return from time $t$ is

$$
G_t = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \cdots.
$$

Here $0\leq\gamma\leq1$. A smaller $\gamma$ emphasizes immediate rewards; a value near one preserves longer-term consequences. Determining which earlier actions deserve credit for a later reward is the **credit-assignment problem**.

## Values and Q-learning

The action-value function $Q(s,a)$ estimates the expected future return after taking action $a$ in state $s$ and then continuing from the resulting state. It gives the agent one long-term usefulness estimate for every state–action choice.

In tabular Q-learning, these values are not known in advance. The Q-table is created before training, commonly with every entry set to zero. During interaction, each observed transition $(s,a,r,s')$ supplies a new target:

$$
y = r + \gamma\max_{a'}Q(s',a')
$$

for a non-terminal transition, followed by

$$
Q(s,a) \leftarrow Q(s,a) + \alpha\bigl(y-Q(s,a)\bigr).
$$

The term $y-Q(s,a)$ is the temporal-difference error: the difference between the new target and the current estimate. The learning rate $\alpha$ controls how far the estimate moves toward that target. For a terminal transition there is no future state to value, so the target is only $r$.

The Q-values can become useful even when every ordinary transition has reward zero. Consider $A\rightarrow B\rightarrow Goal$, where only entering the goal gives reward $1$. A successful episode first increases the value of the final action at $B$. On a later visit, the transition from $A$ can then receive the positive target

$$
0 + \gamma\max_{a'}Q(B,a').
$$

In this way, the goal reward gradually propagates backward through earlier decisions. Before the goal is discovered, however, all observed rewards and Q-values may remain zero, which is why exploration is necessary.

After learning, the agent can act greedily by selecting the action with the greatest Q-value:

$$
a = \arg\max_a Q(s,a).
$$

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
