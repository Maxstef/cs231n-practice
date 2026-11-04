# Learned dynamics, planning, and model-predictive control

Model-based control uses a transition model to predict how actions change the state:

$$
\hat{s}_{t+1}=f_\theta(s_t,a_t).
$$

The model may be known from physics or learned from transition data. A learned model turns dynamics prediction into supervised learning: $(s_t,a_t)$ is the input and the observed $s_{t+1}$ is the target.

The dynamics model is not itself a policy. It predicts the consequence of a proposed action; a controller or planner compares those predicted consequences and decides which action to execute. Model-free control may map a state directly to an action, while model-based control inserts prediction and planning between them.

## Planning with a model

A planner compares possible future action sequences without executing all of them in the real environment:

1. propose candidate action sequences;
2. predict their state trajectories with the model;
3. assign each trajectory a cost or reward;
4. choose the best candidate.

Random shooting obtains candidates by sampling. It is easy to understand and parallelize, but it only finds the best sequence among the samples rather than the globally optimal sequence. Sampling actions in short constant blocks can reduce the search dimension and produce coherent acceleration and braking phases, at the cost of less precise action timing.

The trajectory score commonly combines several manually weighted objectives, such as intermediate distance to the target, terminal distance, terminal velocity, and action effort. These weights are task-dependent hyperparameters: they express priorities and should account for the different numerical scales of terms summed once versus across the entire horizon.

## One-step accuracy is not enough

A model can predict one transition accurately yet drift during a long imagined rollout. After the first step, its prediction becomes the next input. Small errors therefore alter later inputs and can accumulate. Planning may also propose states or actions unlike those in the training data, where predictions are less reliable.

This creates a model-exploitation risk: the planner may choose a trajectory that looks excellent only because it uses a region where the model is wrong.

## Model-predictive control

Model-predictive control repeatedly:

1. observes the current real state;
2. plans over a finite horizon;
3. executes only the first action;
4. observes the actual result;
5. replans from the new state.

Replanning anchors every new plan to reality. It limits the effect of long model rollouts and responds to disturbances that were absent from the previous plan.

An open-loop controller instead plans once and executes the whole sequence. It cannot revise later actions after unexpected motion or model error. Long open-loop plans are difficult both because prediction error accumulates and because searching over more future actions creates a higher-dimensional optimization problem. MPC repeatedly solves a shorter problem and corrects it with real observations.

## Practical trade-offs

- A longer horizon can anticipate delayed effects but accumulates more model error.
- More candidate sequences improve search but require more computation.
- Broader transition data improves model coverage but costs interaction.
- Planning outside the transition model's training range relies on extrapolation and can produce unreliable trajectories.
- Short replanning intervals improve feedback but increase planning frequency.
- Ensembles, uncertainty penalties, and constraints can discourage unsafe or unreliable predicted regions.

[Notebook 71](../../notebooks/71_learned_dynamics_and_model_predictive_control.ipynb) demonstrates these ideas with a learned point-mass model and random-shooting MPC.

## Resources

- [CS231N Spring 2025 schedule](https://cs231n.stanford.edu/2025/schedule.html)
- [Model Predictive Control: Theory, Computation, and Design](https://sites.engineering.ucsb.edu/~jbraw/mpc/)
