# Robot-learning notes: recommended reading order

These notes move from the robot's interaction with the world to learning methods, planning, general-purpose policies, and evaluation.

## Recommended sequence

1. [Robot perception: embodied, active, and situated](robot-perception-embodied-active-and-situated.md)  
   Begin with why robot perception differs from passive image recognition and why state and observation are not always identical.

2. [Agent–environment interaction and reinforcement learning](agent-environment-mdp-and-reinforcement-learning.md)  
   Learn the interaction loop, MDP terminology, returns, Q-values, exploration, and delayed rewards.

3. [Reinforcement learning versus supervised learning](reinforcement-learning-vs-supervised-learning.md)  
   Compare fixed labeled datasets with stochastic, policy-dependent interaction and understand credit assignment.

4. [Imitation learning, behavioral cloning, and DAgger](imitation-learning-behavioral-cloning-and-dagger.md)  
   Study learning from expert actions, closed-loop covariate shift, compounding errors, and iterative dataset aggregation.

5. [Inverse reinforcement learning and implicit behavior cloning](inverse-reinforcement-learning-and-implicit-behavior-cloning.md)  
   Contrast copying expert actions with inferring preferences or learning an action-energy function.

6. [Explicit, implicit, and diffusion policies](explicit-implicit-and-diffusion-policies.md)  
   Compare direct action prediction, energy-based action selection, and iterative diffusion-based generation.

7. [Learned dynamics, planning, and model-predictive control](learned-dynamics-planning-and-mpc.md)  
   Separate predicting action consequences from choosing actions, then study planning, model error, and replanning.

8. [Visual goal-conditioned policies and VLA intuition](visual-goal-conditioned-policies-and-vla.md)  
   Introduce conditional behavior from visual goals or instructions and connect it to vision-language-action models.

9. [Robotic foundation models and cross-embodiment data](robotic-foundation-models-and-cross-embodiment-data.md)  
   Scale the VLA idea across tasks, datasets, and different robot bodies.

10. [Robot-learning evaluation and sim-to-real](robot-learning-evaluation-and-sim-to-real.md)  
    Finish with real-world success metrics, simulation limitations, safety, failure analysis, and transfer to deployment.

## Matching practice

- [Notebook 69 — Agent–environment interaction and tabular RL](../../notebooks/69_agent_environment_and_tabular_rl.ipynb)
- [Notebook 70 — Imitation learning and covariate shift](../../notebooks/70_imitation_learning_and_covariate_shift.ipynb)
- [Notebook 71 — Learned dynamics and model-predictive control](../../notebooks/71_learned_dynamics_and_model_predictive_control.ipynb)
- [Notebook 72 — Visual goal-conditioned policies and VLA intuition](../../notebooks/72_visual_goal_conditioned_policies_and_vla_intuition.ipynb)

The inverse-RL, implicit-policy, diffusion-policy, foundation-model, and sim-to-real material remains primarily conceptual in this project.
