# Robot-learning evaluation and sim-to-real

Robot policies ultimately act in a physical world, so validation loss or offline action accuracy is not enough. Small prediction errors can compound during a rollout, and successful behavior may require recovery from disturbances or unfamiliar states.

## Real-world evaluation

Real trials provide the most relevant evidence but are expensive, slow, noisy, and sometimes unsafe. Useful measurements include:

- task success rate over repeated trials;
- completion time and intervention rate;
- robustness to changed objects, viewpoints, lighting, and initial states;
- safety violations and characteristic failure modes.

Training loss may correlate weakly with these measures because the training objective, test task, and rollout horizon are different.

## Simulation and the sim-to-real gap

Simulation permits fast, repeatable, and safe evaluation, but its physics, sensors, objects, and human interactions are imperfect. A policy can exploit simulator-specific details that do not exist in reality; this difference is the **sim-to-real gap**.

Domain randomization, more realistic asset generation, calibration with real data, and mixed simulation/real training can reduce the gap. They cannot guarantee transfer, so final claims still need real-world testing.

## No single ImageNet-style score

Robotics lacks one universal benchmark because success depends on embodiment, task definitions, environments, and safety requirements. Standardized suites are valuable, but they only approximate the diversity and messiness of deployment.

Related notes:

- [Learned dynamics, planning, and model-predictive control](learned-dynamics-planning-and-mpc.md)
- [Visual goal-conditioned policies and VLA intuition](visual-goal-conditioned-policies-and-vla.md)
