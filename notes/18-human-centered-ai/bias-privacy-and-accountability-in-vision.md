# Bias, privacy, and accountability in vision

Computer-vision systems inherit choices about what data is collected, how it is labeled, which errors are measured, and where the system is deployed. A high aggregate benchmark score can hide serious failures for particular groups or situations.

## Sources of bias

- datasets may underrepresent some people, environments, or visual conditions;
- labels may encode social assumptions or inconsistent human judgments;
- objectives may optimize average accuracy while ignoring unequal error costs;
- deployment data may differ from the development benchmark;
- feedback loops may reinforce earlier model decisions.

Bias is therefore not only a property of model weights. It can enter through the complete data and decision pipeline.

## Documentation and auditing

Useful practices include dataset documentation, subgroup evaluation, error analysis, model cards, internal audits, and clear ownership of deployment decisions. Evaluation should ask who benefits, who bears risk, and whether errors can be challenged or corrected.

Fairness cannot usually be reduced to one universal metric. Different fairness goals may conflict, so the choice of metric is itself a value judgment that should be made explicit.

## Privacy is not simply lower accuracy

Sometimes a system should intentionally avoid capturing identifying information. Possible approaches include:

- collecting only task-relevant signals;
- blurring faces or masking bodies;
- using low-resolution, depth, event, or purpose-built sensors;
- processing data locally and retaining less raw footage;
- federated learning or cryptographic techniques where appropriate.

Each method protects different information and has different failure modes. Blurring after recording, for example, does not prevent the original sensor from capturing sensitive pixels. Privacy should be designed into data collection, storage, access, and deletion—not added only to the final visualization.

## Main intuition

Responsible vision requires more than removing names from a dataset. It requires data minimization, representative evaluation, security, transparency, governance, and meaningful recourse for affected people.

