# CS231n Practice

An independent, hands-on learning compendium for Stanford's **CS231n: Deep
Learning for Computer Vision (Spring 2025)**.

The project follows the complete 18-lecture course while emphasizing active
practice: mathematical intuition, tensor-shape reasoning, small
implementations, correctness checks, controlled experiments, visualizations,
failure analysis, and written reflection.

## Project milestone

The main course sequence is complete. The repository contains:

- **74 Jupyter notebooks**, numbered `00` through `73`;
- reusable NumPy and PyTorch implementations under `cs231n_practice/`;
- automated unit tests and numerical gradient checks;
- a topic-organized notes compendium with lecture references;
- experiments using CIFAR-10 and small synthetic datasets where appropriate.

This is an educational implementation project, not a reproduction of
Stanford's official assignments or a collection of production-ready models.

## Topics covered

The notebook sequence progresses through:

1. NumPy, images, k-nearest neighbors, and vectorized distances;
2. linear classifiers, SVM and softmax losses, gradients, SGD, and model
   selection;
3. computational graphs, backpropagation, neural networks, normalization, and
   dropout;
4. convolution, pooling, CNN architectures, residual connections, and
   CIFAR-10 experiments;
5. RNNs, BPTT, LSTMs, GRUs, sequence layers, and image captioning;
6. attention, transformers, positional encoding, and Vision Transformers;
7. localization, object detection, segmentation, and network visualization;
8. video representation, 3D convolution, motion, and temporal attention;
9. distributed training concepts and self-supervised learning;
10. autoregressive models, autoencoders, VAEs, GANs, diffusion, and flow;
11. 3D representations, point-cloud learning, and neural implicit fields;
12. vision-language learning, multimodal fusion, and zero-shot classification;
13. reinforcement learning, imitation learning, MPC, and VLA intuition;
14. human-centered evaluation, robustness, abstention, privacy, and model
    documentation.

See [ROADMAP.md](ROADMAP.md) for the detailed learning sequence and completion
status.

## Learning workflow

Most topics follow the same progression:

1. establish the problem and notation;
2. derive the important equations and track tensor shapes;
3. implement the core operation from basic array or tensor primitives;
4. verify it with small examples, assertions, tests, or gradient checks;
5. run a controlled experiment and visualize the behavior;
6. interpret results, limitations, and common failure modes;
7. answer short reflection questions;
8. move stable logic from the notebook into a reusable module.

The notebooks intentionally retain explanatory steps that a compact library
would normally hide.

## Repository structure

```text
cs231n-practice/
├── notebooks/          # 00–73: explanations, exercises, and experiments
├── cs231n_practice/    # Reusable classifiers, layers, models, and helpers
│   └── classifiers/    # Reusable model implementations
├── tests/              # Unit tests, shape checks, and gradient checks
├── notes/              # Topic-organized conceptual and mathematical notes
├── data/               # Local downloaded datasets; not committed
├── ROADMAP.md          # Topic sequence and completion status
├── environment.yml     # Conda environment definition
└── pyproject.toml      # Package metadata and dependency ranges
```

Lecture screenshots used as private study references live under
`notes/_references/` and are excluded from version control. Public notes are
written in original language rather than reproducing lecture slides.

## Where to start

- Follow notebooks in numerical order for the complete learning path.
- Use [ROADMAP.md](ROADMAP.md) for a higher-level view of the curriculum.
- Browse `notes/` for focused review without rerunning experiments.
- Inspect `cs231n_practice/` after a notebook to see the reusable version of an
  implementation.
- Read the corresponding file in `tests/` to see the expected behavior and
  edge cases.

Some note directories include a `README.md` with a recommended reading order,
such as the [robot-learning notes](notes/17-robot-learning/README.md).

## Environment setup

The project targets CPython 3.11. Conda creates the local environment and
selects the Python version; package dependencies are declared in
`pyproject.toml` and installed with `pip` through `environment.yml`.

```bash
conda env create --prefix ./.venv -f environment.yml
conda activate ./.venv
python --version
python -m pytest
jupyter lab
```

The current environment has been verified with Python 3.11.15, NumPy 2.3.5,
and PyTorch 2.13.0. Compatible stable dependency ranges remain defined in
`pyproject.toml`.

Datasets are downloaded into `data/` by the notebooks that need them. Training
times and exact numerical results depend on hardware, library versions, and
the selected device.

## Verification

Run the complete automated test suite from the repository root:

```bash
python -m pytest
```

The tests cover reusable layers and models, input validation, tensor shapes,
forward and backward passes, numerical gradients, masking behavior, sampling,
and experiment helpers. Notebook outputs supplement these tests with visual
and empirical checks.

## References and attribution

- [CS231n Spring 2025 schedule](https://cs231n.stanford.edu/2025/schedule.html)
- [CS231n Spring 2025 assignments](https://cs231n.stanford.edu/2025/assignments.html)

This is an independent study repository and is not an official Stanford
course resource. CIFAR-10 and other external datasets, papers, and resources
remain subject to their original licenses and terms.
