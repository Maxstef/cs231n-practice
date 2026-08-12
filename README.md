# CS231n Practice

A hands-on learning compendium for Stanford's **CS231n: Deep Learning for
Computer Vision (Spring 2025)**.

The aim is not just to run existing models. Each topic will move from the
underlying idea and mathematics to a small implementation, correctness checks,
experiments, and a short written reflection.

## Learning workflow

For each topic:

1. State the problem and establish a simple baseline.
2. Derive the important equations and track tensor shapes.
3. Implement the core algorithm with NumPy when practical.
4. Check the implementation with small examples and automated tests.
5. Reimplement or apply the idea with PyTorch.
6. Run a controlled experiment and visualize the result.
7. Record conclusions, failure cases, and review questions.

The project is developed in small increments. Notebooks will explain and
demonstrate ideas; reusable logic will live in Python modules rather than being
hidden inside notebook cells.

## Planned structure

```text
cs231n-practice/
├── notebooks/          # Explanations, experiments, and visualizations
├── cs231n_practice/    # Reusable implementations
├── tests/              # Correctness and gradient checks
├── exercises/          # Small focused practice problems
├── notes/              # Summaries and reference material
└── data/               # Local datasets (not committed)
```

See [ROADMAP.md](ROADMAP.md) for the topic sequence and progress.

## Environment setup

The project targets CPython 3.11. Conda is used only to create the environment
and select the Python version; project dependencies are declared in
`pyproject.toml` and installed with `pip`.

```bash
conda env create --prefix ./.venv -f environment.yml
conda activate ./.venv
python --version
python -m pytest
jupyter lab
```

The project supports Python `3.11`. Dependency ranges in `pyproject.toml` allow
`pip` to select compatible stable releases. This setup has been verified with
Python 3.11.15, NumPy 2.3.5, and PyTorch 2.13.0.

## References

- [CS231n Spring 2025 schedule](https://cs231n.stanford.edu/2025/schedule.html)
- [CS231n Spring 2025 assignments](https://cs231n.stanford.edu/2025/assignments.html)

This is an independent study repository and is not an official Stanford
course resource.
