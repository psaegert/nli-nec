<h1 align="center" style="margin-top: 0px;">Named Entity Classification through Natural Language Inference with Transformers</h1>
<h2 align="center" style="margin-top: 0px;">Formal Semantics: Student Project</h2>

<div align="center">

[![Pytest](https://github.com/psaegert/nli-nec/actions/workflows/pytest.yml/badge.svg)](https://github.com/psaegert/nli-nec/actions/workflows/pytest.yml)
[![Code Quality](https://github.com/psaegert/nli-nec/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/psaegert/nli-nec/actions/workflows/pre-commit.yml)

</div>

# Requirements
- Python 3.10

# Getting Started

## Create a virtual environment
```bash
conda create -n nli-nec python=3.10
conda activate nli-nec
```

## Install the package
```bash
pip install -e .
```

# Contributing
I use
- [flake8](https://pypi.org/project/flake8/) to enforce linting
- [mypy](https://pypi.org/project/mypy/) to enforce static typing
- [isort](https://pypi.org/project/isort/) to enforce import sorting
- [pytest](https://pypi.org/project/pytest/) to run tests against our code (see `tests/`)

To set up the pre-commit hooks, run the following command:
```bash
pre-commit install
```

To run the pre-commit hooks manually, run the following command:
```bash
pre-commit run --all-files
```

To run the tests, use the following command:
```bash
pytest
```