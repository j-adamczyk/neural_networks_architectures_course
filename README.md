# Neural Network Architectures - labs

## Topics

1. [Convolutional architectures](lab1)
2. [Transformers](lab2)

## Environment setup

Every lab has its own environment, defined in its directory (`pyproject.toml` and `uv.lock`).

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then create the virtual environment
of a lab with all dependencies (including the dev ones: Jupyter Notebook, ruff and pre-commit). An example
for the first lab:
```shell
cd lab1
uv sync
```

Run Jupyter Notebook:
```shell
uv run jupyter notebook
```

Install the pre-commit hooks (ruff linter and formatter, run on every commit, configured in `ruff.toml`):
```shell
uv run pre-commit install
```

**Google Colab.** All packages used in the labs are preinstalled on Colab. `requirements.txt` in every lab
directory contains the exact versions from its `uv.lock`, without dev dependencies, for use with
`pip install -r requirements.txt`.
