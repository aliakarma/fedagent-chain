# Contributing to FedAgent-Chain

Thank you for your interest in contributing to FedAgent-Chain!
This guide covers the contribution process for code, documentation, and research extensions.

## Code of Conduct

Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.

## Quick Start (30 minutes to first contribution)

```bash
# 1. Fork and clone
git clone https://github.com/YOUR_USERNAME/fedagent-chain.git
cd fedagent-chain

# 2. Create conda environment
conda env create -f environment.yml
conda activate fedagent-chain

# 3. Install in editable mode with dev dependencies
pip install -e .
pip install -r requirements-dev.txt

# 4. Install pre-commit hooks
pre-commit install

# 5. Create a feature branch
git checkout -b feature/my-contribution develop

# 6. Verify everything works
make test-fast
```

## Development Workflow

### Making Changes

1. Make your changes in the appropriate `src/` submodule.
2. Add or update docstrings (NumPy-style) for all public functions.
3. Write tests in `tests/unit/` or `tests/integration/`.
4. Run quality checks: `make quality`
5. Run tests: `make test-unit`

### Code Standards

- All code must pass `ruff check .` (linting) and `black --check .` (formatting).
- All public functions must have NumPy-style docstrings with Parameters and Returns sections.
- New features must include unit tests with ≥80% line coverage.
- Commits must follow [Conventional Commits](https://www.conventionalcommits.org/) specification.
- **No hardcoded hyperparameters** — add new params to the appropriate YAML in `configs/`.

### Commit Message Format

```
<type>(<scope>): <imperative summary under 72 chars>

[Optional body]

[Optional footer: Closes #issue]
```

Valid types: `feat`, `fix`, `docs`, `refactor`, `test`, `perf`, `build`, `ci`, `chore`.

Examples:
```
feat(federated): add SCAFFOLD algorithm as alternative aggregator
fix(blockchain): correct UTF-8 encoding in SHA-256 hash computation
docs(agents): update EmploymentAgent docstring with formula reference
test(privacy): add edge case for zero-norm update in clip_update
```

### Pull Request Process

1. Ensure CI passes on your fork before opening a PR.
2. Reference related issues with `Closes #N`.
3. Fill out the PR template completely.
4. Request review from at least one maintainer.
5. Squash commits before merging if the history is noisy.

## Adding a New Federated Learning Algorithm

1. Create `src/federated/my_algorithm.py` inheriting from `FedAvgAggregator`.
2. Add a config file at `configs/federated/my_algorithm.yaml`.
3. Add unit tests in `tests/unit/test_my_algorithm.py`.
4. Document the algorithm with references to the paper it implements.

## Adding a New Agentic Service

1. Create `src/agents/my_agent.py` inheriting from `BaseAgent`.
2. Add a config file at `configs/agents/my_agent.yaml`.
3. Register the agent in `src/agents/__init__.py`.
4. Add unit tests in `tests/unit/test_agents.py`.

## Reporting Issues

Use GitHub Issues with the appropriate template:
- **Bug report**: For code errors, import failures, or unexpected output.
- **Reproducibility issue**: For paper results that cannot be replicated.
- **Feature request**: For new algorithms, agents, or evaluation metrics.

## Acknowledgement

All contributors will be acknowledged in `CHANGELOG.md` and in the repository's
contributor list. Significant research contributions may be acknowledged in
future paper extensions.
