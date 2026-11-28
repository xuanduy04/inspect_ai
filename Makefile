.PHONY: hooks
hooks:
	uv run --frozen pre-commit install

.PHONY: ruff
ruff:
	ruff check --fix
	ruff format

.PHONY: mypy
mypy:
	mypy --exclude tests/test_package src tests

.PHONY: check
check: ruff mypy

.PHONY: test
test:
	pytest

.PHONY: test-parallel
test-parallel:
	pytest -n auto

include docs/evals/inspect-evals.mk
