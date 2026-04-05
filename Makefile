.PHONY: test lint typecheck format check install

test:
	pytest --cov --cov-report=term-missing

lint:
	ruff check .
	ruff format --check .

typecheck:
	mypy carmel Carmel.py

format:
	ruff format .
	ruff check --fix .

check: lint typecheck test

install:
	pip install -e ".[dev]"
