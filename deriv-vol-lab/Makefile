.PHONY: lint test run

lint:
	ruff check src tests
	ruff format --check src tests
	mypy src

test:
	pytest

run:
	python -m deriv_vol_lab
