PYTHON ?= python

.PHONY: install hooks lint format typecheck test smoke check container
install:
	$(PYTHON) -m pip install --no-deps "torch>=2.8,<3" --index-url https://download.pytorch.org/whl/cpu
	$(PYTHON) -m pip install ".[dev,imaging,preprocessing,bias,training]"

hooks:
	$(PYTHON) -m pre_commit install

lint:
	$(PYTHON) -m ruff check neurovision tests
	$(PYTHON) -m ruff format --check neurovision tests

format:
	$(PYTHON) -m ruff format neurovision tests

typecheck:
	$(PYTHON) -m mypy

test:
	$(PYTHON) -m unittest discover -s tests -v

smoke:
	$(PYTHON) -m neurovision

check: lint typecheck test smoke
	$(PYTHON) -m pip check

container:
	docker compose run --build --rm pipeline
