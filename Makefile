PYTHON ?= python

.PHONY: install lint format test smoke check container
install:
	$(PYTHON) -m pip install ".[dev]"

lint:
	$(PYTHON) -m ruff check neurovision tests
	$(PYTHON) -m ruff format --check neurovision tests

format:
	$(PYTHON) -m ruff format neurovision tests

test:
	$(PYTHON) -m unittest discover -s tests -v

smoke:
	$(PYTHON) -m neurovision

check: lint test smoke
	$(PYTHON) -m pip check

container:
	docker compose run --build --rm pipeline
