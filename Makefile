.PHONY: install run setup

VENV_PYTHON := ./.venv/bin/python
VENV_PIP := ./.venv/bin/pip

install:
	@echo "--- Installing dependencies ---"

	$(VENV_PIP) install -r requirements.txt

run: install
	@echo "--- Running the application ---"
	$(VENV_PYTHON) tremoranalyzer.py

setup: install run
