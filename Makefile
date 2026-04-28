.PHONY: install run score setup

VENV_PYTHON := ./venv/bin/python
VENV_PIP := ./venv/bin/pip

install:
	@echo "--- Installing dependencies ---"
	$(VENV_PIP) install -r requirements.txt

run: install
	@echo "--- Training Model (TensorFlow) ---"
	$(VENV_PYTHON) train_physionet_augmented.py

score: install
	@echo "--- Generating Performance Metrics & Scores ---"
	$(VENV_PYTHON) evaluate_physionet_augmented.py

setup: install run score
