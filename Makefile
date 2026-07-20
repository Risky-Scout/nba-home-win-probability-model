PYTHON ?= python3
DATA ?= data/nba-win-probability-data.csv

.PHONY: setup select predict report test verify reproduce clean

setup:
	uv sync --frozen

select:
	$(PYTHON) -m nba_wp.cli select --data "$(DATA)" --config configs/model.yaml --artifact-dir artifacts/current

predict:
	$(PYTHON) -m nba_wp.cli predict --data "$(DATA)" --config configs/model.yaml \
	  --output predictions/april_predictions.csv \
	  --rolling-output predictions/april_predictions_rolling_scenario.csv

report:
	$(PYTHON) -m nba_wp.cli report --data "$(DATA)" --config configs/model.yaml

test:
	NBA_DATA_PATH="$(DATA)" $(PYTHON) -m pytest

verify:
	$(PYTHON) validate_submission.py --root . --data "$(DATA)" --recompute
	$(PYTHON) scripts/verify_predictions.py
	$(PYTHON) -m ruff check src scripts tests --select E9,F63,F7,F82,F401,F821

reproduce:
	$(PYTHON) run_submission.py --root . --data "$(DATA)" --mode full

clean:
	rm -rf .pytest_cache src/nba_wp/__pycache__ scripts/__pycache__ tests/__pycache__
