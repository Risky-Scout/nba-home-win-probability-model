PYTHON ?= python
DATA ?= data/nba-win-probability-data.csv

.PHONY: setup audit select score evidence test verify reproduce clean

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements-dev.txt

audit:
	$(PYTHON) -m scripts.data_audit --data "$(DATA)" --artifact-dir artifacts/current

select:
	$(PYTHON) -m scripts.select_model --data "$(DATA)" --config-dir configs --artifact-dir artifacts/current

score:
	$(PYTHON) -m scripts.score_final --data "$(DATA)" --selected-spec artifacts/current/selected_spec_pre_march.json --output-dir outputs --artifact-dir artifacts/current --figure-dir figures

test:
	NBA_DATA_PATH="$(DATA)" $(PYTHON) -m pytest

verify:
	$(PYTHON) validate_submission.py --root . --data "$(DATA)" --recompute
	$(PYTHON) -m ruff check nba_wp scripts tests --select E9,F63,F7,F82,F401,F821

reproduce:
	$(PYTHON) run_submission.py --root . --data "$(DATA)" --mode full

clean:
	rm -rf .pytest_cache nba_wp/__pycache__ scripts/__pycache__ tests/__pycache__
