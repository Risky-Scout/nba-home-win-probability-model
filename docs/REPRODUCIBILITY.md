# Reproducibility

## Supported environment

- Python 3.11-3.13
- Python 3.12 recommended
- deterministic scikit-learn solvers with fixed tolerance and random state

Pinned dependencies are in `requirements.txt` and `requirements-dev.txt`.

## Input contract

Expected assignment-file SHA-256:

```text
fc4d66b4a7d08cf975fe7b82f22e95cce37b84976536e627621d5238481638c3
```

The raw CSV is intentionally not committed. Place it anywhere and pass an
absolute path.

## macOS setup

```bash
bash scripts/bootstrap_macos.sh
source .venv/bin/activate
```

## Complete reproduction

```bash
python run_submission.py   --root .   --data "/absolute/path/to/nba-win-probability-data.csv"   --mode full
```

This regenerates the audit, March selection, selected specification, model
outputs, evidence tables, figures, and integrity manifest.

## Verification levels

### Level 1 - saved-artifact consistency

```bash
python validate_submission.py   --root .   --data "/absolute/path/to/nba-win-probability-data.csv"
```

Checks:

- schema and row count;
- pregame record reconciliation;
- selection cutoff;
- zero April rows in selection;
- prediction counts;
- direct metric recomputation from saved prices.

### Level 2 - full probability rebuild

```bash
python validate_submission.py   --root .   --data "/absolute/path/to/nba-win-probability-data.csv"   --recompute
```

This rebuilds the selected features, refits the deployed Elo-only champion, and
compares every saved March and April probability within `1e-12`.

### Level 3 - source-to-selection reproduction

```bash
make reproduce DATA="/absolute/path/to/nba-win-probability-data.csv"
```

This also reruns the complete March candidate grid.

## Expected runtime

Runtime depends on hardware. The expensive operation is repeated daily
Bradley-Terry fitting across five architecture candidates. The scorer uses only
the selected architecture and is materially faster than full selection.

For an interview, run the fast validator first. Rerun `--mode score` when asked
to demonstrate fitting and pricing, and rerun `--mode select` when asked to
prove where the selected parameters came from.

## Floating point

CSV probabilities are stored at normal double precision. Tests use absolute
tolerance `1e-12`. A different BLAS implementation or dependency version can
produce smaller harmless differences, which is why versions are pinned.
