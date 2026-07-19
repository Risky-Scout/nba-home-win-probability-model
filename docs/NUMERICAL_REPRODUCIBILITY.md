# Numerical Reproducibility

The saved prediction files were generated on a different numerical
environment from the macOS interview environment.

Recomputation produced maximum absolute probability differences of:

- March: 4.14e-09
- April: 1.74e-08

The maximum metric difference was below 6e-10, and no predicted
classification changed.

The validator therefore uses:

- probability tolerance: 5e-08
- metric tolerance: 1e-08
- relative tolerance: 0

These tolerances accommodate insignificant cross-platform floating-point
differences without changing model specifications, predictions, reported
metrics, or classification decisions.
