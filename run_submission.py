
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce the complete NBA home-win probability submission."
    )
    parser.add_argument("--data", required=True)
    parser.add_argument(
        "--mode",
        choices=["full", "score", "audit", "select", "verify"],
        default="full",
    )
    parser.add_argument("--root", default=".")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    data = str(Path(args.data).resolve())
    python = sys.executable
    started = time.perf_counter()

    if args.mode in {"full", "audit"}:
        run(
            [
                python,
                "-m",
                "scripts.data_audit",
                "--data",
                data,
                "--artifact-dir",
                str(root / "artifacts"),
            ]
        )

    if args.mode in {"full", "select"}:
        run(
            [
                python,
                "-m",
                "scripts.select_model",
                "--data",
                data,
                "--config-dir",
                str(root / "configs"),
                "--artifact-dir",
                str(root / "artifacts"),
            ]
        )

    if args.mode in {"full", "score"}:
        run(
            [
                python,
                "-m",
                "scripts.score_final",
                "--data",
                data,
                "--selected-spec",
                str(root / "artifacts" / "selected_spec.json"),
                "--output-dir",
                str(root / "outputs"),
                "--artifact-dir",
                str(root / "artifacts"),
                "--figure-dir",
                str(root / "figures"),
            ]
        )

    if args.mode == "full":
        # Nested rolling-origin audit (policy-matched frozen + daily; blend,
        # calibration and schedule challengers) is the out-of-sample evidence.
        run([python, "-m", "scripts.nested_validation", "--data", data,
             "--artifact-dir", str(root / "artifacts"), "--figure-dir", str(root / "figures")])
        # Calibration-risk investigation: raw Elo vs identity-shrunk Platt and
        # Beta calibrators under both policies with strict promotion gates. Runs
        # after the nested audit and BEFORE the manifest so its decision artifact
        # is hashed into the manifest. Writes artifacts/calibration_challenger_decision.json.
        run([python, "-m", "scripts.calibration_challenger", "--data", data,
             "--artifact-dir", str(root / "artifacts")])
        # Rebuild the Excel twin and emit the machine-readable reconciliation.
        run([python, "-m", "scripts.rebuild_full_workbook"])
        run([python, "-m", "scripts.workbook_reconciliation"])
        # Committed test report artifact.
        report = root / "artifacts" / "pytest_report.txt"
        with report.open("w") as handle:
            print("+ python -m pytest -rA", flush=True)
            subprocess.run([python, "-m", "pytest", "-rA"], check=True, stdout=handle, stderr=subprocess.STDOUT)
        run(
            [
                python,
                "-m",
                "scripts.generate_manifest",
                "--root",
                str(root),
                "--output",
                "artifacts/manifest.sha256",
            ]
        )

    if args.mode in {"full", "verify"}:
        run(
            [
                python,
                str(root / "validate_submission.py"),
                "--root",
                str(root),
                "--data",
                data,
            ]
        )

    elapsed = time.perf_counter() - started
    print(f"Completed mode={args.mode} in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    main()
