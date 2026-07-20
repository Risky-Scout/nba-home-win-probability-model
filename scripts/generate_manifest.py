
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


EXCLUDED_PARTS = {".git", ".venv", "__pycache__", ".pytest_cache"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default="artifacts/current/manifest.sha256")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output = (root / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.resolve() == output:
            continue
        if any(part in EXCLUDED_PARTS for part in path.parts):
            continue
        relative = path.relative_to(root)
        if (
            relative.parts
            and relative.parts[0] == "data"
            and path.suffix.lower() == ".csv"
        ):
            continue
        lines.append(f"{sha256(path)}  {relative.as_posix()}")
    output.write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(lines)} hashes to {output}")


if __name__ == "__main__":
    main()
