from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from .metrics import summarize_runs


def _iter_result_paths(root: Path) -> Iterable[Path]:
    if root.is_file() and root.suffix.lower() == ".json":
        yield root
    elif root.is_dir():
        for path in sorted(root.rglob("results.json")):
            yield path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute EPR/VSR/OS metrics from VizFlow results.json artifacts."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Paths to results.json files or directories containing them.",
    )
    parser.add_argument(
        "--output",
        help="Optional path to write JSON summary (defaults to stdout).",
    )
    args = parser.parse_args()

    result_paths = []
    for raw in args.inputs:
        for path in _iter_result_paths(Path(raw)):
            result_paths.append(path)

    if not result_paths:
        raise SystemExit("No results.json files found in provided inputs.")

    summary = summarize_runs(result_paths)
    output_text = json.dumps(summary, ensure_ascii=False, indent=2)

    if args.output:
        Path(args.output).write_text(output_text, encoding="utf-8")
    else:
        print(output_text)


if __name__ == "__main__":
    main()
