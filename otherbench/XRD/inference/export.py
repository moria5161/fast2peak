"""Export final numerical results and QC provenance to public docs."""

import argparse
import csv
from pathlib import Path
from shutil import copy2

from otherbench.XRD.utils.config import XRD_ROOT
from otherbench.XRD.utils.reporting import save_macro_average


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, default=XRD_ROOT / "outputs" / "evaluation")
    parser.add_argument("--data-dir", type=Path, default=XRD_ROOT / "datasets" / "benchmark")
    parser.add_argument("--output-dir", type=Path, default=XRD_ROOT / "docs")
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = args.evaluation_dir.resolve()
    with (source / "leaderboard.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    save_macro_average(rows, output)
    copy2(source / "leaderboard.csv", output / "leaderboard.csv")
    copy2(source / "summary.json", output / "evaluation_summary.json")
    copy2(args.data_dir / "qc_manifest.csv", output / "qc_manifest.csv")
    copy2(args.data_dir / "summary.json", output / "data_summary.json")
    print(f"Published numerical results: {output}")
