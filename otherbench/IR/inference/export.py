"""Export computed metric artifacts to a local output directory."""

import argparse
import json
from pathlib import Path
from shutil import copy2

from otherbench.IR.utils.config import IR_ROOT
from otherbench.IR.utils.metrics import export_leaderboard


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, default=IR_ROOT / "outputs" / "evaluation")
    parser.add_argument("--output-dir", type=Path, default=IR_ROOT / "outputs" / "exported")
    args = parser.parse_args(argv)
    source = args.evaluation_dir.resolve() / "summary.json"
    summary = json.loads(source.read_text(encoding="utf-8"))
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    copy2(source, output / "evaluation_summary.json")
    export_leaderboard(summary, output)
    print(f"Published metric files: {output}")
