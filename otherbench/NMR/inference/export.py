"""Export computed metrics without overwriting the supplied manuscript Table S4."""

import argparse
from pathlib import Path
from shutil import copy2
from otherbench.NMR.utils.config import NMR_ROOT


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, default=NMR_ROOT / "outputs/evaluation")
    parser.add_argument("--output-dir", type=Path, default=NMR_ROOT / "outputs" / "exported")
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("leaderboard.csv", "overall_leaderboard.csv", "summary.json"):
        destination = "evaluation_summary.json" if name == "summary.json" else name
        copy2(args.evaluation_dir / name, args.output_dir / destination)
    print(f"Exported computed metrics to {args.output_dir}")
