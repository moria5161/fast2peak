"""Assemble the final Table S4 from the two historical native-6400 evaluations."""

import argparse
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
from otherbench.NMR.utils.reporting import read_csv, save_results


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-metrics", type=Path, required=True)
    parser.add_argument("--baseline-predictions", type=Path, required=True)
    parser.add_argument("--f2p-metrics", type=Path, required=True)
    parser.add_argument("--f2p-predictions", type=Path, required=True)
    parser.add_argument("--f2p-name", default="F2P-depth4-wd0-mask0.03")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    baseline = [r for r in read_csv(args.baseline_metrics) if r["method"] != "F2P"]
    f2p = [{**r, "method": "F2P"} for r in read_csv(args.f2p_metrics) if r["method"] == args.f2p_name]
    if len(f2p) != 450 or len(baseline) != 2250:
        raise ValueError(f"Expected 450 final F2P and 2250 other rows; got {len(f2p)}, {len(baseline)}")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = baseline + f2p
    leaderboard, overall = save_results(rows, output)
    with h5py.File(args.baseline_predictions) as old, h5py.File(args.f2p_predictions) as final, h5py.File(output / "denoised.h5", "w") as dest:
        for key in ("gt", "ppm"):
            if not np.array_equal(old[key][:], final[key][:]):
                raise ValueError(f"Historical evaluations disagree on {key}")
            old.copy(key, dest)
        groups = dest.create_group("conditions")
        for condition in dict.fromkeys(r["condition"] for r in rows):
            group = groups.create_group(condition)
            source = old["conditions"][condition]
            if not np.array_equal(source["noisy"][:], final[condition]["RAW"][:]):
                raise ValueError(f"Historical evaluations disagree on noisy input: {condition}")
            for method in ("noisy", "WT", "Cadzow", "LD-Net", "DN-Unet"):
                source.copy(method, group)
            group.create_dataset("F2P", data=final[condition][args.f2p_name][:], compression="gzip", shuffle=True)
    summary = {"samples_per_condition": 150, "conditions": 3, "evaluated_pairs": 450,
               "assembly": "five methods from native6400 evaluation_all; F2P from depth4 mask0.03 epochs400 comparison",
               "f2p_checkpoint_epoch": 396,
               "source_files": {key: {"path": str(getattr(args, key)), "sha256": file_hash(getattr(args, key))}
                                for key in ("baseline_metrics", "baseline_predictions", "f2p_metrics", "f2p_predictions")},
               "leaderboard": leaderboard, "overall": overall}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Assembled {len(rows)} metric rows for six final methods: {output}")
