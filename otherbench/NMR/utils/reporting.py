"""Numerical metric aggregation and exports for the final six-method benchmark."""

import csv
from pathlib import Path
from otherbench.NMR.utils.metrics import aggregate

METHOD_ORDER = ["RAW", "WT", "Cadzow", "LD-Net", "DN-Unet", "F2P"]


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_results(rows, output):
    conditions = list(dict.fromkeys(row["condition"] for row in rows))
    methods = [name for name in METHOD_ORDER if any(r["method"] == name for r in rows)]
    leaderboard = [{"condition": condition, "method": method,
                    **aggregate([r for r in rows if r["condition"] == condition and r["method"] == method])}
                   for condition in conditions for method in methods]
    overall = [{"condition": "overall", "method": method,
                **aggregate([r for r in rows if r["method"] == method])}
               for method in methods]
    write_csv(output / "metrics.csv", rows)
    write_csv(output / "leaderboard.csv", leaderboard)
    write_csv(output / "overall_leaderboard.csv", overall)
    return leaderboard, overall
