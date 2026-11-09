"""Summarize final-epoch metrics (mean +/- std over seeds) for one or more experiments.

Usage:
    python experiments/summarize.py results/baseline_fmnist results/baseline_fmnist_dropout

A run is any folder that holds seed*/history.csv, so each value of a sweep is its own run.
Within an experiment, the run with the highest mean val_acc is marked "(best val)". That is the
value to report, because it was chosen without looking at test data. The output is a Markdown
table, ready to paste into docs/progress-log.md or a pull request.
"""

import argparse
from pathlib import Path

import pandas as pd

# Column -> (header, scale, decimals)
METRICS = {
    "train_acc": ("train acc %", 100, 1),
    "val_acc": ("val acc %", 100, 1),
    "test_acc": ("test acc %", 100, 1),
    "gap": ("gap (points)", 100, 1),
    "test_loss": ("test loss", 1, 2),
}


def final_metrics(run_dir: Path) -> pd.DataFrame:
    """Last row of every seed's history, one row per seed."""
    rows = [pd.read_csv(f).iloc[-1] for f in sorted(run_dir.glob("seed*/history.csv"))]
    return pd.DataFrame(rows)[list(METRICS)]


def sort_key(run_dir: Path) -> tuple:
    """Order sweep folders such as "dropout=0.25" by their numeric value."""
    _, _, value = run_dir.name.partition("=")
    try:
        return (0, float(value))
    except ValueError:
        return (1, run_dir.name)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("exp_dirs", nargs="+", type=Path)
    args = p.parse_args()

    print("| run | seeds | " + " | ".join(h for h, _, _ in METRICS.values()) + " |")
    print("|---" * (len(METRICS) + 2) + "|")
    for exp in args.exp_dirs:
        runs = sorted({f.parent.parent for f in exp.glob("**/seed*/history.csv")}, key=sort_key)
        if not runs:
            raise SystemExit(f"No seed*/history.csv under {exp}")
        finals = {r: final_metrics(r) for r in runs}
        best = max(runs, key=lambda r: finals[r]["val_acc"].mean())
        for r, df in finals.items():
            name = r.relative_to(exp.parent).as_posix()
            if len(runs) > 1 and r == best:
                name += " (best val)"
            cells = [
                f"{scale * df[c].mean():.{d}f} ± {scale * df[c].std(ddof=0):.{d}f}"
                for c, (_, scale, d) in METRICS.items()
            ]
            print(f"| {name} | {len(df)} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
