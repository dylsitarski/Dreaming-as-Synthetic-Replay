"""Plot accuracy curves for every seed of an experiment.

Usage:
    python experiments/plot_history.py results/baseline_fmnist
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("exp_dir")
    args = p.parse_args()
    exp = Path(args.exp_dir)

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = {"train_acc": "tab:blue", "val_acc": "tab:orange", "test_acc": "tab:green"}
    for i, f in enumerate(sorted(exp.glob("seed*/history.csv"))):
        df = pd.read_csv(f)
        for col, c in colors.items():
            ax.plot(df["epoch"], df[col], color=c, alpha=0.6, label=col if i == 0 else None)
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_title(exp.name)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = exp / "curves.png"
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
