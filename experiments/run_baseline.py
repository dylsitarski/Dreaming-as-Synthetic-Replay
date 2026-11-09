"""Train the wake-phase baseline and save metrics plus a checkpoint for each seed.

Usage:
    python experiments/run_baseline.py --config configs/baseline_fmnist.yaml --seeds 0 1 2

Outputs go to results/<name>/seed<k>/: history.csv, summary.json, config.yaml, and wake.pt
(checkpoints are git-ignored). If the config has a `sweep` block, each swept value is trained in
turn and saved under results/<name>/<key>=<value>/seed<k>/.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import yaml

from dreamreplay.data import load_splits, make_loader
from dreamreplay.models import build_model
from dreamreplay.train import build_optimizer, fit
from dreamreplay.utils import expand_sweep, get_device, load_config, set_seed


def run(cfg: dict, seed: int, out_root: Path) -> dict:
    set_seed(seed)
    device = get_device(cfg.get("device", "auto"))
    tcfg = cfg["train"]

    splits = load_splits(seed=seed, **cfg["data"])
    train_loader = make_loader(splits.train, tcfg["batch_size"], shuffle=True, seed=seed)
    eval_loaders = {
        "train": make_loader(splits.train, tcfg["eval_batch_size"], shuffle=False),
        "val": make_loader(splits.val, tcfg["eval_batch_size"], shuffle=False),
        "test": make_loader(splits.test, tcfg["eval_batch_size"], shuffle=False),
    }

    model = build_model(cfg["model"], splits.in_channels, splits.num_classes, splits.image_size)
    model.to(device)
    opt = build_optimizer(model, cfg["optim"])

    history = fit(
        model, train_loader, eval_loaders, opt, tcfg["epochs"], device,
        phase="wake", log_every=tcfg.get("log_every", 1),
    )

    out = out_root / cfg["name"] / cfg.get("run", "") / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    torch.save({"model": model.state_dict(), "optim": opt.state_dict(), "seed": seed},
               out / "wake.pt")

    final = history[-1]
    best_val = max(history, key=lambda r: r["val_acc"])
    summary = {
        "seed": seed,
        "final": {k: final[k] for k in ("epoch", "train_acc", "val_acc", "test_acc", "gap")},
        "best_val_epoch": best_val["epoch"],
        "test_acc_at_best_val": best_val["test_acc"],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--out", default="results")
    args = p.parse_args()

    cfg = load_config(args.config)
    for run_cfg in expand_sweep(cfg):
        summaries = [run(run_cfg, s, Path(args.out)) for s in args.seeds]

        df = pd.DataFrame([{"seed": s["seed"], **s["final"]} for s in summaries])
        label = f" for {run_cfg['run']}" if "run" in run_cfg else ""
        print(f"\nFinal metrics across seeds{label} (mean +/- std):")
        for col in ("train_acc", "val_acc", "test_acc", "gap"):
            print(f"  {col:10s} {df[col].mean():.4f} +/- {df[col].std(ddof=0):.4f}")


if __name__ == "__main__":
    main()
