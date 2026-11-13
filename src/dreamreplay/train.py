"""Training and evaluation loops.

`fit` runs one training phase. The wake phase and later the dream phase both call it, each with
its own loader and number of epochs, and both append to the same history.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader


def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    cfg = dict(cfg)
    name = cfg.pop("name", "adam")
    if name == "adam":
        return torch.optim.Adam(model.parameters(), **cfg)
    if name == "adamw":
        # Decoupled weight decay (Loshchilov & Hutter, 2019). Adam's own `weight_decay` is an L2
        # penalty that Adam rescales per parameter, so it does not shrink weights evenly.
        return torch.optim.AdamW(model.parameters(), **cfg)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), **cfg)
    raise ValueError(f"Unknown optimizer {name!r}")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    loss_sum, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
        n += len(y)
    return {"loss": loss_sum / n, "acc": correct / n}


def train_epoch(
    model: nn.Module, loader: DataLoader, opt: torch.optim.Optimizer, device: torch.device
) -> None:
    model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        F.cross_entropy(model(x), y).backward()
        opt.step()


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    eval_loaders: dict[str, DataLoader],
    opt: torch.optim.Optimizer,
    epochs: int,
    device: torch.device,
    phase: str = "wake",
    history: list[dict] | None = None,
    log_every: int = 1,
) -> list[dict]:
    """Train for `epochs` epochs and record metrics on every loader in `eval_loaders`.

    `eval_loaders` should include an un-shuffled copy of the training set under the key "train".
    The generalization gap is then computed from clean training accuracy, not the running
    accuracy measured during training.
    """
    history = [] if history is None else history
    start = history[-1]["epoch"] if history else 0
    for e in range(1, epochs + 1):
        train_epoch(model, train_loader, opt, device)
        if e % log_every and e != epochs:
            continue
        row = {"epoch": start + e, "phase": phase}
        for split, loader in eval_loaders.items():
            for k, v in evaluate(model, loader, device).items():
                row[f"{split}_{k}"] = v
        if "train_acc" in row and "test_acc" in row:
            row["gap"] = row["train_acc"] - row["test_acc"]
        history.append(row)
        print(
            f"[{phase}] epoch {row['epoch']:4d}  "
            + "  ".join(f"{k}={v:.4f}" for k, v in row.items() if isinstance(v, float))
        )
    return history
