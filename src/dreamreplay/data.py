"""Dataset loading as in-memory tensors, with class-balanced subsampling.

Everything is held as tensors (not lazy torchvision datasets). That keeps training fast on CPU
and lets dream strategies work on whole batches of images directly.
"""

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

_DATASETS = {
    "mnist": datasets.MNIST,
    "fashion_mnist": datasets.FashionMNIST,
    "cifar10": datasets.CIFAR10,
}

# Per-channel mean/std of each full training set.
_STATS = {
    "mnist": ((0.1307,), (0.3081,)),
    "fashion_mnist": ((0.2860,), (0.3530,)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
}


@dataclass
class Splits:
    train: TensorDataset
    val: TensorDataset
    test: TensorDataset
    in_channels: int
    num_classes: int
    image_size: int


def _to_tensors(ds) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.as_tensor(ds.data)
    if x.ndim == 3:  # (N, H, W) grayscale
        x = x.unsqueeze(1)
    else:  # CIFAR stores (N, H, W, C)
        x = x.permute(0, 3, 1, 2)
    x = x.float().div(255.0)
    y = torch.as_tensor(ds.targets, dtype=torch.long)
    return x, y


def _normalize(x: torch.Tensor, name: str) -> torch.Tensor:
    mean, std = _STATS[name]
    mean = torch.tensor(mean).view(1, -1, 1, 1)
    std = torch.tensor(std).view(1, -1, 1, 1)
    return (x - mean) / std


def _balanced_indices(y: torch.Tensor, per_class: int, gen: torch.Generator) -> torch.Tensor:
    idx = []
    for c in y.unique():
        cls = (y == c).nonzero(as_tuple=True)[0]
        if per_class > len(cls):
            raise ValueError(f"Requested {per_class} samples of class {c}, only {len(cls)} exist")
        idx.append(cls[torch.randperm(len(cls), generator=gen)[:per_class]])
    return torch.cat(idx)


def load_splits(
    name: str, train_size: int, val_size: int = 5000, root: str = "data", seed: int = 0
) -> Splits:
    """Load a dataset and return a small class-balanced train set, a val set, and the full test set.

    The val set is drawn from training images that are *not* in the train subset, so that
    hyperparameters such as dream distortion strength can be tuned without touching test data.
    """
    if name not in _DATASETS:
        raise ValueError(f"Unknown dataset {name!r}; choose from {list(_DATASETS)}")
    cls = _DATASETS[name]
    x_tr, y_tr = _to_tensors(cls(root, train=True, download=True))
    x_te, y_te = _to_tensors(cls(root, train=False, download=True))
    x_tr, x_te = _normalize(x_tr, name), _normalize(x_te, name)

    num_classes = int(y_tr.max()) + 1
    gen = torch.Generator().manual_seed(seed)
    train_idx = _balanced_indices(y_tr, train_size // num_classes, gen)

    remaining = torch.ones(len(y_tr), dtype=torch.bool)
    remaining[train_idx] = False
    pool = remaining.nonzero(as_tuple=True)[0]
    val_idx = pool[torch.randperm(len(pool), generator=gen)[:val_size]]

    return Splits(
        train=TensorDataset(x_tr[train_idx], y_tr[train_idx]),
        val=TensorDataset(x_tr[val_idx], y_tr[val_idx]),
        test=TensorDataset(x_te, y_te),
        in_channels=x_tr.shape[1],
        num_classes=num_classes,
        image_size=x_tr.shape[-1],
    )


def make_loader(ds: TensorDataset, batch_size: int, shuffle: bool, seed: int = 0) -> DataLoader:
    gen = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=gen)
