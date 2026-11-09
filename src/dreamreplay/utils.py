"""Seeding, config loading (including sweeps), and device selection."""

import copy
import random

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def expand_sweep(cfg: dict) -> list[dict]:
    """Expand a config's optional `sweep` block into one config per value.

    `sweep` maps a single dotted key to a list of values, for example
    `sweep: {model.dropout: [0.25, 0.5]}`. Each returned config has that key set, no `sweep`
    block, and a `run` name such as "dropout=0.25" that names its results folder.
    A config without `sweep` comes back unchanged, as a one-item list.
    """
    sweep = cfg.get("sweep")
    if not sweep:
        return [cfg]
    if len(sweep) != 1:
        raise ValueError(f"A sweep must vary exactly one key, got {list(sweep)}")
    ((key, values),) = sweep.items()
    *parents, leaf = key.split(".")
    runs = []
    for value in values:
        run = copy.deepcopy(cfg)
        del run["sweep"]
        node = run
        for k in parents:
            node = node[k]
        if leaf not in node:
            raise KeyError(f"Sweep key {key!r} is not in the config")
        node[leaf] = value
        run["run"] = f"{leaf}={value}"
        runs.append(run)
    return runs


def get_device(pref: str = "auto") -> torch.device:
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)
