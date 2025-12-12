import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Make the parent directory importable (handles spaces in path)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mlp.model import MLPInitializer


def load_npz(path):
    data = np.load(path, allow_pickle=True)
    return data["X"], data["y"], data["meta"]


def split_by_config(meta, train_ratio=0.7, val_ratio=0.15, seed=0):
    rng = np.random.default_rng(seed)
    cfg_to_indices = defaultdict(list)
    for idx, m in enumerate(meta):
        key = tuple(m["config"])
        cfg_to_indices[key].append(idx)
    cfg_keys = list(cfg_to_indices.keys())
    rng.shuffle(cfg_keys)
    n = len(cfg_keys)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)
    train_cfg = set(cfg_keys[:n_train])
    val_cfg = set(cfg_keys[n_train : n_train + n_val])
    test_cfg = set(cfg_keys[n_train + n_val :])

    splits = {"train": [], "val": [], "test": []}
    for k, idxs in cfg_to_indices.items():
        if k in train_cfg:
            splits["train"].extend(idxs)
        elif k in val_cfg:
            splits["val"].extend(idxs)
        else:
            splits["test"].extend(idxs)
    return splits


def make_loaders(X, y, splits, batch_size=256):
    loaders = {}
    for name, idxs in splits.items():
        xs = torch.tensor(X[idxs], dtype=torch.float32)
        ys = torch.tensor(y[idxs], dtype=torch.float32)
        ds = TensorDataset(xs, ys)
        loaders[name] = DataLoader(ds, batch_size=batch_size, shuffle=name == "train")
    return loaders


def train(args):
    X, y, meta = load_npz(args.data)
    splits = split_by_config(meta, seed=args.seed)
    loaders = make_loaders(X, y, splits, batch_size=args.batch_size)

    input_dim = X.shape[1]
    output_dim = y.shape[1]
    model = MLPInitializer(input_dim, output_dim).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=5, min_lr=1e-5
    )
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_epoch = 0
    early_stop_patience = 15
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0
        n = 0
        for xb, yb in loaders["train"]:
            xb = xb.to(args.device)
            yb = yb.to(args.device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
            n += xb.size(0)
        train_loss = total / max(1, n)

        def eval_split(split):
            model.eval()
            total = 0
            n = 0
            with torch.no_grad():
                for xb, yb in loaders[split]:
                    xb = xb.to(args.device)
                    yb = yb.to(args.device)
                    pred = model(xb)
                    loss = loss_fn(pred, yb)
                    total += loss.item() * xb.size(0)
                    n += xb.size(0)
            return total / max(1, n)

        val_loss = eval_split("val")

        print(f"Epoch {epoch}: train {train_loss:.6f} | val {val_loss:.6f}")
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), args.out)
            with open(args.out + ".meta.json", "w") as f:
                json.dump(
                    {
                        "input_dim": input_dim,
                        "output_dim": output_dim,
                        "hidden": [512, 512],
                        "val_loss": val_loss,
                        "train_loss": train_loss,
                        "best_epoch": epoch,
                    },
                    f,
                    indent=2,
                )
        elif epoch - best_epoch >= early_stop_patience:
            print(f"Early stopping at epoch {epoch} (no val improve for {early_stop_patience} epochs)")
            break

    test_loss = None
    if loaders["test"]:
        test_loss = 0.0
        n = 0
        model.eval()
        with torch.no_grad():
            for xb, yb in loaders["test"]:
                xb = xb.to(args.device)
                yb = yb.to(args.device)
                loss = loss_fn(model(xb), yb)
                test_loss += loss.item() * xb.size(0)
                n += xb.size(0)
        test_loss /= max(1, n)
    print(f"Best val loss: {best_val:.6f} | test loss: {test_loss}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="mlp/ik_dataset.npz")
    parser.add_argument("--out", default="mlp/mlp_initializer.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    train(args)

