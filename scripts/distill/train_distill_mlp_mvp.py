#!/usr/bin/env python3
import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


EPS = 1e-8
FEATURES = ["fire_x", "fire_y", "hrr_peak_kw", "vent_open_ratio", "duration_s"]
TARGET = "teacher_temp_peak"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MLP Student 蒸馏训练 MVP")
    p.add_argument("--cases", type=Path, default=Path("data/meta/cases_mvp.csv"))
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("distill/artifacts/mvp_distill_mlp"),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--val-ratio", type=float, default=0.2)
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_cases(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        required = ["case_id"] + FEATURES
        missing = [c for c in required if c not in fields]
        if missing:
            raise ValueError(f"cases 缺少字段: {missing}")
        return list(reader)


def teacher_fn(x: np.ndarray) -> np.ndarray:
    fire_x = x[:, 0]
    fire_y = x[:, 1]
    hrr = x[:, 2]
    vent = x[:, 3]
    duration = x[:, 4]
    y = (
        20.0
        + 0.028 * hrr
        + 9.0 * vent
        + 0.012 * duration
        + 0.18 * fire_x
        - 0.11 * fire_y
        + 0.0000015 * hrr * duration
        - 0.035 * fire_x * vent
    )
    return y.astype(np.float32)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    nmae = float(mae / (float(np.max(y_true) - np.min(y_true)) + EPS))
    accuracy = float(max(0.0, 1.0 - nmae) * 100.0)
    return {"mae": mae, "nmae": nmae, "accuracy": accuracy}


class MLPStudent(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_cases(args.cases)
    x = np.array([[float(r[f]) for f in FEATURES] for r in rows], dtype=np.float32)
    y = teacher_fn(x)

    # 标准化输入，提升训练稳定性
    x_mean = x.mean(axis=0, keepdims=True)
    x_std = x.std(axis=0, keepdims=True) + 1e-6
    x_n = (x - x_mean) / x_std

    n = len(x_n)
    idx = np.arange(n)
    np.random.shuffle(idx)
    split = int(n * (1.0 - args.val_ratio))
    tr_idx, va_idx = idx[:split], idx[split:]

    y_tr_raw = y[tr_idx]
    y_va_raw = y[va_idx]
    y_mean = float(y_tr_raw.mean())
    y_std = float(y_tr_raw.std() + 1e-6)
    y_tr_n = (y_tr_raw - y_mean) / y_std
    y_va_n = (y_va_raw - y_mean) / y_std

    x_tr = torch.tensor(x_n[tr_idx], dtype=torch.float32)
    y_tr = torch.tensor(y_tr_n, dtype=torch.float32).unsqueeze(1)
    x_va = torch.tensor(x_n[va_idx], dtype=torch.float32)
    y_va = torch.tensor(y_va_n, dtype=torch.float32).unsqueeze(1)

    model = MLPStudent(in_dim=len(FEATURES), hidden_dim=args.hidden_dim)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    history = []
    for ep in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()
        pred = model(x_tr)
        loss = loss_fn(pred, y_tr)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(x_va), y_va).item()
        history.append(
            {
                "epoch": ep,
                "train_mse": float(loss.item()),
                "val_mse": float(val_loss),
            }
        )

    model.eval()
    with torch.no_grad():
        y_tr_pred_n = model(x_tr).squeeze(1).cpu().numpy()
        y_va_pred_n = model(x_va).squeeze(1).cpu().numpy()

    y_tr_pred = y_tr_pred_n * y_std + y_mean
    y_va_pred = y_va_pred_n * y_std + y_mean
    m_tr = compute_metrics(y_tr_raw, y_tr_pred)
    m_va = compute_metrics(y_va_raw, y_va_pred)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "x_mean": x_mean.tolist(),
            "x_std": x_std.tolist(),
            "y_mean": y_mean,
            "y_std": y_std,
            "features": FEATURES,
            "target": TARGET,
            "seed": args.seed,
        },
        args.output_dir / "student_mlp_model.pt",
    )

    with (args.output_dir / "metrics_train.json").open("w", encoding="utf-8") as f:
        json.dump(m_tr, f, ensure_ascii=False, indent=2)
    with (args.output_dir / "metrics_val.json").open("w", encoding="utf-8") as f:
        json.dump(m_va, f, ensure_ascii=False, indent=2)

    with (args.output_dir / "train_history.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_mse", "val_mse"])
        writer.writeheader()
        for row in history:
            writer.writerow(row)

    print(f"MLP 蒸馏完成: {args.output_dir}")
    print(f"train: {m_tr}")
    print(f"val:   {m_va}")


if __name__ == "__main__":
    main()
