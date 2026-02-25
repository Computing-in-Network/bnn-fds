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


class StudentMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim_1: int = 64, hidden_dim_2: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Linear(hidden_dim_2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="真实 FDS 标签蒸馏训练 MVP")
    p.add_argument("--cases", type=Path, default=Path("data/meta/cases_mvp.csv"))
    p.add_argument("--labels", type=Path, default=Path("data/meta/fds_labels_real_mvp.csv"))
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("distill/artifacts/real_mvp"),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--hidden-dim-1", type=int, default=64)
    p.add_argument("--hidden-dim-2", type=int, default=64)
    p.add_argument(
        "--select-model",
        type=str,
        default="mlp",
        choices=["mlp", "linear", "auto"],
        help="主模型选择策略：mlp(默认)、linear、auto(按验证 MAE 选优)",
    )
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_csv(path: Path) -> tuple[list[str], list[dict]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return (reader.fieldnames or []), list(reader)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    value_range = float(np.max(y_true) - np.min(y_true))
    if value_range < 1e-6:
        # 单样本或近常量目标时，回退到幅值尺度避免分母退化
        value_range = max(float(abs(np.mean(y_true))), 1.0)
    nmae = float(mae / (value_range + EPS))
    accuracy = float(max(0.0, 1.0 - nmae) * 100.0)
    return {"mae": mae, "nmae": nmae, "accuracy": accuracy}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    _, cases = load_csv(args.cases)
    _, labels = load_csv(args.labels)
    label_map = {r["case_id"]: float(r["last_sim_time_s"]) for r in labels}

    x_list = []
    y_list = []
    ids = []
    for r in cases:
        cid = r["case_id"]
        if cid not in label_map:
            continue
        x_list.append([float(r[f]) for f in FEATURES])
        y_list.append(label_map[cid])
        ids.append(cid)

    if len(x_list) < 3:
        raise ValueError("有效真实标签样本不足（至少需要 3 条）")

    x = np.array(x_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    # 归一化
    x_mean = x.mean(axis=0, keepdims=True)
    x_std = x.std(axis=0, keepdims=True) + 1e-6
    x_n = (x - x_mean) / x_std

    idx = np.arange(len(x_n))
    np.random.shuffle(idx)
    split = max(1, int(len(idx) * (1.0 - args.val_ratio)))
    tr_idx = idx[:split]
    va_idx = idx[split:] if split < len(idx) else idx[-1:]

    y_tr = y[tr_idx]
    y_va = y[va_idx]
    y_mean = float(y_tr.mean())
    y_std = float(y_tr.std() + 1e-6)

    xtr = torch.tensor(x_n[tr_idx], dtype=torch.float32)
    ytr = torch.tensor((y_tr - y_mean) / y_std, dtype=torch.float32).unsqueeze(1)
    xva = torch.tensor(x_n[va_idx], dtype=torch.float32)
    yva = torch.tensor((y_va - y_mean) / y_std, dtype=torch.float32).unsqueeze(1)

    # 线性基线
    xtr_np = x_n[tr_idx]
    xva_np = x_n[va_idx]
    ytr_n_np = ((y_tr - y_mean) / y_std).astype(np.float32)
    yva_n_np = ((y_va - y_mean) / y_std).astype(np.float32)
    xtr_aug = np.concatenate([xtr_np, np.ones((xtr_np.shape[0], 1), dtype=np.float32)], axis=1)
    theta, _, _, _ = np.linalg.lstsq(xtr_aug, ytr_n_np, rcond=None)
    w_lin = theta[:-1]
    b_lin = float(theta[-1])
    ptr_lin = (xtr_np @ w_lin + b_lin) * y_std + y_mean
    pva_lin = (xva_np @ w_lin + b_lin) * y_std + y_mean
    mtr_lin = metrics(y_tr, ptr_lin)
    mva_lin = metrics(y_va, pva_lin)

    # MLP 模型
    model = StudentMLP(
        in_dim=len(FEATURES),
        hidden_dim_1=args.hidden_dim_1,
        hidden_dim_2=args.hidden_dim_2,
    )
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    history = []
    for ep in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()
        pred = model(xtr)
        loss = loss_fn(pred, ytr)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            v = loss_fn(model(xva), yva).item()
        history.append({"epoch": ep, "train_mse": float(loss.item()), "val_mse": float(v)})

    model.eval()
    with torch.no_grad():
        ptr_n = model(xtr).squeeze(1).cpu().numpy()
        pva_n = model(xva).squeeze(1).cpu().numpy()
    ptr_mlp = ptr_n * y_std + y_mean
    pva_mlp = pva_n * y_std + y_mean
    mtr_mlp = metrics(y_tr, ptr_mlp)
    mva_mlp = metrics(y_va, pva_mlp)

    # 选择主模型：默认优先 MLP，保留 linear 作为可切换基线
    if args.select_model == "linear":
        use_linear = True
    elif args.select_model == "mlp":
        use_linear = False
    else:
        use_linear = mva_lin["mae"] <= mva_mlp["mae"]

    if use_linear:
        mtr, mva = mtr_lin, mva_lin
        model_payload = {
            "model_type": "linear",
            "linear_weights": w_lin.tolist(),
            "linear_bias": b_lin,
        }
    else:
        mtr, mva = mtr_mlp, mva_mlp
        model_payload = {
            "model_type": "mlp",
            "hidden_dim_1": args.hidden_dim_1,
            "hidden_dim_2": args.hidden_dim_2,
            "state_dict": model.state_dict(),
        }

    # 保存产物
    torch.save(
        {
            **model_payload,
            "x_mean": x_mean.tolist(),
            "x_std": x_std.tolist(),
            "y_mean": y_mean,
            "y_std": y_std,
            "features": FEATURES,
            "target": "last_sim_time_s",
            "num_samples": len(x_list),
            "val_metrics_linear": mva_lin,
            "val_metrics_mlp": mva_mlp,
            "train_metrics_linear": mtr_lin,
            "train_metrics_mlp": mtr_mlp,
            "selection_policy": args.select_model,
        },
        args.output_dir / "student_real_mvp.pt",
    )
    (args.output_dir / "metrics_train.json").write_text(
        json.dumps(mtr, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "metrics_val.json").write_text(
        json.dumps(mva, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "train_history.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_mse", "val_mse"])
        writer.writeheader()
        for r in history:
            writer.writerow(r)

    print(f"真实蒸馏训练完成: {args.output_dir}")
    print(
        f"samples={len(x_list)} "
        f"selected={'linear' if use_linear else 'mlp'} policy={args.select_model} "
        f"hidden=({args.hidden_dim_1},{args.hidden_dim_2}) "
        f"train={mtr} val={mva}"
    )


if __name__ == "__main__":
    main()
