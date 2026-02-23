#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


FEATURES = ["fire_x", "fire_y", "hrr_peak_kw", "vent_open_ratio", "duration_s"]


class MLPStudent(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 32):
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MLP 蒸馏模型推理 MVP")
    p.add_argument("--cases", type=Path, default=Path("data/meta/cases_mvp.csv"))
    p.add_argument(
        "--model",
        type=Path,
        default=Path("distill/artifacts/mvp_distill_mlp/student_mlp_model.pt"),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("distill/artifacts/mvp_distill_mlp/infer_preds.csv"),
    )
    return p.parse_args()


def load_cases(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        missing = [c for c in (["case_id"] + FEATURES) if c not in fields]
        if missing:
            raise ValueError(f"cases 缺少字段: {missing}")
        return list(reader)


def main() -> None:
    args = parse_args()
    rows = load_cases(args.cases)
    ckpt = torch.load(args.model, map_location="cpu")

    model = MLPStudent(in_dim=len(FEATURES), hidden_dim=32)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    x = np.array([[float(r[f]) for f in FEATURES] for r in rows], dtype=np.float32)
    x_mean = np.array(ckpt["x_mean"], dtype=np.float32)
    x_std = np.array(ckpt["x_std"], dtype=np.float32)
    y_mean = float(ckpt["y_mean"])
    y_std = float(ckpt["y_std"])

    x_n = (x - x_mean) / x_std
    with torch.no_grad():
        y_n = model(torch.tensor(x_n, dtype=torch.float32)).squeeze(1).cpu().numpy()
    y = y_n * y_std + y_mean

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "student_pred_temp_peak"])
        writer.writeheader()
        for r, pred in zip(rows, y):
            writer.writerow({"case_id": r["case_id"], "student_pred_temp_peak": round(float(pred), 6)})

    print(f"推理完成: {args.output} (rows={len(rows)})")


if __name__ == "__main__":
    main()
