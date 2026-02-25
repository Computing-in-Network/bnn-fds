#!/usr/bin/env python3
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


FEATURES = ["fire_x", "fire_y", "hrr_peak_kw", "vent_open_ratio", "duration_s"]


class StudentMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim_1: int, hidden_dim_2: int):
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
    p = argparse.ArgumentParser(description="真实 FDS 蒸馏模型推理时延基准")
    p.add_argument("--cases", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--num-repeats", type=int, default=5000)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def load_cases(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return np.array([[float(r[c]) for c in FEATURES] for r in rows], dtype=np.float32)


def main() -> None:
    args = parse_args()
    x = load_cases(args.cases)
    if x.shape[0] == 0:
        raise ValueError("cases 为空")

    ckpt = torch.load(args.model, map_location="cpu")
    x_mean = np.array(ckpt["x_mean"], dtype=np.float32)
    x_std = np.array(ckpt["x_std"], dtype=np.float32)
    y_mean = float(ckpt["y_mean"])
    y_std = float(ckpt["y_std"])
    x_n = (x - x_mean) / x_std

    model_type = ckpt.get("model_type", "mlp")
    if model_type == "linear":
        w = np.array(ckpt["linear_weights"], dtype=np.float32)
        b = float(ckpt["linear_bias"])

        def infer_once() -> np.ndarray:
            y_n = x_n @ w + b
            return y_n * y_std + y_mean

    else:
        h1 = int(ckpt.get("hidden_dim_1", 64))
        h2 = int(ckpt.get("hidden_dim_2", h1))
        model = StudentMLP(in_dim=len(FEATURES), hidden_dim_1=h1, hidden_dim_2=h2)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        x_t = torch.tensor(x_n, dtype=torch.float32)

        def infer_once() -> np.ndarray:
            with torch.no_grad():
                y_n = model(x_t).squeeze(1).cpu().numpy()
            return y_n * y_std + y_mean

    for _ in range(args.warmup):
        _ = infer_once()

    latencies_ms = []
    for _ in range(args.num_repeats):
        t0 = time.perf_counter()
        _ = infer_once()
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    arr = np.array(latencies_ms, dtype=np.float64)
    report = {
        "model_type": model_type,
        "num_cases": int(x.shape[0]),
        "num_repeats": int(args.num_repeats),
        "warmup": int(args.warmup),
        "p50_latency_ms_per_batch": float(np.percentile(arr, 50)),
        "p95_latency_ms_per_batch": float(np.percentile(arr, 95)),
        "avg_latency_ms_per_batch": float(arr.mean()),
        "avg_latency_ms_per_case": float(arr.mean() / x.shape[0]),
        "meets_1s_goal": bool(np.percentile(arr, 95) <= 1000.0),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"基准完成: {args.output}")
    print(
        f"type={report['model_type']} cases={report['num_cases']} "
        f"p95_batch={report['p95_latency_ms_per_batch']:.6f}ms "
        f"avg_case={report['avg_latency_ms_per_case']:.6f}ms"
    )


if __name__ == "__main__":
    main()
