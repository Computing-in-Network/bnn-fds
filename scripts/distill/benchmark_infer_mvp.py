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
    p = argparse.ArgumentParser(description="MLP 蒸馏推理延迟基准 MVP")
    p.add_argument("--cases", type=Path, default=Path("data/meta/cases_mvp.csv"))
    p.add_argument(
        "--model",
        type=Path,
        default=Path("distill/artifacts/mvp_distill_mlp/student_mlp_model.pt"),
    )
    p.add_argument("--num-repeats", type=int, default=50)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("distill/artifacts/mvp_distill_mlp/infer_benchmark.json"),
    )
    return p.parse_args()


def load_cases(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    x = np.array([[float(r[f]) for f in FEATURES] for r in rows], dtype=np.float32)
    return x


def main() -> None:
    args = parse_args()
    x = load_cases(args.cases)
    ckpt = torch.load(args.model, map_location="cpu")

    model = MLPStudent(in_dim=len(FEATURES), hidden_dim=32)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    x_mean = np.array(ckpt["x_mean"], dtype=np.float32)
    x_std = np.array(ckpt["x_std"], dtype=np.float32)
    y_mean = float(ckpt["y_mean"])
    y_std = float(ckpt["y_std"])

    x_n = (x - x_mean) / x_std
    x_t = torch.tensor(x_n, dtype=torch.float32)

    # warmup
    with torch.no_grad():
        _ = model(x_t)

    latencies_ms = []
    with torch.no_grad():
        for _ in range(args.num_repeats):
            t0 = time.perf_counter()
            y_n = model(x_t).squeeze(1).cpu().numpy()
            _ = y_n * y_std + y_mean
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

    arr = np.array(latencies_ms, dtype=np.float64)
    p95 = float(np.percentile(arr, 95))
    avg = float(arr.mean())
    p50 = float(np.percentile(arr, 50))

    report = {
        "num_cases": int(x.shape[0]),
        "num_repeats": int(args.num_repeats),
        "p50_latency_ms": p50,
        "p95_latency_ms": p95,
        "avg_latency_ms": avg,
        "meets_1s_goal": p95 <= 1000.0,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"基准完成: {args.output}")
    print(
        f"num_cases={report['num_cases']} repeats={report['num_repeats']} "
        f"p95={report['p95_latency_ms']:.3f}ms avg={report['avg_latency_ms']:.3f}ms"
    )


if __name__ == "__main__":
    main()
