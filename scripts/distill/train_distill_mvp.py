#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import numpy as np


EPS = 1e-8
FEATURES = ["fire_x", "fire_y", "hrr_peak_kw", "vent_open_ratio", "duration_s"]
TARGET = "teacher_temp_peak"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Teacher-Student 蒸馏训练 MVP")
    p.add_argument(
        "--cases",
        type=Path,
        default=Path("data/meta/cases_mvp.csv"),
        help="输入 cases CSV",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("distill/artifacts/mvp_distill"),
        help="蒸馏产物输出目录",
    )
    return p.parse_args()


def load_cases(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"cases 不存在: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        required = ["case_id"] + FEATURES
        missing = [c for c in required if c not in fields]
        if missing:
            raise ValueError(f"cases 缺少字段: {missing}")
        return list(reader)


def teacher_fn(x: np.ndarray) -> np.ndarray:
    # x columns: fire_x, fire_y, hrr_peak_kw, vent_open_ratio, duration_s
    fire_x = x[:, 0]
    fire_y = x[:, 1]
    hrr = x[:, 2]
    vent = x[:, 3]
    duration = x[:, 4]
    # 构造可解释 teacher 温度峰值（近似）
    y = (
        20.0
        + 0.028 * hrr
        + 9.0 * vent
        + 0.012 * duration
        + 0.18 * fire_x
        - 0.11 * fire_y
    )
    return y


def fit_linear_regression(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    # 增广项拟合偏置: y = xw + b
    ones = np.ones((x.shape[0], 1))
    xa = np.concatenate([x, ones], axis=1)
    theta, _, _, _ = np.linalg.lstsq(xa, y, rcond=None)
    w = theta[:-1]
    b = float(theta[-1])
    return w, b


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    abs_err = np.abs(y_true - y_pred)
    mae = float(abs_err.mean())
    value_range = float(y_true.max() - y_true.min())
    nmae = float(mae / (value_range + EPS))
    accuracy = float(max(0.0, 1.0 - nmae) * 100.0)
    return {"mae": mae, "nmae": nmae, "accuracy": accuracy}


def save_teacher_targets(rows: list[dict], y_teacher: np.ndarray, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", TARGET])
        writer.writeheader()
        for r, y in zip(rows, y_teacher):
            writer.writerow({"case_id": r["case_id"], TARGET: round(float(y), 6)})


def save_predictions(rows: list[dict], y_pred: np.ndarray, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "student_pred_temp_peak"])
        writer.writeheader()
        for r, y in zip(rows, y_pred):
            writer.writerow({"case_id": r["case_id"], "student_pred_temp_peak": round(float(y), 6)})


def main() -> None:
    args = parse_args()
    rows = load_cases(args.cases)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    x = np.array([[float(r[f]) for f in FEATURES] for r in rows], dtype=np.float64)
    y_teacher = teacher_fn(x)

    w, b = fit_linear_regression(x, y_teacher)
    y_pred = x @ w + b
    m = metrics(y_teacher, y_pred)

    teacher_path = args.output_dir / "teacher_targets_mvp.csv"
    pred_path = args.output_dir / "student_preds_mvp.csv"
    model_path = args.output_dir / "student_model_mvp.json"
    metrics_path = args.output_dir / "metrics_mvp.json"

    save_teacher_targets(rows, y_teacher, teacher_path)
    save_predictions(rows, y_pred, pred_path)

    model_data = {
        "model": "linear_regression_student",
        "feature_order": FEATURES,
        "target": TARGET,
        "weights": {name: float(val) for name, val in zip(FEATURES, w)},
        "bias": b,
    }
    model_path.write_text(json.dumps(model_data, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics_path.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"蒸馏训练完成: {args.output_dir}")
    print(f"metrics: mae={m['mae']:.6f}, nmae={m['nmae']:.6f}, accuracy={m['accuracy']:.2f}%")


if __name__ == "__main__":
    main()
