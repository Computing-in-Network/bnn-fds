#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path


EPS = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MVP 指标评估脚本")
    parser.add_argument("--y-true", type=Path, required=True, help="真实值 CSV")
    parser.add_argument("--y-pred", type=Path, required=True, help="预测值 CSV")
    parser.add_argument("--latency", type=Path, required=True, help="时延样本文件（毫秒）")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval/reports/metrics_mvp.json"),
        help="评估结果输出 JSON",
    )
    return parser.parse_args()


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        raise FileNotFoundError(f"未找到文件: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV 缺少表头: {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV 没有数据行: {path}")
    return reader.fieldnames, rows


def _to_float(v: str, col: str) -> float:
    try:
        return float(v)
    except ValueError as e:
        raise ValueError(f"列 {col} 含非数值: {v}") from e


def _align_rows(
    true_rows: list[dict[str, str]], pred_rows: list[dict[str, str]]
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if "case_id" in true_rows[0] and "case_id" in pred_rows[0]:
        pred_map = {r["case_id"]: r for r in pred_rows}
        aligned_true = []
        aligned_pred = []
        for tr in true_rows:
            cid = tr["case_id"]
            if cid not in pred_map:
                raise ValueError(f"y_pred 缺少 case_id: {cid}")
            aligned_true.append(tr)
            aligned_pred.append(pred_map[cid])
        return aligned_true, aligned_pred
    if len(true_rows) != len(pred_rows):
        raise ValueError("无 case_id 对齐时，y_true 与 y_pred 行数必须一致")
    return true_rows, pred_rows


def compute_metrics(
    true_fields: list[str],
    true_rows: list[dict[str, str]],
    pred_fields: list[str],
    pred_rows: list[dict[str, str]],
) -> dict:
    if set(true_fields) != set(pred_fields):
        raise ValueError("y_true 与 y_pred 列集合不一致")

    numeric_cols = [c for c in true_fields if c != "case_id"]
    if not numeric_cols:
        raise ValueError("未发现可计算的数值列")

    true_rows, pred_rows = _align_rows(true_rows, pred_rows)

    col_true_values: dict[str, list[float]] = {c: [] for c in numeric_cols}
    col_abs_errors: dict[str, list[float]] = {c: [] for c in numeric_cols}

    total_abs_error = 0.0
    total_count = 0
    for tr, pr in zip(true_rows, pred_rows):
        for col in numeric_cols:
            tv = _to_float(tr[col], col)
            pv = _to_float(pr[col], col)
            ae = abs(tv - pv)
            col_true_values[col].append(tv)
            col_abs_errors[col].append(ae)
            total_abs_error += ae
            total_count += 1

    mae = total_abs_error / total_count

    per_column = {}
    nmae_values = []
    for col in numeric_cols:
        col_mae = sum(col_abs_errors[col]) / len(col_abs_errors[col])
        col_min = min(col_true_values[col])
        col_max = max(col_true_values[col])
        col_range = col_max - col_min
        col_nmae = col_mae / (col_range + EPS)
        nmae_values.append(col_nmae)
        per_column[col] = {
            "mae": col_mae,
            "nmae": col_nmae,
            "min_true": col_min,
            "max_true": col_max,
        }

    nmae = sum(nmae_values) / len(nmae_values)
    accuracy = max(0.0, 1.0 - nmae) * 100.0
    return {
        "num_samples": len(true_rows),
        "num_targets": len(numeric_cols),
        "mae": mae,
        "nmae": nmae,
        "accuracy": accuracy,
        "per_column": per_column,
    }


def compute_p95_latency(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"未找到时延文件: {path}")
    vals = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            vals.append(float(s))
    if not vals:
        raise ValueError("时延文件为空")
    vals.sort()
    idx = max(0, math.ceil(0.95 * len(vals)) - 1)
    return {"num_latency_samples": len(vals), "p95_latency_ms": vals[idx]}


def main() -> None:
    args = parse_args()
    true_fields, true_rows = _read_csv(args.y_true)
    pred_fields, pred_rows = _read_csv(args.y_pred)
    metrics = compute_metrics(true_fields, true_rows, pred_fields, pred_rows)
    lat = compute_p95_latency(args.latency)

    result = {
        "input": {
            "y_true": str(args.y_true),
            "y_pred": str(args.y_pred),
            "latency": str(args.latency),
        },
        "metrics": {
            "mae": metrics["mae"],
            "nmae": metrics["nmae"],
            "accuracy": metrics["accuracy"],
            "p95_latency_ms": lat["p95_latency_ms"],
        },
        "summary": {
            "num_samples": metrics["num_samples"],
            "num_targets": metrics["num_targets"],
            "num_latency_samples": lat["num_latency_samples"],
        },
        "per_column": metrics["per_column"],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"评估完成，输出: {args.output}")


if __name__ == "__main__":
    main()
