#!/usr/bin/env python3
import argparse
import csv
import json
import random
from pathlib import Path


REQUIRED_FIELDS = [
    "fire_x",
    "fire_y",
    "hrr_peak_kw",
    "vent_open_ratio",
    "duration_s",
]


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    if "ranges" not in config:
        raise ValueError("配置缺少 ranges 字段")
    for field in REQUIRED_FIELDS:
        if field not in config["ranges"]:
            raise ValueError(f"配置缺少参数范围: {field}")
        bounds = config["ranges"][field]
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ValueError(f"参数范围格式错误: {field}")
        if bounds[0] >= bounds[1]:
            raise ValueError(f"参数范围必须满足 min < max: {field}")
    return config


def sample_cases(n_cases: int, config: dict) -> list[dict]:
    seed = int(config.get("seed", 42))
    random.seed(seed)
    ranges = config["ranges"]
    rows = []
    for i in range(1, n_cases + 1):
        row = {"case_id": f"case_{i:06d}"}
        for field in REQUIRED_FIELDS:
            low, high = ranges[field]
            row[field] = round(random.uniform(low, high), 6)
        rows.append(row)
    return rows


def write_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["case_id"] + REQUIRED_FIELDS
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 FDS 数据引擎 MVP 工况清单")
    parser.add_argument(
        "--n-cases",
        type=int,
        default=100,
        help="生成工况数量，例如 100 或 1000",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data/case_space_mvp.json"),
        help="参数空间配置文件（JSON）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/meta/cases_mvp.csv"),
        help="输出 CSV 路径",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_cases <= 0:
        raise ValueError("n-cases 必须为正整数")
    config = load_config(args.config)
    rows = sample_cases(args.n_cases, config)
    write_csv(rows, args.output)
    print(f"已生成 {len(rows)} 条工况: {args.output}")


if __name__ == "__main__":
    main()
