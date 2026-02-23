#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="构建过滤后的真实 FDS 标签数据集")
    p.add_argument(
        "--input",
        type=Path,
        default=Path("data/meta/fds_labels_real_mvp.csv"),
        help="原始标签 CSV",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/meta/fds_labels_real_mvp_filtered.csv"),
        help="过滤后标签 CSV",
    )
    p.add_argument(
        "--min-time-step",
        type=int,
        default=50,
        help="最小时间步门槛，小于该值的样本会被剔除",
    )
    p.add_argument(
        "--min-sim-time",
        type=float,
        default=1.0,
        help="最小模拟时长（秒）门槛",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with args.input.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    kept = []
    for r in rows:
        ts = int(float(r["last_time_step"]))
        st = float(r["last_sim_time_s"])
        if ts < args.min_time_step:
            continue
        if st < args.min_sim_time:
            continue
        kept.append(
            {
                "case_id": r["case_id"],
                "last_time_step": ts,
                "last_sim_time_s": st,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["case_id", "last_time_step", "last_sim_time_s"]
        )
        writer.writeheader()
        for r in kept:
            writer.writerow(r)

    print(
        f"过滤完成: {args.output} (input={len(rows)} kept={len(kept)} "
        f"min_time_step={args.min_time_step} min_sim_time={args.min_sim_time})"
    )


if __name__ == "__main__":
    main()
