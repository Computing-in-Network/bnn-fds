#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path


TIME_STEP_RE = re.compile(
    r"Time Step:\s*(\d+),\s*Simulation Time:\s*([0-9]*\.?[0-9]+)\s*s"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="从真实 FDS run.log 提取监督标签")
    p.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("data/fds_outputs"),
        help="FDS 输出根目录",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/meta/fds_labels_real_mvp.csv"),
        help="标签 CSV 输出路径",
    )
    p.add_argument(
        "--min-sim-time",
        type=float,
        default=0.5,
        help="最小有效模拟时长（秒），低于该值的样本剔除",
    )
    return p.parse_args()


def parse_one(log_path: Path) -> tuple[str, str]:
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    last_step = ""
    last_time = ""
    for m in TIME_STEP_RE.finditer(text):
        last_step = m.group(1)
        last_time = m.group(2)
    return last_step, last_time


def main() -> None:
    args = parse_args()
    rows = []

    if args.outputs_dir.exists():
        for case_dir in sorted(args.outputs_dir.iterdir()):
            if not case_dir.is_dir():
                continue
            case_id = case_dir.name
            log_path = case_dir / "run.log"
            if not log_path.exists():
                continue
            last_step, last_time = parse_one(log_path)
            if not last_time:
                continue
            sim_time = float(last_time)
            if sim_time < args.min_sim_time:
                continue
            rows.append(
                {
                    "case_id": case_id,
                    "last_time_step": int(last_step),
                    "last_sim_time_s": sim_time,
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["case_id", "last_time_step", "last_sim_time_s"]
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"标签提取完成: {args.output} (rows={len(rows)})")


if __name__ == "__main__":
    main()
