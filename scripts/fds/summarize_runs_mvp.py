#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path


TIME_STEP_RE = re.compile(
    r"Time Step:\s*(\d+),\s*Simulation Time:\s*([0-9]*\.?[0-9]+)\s*s"
)

ERROR_HINT_PATTERNS = [
    "error while loading shared libraries",
    "not found",
    "ERROR",
    "Fatal error",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="汇总 FDS 运行日志为结构化 CSV")
    p.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("data/fds_outputs"),
        help="FDS 输出根目录（包含 case 子目录）",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/meta/fds_run_summary_mvp.csv"),
        help="摘要 CSV 输出路径",
    )
    return p.parse_args()


def parse_run_log(log_path: Path) -> dict:
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    last_step = ""
    last_time = ""
    for match in TIME_STEP_RE.finditer(text):
        last_step = match.group(1)
        last_time = match.group(2)

    error_hint = ""
    for line in reversed(text.splitlines()):
        if any(k in line for k in ERROR_HINT_PATTERNS):
            error_hint = line.strip()[:200]
            break

    status = "success"
    if "STOP: FDS completed successfully" in text:
        status = "success"
    elif error_hint:
        status = "failed"
    elif not last_step:
        status = "unknown"

    return {
        "status": status,
        "last_time_step": last_step,
        "last_sim_time_s": last_time,
        "error_hint": error_hint,
    }


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
                rows.append(
                    {
                        "case_id": case_id,
                        "status": "missing_log",
                        "last_time_step": "",
                        "last_sim_time_s": "",
                        "error_hint": "run.log not found",
                    }
                )
                continue
            parsed = parse_run_log(log_path)
            rows.append({"case_id": case_id, **parsed})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_id",
                "status",
                "last_time_step",
                "last_sim_time_s",
                "error_hint",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"摘要生成完成: {args.output} (rows={len(rows)})")


if __name__ == "__main__":
    main()
