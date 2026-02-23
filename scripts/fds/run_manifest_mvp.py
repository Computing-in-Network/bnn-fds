#!/usr/bin/env python3
import argparse
import csv
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


REQUIRED_COLUMNS = ["case_id", "fds_input_path", "output_dir", "run_cmd"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FDS manifest 批量执行器 MVP")
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/meta/fds_manifest_mvp.csv"),
        help="输入 manifest CSV",
    )
    p.add_argument(
        "--output-report",
        type=Path,
        default=Path("data/meta/fds_run_report_mvp.json"),
        help="输出运行报告 JSON",
    )
    p.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="最多执行前 N 条，0 表示全部",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印并记录计划，不实际执行 run_cmd",
    )
    p.add_argument(
        "--timeout-s",
        type=int,
        default=0,
        help="单任务超时秒数，0 表示不设超时",
    )
    return p.parse_args()


def load_manifest(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"manifest 不存在: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        missing = [c for c in REQUIRED_COLUMNS if c not in fields]
        if missing:
            raise ValueError(f"manifest 缺少字段: {missing}")
        rows = list(reader)
    if not rows:
        raise ValueError("manifest 为空")
    return rows


def resolve_fds_cmd(cmd: str) -> str:
    stripped = cmd.lstrip()
    if not stripped.startswith("fds "):
        return cmd
    if shutil.which("fds"):
        return cmd
    fallback = os.path.expanduser("~/FDS/FDS6/bin/fds")
    if os.path.exists(fallback):
        return stripped.replace("fds ", f"{fallback} ", 1)
    return cmd


def with_fds_env(cmd: str) -> str:
    fds_vars = os.path.expanduser("~/FDS/FDS6/bin/FDS6VARS.sh")
    smv_vars = os.path.expanduser("~/FDS/FDS6/bin/SMV6VARS.sh")
    prefix_parts = []
    if os.path.exists(fds_vars):
        prefix_parts.append(f"source {fds_vars}")
    if os.path.exists(smv_vars):
        prefix_parts.append(f"source {smv_vars}")
    if not prefix_parts:
        return cmd
    return " && ".join(prefix_parts) + " && " + cmd


def run_one(row: dict, dry_run: bool, timeout_s: int) -> dict:
    case_id = row["case_id"]
    cmd = with_fds_env(resolve_fds_cmd(row["run_cmd"]))
    output_dir = Path(row["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        return {
            "case_id": case_id,
            "status": "skipped",
            "reason": "dry_run",
            "cmd": cmd,
            "output_dir": str(output_dir),
        }

    start = datetime.now(timezone.utc)
    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            executable="/bin/bash",
            capture_output=True,
            text=True,
            timeout=timeout_s if timeout_s > 0 else None,
        )
        end = datetime.now(timezone.utc)
        status = "success" if proc.returncode == 0 else "failed"
        return {
            "case_id": case_id,
            "status": status,
            "return_code": proc.returncode,
            "cmd": cmd,
            "output_dir": str(output_dir),
            "started_at": start.isoformat(),
            "finished_at": end.isoformat(),
            "stdout_tail": proc.stdout[-500:] if proc.stdout else "",
            "stderr_tail": proc.stderr[-500:] if proc.stderr else "",
        }
    except subprocess.TimeoutExpired:
        end = datetime.now(timezone.utc)
        return {
            "case_id": case_id,
            "status": "failed",
            "return_code": -1,
            "reason": f"timeout_{timeout_s}s",
            "cmd": cmd,
            "output_dir": str(output_dir),
            "started_at": start.isoformat(),
            "finished_at": end.isoformat(),
            "stdout_tail": "",
            "stderr_tail": "",
        }


def main() -> None:
    args = parse_args()
    rows = load_manifest(args.manifest)
    if args.max_cases > 0:
        rows = rows[: args.max_cases]

    items = []
    success = failed = skipped = 0
    for row in rows:
        result = run_one(row, args.dry_run, args.timeout_s)
        items.append(result)
        if result["status"] == "success":
            success += 1
        elif result["status"] == "failed":
            failed += 1
        else:
            skipped += 1

    report = {
        "manifest": str(args.manifest),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": args.dry_run,
        "summary": {
            "total": len(rows),
            "success": success,
            "failed": failed,
            "skipped": skipped,
        },
        "items": items,
    }

    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    with args.output_report.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"执行完成，报告输出: {args.output_report}")
    print(
        f"summary: total={len(rows)} success={success} failed={failed} skipped={skipped} dry_run={args.dry_run}"
    )


if __name__ == "__main__":
    main()
