#!/usr/bin/env python3
import argparse
import csv
import shlex
from pathlib import Path


REQUIRED_INPUT_FIELDS = [
    "case_id",
    "fire_x",
    "fire_y",
    "hrr_peak_kw",
    "vent_open_ratio",
    "duration_s",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="根据 case 清单生成 FDS 任务 manifest")
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("data/meta/cases_mvp.csv"),
        help="输入工况 CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/meta/fds_manifest_mvp.csv"),
        help="输出 manifest CSV",
    )
    parser.add_argument(
        "--fds-input-root",
        type=Path,
        default=Path("data/fds_inputs"),
        help="FDS 输入文件根目录",
    )
    parser.add_argument(
        "--fds-output-root",
        type=Path,
        default=Path("data/fds_outputs"),
        help="FDS 输出目录根路径",
    )
    return parser.parse_args()


def load_cases(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"未找到输入文件: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        missing = [name for name in REQUIRED_INPUT_FIELDS if name not in fields]
        if missing:
            raise ValueError(f"输入 CSV 缺少字段: {missing}")
        return list(reader)


def build_manifest_rows(
    cases: list[dict], fds_input_root: Path, fds_output_root: Path
) -> list[dict]:
    rows = []
    for case in cases:
        case_id = case["case_id"]
        fds_input_path = (fds_input_root / f"{case_id}.fds").resolve().as_posix()
        output_dir = (fds_output_root / case_id).resolve().as_posix()
        out_q = shlex.quote(output_dir)
        in_q = shlex.quote(fds_input_path)
        # 在 case 专属目录执行 FDS，确保 _devc.csv 等输出与 run.log 同目录便于后处理
        run_cmd = f"mkdir -p {out_q} && cd {out_q} && fds {in_q} > run.log 2>&1"
        rows.append(
            {
                "case_id": case_id,
                "fds_input_path": fds_input_path,
                "output_dir": output_dir,
                "run_cmd": run_cmd,
            }
        )
    return rows


def write_manifest(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["case_id", "fds_input_path", "output_dir", "run_cmd"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    cases = load_cases(args.cases)
    rows = build_manifest_rows(cases, args.fds_input_root, args.fds_output_root)
    write_manifest(rows, args.output)
    print(f"已生成 manifest，共 {len(rows)} 条: {args.output}")


if __name__ == "__main__":
    main()
