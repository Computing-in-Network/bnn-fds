#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


REQUIRED_COLUMNS = [
    "case_id",
    "fire_x",
    "fire_y",
    "hrr_peak_kw",
    "vent_open_ratio",
    "duration_s",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="根据 cases 生成最小 FDS 输入文件")
    p.add_argument(
        "--cases",
        type=Path,
        default=Path("data/meta/cases_mvp.csv"),
        help="cases CSV 路径",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/fds_inputs"),
        help="输出 .fds 文件目录",
    )
    p.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="最多生成前 N 条，0 表示全部",
    )
    return p.parse_args()


def load_cases(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"cases 文件不存在: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        missing = [c for c in REQUIRED_COLUMNS if c not in fields]
        if missing:
            raise ValueError(f"cases 缺少字段: {missing}")
        return list(reader)


def build_fds_text(case: dict) -> str:
    case_id = case["case_id"]
    fire_x = float(case["fire_x"])
    fire_y = float(case["fire_y"])
    hrr_peak_kw = float(case["hrr_peak_kw"])
    vent_open_ratio = float(case["vent_open_ratio"])
    duration_s = float(case["duration_s"])

    # 使用 10m x 10m x 3m 的最小单房间示例
    x1 = max(0.5, min(9.5, fire_x - 0.25))
    x2 = max(0.5, min(9.5, fire_x + 0.25))
    y1 = max(0.5, min(9.5, fire_y - 0.25))
    y2 = max(0.5, min(9.5, fire_y + 0.25))
    z1 = 0.0
    z2 = 0.1

    vent_width = 1.0 + 1.0 * max(0.0, min(1.0, vent_open_ratio))
    vx1 = 0.0
    vx2 = 0.0
    vy1 = 4.5
    vy2 = 4.5 + vent_width
    vz1 = 0.0
    vz2 = 2.0

    dev_x = 5.0
    dev_y = 5.0
    dev_z = 1.6
    fire_dev_x = max(0.5, min(9.5, fire_x))
    fire_dev_y = max(0.5, min(9.5, fire_y))
    fire_dev_z = 1.6

    txt = f"""&HEAD CHID='{case_id}', TITLE='MVP auto-generated case' /
&TIME T_END={duration_s:.1f} /
&DUMP DT_DEVC=1.0 /
&MESH IJK=40,40,12, XB=0.0,10.0, 0.0,10.0, 0.0,3.0 /
&REAC FUEL='PROPANE' /
&SURF ID='BURNER', HRRPUA={hrr_peak_kw:.2f}, COLOR='RED' /
&OBST XB={x1:.3f},{x2:.3f}, {y1:.3f},{y2:.3f}, {z1:.3f},{z2:.3f}, SURF_ID='BURNER' /
&VENT XB={vx1:.3f},{vx2:.3f}, {vy1:.3f},{vy2:.3f}, {vz1:.3f},{vz2:.3f}, SURF_ID='OPEN' /
&DEVC ID='TEMP_CENTER', QUANTITY='TEMPERATURE', XYZ={dev_x:.3f},{dev_y:.3f},{dev_z:.3f} /
&DEVC ID='TEMP_FIRE', QUANTITY='TEMPERATURE', XYZ={fire_dev_x:.3f},{fire_dev_y:.3f},{fire_dev_z:.3f} /
&DEVC ID='O2_VOL', QUANTITY='VOLUME FRACTION', SPEC_ID='OXYGEN', XYZ={dev_x:.3f},{dev_y:.3f},{dev_z:.3f} /
&DEVC ID='CO2_VOL', QUANTITY='VOLUME FRACTION', SPEC_ID='CARBON DIOXIDE', XYZ={dev_x:.3f},{dev_y:.3f},{dev_z:.3f} /
&DEVC ID='CO_VOL', QUANTITY='VOLUME FRACTION', SPEC_ID='CARBON MONOXIDE', XYZ={dev_x:.3f},{dev_y:.3f},{dev_z:.3f} /
&TAIL /
"""
    return txt


def main() -> None:
    args = parse_args()
    rows = load_cases(args.cases)
    if args.max_cases > 0:
        rows = rows[: args.max_cases]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for row in rows:
        case_id = row["case_id"]
        out = args.output_dir / f"{case_id}.fds"
        out.write_text(build_fds_text(row), encoding="utf-8")

    print(f"生成完成: {len(rows)} 个 .fds 文件 -> {args.output_dir}")


if __name__ == "__main__":
    main()
