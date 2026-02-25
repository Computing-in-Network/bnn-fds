#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="从 FDS _devc.csv 提取业务标签（气体浓度 + 火情）")
    p.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("data/fds_outputs"),
        help="FDS case 输出目录（每个 case 子目录内应有 *_devc.csv）",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/meta/fds_business_labels_mvp.csv"),
        help="业务标签输出 CSV 路径",
    )
    p.add_argument(
        "--co-alarm-ppm",
        type=float,
        default=50.0,
        help="CO 报警阈值（ppm）",
    )
    p.add_argument(
        "--fire-temp-alarm-c",
        type=float,
        default=60.0,
        help="火情高温阈值（摄氏度）",
    )
    return p.parse_args()


def _is_float(v: str) -> bool:
    try:
        float(v)
        return True
    except ValueError:
        return False


def _normalize_col(name: str) -> str:
    return name.strip().strip('"').strip().lower()


def _pick_col(cols: list[str], key: str) -> int:
    for i, c in enumerate(cols):
        if key in c:
            return i
    return -1


def parse_devc_csv(path: Path) -> dict[str, list[float]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return {}

    header_idx = -1
    for i, row in enumerate(rows):
        norm = [_normalize_col(x) for x in row]
        if any(x == "time" for x in norm):
            header_idx = i
            break
    if header_idx < 0:
        return {}

    raw_header = rows[header_idx]
    cols = [_normalize_col(x) for x in raw_header]
    start_idx = header_idx + 1
    if start_idx < len(rows):
        first = rows[start_idx][0].strip() if rows[start_idx] else ""
        if not _is_float(first):
            start_idx += 1

    out: dict[str, list[float]] = {c: [] for c in cols}
    for row in rows[start_idx:]:
        if not row:
            continue
        if len(row) < len(cols):
            continue
        if not _is_float(row[0].strip()):
            continue
        for i, c in enumerate(cols):
            v = row[i].strip()
            if _is_float(v):
                out[c].append(float(v))
    return out


def main() -> None:
    args = parse_args()
    rows = []

    if args.outputs_dir.exists():
        for case_dir in sorted(args.outputs_dir.iterdir()):
            if not case_dir.is_dir():
                continue
            case_id = case_dir.name
            devc = case_dir / f"{case_id}_devc.csv"
            if not devc.exists():
                candidates = sorted(case_dir.glob("*_devc.csv"))
                if not candidates:
                    continue
                devc = candidates[0]

            data = parse_devc_csv(devc)
            if not data:
                continue

            cols = list(data.keys())
            i_time = _pick_col(cols, "time")
            i_temp_center = _pick_col(cols, "temp_center")
            i_temp_fire = _pick_col(cols, "temp_fire")
            i_o2 = _pick_col(cols, "o2_vol")
            i_co2 = _pick_col(cols, "co2_vol")
            i_co = _pick_col(cols, "co_vol")
            if min(i_time, i_temp_center, i_temp_fire, i_o2, i_co2, i_co) < 0:
                continue

            t = data[cols[i_time]]
            temp_center = data[cols[i_temp_center]]
            temp_fire = data[cols[i_temp_fire]]
            o2 = data[cols[i_o2]]
            co2 = data[cols[i_co2]]
            co = data[cols[i_co]]
            if not t:
                continue

            co_max_vol = max(co)
            row = {
                "case_id": case_id,
                "t_end_s": t[-1],
                "gas_o2_min_vol_frac": min(o2),
                "gas_o2_end_vol_frac": o2[-1],
                "gas_co2_max_vol_frac": max(co2),
                "gas_co2_end_vol_frac": co2[-1],
                "gas_co_max_vol_frac": co_max_vol,
                "gas_co_end_vol_frac": co[-1],
                "gas_co_max_ppm": co_max_vol * 1_000_000.0,
                "fire_temp_center_peak_c": max(temp_center),
                "fire_temp_center_end_c": temp_center[-1],
                "fire_temp_fire_peak_c": max(temp_fire),
                "fire_temp_fire_end_c": temp_fire[-1],
            }
            row["gas_co_alarm_50ppm"] = 1 if row["gas_co_max_ppm"] >= args.co_alarm_ppm else 0
            row["fire_high_temp_alarm"] = (
                1 if row["fire_temp_fire_peak_c"] >= args.fire_temp_alarm_c else 0
            )
            rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "case_id",
            "t_end_s",
            "gas_o2_min_vol_frac",
            "gas_o2_end_vol_frac",
            "gas_co2_max_vol_frac",
            "gas_co2_end_vol_frac",
            "gas_co_max_vol_frac",
            "gas_co_end_vol_frac",
            "gas_co_max_ppm",
            "gas_co_alarm_50ppm",
            "fire_temp_center_peak_c",
            "fire_temp_center_end_c",
            "fire_temp_fire_peak_c",
            "fire_temp_fire_end_c",
            "fire_high_temp_alarm",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"业务标签提取完成: {args.output} (rows={len(rows)})")


if __name__ == "__main__":
    main()
