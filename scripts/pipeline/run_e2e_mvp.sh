#!/usr/bin/env bash
set -euo pipefail

MAX_CASES=10
DRY_RUN=0
N_CASES=100
TIMEOUT_S=5

while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-cases)
      MAX_CASES="$2"
      shift 2
      ;;
    --n-cases)
      N_CASES="$2"
      shift 2
      ;;
    --timeout-s)
      TIMEOUT_S="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    *)
      echo "未知参数: $1"
      echo "用法: $0 [--max-cases N] [--n-cases N] [--timeout-s N] [--dry-run]"
      exit 1
      ;;
  esac
done

echo "[1/5] 生成 cases..."
python3 scripts/data/generate_cases.py --n-cases "${N_CASES}" --output data/meta/cases_mvp.csv

echo "[2/5] 生成 manifest..."
python3 scripts/data/build_fds_manifest.py \
  --cases data/meta/cases_mvp.csv \
  --output data/meta/fds_manifest_mvp.csv

echo "[3/5] 生成 FDS 输入文件..."
python3 scripts/fds/generate_fds_inputs_mvp.py \
  --cases data/meta/cases_mvp.csv \
  --output-dir data/fds_inputs \
  --max-cases "${MAX_CASES}"

echo "[4/5] 执行 manifest..."
RUN_ARGS=(
  --manifest data/meta/fds_manifest_mvp.csv
  --output-report data/meta/fds_run_report_mvp.json
  --max-cases "${MAX_CASES}"
  --timeout-s "${TIMEOUT_S}"
)
if [[ "${DRY_RUN}" -eq 1 ]]; then
  RUN_ARGS+=(--dry-run)
fi
python3 scripts/fds/run_manifest_mvp.py "${RUN_ARGS[@]}"

echo "[5/5] 汇总运行结果..."
python3 scripts/fds/summarize_runs_mvp.py \
  --outputs-dir data/fds_outputs \
  --output data/meta/fds_run_summary_mvp.csv

echo "生成流程摘要..."
python3 - <<'PY'
import csv, json
from pathlib import Path

report = Path("data/meta/fds_run_report_mvp.json")
summary_csv = Path("data/meta/fds_run_summary_mvp.csv")
out = Path("data/meta/pipeline_mvp_summary.json")

run_report = json.loads(report.read_text(encoding="utf-8"))
rows = list(csv.DictReader(summary_csv.open("r", encoding="utf-8")))
run_case_ids = {item.get("case_id") for item in run_report.get("items", [])}
rows = [r for r in rows if r.get("case_id") in run_case_ids]

status_counts = {}
for r in rows:
    s = r.get("status", "unknown")
    status_counts[s] = status_counts.get(s, 0) + 1

result = {
    "run_report_summary": run_report.get("summary", {}),
    "dry_run": run_report.get("dry_run", False),
    "summary_rows": len(rows),
    "status_counts": status_counts,
}
out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"输出: {out}")
PY

echo "E2E MVP 完成。"
