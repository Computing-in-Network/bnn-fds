# scripts/fds 使用说明

## 脚本
- `scripts/fds/run_manifest_mvp.py`
- `scripts/fds/generate_fds_inputs_mvp.py`
- `scripts/fds/summarize_runs_mvp.py`
- `scripts/fds/extract_real_labels_mvp.py`
- `scripts/fds/build_real_labels_dataset_mvp.py`

## 作用
读取 `fds_manifest`，顺序执行任务（或 dry-run），并输出运行报告 JSON。
也可以从 `cases_mvp.csv` 自动生成最小 `.fds` 输入文件。

## 先生成输入文件
```bash
python3 scripts/fds/generate_fds_inputs_mvp.py \
  --cases data/meta/cases_mvp.csv \
  --output-dir data/fds_inputs \
  --max-cases 10
```

## 示例
```bash
# 仅预演，不执行命令
python3 scripts/fds/run_manifest_mvp.py \
  --manifest data/meta/fds_manifest_mvp.csv \
  --output-report data/meta/fds_run_report_mvp.json \
  --max-cases 10 \
  --dry-run

# 实际执行（需确保 .fds 文件存在且环境已加载 fds）
python3 scripts/fds/run_manifest_mvp.py \
  --manifest data/meta/fds_manifest_mvp.csv \
  --output-report data/meta/fds_run_report_mvp.json \
  --max-cases 3 \
  --timeout-s 60
```

## 报告字段
- `dry_run`
- `summary.total`
- `summary.success`
- `summary.failed`
- `summary.skipped`

## 说明
- 若当前 shell 未加载 `fds` 到 PATH，脚本会尝试回退到 `~/FDS/FDS6/bin/fds`。

## 日志摘要
```bash
python3 scripts/fds/summarize_runs_mvp.py \
  --outputs-dir data/fds_outputs \
  --output data/meta/fds_run_summary_mvp.csv
```

## 真实标签提取
```bash
python3 scripts/fds/extract_real_labels_mvp.py \
  --outputs-dir data/fds_outputs \
  --output data/meta/fds_labels_real_mvp.csv
```

## 标签质量门控
```bash
python3 scripts/fds/build_real_labels_dataset_mvp.py \
  --input data/meta/fds_labels_real_mvp.csv \
  --output data/meta/fds_labels_real_mvp_filtered.csv \
  --min-time-step 50 \
  --min-sim-time 1.0
```
