# scripts/fds 使用说明

## 脚本
- `scripts/fds/run_manifest_mvp.py`
- `scripts/fds/generate_fds_inputs_mvp.py`
- `scripts/fds/summarize_runs_mvp.py`
- `scripts/fds/extract_real_labels_mvp.py`
- `scripts/fds/build_real_labels_dataset_mvp.py`
- `scripts/fds/extract_business_labels_mvp.py`

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

## 业务标签提取（气体浓度 + 火情）
```bash
python3 scripts/fds/extract_business_labels_mvp.py \
  --outputs-dir data/fds_outputs \
  --output data/meta/fds_business_labels_mvp.csv
```

输出字段示例：
- 气体浓度类：`gas_co_max_ppm`、`gas_co2_max_vol_frac`、`gas_o2_min_vol_frac`
- 火情类：`fire_temp_fire_peak_c`、`fire_temp_center_peak_c`

## 业务链路最小验收（建议先跑 1 组）
```bash
# 1) 生成 1 组输入
python3 scripts/fds/generate_fds_inputs_mvp.py \
  --cases data/meta/cases_mvp.csv \
  --output-dir data/fds_inputs \
  --max-cases 1

# 2) 生成 manifest（确保输出落在 case 目录）
python3 scripts/data/build_fds_manifest.py \
  --cases data/meta/cases_mvp.csv \
  --output data/meta/fds_manifest_mvp.csv \
  --fds-input-root data/fds_inputs \
  --fds-output-root data/fds_outputs

# 3) 运行 FDS
python3 scripts/fds/run_manifest_mvp.py \
  --manifest data/meta/fds_manifest_mvp.csv \
  --output-report data/meta/fds_run_report_mvp.json \
  --max-cases 1 \
  --timeout-s 600

# 4) 提取业务标签
python3 scripts/fds/extract_business_labels_mvp.py \
  --outputs-dir data/fds_outputs \
  --output data/meta/fds_business_labels_mvp.csv
```

验收要点：
- 每个 case 目录中存在 `*_devc.csv`。
- `data/meta/fds_business_labels_mvp.csv` 至少包含：
  - `gas_co_max_ppm`、`gas_co2_max_vol_frac`、`gas_o2_min_vol_frac`
  - `fire_temp_fire_peak_c`、`fire_temp_center_peak_c`

## 标签质量门控
```bash
python3 scripts/fds/build_real_labels_dataset_mvp.py \
  --input data/meta/fds_labels_real_mvp.csv \
  --output data/meta/fds_labels_real_mvp_filtered.csv \
  --min-time-step 50 \
  --min-sim-time 1.0
```
