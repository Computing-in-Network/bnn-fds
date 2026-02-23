# scripts/fds 使用说明

## 脚本
- `scripts/fds/run_manifest_mvp.py`

## 作用
读取 `fds_manifest`，顺序执行任务（或 dry-run），并输出运行报告 JSON。

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
  --max-cases 3
```

## 报告字段
- `dry_run`
- `summary.total`
- `summary.success`
- `summary.failed`
- `summary.skipped`
