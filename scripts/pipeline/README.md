# scripts/pipeline 使用说明

## 脚本
- `scripts/pipeline/run_e2e_mvp.sh`

## 作用
串联 MVP 流程：
1. `generate_cases`
2. `build_manifest`
3. `generate_fds_inputs`
4. `run_manifest`
5. `summarize_runs`

## 示例
```bash
# dry-run（推荐先执行）
bash scripts/pipeline/run_e2e_mvp.sh --max-cases 10 --n-cases 100 --dry-run

# 小规模真实运行
bash scripts/pipeline/run_e2e_mvp.sh --max-cases 1 --n-cases 10 --timeout-s 5
```

## 输出
- `data/meta/fds_run_report_mvp.json`
- `data/meta/fds_run_summary_mvp.csv`
- `data/meta/pipeline_mvp_summary.json`
