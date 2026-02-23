# ISSUE-0007 FDS 批量执行器 MVP（基于 manifest）

## 状态
- Open

## 背景
当前已经具备工况清单与 `fds_manifest_mvp.csv`，需要一个批量执行器把 manifest 转成可执行任务，并输出汇总报告。

## 目标
- 读取 `data/meta/fds_manifest_mvp.csv`。
- 按任务顺序执行（支持 dry-run）。
- 生成统一执行报告，统计成功/失败/跳过数量。

## 验收标准
- 新增脚本：`scripts/fds/run_manifest_mvp.py`。
- 支持参数：`--manifest`、`--output-report`、`--max-cases`、`--dry-run`。
- 输出报告：`data/meta/fds_run_report_mvp.json`。
- 报告至少包含字段：`total`、`success`、`failed`、`skipped`、`dry_run`。

## 范围
- In Scope：
  - 单机顺序执行。
  - 执行结果记录与汇总。
- Out of Scope：
  - 并行调度、多机分发。
  - 自动重试策略。

## 风险
- manifest 中 `run_cmd` 与本地环境不一致会导致运行失败。
- 输入 `.fds` 文件缺失时任务会失败。

## 任务拆分
- [ ] 实现 manifest 执行器脚本。
- [ ] 增加使用说明。
- [ ] 运行一次 dry-run 并输出报告。
- [ ] 更新 issue 状态并提交。
