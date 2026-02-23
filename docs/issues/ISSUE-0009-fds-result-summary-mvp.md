# ISSUE-0009 FDS 结果摘要提取 MVP

## 状态
- Done

## 背景
当前已经具备输入生成与批量执行能力，下一步需要把运行产物（如 `run.log`）提取为结构化摘要，便于后续训练数据筛选与质量检查。

## 目标
- 从 `data/fds_outputs/*/run.log` 提取关键字段。
- 生成统一摘要 CSV。
- 统计成功/失败案例数量。

## 验收标准
- 新增脚本：`scripts/fds/summarize_runs_mvp.py`。
- 输出：`data/meta/fds_run_summary_mvp.csv`。
- 摘要字段至少包含：`case_id`、`status`、`last_time_step`、`last_sim_time_s`、`error_hint`。

## 范围
- In Scope：
  - 正常日志关键字段解析。
  - 常见错误提示提取（如库缺失、文件缺失、超时）。
- Out of Scope：
  - 完整 FDS 二进制结果解析（S3D/SMV）。
  - 可视化面板。

## 风险
- 不同版本日志格式差异导致解析规则失效。
- 日志不完整时可能提取不到时间步信息。

## 任务拆分
- [x] 实现日志摘要脚本。
- [x] 生成摘要 CSV 样例。
- [x] 更新使用文档。
- [x] 更新 issue 状态并提交。
