# ISSUE-0010 端到端流水线脚本 MVP（生成-执行-汇总）

## 状态
- Done

## 背景
当前各环节脚本已具备（工况生成、manifest 生成、输入生成、执行器、摘要器），但还缺少统一入口，执行链路仍需手工串联。

## 目标
- 提供一个一键脚本串联关键步骤。
- 支持 dry-run 与小样本真实运行。
- 输出统一的流程结果摘要。

## 验收标准
- 新增脚本：`scripts/pipeline/run_e2e_mvp.sh`。
- 至少串联：`generate_cases -> build_manifest -> generate_fds_inputs -> run_manifest -> summarize_runs`。
- 支持参数：`--max-cases`、`--dry-run`。
- 输出最终摘要：`data/meta/pipeline_mvp_summary.json`。

## 范围
- In Scope：
  - 单机顺序流水线执行。
  - 基础错误退出与步骤日志。
- Out of Scope：
  - 并行执行优化。
  - 失败重试与任务恢复。

## 风险
- 各步骤输入输出依赖耦合，任何一步失败会中断全流程。
- 真实运行时长受 FDS 仿真复杂度影响。

## 任务拆分
- [x] 实现 e2e 流水线脚本。
- [x] 跑通一遍 dry-run。
- [x] 生成流程摘要 JSON。
- [x] 更新 issue 状态并提交。
