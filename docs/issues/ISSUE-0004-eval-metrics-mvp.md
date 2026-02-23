# ISSUE-0004 评估指标计算 MVP（Accuracy / MAE / P95）

## 状态
- Done

## 背景
数据引擎与任务清单已具备基础能力，下一步需要落地统一评估口径，确保训练与验收使用同一指标定义。

## 目标
- 实现评估脚本，计算 `MAE`、`NMAE`、`Accuracy`。
- 提供推理时延统计（含 `P95`）工具函数或脚本。
- 输出标准化评估报告（JSON）。

## 验收标准
- 新增脚本：`eval/metrics_mvp.py`。
- 输入支持：`y_true.csv`、`y_pred.csv`、`latency_ms.txt`。
- 输出：`eval/reports/metrics_mvp.json`。
- 至少包含字段：`mae`、`nmae`、`accuracy`、`p95_latency_ms`。

## 范围
- In Scope：
  - 单变量/多变量基础指标聚合。
  - P95 统计与结果持久化。
- Out of Scope：
  - 复杂可视化面板。
  - 线上实时监控接入。

## 风险
- 标签与预测文件字段对齐错误会导致统计失真。
- 时延样本数量不足会影响 P95 稳定性。

## 任务拆分
- [x] 定义输入文件格式与字段对齐规则。
- [x] 实现指标计算脚本。
- [x] 生成示例输入并完成一次评估。
- [x] 输出 JSON 报告并文档化。
