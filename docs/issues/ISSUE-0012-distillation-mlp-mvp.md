# ISSUE-0012 非线性 Student 蒸馏 MVP（MLP）

## 状态
- Done

## 背景
`ISSUE-0011` 已完成线性 student 蒸馏基线。下一步需要引入非线性 student（MLP），并增加训练/验证集拆分，以更接近真实训练流程。

## 目标
- 实现 MLP student 蒸馏训练脚本。
- 增加训练集/验证集划分与指标输出。
- 产出可复现的模型权重与训练曲线摘要。

## 验收标准
- 新增脚本：`scripts/distill/train_distill_mlp_mvp.py`。
- 输出目录：`distill/artifacts/mvp_distill_mlp/`。
- 至少产出：
  - `metrics_train.json`
  - `metrics_val.json`
  - `student_mlp_model.pt`
  - `train_history.csv`
- 指标字段至少包含：`mae`、`nmae`、`accuracy`。

## 范围
- In Scope：
  - 单目标回归（teacher_temp_peak）。
  - 固定随机种子与可复现训练流程。
- Out of Scope：
  - 多任务蒸馏。
  - 超参数自动搜索。

## 风险
- 样本量较小时，MLP 可能过拟合。
- 不同随机种子会引起指标波动。

## 任务拆分
- [x] 实现 MLP 蒸馏训练脚本。
- [x] 输出训练/验证指标与模型文件。
- [x] 更新使用说明。
- [x] 更新 issue 状态并提交。
