# ISSUE-0011 蒸馏训练 MVP（Teacher-Student）

## 状态
- Done

## 背景
数据与执行链路已具备，下一步进入“真实蒸馏训练”阶段，先实现最小可运行的 Teacher-Student 蒸馏流程与指标产出。

## 目标
- 基于 `cases_mvp.csv` 构造 teacher 目标并训练 student 回归模型。
- 输出 student 模型参数与训练评估指标。
- 指标口径与 `docs/spec.md` 保持一致（MAE/NMAE/Accuracy）。

## 验收标准
- 新增脚本：`scripts/distill/train_distill_mvp.py`。
- 输出目录：`distill/artifacts/mvp_distill/`。
- 至少产出：
  - `teacher_targets_mvp.csv`
  - `student_model_mvp.json`
  - `metrics_mvp.json`
- 指标字段至少包含：`mae`、`nmae`、`accuracy`。

## 范围
- In Scope：
  - 线性 student 基线（可解释、可快速验证）。
  - 蒸馏训练与离线评估。
- Out of Scope：
  - 深度网络 student（MLP/FNO）正式训练。
  - 多任务联合蒸馏。

## 风险
- Teacher 目标简化会导致指标偏乐观。
- 线性 student 表达能力有限，后续需升级模型结构。

## 任务拆分
- [x] 实现 teacher 目标构造与 student 训练脚本。
- [x] 生成蒸馏产物与指标。
- [x] 补充使用说明。
- [x] 更新 issue 状态并提交。
