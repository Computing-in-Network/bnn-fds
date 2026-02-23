# ISSUE-0014 真实 FDS 标签蒸馏 MVP（非合成 Teacher）

## 状态
- Done

## 背景
当前蒸馏训练使用的是合成 teacher 函数，尚未形成“基于真实 FDS 输出标签”的完整蒸馏闭环。

## 目标
- 从真实 FDS 运行日志提取监督标签。
- 基于真实标签训练 student 模型。
- 输出训练/验证指标，形成第一版真实蒸馏结果。

## 验收标准
- 新增脚本：
  - `scripts/fds/extract_real_labels_mvp.py`
  - `scripts/distill/train_distill_real_mvp.py`
- 产出文件：
  - `data/meta/fds_labels_real_mvp.csv`
  - `distill/artifacts/real_mvp/metrics_train.json`
  - `distill/artifacts/real_mvp/metrics_val.json`
  - `distill/artifacts/real_mvp/student_real_mvp.pt`
- 指标字段包含：`mae`、`nmae`、`accuracy`。

## 范围
- In Scope：
  - 基于 `run.log` 解析标签（首版标签：`last_sim_time_s`）。
  - 训练/验证拆分与蒸馏训练。
- Out of Scope：
  - 多标签联合训练。
  - 全场 3D 张量监督。

## 风险
- 样本数量不足会导致指标波动大。
- `run.log` 解析依赖日志格式，版本变化可能影响稳定性。

## 任务拆分
- [x] 实现真实标签提取脚本。
- [x] 生成标签 CSV 并完成数据校验。
- [x] 实现真实标签蒸馏训练并输出指标。
- [x] 更新 issue 状态并提交。
