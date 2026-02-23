# ISSUE-0015 真实 FDS 蒸馏质量提升 MVP（扩样本+标签质量门控）

## 状态
- Open

## 背景
`ISSUE-0014` 已打通真实 FDS 标签蒸馏闭环，但验证集表现不稳定，核心原因是样本数量偏少且标签质量受超时截断影响。

## 目标
- 扩展真实 FDS 标签样本规模。
- 增加标签质量门控（最小时间步阈值）。
- 基于门控后的标签重训 student，输出更稳定的验证指标。

## 验收标准
- 新增脚本：
  - `scripts/fds/build_real_labels_dataset_mvp.py`
- 产出：
  - `data/meta/fds_labels_real_mvp_filtered.csv`
  - `distill/artifacts/real_mvp_filtered/metrics_train.json`
  - `distill/artifacts/real_mvp_filtered/metrics_val.json`
  - `distill/artifacts/real_mvp_filtered/student_real_mvp.pt`
- 标签门控规则可配置（至少包含 `min_time_step`）。

## 范围
- In Scope：
  - 真实运行结果过滤与训练集构建。
  - 重新蒸馏训练并输出指标。
- Out of Scope：
  - 复杂多标签联合蒸馏。
  - 大规模并行仿真调度优化。

## 风险
- 超时策略仍可能引入系统性偏差。
- 样本分布偏斜会影响泛化稳定性。

## 任务拆分
- [ ] 生成并过滤真实标签数据集。
- [ ] 重训真实蒸馏模型并输出指标。
- [ ] 更新文档与执行说明。
- [ ] 更新 issue 状态并提交。
