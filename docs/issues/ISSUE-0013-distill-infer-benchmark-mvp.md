# ISSUE-0013 蒸馏模型推理接入与 P95 基准测试 MVP

## 状态
- Open

## 背景
`ISSUE-0012` 已完成 MLP student 蒸馏训练。下一步需要打通推理入口并测量延迟，验证是否满足秒级推理目标。

## 目标
- 提供 MLP 蒸馏模型推理脚本。
- 提供推理延迟基准脚本（输出 P95）。
- 产出结构化基准报告。

## 验收标准
- 新增脚本：
  - `scripts/distill/infer_distill_mlp_mvp.py`
  - `scripts/distill/benchmark_infer_mvp.py`
- 输出文件：
  - `distill/artifacts/mvp_distill_mlp/infer_preds.csv`
  - `distill/artifacts/mvp_distill_mlp/infer_benchmark.json`
- 报告字段至少包含：
  - `num_cases`
  - `num_repeats`
  - `p95_latency_ms`
  - `avg_latency_ms`

## 范围
- In Scope：
  - 单机 CPU 推理基准。
  - 基线模型加载与批量/逐样本推理。
- Out of Scope：
  - ONNX/TensorRT 部署优化。
  - GPU 高性能推理优化。

## 风险
- 小样本基准波动较大，需要重复次数平滑。
- 不同硬件环境下 P95 不可直接横向比较。

## 任务拆分
- [ ] 实现推理脚本。
- [ ] 实现 P95 基准脚本。
- [ ] 运行基准并输出报告。
- [ ] 更新 issue 状态并提交。
