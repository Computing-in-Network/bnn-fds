# PR 说明（2026-02-25）

## 本次目标
- 跑通 5 组真实 FDS 数据闭环（生成 -> 运行 -> 标签 -> 蒸馏 -> 时延对比）。
- 将真实蒸馏主模型默认策略调整为 Tiny MLP（64x64）。
- 更新后续 50000 组扩参与规模化计划。

## 代码与脚本变更
1. 真实蒸馏训练脚本升级：
- 文件：`scripts/distill/train_distill_real_mvp.py`
- 新增参数：
  - `--hidden-dim-1`（默认 64）
  - `--hidden-dim-2`（默认 64）
  - `--select-model`（`mlp`/`linear`/`auto`，默认 `mlp`）
- 目的：
  - 以 Tiny MLP（64x64）作为默认主模型；
  - 线性模型保留为对照基线；
  - 自动策略可按验证 MAE 选优。

2. 新增真实标签模型推理时延基准脚本：
- 文件：`scripts/distill/benchmark_real_mvp.py`
- 能力：
  - 兼容 `student_real_mvp.pt` 的 `linear`/`mlp` 两类模型；
  - 输出批级/样本级 p50/p95/avg 时延。

3. 文档更新：
- `scripts/distill/README.md`
- `docs/issues/ISSUE-0016-fds-param-expansion-and-50000-scale-plan.md`

## 今日测试数据与结果
1. FDS 5 组烟雾测试：
- 相关文件：
  - `data/meta/cases_smoke_5.csv`
  - `data/meta/fds_manifest_smoke_5.csv`
  - `data/fds_inputs_smoke/`
  - `data/fds_outputs_smoke/`
  - `data/meta/fds_run_report_smoke_5.json`
  - `data/meta/fds_run_summary_smoke_5.csv`
  - `data/meta/fds_labels_real_smoke_5.csv`
  - `data/meta/fds_labels_real_smoke_5_filtered.csv`
- 结果：`total=5, success=5, failed=0`

2. 蒸馏训练与对比：
- 线性/自动策略产物：`distill/artifacts/real_smoke_5/`
- Tiny MLP(64x64) 产物：`distill/artifacts/real_smoke_5_mlp64/`
- 对比报告：
  - `data/meta/compare_fds_bnn_smoke_5.json`
  - `data/meta/compare_fds_bnn_smoke_5_mlp64.json`

3. 关键指标（Tiny MLP 64x64，5 组样本）：
- 训练耗时：`3.45 s`
- 推理时延：
  - `avg ≈ 0.0716 ms/批(5组)`
  - `p95 ≈ 0.0802 ms/批(5组)`
- FDS 平均墙钟：`≈ 101.43 s/组`
- 结论：推理时延显著低于 1s 目标（毫秒级）。

## 后续计划
- 进入 M2：扩展输入参数到 10~12 维（空气/风向等）。
- 进入 M3/M4：按分批与并行方案推进 50000 组规模化执行与蒸馏评估。
