# ISSUE-0008 自动生成最小 FDS 输入文件 MVP

## 状态
- Open

## 背景
当前已有工况清单与执行器，但 `data/fds_inputs/*.fds` 尚未自动生成，导致真实执行阶段仍需手工准备输入文件。

## 目标
- 从 `cases_mvp.csv` 自动生成最小可运行 `.fds` 文件。
- 与 `fds_manifest_mvp.csv` 的 `fds_input_path` 路径约定保持一致。
- 支持限制生成数量（便于小规模验证）。

## 验收标准
- 新增脚本：`scripts/fds/generate_fds_inputs_mvp.py`。
- 输入：`data/meta/cases_mvp.csv`。
- 输出目录：`data/fds_inputs/`。
- 生成文件命名：`<case_id>.fds`。
- 支持参数：`--cases`、`--output-dir`、`--max-cases`。

## 范围
- In Scope：
  - 最小房间场景模板生成。
  - 火源位置与热释放率参数映射。
- Out of Scope：
  - 复杂几何与多房间。
  - 材料库与边界条件细化。

## 风险
- 简化模板与真实目标场景存在偏差。
- 参数映射若不合理，可能导致求解不稳定。

## 任务拆分
- [ ] 编写 `.fds` 模板生成脚本。
- [ ] 生成样例输入文件并检查格式。
- [ ] 补充使用文档。
- [ ] 更新 issue 状态并提交。
