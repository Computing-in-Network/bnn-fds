# ISSUE-0003 FDS 批量任务清单与命令编排 MVP

## 状态
- Open

## 背景
`ISSUE-0002` 已产出工况清单 CSV，下一步需要把工况映射成可执行任务列表，供后续批量 FDS 运行器消费。

## 目标
- 根据 `cases_mvp.csv` 生成任务清单（manifest）。
- 输出每个 case 的输入文件路径、输出目录和建议执行命令。
- 保持与后续调度器解耦（先输出静态清单）。

## 验收标准
- 新增脚本：`scripts/data/build_fds_manifest.py`。
- 输入：`data/meta/cases_mvp.csv`。
- 输出：`data/meta/fds_manifest_mvp.csv`。
- 字段至少包含：`case_id`、`fds_input_path`、`output_dir`、`run_cmd`。

## 范围
- In Scope：
  - 清单构建与字段校验。
  - 可配置输入/输出根目录。
- Out of Scope：
  - 真正执行 FDS。
  - 多机调度与失败重试机制。

## 风险
- FDS 输入文件模板未统一，可能导致命令字段后续变更。
- 路径规范不一致会增加后续集成成本。

## 任务拆分
- [ ] 定义 manifest 字段与目录约定。
- [ ] 实现 manifest 生成脚本。
- [ ] 增加使用文档与示例。
- [ ] 运行自检并提交样例输出。
