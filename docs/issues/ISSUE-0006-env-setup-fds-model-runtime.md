# ISSUE-0006 开发环境安装（FDS 与模型运行时）

## 状态
- Done

## 背景
当前仓库已完成流程、数据清单与评估 MVP，但本机缺少 `fds`/`smokeview`，无法进入真实仿真数据阶段。

## 目标
- 安装并验证 FDS 与 Smokeview 可执行程序。
- 校验模型运行时依赖（Python、Torch、TensorFlow、NumPy）可用。
- 输出环境检查结果文档，作为后续训练与仿真的基线。

## 验收标准
- `command -v fds` 返回可执行路径。
- `command -v smokeview` 返回可执行路径。
- `python3 -c "import torch,tensorflow,numpy"` 执行成功。
- 新增文档：`docs/setup/env-check.md`，记录版本与验证命令。

## 范围
- In Scope：
  - 软件安装与命令行可用性验证。
  - 环境结果记录。
- Out of Scope：
  - FDS 示例算例验证。
  - GPU/CUDA 优化调优。

## 风险
- 安装脚本与系统发行版不兼容。
- 系统权限不足导致安装失败。

## 任务拆分
- [x] 获取并安装 FDS/Smokeview。
- [x] 校验模型运行时依赖。
- [x] 记录安装与验证结果。
- [x] 提交并推送变更。
