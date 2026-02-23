# bnn-fds

面向 FDS（Fire Dynamics Simulator）蒸馏加速的神经网络项目。  
项目目标是实现：
- 推理时延达到秒级（`P95 <= 1.0s`）。
- 测试集准确率不低于 85%（统一口径见 `docs/spec.md`）。

## 开发规范
- 严格 Git Flow：`main` / `develop` / `feature-*` / `release-*` / `hotfix-*`。
- 严格 Issue 驱动：先创建 Issue，再创建 `feature/*` 分支开发。

详细规范：
- 技术方案：`docs/spec.md`
- Git Flow：`docs/gitflow.md`
- Issue 流程：`docs/issue-flow.md`

## 项目结构
- `configs/`：训练、评估、数据处理配置。
- `data/`：数据目录（原始数据与处理后数据索引）。
- `train/`：训练相关脚本与入口。
- `eval/`：评估与验收脚本。
- `scripts/`：工程脚本（含 gitflow 助手脚本）。
- `docs/`：规范、流程与 issue 台账。

## 当前状态
- 已完成仓库 Git Flow 初始化。
- 已创建 `ISSUE-0001` 并进入 `feature/0001-project-bootstrap` 开发分支。

## 快速自检
```bash
bash scripts/bootstrap/check_structure.sh
```
