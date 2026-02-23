# Git Flow 开发流程规范（严格执行）

## 1. 目的与适用范围
本规范用于约束 `bnn-fds` 项目的日常开发、发布与紧急修复流程。  
所有成员必须遵循本规范，确保研发节奏稳定、发布可追溯、质量可审计。

适用范围：
- 代码开发与重构。
- 文档更新（需求、设计、测试、运维）。
- 发布版本与热修复。

## 2. 分支模型（标准 Git Flow）
- `main`
  - 仅保存可发布、可回滚的稳定版本。
  - 每次发布必须打版本标签（如 `v0.1.0`）。
- `develop`
  - 日常集成分支，汇总已完成的功能分支。
  - 不允许直接提交，必须通过 PR 合并。
- `feature/*`
  - 功能开发分支，从 `develop` 拉出，完成后回合并到 `develop`。
  - 命名规范：`feature/<issue-id>-<short-name>`。
  - 示例：`feature/23-data-pipeline-mvp`。
- `release/*`
  - 发布准备分支，从 `develop` 拉出，完成发布检查后合并到 `main` 与 `develop`。
  - 命名规范：`release/<version>`，示例：`release/0.1.0`。
- `hotfix/*`
  - 线上紧急修复，从 `main` 拉出，修复后回合并到 `main` 与 `develop`。
  - 命名规范：`hotfix/<version>-<short-name>`。

## 3. 保护规则（强制）
- 禁止直接向 `main`、`develop` 推送。
- 禁止绕过 PR 直接合并。
- 禁止未创建 Issue 就开始开发。
- PR 至少满足：
  - 通过自动化检查（至少包含格式/静态检查/单测中的可用项）。
  - 至少 1 名审阅者通过。
  - 关联需求或 issue 编号。
- 未经过评审的文档与代码不得进入 `develop`。

## 4. 提交与 PR 规范
### 4.1 Commit 规范
推荐使用 Conventional Commits：
- `feat:` 新功能
- `fix:` 缺陷修复
- `docs:` 文档变更
- `refactor:` 重构
- `test:` 测试相关
- `chore:` 构建或工具调整

示例：
- `feat(train): 增加蒸馏训练入口与配置解析`
- `docs(spec): 明确秒级推理与85%准确率验收标准`

### 4.2 PR 描述模板（建议）
- 背景与目标
- 变更范围
- 风险评估
- 测试与验证结果
- 回滚方案
- 关联 issue

## 5. 日常开发流程（feature）
1. 先创建 Issue：
   - 平台 Issue + 仓库内 `docs/issues/ISSUE-xxxx-*.md`。
2. 同步分支：
   - `git checkout develop`
   - `git pull origin develop`
3. 创建功能分支：
   - 推荐：`bash scripts/gitflow/start_issue.sh <issue-id> <short-name>`
   - 或手动：`git checkout -b feature/<issue-id>-<short-name>`
4. 开发与本地验证：
   - 完成代码与文档。
   - 运行可用检查项（格式、静态检查、单测）。
5. 提交与推送：
   - `git add ...`
   - `git commit -m "feat(...): ... (ISSUE-xxxx)"`
   - `git push origin feature/<...>`
6. 发起 PR 合并至 `develop`：
   - 补齐 PR 模板内容。
   - 处理审阅意见后再合并。

## 6. 发布流程（release）
1. 从 `develop` 创建：
   - `git checkout -b release/<version> develop`
2. 仅允许做发布准备项：
   - 版本号、发布说明、必要的小修复。
3. 验证通过后：
   - 合并 `release/<version>` -> `main`
   - 打标签：`git tag -a v<version> -m "release v<version>"`
   - 推送 `main` 与 tag。
4. 回合并：
   - 合并 `release/<version>` -> `develop`，避免分支漂移。

## 7. 紧急修复流程（hotfix）
1. 从 `main` 创建：
   - `git checkout -b hotfix/<version>-<short-name> main`
2. 完成修复与最小验证后：
   - 合并到 `main`，打补丁标签（如 `v0.1.1`）。
3. 必须回合并到 `develop`：
   - 防止后续版本回归同类问题。

## 8. 当前仓库启动 Git Flow 的步骤（仓库尚无提交时）
当前仓库若处于“无初始提交”状态，请先执行初始化：
1. 在 `main` 完成首次基线提交（README/规范文档/目录骨架）。
2. 推送 `main` 到远端。
3. 创建并推送 `develop`：
   - `git checkout -b develop`
   - `git push -u origin develop`
4. 后续开发全部基于 `feature/*` 从 `develop` 拉分支。

## 9. 文档策略（中文详细化）
- 所有核心文档必须使用中文，且描述完整：
  - 背景、目标、输入输出、边界条件、指标口径、风险与回滚。
- 文档与代码同仓同版本管理，变更必须在同一 PR 中可追溯。
- 关键阈值（如准确率、时延）必须写入规范文档，不允许仅口头约定。

## 10. 执行与审计
- 每个里程碑结束后进行一次流程审计：
  - 分支命名是否合规。
  - 是否存在直接提交到保护分支。
  - 是否存在无验证合并。
  - 文档是否与实现一致。
- 发现违规时，必须在下一个工作日内完成纠正并记录原因。
