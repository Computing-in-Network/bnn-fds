# Issue 驱动开发流程（强制）

## 1. 原则
- 所有开发任务必须先创建 Issue，再开始编码。
- 未关联 Issue 的分支、提交、PR 一律视为不合规。
- 全流程遵循 Git Flow：`feature/* -> develop -> release/* -> main`。

## 2. 标准流程
1. 创建 Issue：
   - 在平台（Gitea/GitHub）创建 Issue，使用功能模板。
   - 在仓库内同步一份 issue 文档到 `docs/issues/`。
2. 创建分支：
   - 从 `develop` 拉出 `feature/<issue-id>-<short-name>`。
3. 开发与提交：
   - 每次提交信息必须包含 issue 编号。
   - 示例：`feat(data): 增加参数采样脚本 (ISSUE-0001)`。
4. 提交 PR：
   - `feature/*` 仅允许合并到 `develop`。
   - PR 描述必须链接 issue，并附验收结果。
5. 关闭 Issue：
   - PR 合并后关闭 Issue。
   - 更新 `docs/issues/ISSUE-xxxx-*.md` 状态为 `Done`。

## 3. 命名规范
- Issue 编号：`ISSUE-0001`、`ISSUE-0002`。
- 分支命名：`feature/0001-<short-name>`。
- 文档命名：`docs/issues/ISSUE-0001-<slug>.md`。

## 4. 最低门槛
- 每个 Issue 必须包含：
  - 背景、目标、验收标准、范围、风险、任务拆分。
- 每个 PR 必须包含：
  - 关联 issue、变更说明、测试结果、回滚策略。

## 5. 禁止项
- 禁止从 `main` 直接拉 `feature/*`。
- 禁止将 `feature/*` 直接合并到 `main`。
- 禁止无 issue 编号提交。
