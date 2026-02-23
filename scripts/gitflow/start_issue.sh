#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "用法: $0 <issue-id(如0001)> <short-name>"
  exit 1
fi

ISSUE_ID="$1"
SHORT_NAME="$2"
ISSUE_FILE_GLOB="docs/issues/ISSUE-${ISSUE_ID}-*.md"
BRANCH="feature/${ISSUE_ID}-${SHORT_NAME}"

if ! ls ${ISSUE_FILE_GLOB} >/dev/null 2>&1; then
  echo "错误: 未找到 issue 文档 ${ISSUE_FILE_GLOB}"
  echo "请先创建 issue，再开始开发。"
  exit 2
fi

CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
if [[ "${CURRENT_BRANCH}" != "develop" ]]; then
  echo "错误: 当前分支是 ${CURRENT_BRANCH}，请先切换到 develop。"
  exit 3
fi

if git rev-parse --verify "${BRANCH}" >/dev/null 2>&1; then
  echo "错误: 分支 ${BRANCH} 已存在。"
  exit 4
fi

git checkout -b "${BRANCH}"
echo "已创建分支: ${BRANCH}"
echo "请确保提交信息包含 ISSUE-${ISSUE_ID}"
