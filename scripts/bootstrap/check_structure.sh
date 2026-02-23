#!/usr/bin/env bash
set -euo pipefail

required_dirs=(
  "configs"
  "data"
  "train"
  "eval"
  "scripts"
  "docs"
  "docs/issues"
)

echo "开始检查项目结构..."
for d in "${required_dirs[@]}"; do
  if [[ -d "${d}" ]]; then
    echo "[OK] ${d}"
  else
    echo "[FAIL] 缺少目录: ${d}"
    exit 1
  fi
done

required_files=(
  "README.md"
  "docs/spec.md"
  "docs/gitflow.md"
  "docs/issue-flow.md"
)

for f in "${required_files[@]}"; do
  if [[ -f "${f}" ]]; then
    echo "[OK] ${f}"
  else
    echo "[FAIL] 缺少文件: ${f}"
    exit 2
  fi
done

echo "结构检查通过。"
