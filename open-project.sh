#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/Users/joey/work/ai-fullstack-starter"

if [ ! -d "$PROJECT_DIR" ]; then
  echo "[error] 未找到项目目录: $PROJECT_DIR"
  exit 1
fi

cd "$PROJECT_DIR"

if command -v code >/dev/null 2>&1; then
  code .
else
  open .
  echo "[info] 未找到 'code' 命令，已在 Finder 打开项目目录。"
fi

exec "$SHELL" -l
