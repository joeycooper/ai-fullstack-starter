#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/Users/joey/work/ai-fullstack-starter"
BACKEND_DIR="$ROOT_DIR/backend"
FRONTEND_DIR="$ROOT_DIR/frontend"

BACKEND_PY="$BACKEND_DIR/.venv/bin/python"

if [ ! -x "$BACKEND_PY" ]; then
  echo "[error] 未找到虚拟环境: $BACKEND_PY"
  echo "请先在 $BACKEND_DIR 创建并安装依赖"
  exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "[error] 未找到 npm，请先安装 Node.js"
  exit 1
fi

if ! command -v cloudflared >/dev/null 2>&1; then
  echo "[error] 未找到 cloudflared，请先安装: brew install cloudflared"
  exit 1
fi

cleanup() {
  if [[ -n "${BACKEND_PID:-}" ]]; then
    kill "$BACKEND_PID" >/dev/null 2>&1 || true
  fi
  if [[ -n "${FRONTEND_PID:-}" ]]; then
    kill "$FRONTEND_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

echo "[info] 启动后端..."
cd "$BACKEND_DIR"
"$BACKEND_PY" -m uvicorn main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

sleep 1

echo "[info] 启动前端..."
cd "$FRONTEND_DIR"
if [ ! -d node_modules ]; then
  npm install
fi
npm run dev -- --host 0.0.0.0 --port 5173 &
FRONTEND_PID=$!

sleep 1

echo "[info] 打开本地预览..."
open "http://127.0.0.1:5173" >/dev/null 2>&1 || true

echo "[info] 启动 Cloudflare Tunnel..."
cloudflared tunnel --url http://127.0.0.1:5173 2>&1 | awk '
  {
    print $0
    if (!opened) {
      match($0, /https:\/\/[a-z0-9-]+\.trycloudflare\.com/)
      if (RSTART > 0) {
        url = substr($0, RSTART, RLENGTH)
        system("open \"" url "\" >/dev/null 2>&1")
        opened = 1
        print "[info] 已自动打开公网地址: " url
      }
    }
  }
'
