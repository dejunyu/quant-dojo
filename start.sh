#!/usr/bin/env bash
# Quant Dojo 一键启动脚本
#
#   ./start.sh           启动 Web 平台（构建前端 + 起服务）
#   ./start.sh dev       开发模式（后端 8000 + 前端热更新 5173）
#   ./start.sh demo      只跑命令行全流程自检
#   ./start.sh test      跑全部测试
set -euo pipefail

cd "$(dirname "$0")"
PY=".venv/bin/python"
export PYTHONPATH="$PWD/engine"

if [ ! -x "$PY" ]; then
  echo "未找到虚拟环境 .venv，正在创建…"
  uv venv .venv --python 3.12
  uv pip install --python "$PY" -r requirements.txt
fi

case "${1:-serve}" in
  dev)
    echo "后端: http://127.0.0.1:8000   前端: http://127.0.0.1:5173"
    "$PY" -m qd serve --reload &
    BACKEND=$!
    trap 'kill $BACKEND 2>/dev/null || true' EXIT
    cd web
    [ -d node_modules ] || npm install
    npm run dev
    ;;
  demo)
    "$PY" -m qd demo
    ;;
  test)
    "$PY" -m pytest tests/ -q
    ;;
  serve|*)
    # 构建前端（如已构建则跳过）
    if [ ! -d web/dist ] || [ web/src -nt web/dist/index.html ]; then
      echo "构建前端…"
      (cd web && { [ -d node_modules ] || npm install; } && npm run build)
    fi
    echo ""
    echo "=================================================================="
    echo "  Quant Dojo 已启动：  http://127.0.0.1:8000"
    echo "  按 Ctrl+C 停止"
    echo "=================================================================="
    exec "$PY" -m qd serve
    ;;
esac
