#!/bin/bash
# ============================================================
# 每日自选盯盘
# ============================================================

WORKSPACE="/home/lyc/.openclaw/workspace/stock_trend"
PYTHON="/home/lyc/.venv/bin/python"
DATE=$(date +%Y-%m-%d)

find ~/.openclaw/workspace/stock_trend -name "__pycache__" -exec rm -rf {} +

echo "========== 每日自选盯盘 [${DATE}] =========="
${PYTHON} ${WORKSPACE}/mootdx/mootdx_volume_monitor.py --date ${DATE} --news --tavily
