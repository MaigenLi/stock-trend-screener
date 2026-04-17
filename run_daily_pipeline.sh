#!/bin/bash
# ============================================================
# 每日收盘后流水线脚本
# 依次执行：缓存更新 → 信号验证 → 反馈跟踪 → 选股 → 收盘报告
# 每步之间等待 60 秒
# ============================================================

LOG_DIR="/home/lyc/.openclaw/workspace/stock_reports"
WORKSPACE="/home/lyc/.openclaw/workspace/stock_trend"
PYTHON="/home/lyc/.venv/bin/python"
DATE=$(date +%Y-%m-%d)

echo "========== 每日收盘流水线 [${DATE}] =========="

# 步骤1：缓存基本面数据
echo "[$(date '+%H:%M:%S')] 步骤1/6: cache_fundamental.py"
${PYTHON} ${WORKSPACE}/cache_fundamental.py >> ${LOG_DIR}/fundamental_cache.log 2>&1
echo "[$(date '+%H:%M:%S')] 步骤1完成，休息60秒..."
sleep 60

# 步骤2：缓存前复权日线（默认开启验证）
echo "[$(date '+%H:%M:%S')] 步骤2/6: cache_qfq_daily.py --refresh"
${PYTHON} ${WORKSPACE}/cache_qfq_daily.py --refresh >> ${LOG_DIR}/qfq_cache.log 2>&1
echo "[$(date '+%H:%M:%S')] 步骤2完成，休息60秒..."
sleep 60

# 步骤3：信号验证
echo "[$(date '+%H:%M:%S')] 步骤3/6: signal_validator.py"
${PYTHON} ${WORKSPACE}/signal_validator.py >> ${LOG_DIR}/signal_validation.log 2>&1
echo "[$(date '+%H:%M:%S')] 步骤3完成，休息60秒..."
sleep 60

# 步骤4：反馈跟踪
echo "[$(date '+%H:%M:%S')] 步骤4/6: feedback_tracker.py"
${PYTHON} ${WORKSPACE}/feedback_tracker.py >> ${LOG_DIR}/feedback_tracker.log 2>&1
echo "[$(date '+%H:%M:%S')] 步骤4完成，休息60秒..."
sleep 60

# 步骤5：选股筛选
echo "[$(date '+%H:%M:%S')] 步骤5/6: gain_turnover_screen.py --check-fundamental --sector-bonus"
#${PYTHON} ${WORKSPACE}/gain_turnover_screen.py --check-fundamental --sector-bonus --check-volume-surge --days 3 --max-gain 8 --top-n 200 >> ${LOG_DIR}/daily_screen.log 2>&1
${PYTHON} ${WORKSPACE}/triple_screen.py --check-fundamental --check-volume-surge --sector-bonus --days 3 --min-gain 2 --max-gain 8 >> ${LOG_DIR}/daily_screen.log 2>&1
echo "[$(date '+%H:%M:%S')] 步骤5完成，休息60秒..."
sleep 60

# 步骤6：收盘报告
echo "[$(date '+%H:%M:%S')] 步骤6/6: closing_report.py"
${PYTHON} ${WORKSPACE}/closing_report.py >> ${LOG_DIR}/closing_report.log 2>&1
echo "[$(date '+%H:%M:%S')] 步骤6完成，休息60秒..."
sleep 60

# 步骤7：趋势强势股扫描
echo "[$(date '+%H:%M:%S')] 步骤7/7: trend_strong_screen.py --top-n 100"
${PYTHON} ${WORKSPACE}/trend_strong_screen.py --top-n 100 >> ${LOG_DIR}/trend_strong.log 2>&1

echo "========== 每日收盘流水线完成 [$(date '+%H:%M:%S')] =========="
