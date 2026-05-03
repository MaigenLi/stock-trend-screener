#!/bin/bash
# ============================================================
# 每日收盘后流水线脚本
# 依次执行：缓存更新 → 信号验证 → 反馈跟踪 → 板块缓存 → 选股 → 收盘报告 → 趋势强势股
# 每步之间等待 60 秒
# ============================================================

LOG_DIR="/home/lyc/.openclaw/workspace/stock_trend/output/log"
WORKSPACE="/home/lyc/.openclaw/workspace/stock_trend"
PYTHON="/home/lyc/.venv/bin/python"
DATE=$(date +%Y-%m-%d)

find ~/.openclaw/workspace/stock_trend -name "__pycache__" -exec rm -rf {} +

echo "========== 每日收盘流水线 [${DATE}] =========="

# 步骤1：缓存基本面数据
echo "[$(date '+%H:%M:%S')] 步骤1/8: cache_fundamental.py"
${PYTHON} ${WORKSPACE}/cache_fundamental.py >> ${LOG_DIR}/fundamental_cache.log 2>&1
echo "[$(date '+%H:%M:%S')] 步骤1完成，休息60秒..."
sleep 60

# 步骤2：缓存前复权日线（默认开启验证）
echo "[$(date '+%H:%M:%S')] 步骤2/8: cache_qfq_daily.py --date ${DATE} --refresh"
${PYTHON} ${WORKSPACE}/cache_qfq_daily.py --date ${DATE} --refresh >> ${LOG_DIR}/qfq_cache.log 2>&1
echo "[$(date '+%H:%M:%S')] 步骤2完成，休息60秒..."
sleep 60

# 步骤3：信号验证
echo "[$(date '+%H:%M:%S')] 步骤3/8: signal_validator.py --date ${DATE}"
#${PYTHON} ${WORKSPACE}/signal_validator.py --date ${DATE} >> ${LOG_DIR}/signal_validation.log 2>&1
echo "[$(date '+%H:%M:%S')] 步骤3完成，休息60秒..."
sleep 60

# 步骤4：反馈跟踪
echo "[$(date '+%H:%M:%S')] 步骤4/8: feedback_tracker.py"
#${PYTHON} ${WORKSPACE}/feedback_tracker.py >> ${LOG_DIR}/feedback_tracker.log 2>&1
echo "[$(date '+%H:%M:%S')] 步骤4完成，休息60秒..."
sleep 60

# 步骤5：刷新板块热点缓存
echo "[$(date '+%H:%M:%S')] 步骤5/8: refresh_sector_cache.py"
#${PYTHON} ${WORKSPACE}/refresh_sector_cache.py >> ${LOG_DIR}/refresh_sector_cache.log 2>&1
echo "[$(date '+%H:%M:%S')] 步骤5完成，休息60秒..."
sleep 60

# 步骤6：选股筛选
echo "[$(date '+%H:%M:%S')] 步骤6/8: triple_screen.py --date ${DATE}"
#${PYTHON} ${WORKSPACE}/triple_screen.py --date ${DATE} >> ${LOG_DIR}/daily_screen.log 2>&1
sleep 60

${PYTHON} ${WORKSPACE}/review_screen/screen_double.py --date ${DATE} --top-n 250 --mode winner --gain20 20 --turnover 6 >> ${LOG_DIR}/screen_double.log 2>&1
echo "[$(date '+%H:%M:%S')] 步骤6完成，休息60秒..."
sleep 60

# 步骤7：收盘报告
echo "[$(date '+%H:%M:%S')] 步骤7/8: closing_report.py"
#${PYTHON} ${WORKSPACE}/closing_report.py >> ${LOG_DIR}/closing_report.log 2>&1
echo "[$(date '+%H:%M:%S')] 步骤7完成，休息60秒..."
sleep 60

# 步骤8：趋势强势股扫描
echo "[$(date '+%H:%M:%S')] 步骤8/8: trend_strong_screen.py --top-n 100"
#${PYTHON} ${WORKSPACE}/trend_strong_screen.py --top-n 100 >> ${LOG_DIR}/trend_strong.log 2>&1

echo "========== 每日收盘流水线完成 [$(date '+%H:%M:%S')] =========="

