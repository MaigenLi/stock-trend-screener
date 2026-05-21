#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
screen_trend.py — N条件选股器
============================================================

使用方法：
  python screen_trend.py                    # 全市场扫描（默认最新日期）
  python screen_trend.py --date 2026-05-08   # 指定信号日扫描
  python screen_trend.py --top 30             # 显示前30只
  python screen_trend.py --min-turnover 3     # 换手率门槛3%
  python screen_trend.py --code sh603629     # 单票分析
  python screen_trend.py --blk '/mnt/d/new_tdx/T0002/blocknew/RMG.blk'   # 指定通达信板块文件扫描
"""

import sys
import argparse
from pathlib import Path
from datetime import date, datetime

WORKSPACE = Path.home() / ".openclaw/workspace"
sys.path.insert(0, str(WORKSPACE / "stock_trend"))
CACHE_DIR   = WORKSPACE / ".cache" / "qfq_daily"
RAW_CACHE_DIR = WORKSPACE / ".cache" / "raw_daily"

import numpy as np
import pandas as pd
from gain_turnover import (
    normalize_symbol,
    load_stock_names_akshare,
    get_all_stock_codes_akshare,
    load_qfq_history,
    load_raw_history,
    rolling_mean,
)

def get_weekday_cn(date_input):
    """
    返回输入日期对应的中文星期几。

    参数:
        date_input: 可以是字符串（格式 'YYYY-MM-DD'）、datetime.date 对象或 datetime.datetime 对象

    返回:
        字符串，例如 '星期一'、'星期二' ... '星期日'
    """
    # 处理字符串输入
    if isinstance(date_input, str):
        date_obj = datetime.strptime(date_input, "%Y-%m-%d").date()
    # 处理 datetime 对象
    elif isinstance(date_input, datetime):
        date_obj = date_input.date()
    # 处理 date 对象
    elif isinstance(date_input, date):
        date_obj = date_input
    else:
        raise TypeError("date_input 必须是字符串 'YYYY-MM-DD'、datetime.date 或 datetime.datetime 对象")

    weekday_num = date_obj.weekday()
    if weekday_num >= 5:   # 5=周六, 6=周日
        return None

    weekdays_cn = ["星期一", "星期二", "星期三", "星期四", "星期五"]
    return weekdays_cn[weekday_num]   # 工作日索引 0~4

# ── 预加载缓存（batch模式加速用）──
_price = {}   # code -> DataFrame

def preload(signal_date=None, data_mode="raw"):
    """
    预加载所有缓存CSV到内存，batch扫描时直接查dict省去每次读文件。
    signal_date: 只保留<=signal_date的历史数据
    data_mode: "raw" 或 "qfq"
    """
    import pandas as pd
    global _price

    # 确保 _price 已初始化
    if "_price" not in globals():
        _price = {}

    cache_dir = RAW_CACHE_DIR if data_mode == "raw" else CACHE_DIR
    if data_mode == "raw":
        files = [f for f in cache_dir.glob("*.csv") if not f.name.endswith("_qfq.csv")]
    else:
        files = list(cache_dir.glob("*_qfq.csv"))

    loaded = 0
    for f in files:
        code_raw = f.stem.replace("_qfq", "")
        key = normalize_symbol(code_raw)
        try:
            df = pd.read_csv(f, dtype={"date": str})
            df["date"] = pd.to_datetime(df["date"])

            # ----- 先切片，再计算指标 -----
            if signal_date:
                df = df[df["date"] <= pd.to_datetime(signal_date)]

            if df.empty:
                continue  # 切片后没数据就跳过

            #close = df["close"].values.astype(float)
            #df["_ma5"]  = rolling_mean(close, 5)
            #df["_ma10"] = rolling_mean(close, 10)
            #df["_ma20"] = rolling_mean(close, 20)
            #df["_ma60"] = rolling_mean(close, 60)
            #df["_atr_pct"] = calc_atr_percent(df, 14)

            #n = len(df)
            #ma5_vals  = df["_ma5"].values.astype(float)
            #ma10_vals = df["_ma10"].values.astype(float)
            #ma20_vals = df["_ma20"].values.astype(float)
            #atr_vals  = df["_atr_pct"].values.astype(float)

            # 为每个位置计算斜率，并除以对应位置的ATR
            #ma5_slope_atr  = np.array([_lr_slope(ma5_vals,  idx) / (atr_vals[idx] + 1e-12) for idx in range(n)])
            #ma10_slope_atr = np.array([_lr_slope(ma10_vals, idx) / (atr_vals[idx] + 1e-12) for idx in range(n)])
            #ma20_slope_atr = np.array([_lr_slope(ma20_vals, idx) / (atr_vals[idx] + 1e-12) for idx in range(n)])

            # 可选：存回DataFrame
            #df["_ma5_slope_atr"]  = ma5_slope_atr
            #df["_ma10_slope_atr"] = ma10_slope_atr
            #df["_ma20_slope_atr"] = ma20_slope_atr

            _price[key] = df
            loaded += 1

        except Exception as e:
            print(f"加载 {f.name} 失败: {e}")

    print(f"预加载 {loaded} 只（{data_mode}），范围≤{signal_date or '最新'}", flush=True)


def _load_df(code, end_date=None, data_mode="raw"):
    """优先从内存缓存取，未命中则兜底用load_history"""
    key = normalize_symbol(code)
    df = _price.get(key)
    if df is None:
        return load_history(code, end_date=end_date, data_mode=data_mode)
    return df

# ── 参数 ──────────────────────────────────────────────────
RED_CROSS = '\033[91m✗\033[0m'
GREEN_CHECK = '\033[92m✓\033[0m'

TURNOVER_LEN  = 5     # 近5日

# ── 换手率-市值幂律适配 ───────────────────────────────
LOAD_MODE = "raw"   # 默认使用原始不复权数据 ("qfq"=前复权)

def load_history(code: str, end_date: str = None, data_mode: str = "raw", refresh: bool = False):
    """统一加载接口。data_mode: 'raw'=原始复权, 'qfq'=前复权"""
    if data_mode == "qfq":
        return load_qfq_history(code, end_date=end_date, refresh=refresh)
    return load_raw_history(code, start_date=None, end_date=end_date, refresh=refresh)


# 公式: TURNOVER_MIN = TURN_BASE × (TURN_CAP_REF / 市值)^TURN_POWER
# 锚点: 30亿→8%, 50亿→6%, 100亿→5%, 200亿→3.5%, 500亿→2.7%
TURN_POWER    = 0.38  # 幂指数（由4个锚点拟合得出）
TURN_CAP_REF  = 100.0  # 参考市值（亿元）
TURN_BASE     = 5.0    # 参考市值对应的换手率(%)
TURN_FLOOR    = 2.0    # 换手率下限兜底

def get_turnover_min(cap_yi: float) -> float:
    """根据市值计算自适应换手率下限（幂律衰减）"""
    if cap_yi <= 0:
        return TURN_BASE  # 兜底：无效市值用基准换手
    return max(TURN_FLOOR, TURN_BASE * (TURN_CAP_REF / cap_yi) ** TURN_POWER)

TURNOVER_MIN  = TURN_BASE  # 向后兼容（实际使用get_turnover_min）
MKT_CAP_MIN   = 25.0  # 流通市值下限（亿元）
GAIN_DAY_MIN  = -2.0  # 当日涨幅下限%%
GAIN_DAY_MAX  = 20.1   # 上调：允许涨停/接近涨停
MA_LEN        = 20    # 计算MA20需要
MA_LEN_60     = 60    # 计算MA60需要

# ── 趋势斜率阈值 ─────────────────────────────────────────
SLOPE_MA_MID     = 1.2   # MA 3日斜率中间值（每日相对MA均价的涨幅）

SLOPE_MA5     = 0.5   # MA5 3日斜率 > 0.5%（每日相对MA均价的涨幅）
SLOPE_MA10    = 0.35   # MA10 3日斜率 > 0.35%（均线越大变化越慢，阈值更高）
SLOPE_MA20    = 0.35   # MA20 3日斜率 > 0.35%（长周期均线，阈值最低）

SLOPE_MA5_MAX     = 1.8   # MA5 3日斜率 < 1.8%（每日相对MA均价的涨幅）
SLOPE_MA10_MAX    = 1.8   # MA10 3日斜率 < 1.8%（均线越大变化越慢，阈值更高）
SLOPE_MA20_MAX    = 2.0   # MA20 3日斜率 < 2.0%（长周期均线，阈值最低）



CLOSE_NEAR_HIGH_RATIO = 0.95   # 当前收盘 > 20日最高收盘 × CLOSE_NEAR_HIGH_RATIO

# 量能阈值
VOL_RATIO5_MAX = 1.10   # 近5日均量比（当日/前5日均）：超过此值视为放量，不建议追高

# -- 第一套参数 慢牛、稳健成长股 --date 2026-05-06   --code 301013
#GAIN20_MIN    = 20.0  # 下调：目标股20日涨幅多在7~12%
#GAIN20_MIN    = 18.0  # 下调：目标股20日涨幅多在7~12%
GAIN20_MIN    = 14.0  # 下调：目标股20日涨幅多在7~12%
GAIN20_MAX    = 38.0  # 目标股20日涨幅可达60%
#GAIN5_MIN     = 5  # 目标股5日涨幅多在2~25%）
GAIN5_MIN     = 3.5  # 目标股5日涨幅多在2~25%）
GAIN5_MAX     = 13.8   # 目标股5日涨幅可达53%

# 日变
#MA5_DAILY  = 1.12
MA5_DAILY  = 0.72
MA10_DAILY = 0.40
MA20_DAILY = 0.65

# 均线斜率 / ATR波动率
#SLOPE_MA5_ATR     = 0.170  # 下限
SLOPE_MA5_ATR     = 0.130  # 下限
SLOPE_MA10_ATR    = 0.050  # 下限
SLOPE_MA20_ATR    = 0.124  # 下限

SLOPE_MA5_ATR_MAX     = 0.390   # 上限
SLOPE_MA10_ATR_MAX    = 0.300   # 上限
SLOPE_MA20_ATR_MAX    = 0.350   # 上限


# -- 第二套参数 主升浪、短线强势股 --
GAIN20_MIN2 = 25.0
GAIN20_MAX2 = 65.0
GAIN5_MIN2 = 8.0
GAIN5_MAX2 = 25.0

# 日变
MA5_DAILY2  = 1.80
MA10_DAILY2 = 1.20
MA20_DAILY2 = 1.20

# 均线斜率 / ATR波动率
SLOPE_MA5_ATR2     = 0.400  # 下限
SLOPE_MA10_ATR2    = 0.240  # 下限
SLOPE_MA20_ATR2    = 0.200  # 下限

SLOPE_MA5_ATR_MAX2     = 0.650   # 上限
SLOPE_MA10_ATR_MAX2    = 0.420   # 上限
SLOPE_MA20_ATR_MAX2    = 0.330   # 上限

# -- 第三套参数 中期趋势延续但短期休整/平台蓄势股 --date 2026-05-19 --code 688449
GAIN20_MIN3 = 25.0
GAIN20_MAX3 = 70.0
GAIN5_MIN3 = -3.0   #（允许轻微回调）
GAIN5_MAX3 = 8.0

# 日变
MA5_DAILY3  = 1.80
MA10_DAILY3 = 2.50
MA20_DAILY3 = 2.20

# 均线斜率 / ATR波动率
SLOPE_MA5_ATR3     = 0.100  # 下限
SLOPE_MA10_ATR3    = 0.350  # 下限
SLOPE_MA20_ATR3    = 0.250  # 下限

SLOPE_MA5_ATR_MAX3     = 0.180   # 上限
SLOPE_MA10_ATR_MAX3    = 0.400   # 上限
SLOPE_MA20_ATR_MAX3    = 0.350   # 上限

# ═══════════════════════════════════════════════════════════════
# 特殊通道参数 — 涨停蓄势后突破
# ═══════════════════════════════════════════════════════════════
# 信号日(i)涨停：close(i)为20日最高，且满足均线多头+MA5斜率加速
LIMITUP_I_SLOPE_MA5   = 0.2   # ma5_slope
LIMITUP_I_DAILY_MA5   = 1.0   # ma5_daily  > 1.0%
#LIMITUP_I_SLOPE_MA10   = 0.25   # ma10_slope
LIMITUP_I_SLOPE_MA10   = 0.45   # ma10_slope
LIMITUP_I_SLOPE_MA20   = 0.23   # ma20_slope
LIMITUP_I_DAILY_MA10   = 0.5   # ma5_daily  < 0.5%
LIMITUP_I_VOLUME       = 1.2   # 量比放大 1.2
# 前1天(i-1)：涨幅<6%，ma5_slope在0~0.7%，ma5_daily在0~0.5%
#LIMITUP_I1_SLOPE_MA5_U = 0.15   # ma5_slope
LIMITUP_I1_SLOPE_MA5_U = 0.23   # ma5_slope
LIMITUP_I1_DAILY_U     = 1.5   # ma5_daily  < 1.0%
# 前2天(i-2)：涨幅<2%，ma5_slope<0.3%，ma5_daily<0.2%
LIMITUP_I2_SLOPE_MA5_U = 0.18   # ma5_slope
LIMITUP_I2_DAILY_U     = 1.0   # ma5_daily  < 1.0%

LIMITUP_GAIN_DAY_MAX  = 3.0   #

"""
============================================================
选股条件（同时满足）：
  [1] MA5>MA10>MA20 且 MA5>MA60, MA10>MA60 (多头排列)
  [2] 两层：MA日变>{MA5_DAILY}%% + 3点回归斜率>阈值
  [3] 流通市值 >= {MKT_CAP_MIN:.0f}亿
  [4] 近5日平均换手率 >= 自适应（幂律：30亿→8.9%，100亿→5%，越大越低）
  [5] 20日涨幅 {GAIN20_MIN:.0f}%%~{GAIN20_MAX:.0f}%%
  [6] 当日涨幅 {GAIN_DAY_MIN:.0f}%%~{GAIN_DAY_MAX:.0f}%%
  [7] 收盘 ≥ 20日最高 × {CLOSE_NEAR_HIGH_RATIO}
  [8] 5日涨幅 {GAIN5_MIN:.0f}%%~{GAIN5_MAX:.0f}%%
============================================================
"""

# 计算 ATR%
def calc_atr_percent(df, period=14):
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    close = df["close"].values.astype(float)

    tr = np.zeros(len(close))

    tr[0] = high[0] - low[0]

    for i in range(1, len(close)):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1]),
        )

    atr = rolling_mean(tr, period)

    # ATR转百分比
    atr_pct = atr / close * 100.0

    return atr_pct

# ══════════════════════════════════════════════════════════════════
# 条件检查函数（统一入口，所有阈值在此维护）
# 每个函数签名: (idx, ..., derived_values) -> (bool_ok, str_description)
# 两处调用: _check_ma_conditions(输出描述) / check_ma(筛选+返回dict)
# ══════════════════════════════════════════════════════════════════

def _cond1_ma_line(idx, ma5, ma10, ma20, ma60, close):
    """① MA5>MA10>MA20 且 MA5>MA60, MA10>MA60（多头排列）"""
    ok = ma5[idx] > ma10[idx] > ma20[idx] and ma5[idx] > ma60[idx] and ma10[idx] > ma60[idx]
    if not ok:
        return False, f"MA排列不满足：MA5={ma5[idx]:.2f} MA10={ma10[idx]:.2f} MA20={ma20[idx]:.2f} MA60={ma60[idx]:.2f}"
    if ma20[idx] < ma60[idx]:
        # 豁免：只要 MA5 正在上升（今日MA5 > 昨日MA5），均线系统整体健康即可通过
        if ma5[idx] > ma5[idx-1]:
            return True, f"MA5>{ma5[idx]:.2f}>MA10>{ma10[idx]:.2f}>MA20>{ma20[idx]:.2f} 且 >MA60={ma60[idx]:.2f} (MA5上升中)"
        return False, f"MA20<MA60且不满足豁免条件"
    if close[idx] < ma5[idx] and close[idx-1] < ma5[idx-1]:
        # 豁免：如果MA5正在上升（今日MA5 > 昨日MA5），允许收盘稍低于MA5
        if ma5[idx] > ma5[idx-1]:
            return True, f"MA5>{ma5[idx]:.2f}>MA10>{ma10[idx]:.2f}>MA20>{ma20[idx]:.2f} 且 >MA60={ma60[idx]:.2f} (MA5上升中)"
        return False, f"最近两天都收盘在MA5下方(MA5未上升)"
    return True, f"MA5>{ma5[idx]:.2f}>MA10>{ma10[idx]:.2f}>MA20>{ma20[idx]:.2f} 且 >MA60={ma60[idx]:.2f}"

def _cond2a_slope_atr(idx, ma5_slope_atr, ma10_slope_atr, ma20_slope_atr):
    """②-1 均线斜率/ATR波动率在合理区间（方向向上但不过热）"""
    ok5  = SLOPE_MA5_ATR  <= ma5_slope_atr[idx]  < SLOPE_MA5_ATR_MAX
    ok10 = SLOPE_MA10_ATR <= ma10_slope_atr[idx] < SLOPE_MA10_ATR_MAX
    ok20 = SLOPE_MA20_ATR <= ma20_slope_atr[idx] < SLOPE_MA20_ATR_MAX
    ok = ok5 and ok10 and ok20
    if not ok:
        msg = []
        msg.append(f"MA5_atr={ma5_slope_atr[idx]:.3f} [{SLOPE_MA5_ATR:.3f},{SLOPE_MA5_ATR_MAX:.3f}) {GREEN_CHECK if ok5 else RED_CROSS}") 
        msg.append(f" MA10_atr={ma10_slope_atr[idx]:.3f} [{SLOPE_MA10_ATR:.3f},{SLOPE_MA10_ATR_MAX}:.3f) {GREEN_CHECK if ok10 else RED_CROSS}")
        msg.append(f" MA20_atr={ma20_slope_atr[idx]:.3f} [{SLOPE_MA20_ATR:.3f},{SLOPE_MA20_ATR_MAX:.3f}) {GREEN_CHECK if ok20 else RED_CROSS}")
        return False, " ".join(msg)
    return True, f"MA5_atr={ma5_slope_atr[idx]:.3f} [{SLOPE_MA5_ATR:.3f},{SLOPE_MA5_ATR_MAX:.3f}) MA10_atr={ma10_slope_atr[idx]:.3f} [{SLOPE_MA10_ATR:.3f},{SLOPE_MA10_ATR_MAX}:.3f) MA20_atr={ma20_slope_atr[idx]:.3f} [{SLOPE_MA20_ATR:.3f},{SLOPE_MA20_ATR_MAX:.3f})"

def _cond2a_2_slope_atr(idx, ma5_slope_atr, ma10_slope_atr, ma20_slope_atr):
    """②-1 均线斜率/ATR波动率在合理区间 """
    ok5  = SLOPE_MA5_ATR2  <= ma5_slope_atr[idx]  < SLOPE_MA5_ATR_MAX2
    ok10 = SLOPE_MA10_ATR2 <= ma10_slope_atr[idx] < SLOPE_MA10_ATR_MAX2
    ok20 = SLOPE_MA20_ATR2 <= ma20_slope_atr[idx] < SLOPE_MA20_ATR_MAX2
    ok = ok5 and ok10 and ok20
    if not ok:
        msg = []
        msg.append(f"MA5_atr={ma5_slope_atr[idx]:.3f} [{SLOPE_MA5_ATR2:.3f},{SLOPE_MA5_ATR_MAX2:.3f}) {GREEN_CHECK if ok5 else RED_CROSS}") 
        msg.append(f" MA10_atr={ma10_slope_atr[idx]:.3f} [{SLOPE_MA10_ATR2:.3f},{SLOPE_MA10_ATR_MAX2:.3f}) {GREEN_CHECK if ok10 else RED_CROSS}")
        msg.append(f" MA20_atr={ma20_slope_atr[idx]:.3f} [{SLOPE_MA20_ATR2:.3f},{SLOPE_MA20_ATR_MAX2:.3f}) {GREEN_CHECK if ok20 else RED_CROSS}")
        return False, " ".join(msg)
    return True, f"MA5_atr={ma5_slope_atr[idx]:.3f} [{SLOPE_MA5_ATR2:.3f},{SLOPE_MA5_ATR_MAX2:.3f}) MA10_atr={ma10_slope_atr[idx]:.3f} [{SLOPE_MA10_ATR2:.3f},{SLOPE_MA10_ATR_MAX2}:.3f) MA20_atr={ma20_slope_atr[idx]:.3f} [{SLOPE_MA20_ATR2:.3f},{SLOPE_MA20_ATR_MAX2:.3f})"

def _cond2a_3_slope_atr(idx, ma5_slope_atr, ma10_slope_atr, ma20_slope_atr):
    """②-1 均线斜率/ATR波动率在合理区间 """
    ok5  = SLOPE_MA5_ATR3  <= ma5_slope_atr[idx]  < SLOPE_MA5_ATR_MAX3
    ok10 = SLOPE_MA10_ATR3 <= ma10_slope_atr[idx] < SLOPE_MA10_ATR_MAX3
    ok20 = SLOPE_MA20_ATR3 <= ma20_slope_atr[idx] < SLOPE_MA20_ATR_MAX3
    ok = ok5 and ok10 and ok20
    if not ok:
        msg = []
        msg.append(f"MA5_atr={ma5_slope_atr[idx]:.3f} [{SLOPE_MA5_ATR3:.3f},{SLOPE_MA5_ATR_MAX3:.3f}) {GREEN_CHECK if ok5 else RED_CROSS}") 
        msg.append(f" MA10_atr={ma10_slope_atr[idx]:.3f} [{SLOPE_MA10_ATR3:.3f},{SLOPE_MA10_ATR_MAX3:.3f}) {GREEN_CHECK if ok10 else RED_CROSS}")
        msg.append(f" MA20_atr={ma20_slope_atr[idx]:.3f} [{SLOPE_MA20_ATR3:.3f},{SLOPE_MA20_ATR_MAX3:.3f}) {GREEN_CHECK if ok20 else RED_CROSS}")
        return False, " ".join(msg)
    return True, f"MA5_atr={ma5_slope_atr[idx]:.3f} [{SLOPE_MA5_ATR3:.3f},{SLOPE_MA5_ATR_MAX3:.3f}) MA10_atr={ma10_slope_atr[idx]:.3f} [{SLOPE_MA10_ATR3:.3f},{SLOPE_MA10_ATR_MAX3}:.3f) MA20_atr={ma20_slope_atr[idx]:.3f} [{SLOPE_MA20_ATR3:.3f},{SLOPE_MA20_ATR_MAX3:.3f})"

def _cond2b_ma_daily(idx, ma5_daily, ma10_daily, ma20_daily, close):
    """②-2 MA5/MA10/MA20 日变 >= 阈值（均线向上推，目标股多为缓涨/横走）"""
    ok = ma5_daily >= MA5_DAILY and ma10_daily >= MA10_DAILY and ma20_daily >= MA20_DAILY

    # ── 条件⑤ 5日 20日涨幅 ────────────────────────────────────
    gain5    = (close[idx]/close[idx-5]-1)*100 if idx>=5 else float('nan')
    gain20   = (close[idx]/close[idx-20]-1)*100 if idx>=20 else float('nan')
    ok5 = GAIN5_MIN <= gain5 <= GAIN5_MAX
    ok20 = GAIN20_MIN <= gain20 <= GAIN20_MAX
    msg = []

    msg.append(f"MA5_daily={ma5_daily:.3f}%>={MA5_DAILY:.2f}% MA10_daily={ma10_daily:.3f}%>={MA10_DAILY:.2f}% MA20_daily={ma20_daily:.3f}%>={MA20_DAILY:.2f}% {GREEN_CHECK if ok else RED_CROSS}") 
    msg.append(f" 5日涨幅{gain5:+.1f}% ∈ [{GAIN5_MIN},{GAIN5_MAX}]% {GREEN_CHECK if ok5 else RED_CROSS} 20日涨幅{gain20:+.1f}% ∈ [{GAIN20_MIN},{GAIN20_MAX}]% {GREEN_CHECK if ok20 else RED_CROSS}") 

    if ok and ok5 and ok20:
        return True, " ".join(msg)
    else:
        return False, " ".join(msg)

def _cond2b_2_ma_daily(idx, ma5_daily, ma10_daily, ma20_daily, close):
    """②-2 MA5/MA10/MA20 日变 >= 阈值（主升浪、短线强势股）"""
    ok = ma5_daily >= MA5_DAILY2 and ma10_daily >= MA10_DAILY2 and ma20_daily >= MA20_DAILY2

    # ── 条件⑤ 5日 20日涨幅 ────────────────────────────────────
    gain5    = (close[idx]/close[idx-5]-1)*100 if idx>=5 else float('nan')
    gain20   = (close[idx]/close[idx-20]-1)*100 if idx>=20 else float('nan')
    ok5 = GAIN5_MIN2 <= gain5 <= GAIN5_MAX2
    ok20 = GAIN20_MIN2 <= gain20 <= GAIN20_MAX2
    msg = []

    msg.append(f"MA5_daily={ma5_daily:.3f}%>={MA5_DAILY2:.2f}% MA10_daily={ma10_daily:.3f}%>={MA10_DAILY2:.2f}% MA20_daily={ma20_daily:.3f}%>={MA20_DAILY2:.2f}% {GREEN_CHECK if ok else RED_CROSS}") 
    msg.append(f" 5日涨幅{gain5:+.1f}% ∈ [{GAIN5_MIN2},{GAIN5_MAX2}]% {GREEN_CHECK if ok5 else RED_CROSS} 20日涨幅{gain20:+.1f}% ∈ [{GAIN20_MIN2},{GAIN20_MAX2}]% {GREEN_CHECK if ok20 else RED_CROSS}") 

    if ok and ok5 and ok20:
        return True, " ".join(msg)
    else:
        return False, " ".join(msg)

def _cond2b_3_ma_daily(idx, ma5_daily, ma10_daily, ma20_daily, close):
    """②-3 MA5/MA10/MA20 日变 >= 阈值（中期趋势延续但短期休整/平台蓄势股）"""
    ok = ma5_daily >= MA5_DAILY3 and ma10_daily >= MA10_DAILY3 and ma20_daily >= MA20_DAILY3

    # ── 条件⑤ 5日 20日涨幅 ────────────────────────────────────
    gain5    = (close[idx]/close[idx-5]-1)*100 if idx>=5 else float('nan')
    gain20   = (close[idx]/close[idx-20]-1)*100 if idx>=20 else float('nan')
    ok5 = GAIN5_MIN3 <= gain5 <= GAIN5_MAX3
    ok20 = GAIN20_MIN3 <= gain20 <= GAIN20_MAX3
    msg = []

    msg.append(f"MA5_daily={ma5_daily:.3f}%>={MA5_DAILY3:.2f}% MA10_daily={ma10_daily:.3f}%>={MA10_DAILY3:.2f}% MA20_daily={ma20_daily:.3f}%>={MA20_DAILY3:.2f}% {GREEN_CHECK if ok else RED_CROSS}") 
    msg.append(f" 5日涨幅{gain5:+.1f}% ∈ [{GAIN5_MIN3},{GAIN5_MAX3}]% {GREEN_CHECK if ok5 else RED_CROSS} 20日涨幅{gain20:+.1f}% ∈ [{GAIN20_MIN3},{GAIN20_MAX3}]% {GREEN_CHECK if ok20 else RED_CROSS}") 

    if ok and ok5 and ok20:
        return True, " ".join(msg)
    else:
        return False, " ".join(msg)

def _cond3_mktcap(idx, mktcap):
    """③ 流通市值 >= 下限"""
    ok = mktcap >= MKT_CAP_MIN
    if not ok:
        return False, f"流通市值{mktcap:.1f}亿 < {MKT_CAP_MIN}亿"
    return True, f"流通市值{mktcap:.1f}亿 ≥ {MKT_CAP_MIN}亿"

def _cond4_turnover(idx, turnover_avg, turnover_now, turnover_prev, gain_day, mktcap):
    """④ 近5日平均换手率 >= 自适应下限（或当日放量大涨满足豁免）"""
    turn_thresh = get_turnover_min(mktcap) if mktcap > 0 else TURN_BASE
    ok = turnover_avg >= turn_thresh
    if not ok:
        ok = (((turnover_now >= turnover_avg and turnover_prev >= turnover_avg) or (turnover_now >= turnover_avg and turnover_now >= TURN_BASE)) and gain_day >= 0.1)
        if ok:
            return True, f"换手豁免：当日{turnover_now:.2f}%放量且涨幅{gain_day:.2f}%>=0.1%"
        else:
            if mktcap > 0 and mktcap > 300 and turnover_avg >= TURN_FLOOR:
                return True, f"换手豁免：近5日均换手{turnover_avg:.2f}% 大市值：{mktcap:.0f}亿"
            else:
                return False, f"近5日均换手{turnover_avg:.2f}% < {turn_thresh:.1f}%，不满足豁免条件"
    return True, f"近5日均换手{turnover_avg:.2f}% ≥ {turn_thresh:.1f}%（市值{mktcap:.0f}亿）"

def _cond6_gain_day(idx, gain_day):
    """⑥ 当日涨幅在合理区间"""
    ok = GAIN_DAY_MIN <= gain_day <= GAIN_DAY_MAX
    if not ok:
        return False, f"当日涨幅{gain_day:+.2f}% 不在[{GAIN_DAY_MIN},{GAIN_DAY_MAX}]%"
    return True, f"当日涨幅{gain_day:+.2f}% ∈ [{GAIN_DAY_MIN},{GAIN_DAY_MAX}]%"

def _cond7_close_near_high(idx, close, recent_20_high):
    """⑦ 收盘价接近20日最高"""
    thresh = recent_20_high * CLOSE_NEAR_HIGH_RATIO
    ok = close[idx] >= thresh
    if not ok:
        return False, f"收盘{close[idx]:.2f} < 20日最高{recent_20_high:.2f}×{CLOSE_NEAR_HIGH_RATIO}={thresh:.2f}"
    return True, f"收盘{close[idx]:.2f} ≥ 20日最高×{CLOSE_NEAR_HIGH_RATIO}={thresh:.2f}"

# ── 共用工具：3点线性回归斜率（归一化）──
def _lr_slope(arr, idx):
    """3点线性回归斜率（归一化），arr[idx-2], arr[idx-1], arr[idx]"""
    if idx < 2:
        return -999.0
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([arr[idx-2], arr[idx-1], arr[idx]], dtype=float)
    x_mean, y_mean = x.mean(), y.mean()
    num = np.sum((x - x_mean) * (y - y_mean))
    den = np.sum((x - x_mean) ** 2)
    if den < 1e-12:
        return 0.0
    # 斜率归一化：除以均价，消除股价绝对水平影响
    # 结果 = 每天变化百分比（相对于均线均值），便于跨股票统一阈值
    return (num / den) / (y_mean + 1e-12) * 100.0


# ══════════════════════════════════════════════════════════════════
# 特殊通道条件检查函数（统一入口，所有阈值在此维护）
# ══════════════════════════════════════════════════════════════════

def _cond_lim_涨停(idx, close, limit_ratio):
    """特殊通道①：涨停（信号日收盘=涨停价）"""
    prev_close = close[idx - 1]
    limit_up = round(prev_close * (1 + limit_ratio), 2)
    is_limitup = abs(close[idx] - limit_up) < 0.005
    return is_limitup, limit_up

def _cond_lim_20high(idx, close, recent_20_high):
    """特殊通道①'：收盘为20日最高"""
    return close[idx] >= recent_20_high * 0.9999

def _cond_lim_均线多头(idx, ma5, ma10, ma20, ma60, close):
    """特殊通道②：收盘高于全部均线且 MA5>MA10×0.99、MA5>MA20×0.99"""
    ok1 = close[idx] > ma5[idx] and close[idx] > ma10[idx] and close[idx] > ma20[idx] and close[idx] > ma60[idx]
    ok2 = ma5[idx] > (ma10[idx] * 0.99)
    ok3 = ma5[idx] > (ma20[idx] * 0.99)
    return ok1 and ok2 and ok3

def _cond_lim_ma5_accel(idx, ma5_slope_atr, ma5_daily):
    """特殊通道③：MA5斜率/ATR>阈值 且 MA5日变>阈值（加速）"""
    ok_slope = ma5_slope_atr[idx] > LIMITUP_I_SLOPE_MA5
    ok_daily = ma5_daily > LIMITUP_I_DAILY_MA5
    return ok_slope, ok_daily

def _cond_lim_ma10_slope(idx, ma10_slope_atr):
    """特殊通道③：MA10斜率/ATR < 阈值（不能过快）"""
    return ma10_slope_atr[idx] < LIMITUP_I_SLOPE_MA10

def _cond_lim_ma20_slope(idx, ma20_slope_atr):
    """特殊通道③：MA20斜率/ATR < 阈值（不能过快）"""
    return ma20_slope_atr[idx] < LIMITUP_I_SLOPE_MA20

def _cond_lim_量比(idx, df):
    """特殊通道③：当日放量（量比>=阈值）"""
    vol_today = float(df["volume"].values[idx])
    vol_prev  = float(df["volume"].values[idx - 1]) if idx >= 1 else 0.0
    return vol_prev > 0 and vol_today / vol_prev >= LIMITUP_I_VOLUME

def _cond_lim_i1(idx, close, ma5_slope_atr, ma5_daily_i1, ma5):
    """特殊通道 i-1 日：涨幅<6% 且 MA5斜率/ATR<阈值 且 MA5日变<阈值（蓄势）"""
    gain_i1 = abs(close[idx - 1] / ma5[idx - 1] - 1) * 100
    ok_gain  = gain_i1 < LIMITUP_GAIN_DAY_MAX
    if not ok_gain:
        gain_i1 = (close[idx - 1] / close[idx - 2] - 1) * 100
        ok_gain  = gain_i1 < LIMITUP_GAIN_DAY_MAX
    ok_slope = LIMITUP_I1_SLOPE_MA5_U > ma5_slope_atr[idx - 1]
    ok_daily = LIMITUP_I1_DAILY_U > ma5_daily_i1
    return ok_gain, ok_slope, ok_daily, gain_i1

def _cond_lim_i2(idx, close, ma5_slope_atr, ma5_daily_i2, ma5):
    """特殊通道 i-2 日：涨幅<2% 且 MA5斜率/ATR<阈值 且 MA5日变<阈值（缩量整理）"""
    gain_i2 = (close[idx - 2] / ma5[idx - 2] - 1) * 100
    ok_gain  = gain_i2 < LIMITUP_GAIN_DAY_MAX
    if not ok_gain:
        gain_i2 = (close[idx - 2] / close[idx - 3] - 1) * 100
        ok_gain  = gain_i2 < LIMITUP_GAIN_DAY_MAX
    ok_slope = ma5_slope_atr[idx - 2] < LIMITUP_I2_SLOPE_MA5_U
    ok_daily = ma5_daily_i2 < LIMITUP_I2_DAILY_U
    return ok_gain, ok_slope, ok_daily, gain_i2

def _check_limitup_conditions(df: pd.DataFrame, signal_date: str = None, code: str = None):
    """
    执行特殊通道条件检查，返回行列表。
    """
    n = len(df)
    if n < 67:
        return None

    if signal_date:
        date_map = {str(d)[:10]: idx for idx, d in enumerate(df["date"].values)}
        if signal_date not in date_map:
            return None
        i = date_map[signal_date]
    else:
        i = n - 1

    if i < 3:
        return None

    close = df["close"].values.astype(float)
    # ── 均线（优先用预计算值）─────
    has_pre = ("_ma5" in df.columns and "_ma10" in df.columns
              and "_ma20" in df.columns and "_ma60" in df.columns)
    if has_pre:
        ma5  = df["_ma5"].values.astype(float)
        ma10 = df["_ma10"].values.astype(float)
        ma20 = df["_ma20"].values.astype(float)
        ma60 = df["_ma60"].values.astype(float)
    else:
        ma5  = rolling_mean(close, 5).astype(float)
        ma10 = rolling_mean(close, 10).astype(float)
        ma20 = rolling_mean(close, 20).astype(float)
        ma60 = rolling_mean(close, 60).astype(float)

    if np.isnan(ma5[i]) or np.isnan(ma10[i]) or np.isnan(ma20[i]) or np.isnan(ma60[i]):
        return None

    lines = []
    lines.append(f"【特殊通道 — 信号日 {str(df.iloc[i]['date'])[:10]}】")

    # ── 涨停判断 ──
    code_check = (code or "").lower()
    if code_check.startswith(("sz300", "sh688", "300", "688")):
        limit_ratio = 0.20
    else:
        limit_ratio = 0.10
    is_limitup, limit_up_price = _cond_lim_涨停(i, close, limit_ratio)
    if not is_limitup:
        return None

    recent_20_high = float(np.max(close[i - 19:i + 1])) if i >= 19 else float(np.max(close[:i + 1]))
    is_20high = _cond_lim_20high(i, close, recent_20_high)
    lines.append(f"  ① 涨停 {is_limitup}（涨停价={limit_up_price:.2f}  20日最高={recent_20_high:.2f}） → {GREEN_CHECK if is_limitup else RED_CROSS}")
    lines.append(f"  ①' 20日最高 {is_20high}（close={close[i]:.2f} ≥ {recent_20_high*0.9999:.2f}） → {GREEN_CHECK if is_20high else RED_CROSS}")

    # ── 均线多头 ──
    ok_all = _cond_lim_均线多头(i, ma5, ma10, ma20, ma60, close)
    lines.append(f"  ② close>{ma5[i]:.2f}>{ma10[i]:.2f}>{ma20[i]:.2f} 且 >{ma60[i]:.2f} → {GREEN_CHECK if ok_all else RED_CROSS}")

    # ── ATR ──
    has_pre_atr = "_atr_pct" in df.columns
    atr_pct = df["_atr_pct"].values.astype(float) if has_pre_atr else calc_atr_percent(df, 14)
    if np.isnan(atr_pct[i]) or atr_pct[i] < 0.1:
        return None

    # ── MA斜率（实时算每个索引，保证准确性）─────
    ma5_slope_atr  = np.array([_lr_slope(ma5,  idx) / (atr_pct[idx] + 1e-12) for idx in range(n)])
    ma10_slope_atr = np.array([_lr_slope(ma10, idx) / (atr_pct[idx] + 1e-12) for idx in range(n)])
    ma20_slope_atr = np.array([_lr_slope(ma20, idx) / (atr_pct[idx] + 1e-12) for idx in range(n)])

    ma5_daily_i   = (ma5[i] / ma5[i - 1] - 1) * 100 if i >= 1 else -999
    ok_ma5_slope, ok_ma5_daily = _cond_lim_ma5_accel(i, ma5_slope_atr, ma5_daily_i)
    ok_ma10_slope = _cond_lim_ma10_slope(i, ma10_slope_atr)
    ok_ma20_slope = _cond_lim_ma20_slope(i, ma20_slope_atr)
    lines.append(f"  ③ ma5_slope_atr={ma5_slope_atr[i]:.3f} > {LIMITUP_I_SLOPE_MA5} → {GREEN_CHECK if ok_ma5_slope else RED_CROSS}")
    lines.append(f"     ma5_daily={ma5_daily_i:.3f}% > {LIMITUP_I_DAILY_MA5}% → {GREEN_CHECK if ok_ma5_daily else RED_CROSS}")
    lines.append(f"     ma10_slope_atr={ma10_slope_atr[i]:.3f} < {LIMITUP_I_SLOPE_MA10} → {GREEN_CHECK if ok_ma10_slope else RED_CROSS}")
    lines.append(f"     ma20_slope_atr={ma20_slope_atr[i]:.3f} < {LIMITUP_I_SLOPE_MA20} → {GREEN_CHECK if ok_ma20_slope else RED_CROSS}")

    ok_turnover = _cond_lim_量比(i, df)
    lines.append(f"     当日放量（量比≥{LIMITUP_I_VOLUME}） → {GREEN_CHECK if ok_turnover else RED_CROSS}")

    # ── i-1 日条件 ──
    ma5_daily_i1 = (ma5[i - 1] / ma5[i - 2] - 1) * 100 if i - 1 >= 1 else -999
    ok_gain1, ok_slope1, ok_daily1, gain_i1 = _cond_lim_i1(i, close, ma5_slope_atr, ma5_daily_i1, ma5)
    lines.append(f"【前1天 {str(df.iloc[i-1]['date'])[:10]}】")
    lines.append(f"  涨幅={gain_i1:.2f}% < {LIMITUP_GAIN_DAY_MAX:.2f}% → {GREEN_CHECK if ok_gain1 else RED_CROSS}")
    lines.append(f"  ma5_slope_atr={ma5_slope_atr[i-1]:.3f} < {LIMITUP_I1_SLOPE_MA5_U} → {GREEN_CHECK if ok_slope1 else RED_CROSS}")
    lines.append(f"  ma5_daily={ma5_daily_i1:.3f}% < {LIMITUP_I1_DAILY_U}% → {GREEN_CHECK if ok_daily1 else RED_CROSS}")

    # ── i-2 日条件 ──
    ma5_daily_i2 = (ma5[i - 2] / ma5[i - 3] - 1) * 100 if i - 2 >= 1 else -999
    ok_gain2, ok_slope2, ok_daily2, gain_i2 = _cond_lim_i2(i, close, ma5_slope_atr, ma5_daily_i2, ma5)
    lines.append(f"【前2天 {str(df.iloc[i-2]['date'])[:10]}】")
    lines.append(f"  涨幅={gain_i2:.2f}% < {LIMITUP_GAIN_DAY_MAX:.2f}% → {GREEN_CHECK if ok_gain2 else RED_CROSS}")
    lines.append(f"  ma5_slope_atr={ma5_slope_atr[i-2]:.3f} < {LIMITUP_I2_SLOPE_MA5_U} → {GREEN_CHECK if ok_slope2 else RED_CROSS}")
    lines.append(f"  ma5_daily={ma5_daily_i2:.3f}% < {LIMITUP_I2_DAILY_U}% → {GREEN_CHECK if ok_daily2 else RED_CROSS}")

    passed = (is_limitup and is_20high and ok_all and ok_ma5_slope and ok_ma5_daily
              and ok_ma10_slope and ok_ma20_slope and ok_turnover
              and ok_gain1 and ok_slope1 and ok_daily1
              and ok_gain2 and ok_slope2 and ok_daily2)
    lines.append(f"\n  最终：{'✓ 通过特殊通道' if passed else '✗ 淘汰'}")
    return lines


def _check_ma_conditions(df: pd.DataFrame, signal_date: str = None):
    """
    单票模式正常通道的逐级条件核对。
    先调用 check_ma；结果为 None 时，用本函数重新检查找出失败点。
    与 check_ma 逻辑完全一致（共用 _cond* 函数），返回行列表。
    """
    n = len(df)
    if n < 67:
        return None

    if signal_date:
        date_map = {str(d)[:10]: idx for idx, d in enumerate(df["date"].values)}
        if signal_date not in date_map:
            return None
        i = date_map[signal_date]
    else:
        i = n - 1

    close    = df["close"].values.astype(float)
    turnover = df["true_turnover"].values.astype(float)
    outs     = df["outstanding_share"].values.astype(float) if "outstanding_share" in df.columns else None

    # ── 均线 ──
    has_pre = ("_ma5" in df.columns and "_ma10" in df.columns
              and "_ma20" in df.columns and "_ma60" in df.columns)
    if has_pre:
        ma5  = df["_ma5"].values.astype(float)
        ma10 = df["_ma10"].values.astype(float)
        ma20 = df["_ma20"].values.astype(float)
        ma60 = df["_ma60"].values.astype(float)
    else:
        ma5  = rolling_mean(close, 5).astype(float)
        ma10 = rolling_mean(close, 10).astype(float)
        ma20 = rolling_mean(close, 20).astype(float)
        ma60 = rolling_mean(close, 60).astype(float)

    if np.isnan(ma5[i]) or np.isnan(ma10[i]) or np.isnan(ma20[i]) or np.isnan(ma60[i]):
        return None

    # ── MA斜率 ──
    has_pre_slope = "_ma5_slope_atr" in df.columns
    if has_pre_slope:
        ma5_slope_atr  = df["_ma5_slope_atr"].values.astype(float)
        ma10_slope_atr = df["_ma10_slope_atr"].values.astype(float)
        ma20_slope_atr = df["_ma20_slope_atr"].values.astype(float)
    else:
        has_pre_atr = "_atr_pct" in df.columns
        atr_pct = df["_atr_pct"].values.astype(float) if has_pre_atr else calc_atr_percent(df, 14)
        ma5_slope_atr  = np.array([_lr_slope(ma5,  idx) / (atr_pct[idx] + 1e-12) for idx in range(n)])
        ma10_slope_atr = np.array([_lr_slope(ma10, idx) / (atr_pct[idx] + 1e-12) for idx in range(n)])
        ma20_slope_atr = np.array([_lr_slope(ma20, idx) / (atr_pct[idx] + 1e-12) for idx in range(n)])

    # ── 派生指标 ──
    ma5_d  = (ma5[i]/ma5[i-1]-1)*100 if i>=1 else -999
    ma10_d = (ma10[i]/ma10[i-1]-1)*100 if i>=1 else -999
    ma20_d = (ma20[i]/ma20[i-1]-1)*100 if i>=1 else -999
    gain_day = (close[i]/close[i-1]-1)*100 if i>=1 else float('nan')
    turnover_avg = np.mean(turnover[i-4:i+1]) if i >= 4 else -999.0
    turnover_now  = float(turnover[i])
    turnover_prev = float(turnover[i-1]) if i >= 1 else 0.0
    recent_20_high = float(np.max(close[i-19:i+1])) if i >= 19 else float(np.max(close[:i+1]))
    mc = (close[i]*outs[i]/1e8) if (outs is not None and not np.isnan(outs[i]) and outs[i]>0) else 0.0

    lines = []
    ok1, msg1 = _cond1_ma_line(i, ma5, ma10, ma20, ma60, close)
    lines.append(f"① {msg1} → {GREEN_CHECK if ok1 else RED_CROSS}")

    ok2a, msg2a = _cond2a_slope_atr(i, ma5_slope_atr, ma10_slope_atr, ma20_slope_atr)
    ok2a_2, msg2a_2 = _cond2a_2_slope_atr(i, ma5_slope_atr, ma10_slope_atr, ma20_slope_atr)
    ok2a_3, msg2a_3 = _cond2a_3_slope_atr(i, ma5_slope_atr, ma10_slope_atr, ma20_slope_atr)
    ok2b, msg2b = _cond2b_ma_daily(i, ma5_d, ma10_d, ma20_d, close)
    ok2b_2, msg2b_2 = _cond2b_2_ma_daily(i, ma5_d, ma10_d, ma20_d, close)
    ok2b_3, msg2b_3 = _cond2b_3_ma_daily(i, ma5_d, ma10_d, ma20_d, close)

    if ok2a and ok2b:
        lines.append(f"②-1_1 {msg2a} → {GREEN_CHECK if ok2a else RED_CROSS}")
        lines.append(f"②-2_1 {msg2b} → {GREEN_CHECK if ok2b else RED_CROSS}")
    else:
        if ok2a_2 and ok2b_2:
            lines.append(f"②-1_2 {msg2a_2} → {GREEN_CHECK if ok2a_2 else RED_CROSS}")
            lines.append(f"②-2_2 {msg2b_2} → {GREEN_CHECK if ok2b_2 else RED_CROSS}")
        else:
            if ok2a_3 and ok2b_3:
                lines.append(f"②-1_3 {msg2a_3} → {GREEN_CHECK if ok2a_3 else RED_CROSS}")
                lines.append(f"②-2_3 {msg2b_3} → {GREEN_CHECK if ok2b_3 else RED_CROSS}")
            else:
                lines.append(f"②-1_1 {msg2a} → {GREEN_CHECK if ok2a else RED_CROSS}")
                lines.append(f"②-2_1 {msg2b} → {GREEN_CHECK if ok2b else RED_CROSS}")
                lines.append(f"②-1_2 {msg2a_2} → {GREEN_CHECK if ok2a_2 else RED_CROSS}")
                lines.append(f"②-2_2 {msg2b_2} → {GREEN_CHECK if ok2b_2 else RED_CROSS}")
                lines.append(f"②-1_3 {msg2a_3} → {GREEN_CHECK if ok2a_3 else RED_CROSS}")
                lines.append(f"②-2_3 {msg2b_3} → {GREEN_CHECK if ok2b_3 else RED_CROSS}")

    ok3, msg3 = _cond3_mktcap(i, mc)
    lines.append(f"③ {msg3} → {GREEN_CHECK if ok3 else RED_CROSS}")

    ok4, msg4 = _cond4_turnover(i, turnover_avg, turnover_now, turnover_prev, gain_day, mc)
    lines.append(f"④ {msg4} → {GREEN_CHECK if ok4 else RED_CROSS}")

    ok6, msg6 = _cond6_gain_day(i, gain_day)
    lines.append(f"⑤ {msg6} → {GREEN_CHECK if ok6 else RED_CROSS}")

    ok7, msg7 = _cond7_close_near_high(i, close, recent_20_high)
    lines.append(f"⑥ {msg7} → {GREEN_CHECK if ok7 else RED_CROSS}")

    lines.append(f"\n  最终：{'✓ 通过全部条件' if (ok1 and ((ok2a and ok2b) or (ok2a_2 and ok2b_2)) and ok3 and ok4 and ok6 and ok7) else '✗ 淘汰'}")
    return lines


# ══════════════════════════════════════════════════════════════
# 核心检测
# ══════════════════════════════════════════════════════════════

def check_ma(df: pd.DataFrame,
              signal_date: str = None,
              data_mode: str = "raw",
             ) -> dict | None:

    n = len(df)
    if n < 65:
        return None

    # ── 确定分析索引 ──────────────────────────────
    if signal_date:
        date_map = {str(d)[:10]: idx for idx, d in enumerate(df["date"].values)}
        if signal_date not in date_map:
            return None
        i = date_map[signal_date]
    else:
        i = n - 1   # 默认最新日期

    if i < 64:
        return None

    close    = df["close"].values.astype(float)
    turnover = df["true_turnover"].values.astype(float)

    outs     = df["outstanding_share"].values.astype(float) if "outstanding_share" in df.columns else None

    # ── 计算均线（优先用预计算值，否则实时算）─────
    has_precomputed = ("_ma5" in df.columns and "_ma10" in df.columns
                     and "_ma20" in df.columns and "_ma60" in df.columns)
    if has_precomputed:
        ma5  = df["_ma5"].values.astype(float)
        ma10 = df["_ma10"].values.astype(float)
        ma20 = df["_ma20"].values.astype(float)
        ma60 = df["_ma60"].values.astype(float)
    else:
        ma5  = rolling_mean(close, 5).astype(float)
        ma10 = rolling_mean(close, 10).astype(float)
        ma20 = rolling_mean(close, MA_LEN).astype(float)
        ma60 = rolling_mean(close, MA_LEN_60).astype(float)

    if np.isnan(ma5[i]) or np.isnan(ma10[i]) or np.isnan(ma20[i]) or np.isnan(ma60[i]):
        return None

    # ── 斜率（优先用预计算值，否则实时算每个索引）─────
    has_pre_slope = ("_ma5_slope_atr" in df.columns and "_ma10_slope_atr" in df.columns
                   and "_ma20_slope_atr" in df.columns)
    if has_pre_slope:
        ma5_slope_atr  = df["_ma5_slope_atr"].values.astype(float)
        ma10_slope_atr = df["_ma10_slope_atr"].values.astype(float)
        ma20_slope_atr = df["_ma20_slope_atr"].values.astype(float)
    else:
        has_pre_atr = "_atr_pct" in df.columns
        atr_pct = df["_atr_pct"].values.astype(float) if has_pre_atr else calc_atr_percent(df, 14)
        ma5_slope_atr  = np.array([_lr_slope(ma5,  idx) / (atr_pct[idx] + 1e-12) for idx in range(n)])
        ma10_slope_atr = np.array([_lr_slope(ma10, idx) / (atr_pct[idx] + 1e-12) for idx in range(n)])
        ma20_slope_atr = np.array([_lr_slope(ma20, idx) / (atr_pct[idx] + 1e-12) for idx in range(n)])

    # ── 派生指标（所有条件函数共用）─────────────────────
    ma5_daily  = (ma5[i]  / ma5[i-1]  - 1) * 100 if i >= 1 else -999
    ma10_daily = (ma10[i] / ma10[i-1] - 1) * 100 if i >= 1 else -999
    ma20_daily = (ma20[i] / ma20[i-1] - 1) * 100 if i >= 1 else -999
    gain_day   = (close[i] / close[i-1] - 1) * 100.0
    gain5      = (close[i] / close[i-5] - 1) * 100.0
    gain20     = (close[i] / close[i-20] - 1) * 100.0 if i >= 20 else float('nan')
    turnover_avg = np.mean(turnover[i - TURNOVER_LEN + 1 : i + 1])
    turnover_now  = float(turnover[i])
    turnover_prev = float(turnover[i-1]) if i >= 1 else 0.0
    recent_20_high = float(np.max(close[i - 19:i + 1])) if i >= 19 else float(np.max(close[:i+1]))

    # ── 条件① ────────────────────────────────────────────
    ok1, _ = _cond1_ma_line(i, ma5, ma10, ma20, ma60, close)
    if not ok1:
        return None

    # ── 条件②-1 斜率/ATR ──────────────────────────────────
    ok2a, _ = _cond2a_slope_atr(i, ma5_slope_atr, ma10_slope_atr, ma20_slope_atr)
    ok2a_2, _ = _cond2a_2_slope_atr(i, ma5_slope_atr, ma10_slope_atr, ma20_slope_atr)
    ok2a_3, _ = _cond2a_3_slope_atr(i, ma5_slope_atr, ma10_slope_atr, ma20_slope_atr)
    if not (ok2a or ok2a_2 or ok2a_3):
        return None

    # ── 条件②-2 日变 ─────────────────────────────────────
    # ── 条件⑤⑧ 5日 20日涨幅 ────────────────────────────────────
    ok2b, _ = _cond2b_ma_daily(i, ma5_daily, ma10_daily, ma20_daily, close)
    ok2b_2, _ = _cond2b_2_ma_daily(i, ma5_daily, ma10_daily, ma20_daily, close)
    ok2b_3, _ = _cond2b_3_ma_daily(i, ma5_daily, ma10_daily, ma20_daily, close)
    if not (ok2a and ok2b):
        if not (ok2a_2 and ok2b_2):
            if not (ok2a_3 and ok2b_3):
                return None

    # ── 条件③ 市值 ──────────────────────────────────────
    if outs is None or np.isnan(outs[i]) or outs[i] <= 0:
        mktcap = MKT_CAP_MIN
    else:
        mktcap = close[i] * outs[i] / 1e8
    ok3, _ = _cond3_mktcap(i, mktcap)
    if not ok3:
        return None

    # ── 条件④ 换手率 ─────────────────────────────────────
    ok4, _ = _cond4_turnover(i, turnover_avg, turnover_now, turnover_prev, gain_day, mktcap)
    if not ok4:
        return None

    # ── 条件⑤ 当日涨幅 ───────────────────────────────────
    ok6, _ = _cond6_gain_day(i, gain_day)
    if not ok6:
        return None

    # ── 条件⑥ 收盘近20日最高 ──────────────────────────────
    ok7, _ = _cond7_close_near_high(i, close, recent_20_high)
    if not ok7:
        return None

    # ── 计算辅助指标（用于展示）──────────────
    ma5_rise   = ma5_daily  >= MA5_DAILY
    ma10_rise  = ma10_daily >= MA10_DAILY
    ma20_rise  = ma20_daily >= MA20_DAILY
    ma5_chg5d  = (ma5[i]  - ma5[i-5])  / ma5[i-5]  * 100.0 if i >= 5  else 0.0
    ma10_chg5d = (ma10[i] - ma10[i-5]) / ma10[i-5] * 100.0 if i >= 5  else 0.0
    ma20_chg5d = (ma20[i] - ma20[i-5]) / ma20[i-5] * 100.0 if i >= 5  else 0.0
    turnover_today = turnover[i]
    vol5_avg  = np.mean(df["volume"].values[i-4:i+1])
    vol5_prev = np.mean(df["volume"].values[i-9:i-4]) if i >= 10 else np.mean(df["volume"].values[max(0,i-9):i])
    vol_ratio = (vol5_avg / vol5_prev) if vol5_prev > 0 else 0.0
    spread = (ma5[i]/ma10[i] - 1)*100 + (ma10[i]/ma20[i] - 1)*100
    return {
        "date":          str(df["date"].values[i])[:10],
        "close":         round(close[i], 2),
        "ma5":           round(ma5[i], 2),
        "ma10":          round(ma10[i], 2),
        "ma20":          round(ma20[i], 2),
        "ma5_rise":      ma5_rise,
        "ma10_rise":     ma10_rise,
        "ma20_rise":     ma20_rise,
        "ma5_daily":     round(ma5_daily, 3),
        "ma10_daily":    round(ma10_daily, 3),
        "ma20_daily":    round(ma20_daily, 3),
        "ma5_slope_atr":     round(ma5_slope_atr[i], 3),
        "ma10_slope_atr":    round(ma10_slope_atr[i], 3),
        "ma20_slope_atr":    round(ma20_slope_atr[i], 3),
        "ma5_chg5d":     round(ma5_chg5d, 2),
        "ma10_chg5d":    round(ma10_chg5d, 2),
        "ma20_chg5d":    round(ma20_chg5d, 2),
        "gain5":         round(gain5, 1),
        "gain20":        round(gain20, 1),
        "turnover_avg":  round(turnover_avg, 2),
        "turnover_today":round(turnover_today, 2),
        "vol_ratio":     round(vol_ratio, 2),
        "spread":        round(spread, 2),
        "mktcap":        round(mktcap, 0),
        "ok1": ok1,
        "ok2a": ok2a,
        "ok2b": ok2b,
        "ok2a_2": ok2a_2,
        "ok2b_2": ok2b_2,
        "ok2a_3": ok2a_3,
        "ok2b_3": ok2b_3,
    }

# 全局log_file用于scan()中的进度输出（终端+文件双写）
_log_file = None

# ══════════════════════════════════════════════════════════════
# 批量扫描
# ══════════════════════════════════════════════════════════════

def check_limitup_channel(df: pd.DataFrame, signal_date: str = None, code: str = None) -> dict | None:
    """特殊通道：涨停蓄势后突破（使用原始不复权数据）"""
    n = len(df)
    if n < 67:   # 需要 i-2 可用，最少要3天空闲
        return None

    if signal_date:
        date_map = {str(d)[:10]: idx for idx, d in enumerate(df["date"].values)}
        if signal_date not in date_map:
            return None
        i = date_map[signal_date]
    else:
        i = n - 1

    if i < 3:
        return None

    close    = df["close"].values.astype(float)
    turnover = df["true_turnover"].values.astype(float)
    outs     = df["outstanding_share"].values.astype(float) if "outstanding_share" in df.columns else None

    # ── 均线（优先用预计算值）─────
    has_precomputed = ("_ma5" in df.columns and "_ma10" in df.columns
                     and "_ma20" in df.columns and "_ma60" in df.columns)
    if has_precomputed:
        ma5  = df["_ma5"].values.astype(float)
        ma10 = df["_ma10"].values.astype(float)
        ma20 = df["_ma20"].values.astype(float)
        ma60 = df["_ma60"].values.astype(float)
    else:
        ma5  = rolling_mean(close, 5).astype(float)
        ma10 = rolling_mean(close, 10).astype(float)
        ma20 = rolling_mean(close, 20).astype(float)
        ma60 = rolling_mean(close, 60).astype(float)

    if np.isnan(ma5[i]) or np.isnan(ma10[i]) or np.isnan(ma20[i]) or np.isnan(ma60[i]):
        return None

    # ── 斜率（优先用预计算值，否则实时算每个索引）─────
    has_pre_slope = ("_ma5_slope_atr" in df.columns and "_ma10_slope_atr" in df.columns
                   and "_ma20_slope_atr" in df.columns)
    if has_pre_slope:
        ma5_slope_atr  = df["_ma5_slope_atr"].values.astype(float)
        ma10_slope_atr = df["_ma10_slope_atr"].values.astype(float)
        ma20_slope_atr = df["_ma20_slope_atr"].values.astype(float)
    else:
        has_pre_atr = "_atr_pct" in df.columns
        atr_pct = df["_atr_pct"].values.astype(float) if has_pre_atr else calc_atr_percent(df, 14)
        ma5_slope_atr  = np.array([_lr_slope(ma5,  idx) / (atr_pct[idx] + 1e-12) for idx in range(n)])
        ma10_slope_atr = np.array([_lr_slope(ma10, idx) / (atr_pct[idx] + 1e-12) for idx in range(n)])
        ma20_slope_atr = np.array([_lr_slope(ma20, idx) / (atr_pct[idx] + 1e-12) for idx in range(n)])

    # ── i日 ──────────────────────────────────────────────
    # 根据代码前缀判断板块
    # code 可能是纯码(如300263)或带前缀(如sz300263)，统一取纯码判断
    code_check = code.lower() if code else ""
    if code_check.startswith(("sz300", "sh688", "300", "688")):
        limit_ratio = 0.20   # 创业板 / 科创板
    else:
        limit_ratio = 0.10   # 主板 / 北交所

    is_limitup, limit_up_price = _cond_lim_涨停(i, close, limit_ratio)
    if not is_limitup:
        return None

    recent_20_high = float(np.max(close[i - 19:i + 1])) if i >= 19 else float(np.max(close[:i + 1]))
    is_20high = _cond_lim_20high(i, close, recent_20_high)
    if not is_20high:
        return None

    if not _cond_lim_均线多头(i, ma5, ma10, ma20, ma60, close):
        return None

    ma5_daily_i   = (ma5[i] / ma5[i-1] - 1) * 100 if i >= 1 else -999

    ok_ma5_slope, ok_ma5_daily = _cond_lim_ma5_accel(i, ma5_slope_atr, ma5_daily_i)
    if not ok_ma5_slope or not ok_ma5_daily:
        return None
    if not _cond_lim_ma10_slope(i, ma10_slope_atr):
        return None
    if not _cond_lim_ma20_slope(i, ma20_slope_atr):
        return None

    # ── 量比：当日成交量 / 前一日成交量 >= LIMITUP_I_VOLUME ─────────────────────
    if not _cond_lim_量比(i, df):
        return None

    # ── i-1日 ─────────────────────────────────────────────
    ma5_daily_i1 = (ma5[i-1] / ma5[i-2] - 1) * 100 if i-1 >= 1 else -999

    ok_gain1, ok_slope1, ok_daily1, gain_i1 = _cond_lim_i1(i, close, ma5_slope_atr, ma5_daily_i1, ma5)
    if not (ok_gain1 and ok_slope1 and ok_daily1):
        return None

    # ── i-2日 ─────────────────────────────────────────────
    ma5_daily_i2 = (ma5[i-2] / ma5[i-3] - 1) * 100 if i-2 >= 1 else -999

    ok_gain2, ok_slope2, ok_daily2, gain_i2 = _cond_lim_i2(i, close, ma5_slope_atr, ma5_daily_i2, ma5)
    if not (ok_gain2 and ok_slope2 and ok_daily2):
        return None

    # ── 市值 & 换手（用于展示）─────────────────────────────
    mktcap = 0.0
    if outs is not None and not np.isnan(outs[i]) and outs[i] > 0:
        mktcap = close[i] * outs[i] / 1e8

    gain20 = (close[i] / close[i-20] - 1) * 100 if i >= 20 else float('nan')
    gain5  = (close[i] / close[i-5]  - 1) * 100 if i >= 5  else float('nan')
    turnover_avg = np.mean(turnover[i-4:i+1])
    vol5_avg  = np.mean(df["volume"].values[i-4:i+1])
    vol5_prev = np.mean(df["volume"].values[i-9:i-4]) if i >= 10 else np.mean(df["volume"].values[max(0,i-9):i])

    vol_ratio = (vol5_avg / vol5_prev) if vol5_prev > 0 else 0.0

    return {
        "date":           str(df["date"].values[i])[:10],
        "close":          round(close[i], 2),
        "ma5":            round(ma5[i], 2),
        "ma10":           round(ma10[i], 2),
        "ma20":           round(ma20[i], 2),
        "gain20":         round(gain20, 1),
        "gain5":          round(gain5, 1),
        "turnover_avg":   round(turnover_avg, 2),
        "turnover_today": round(float(turnover[i]), 2),
        "vol_ratio":      round(vol_ratio, 2),
        "mktcap":         round(mktcap, 0),
        # 特殊通道标记
        "limit_up_price": limit_up_price,
        "recent_20_high": recent_20_high,
        "_limitup": True,
        "ma5_slope_atr":      round(ma5_slope_atr[i], 3),
        "ma5_daily":      round(ma5_daily_i, 3),
        "ma10_slope_atr":     round(ma10_slope_atr[i], 3),
        "ma20_slope_atr":     round(ma20_slope_atr[i], 3),
        "gain_i1":        round(gain_i1, 2),
        "gain_i2":        round(gain_i2, 2),
        "ma5_slope_atr_i1":   round(ma5_slope_atr[i-1], 3),
        "ma5_daily_i1":   round(ma5_daily_i1, 3),
        "ma5_slope_atr_i2":   round(ma5_slope_atr[i-2], 3),
        "ma5_daily_i2":   round(ma5_daily_i2, 3),
    }



def scan(codes, names, top, signal_date=None, data_mode="raw"):
    global _log_file
    limitup_results = []
    normal_results = []
    for idx, code in enumerate(codes):
        try:
            df = _load_df(code, end_date=signal_date, data_mode=data_mode)
            name = names.get(code, "")
            # 排除ST
            if "ST" in name.upper() or "*ST" in name or "S*ST" in name:
                continue
            # 特殊通道先检
            r_lim = check_limitup_channel(df, signal_date=signal_date, code=code)
            if r_lim:
                r_lim["code"] = code
                r_lim["name"] = name
                limitup_results.append(r_lim)
                continue
            # 正常通道
            r = check_ma(df, signal_date=signal_date, data_mode=data_mode)
            if r:
                r["code"] = code
                r["name"] = name
                normal_results.append(r)
        except Exception:
            sys.__stdout__.write(f"[ERR] {code}: {e}\n")
            pass

        if (idx + 1) % 1000 == 0:
            msg = "  已扫描 %d 只 ... 特殊通道 %d 只 正常 %d 只\n" % (idx+1, len(limitup_results), len(normal_results))
            sys.__stdout__.write(msg)
            sys.__stdout__.flush()

    # 特殊通道排在最前面
    limitup_results.sort(key=lambda x: -x["gain20"])
    normal_results.sort(key=lambda x: -x["gain20"])
    total_lim = len(limitup_results)
    total_nor = len(normal_results)
    msg = "\n  [OK] 扫描完毕  特殊通道 %d 只  正常通道 %d 只（各取前 %d 只）\n\n" % (total_lim, total_nor, top)
    sys.__stdout__.write(msg)
    sys.__stdout__.flush()

    return limitup_results[:top] + normal_results[:top]


def banner():
    print(f"""
+======================================================================+
|                    条件选股器
+======================================================================+
|  ★ 特殊通道：涨停蓄势后突破（排在最前，🔴标记）
|     信号日涨停=20日最高，均线多头，ma5_slope>{LIMITUP_I_SLOPE_MA5 }%，ma5_daily>{LIMITUP_I_DAILY_MA5}%
|     前1天涨幅<6%，ma5_slope∈(0,{LIMITUP_I1_SLOPE_MA5_U}%)，ma5_daily∈(0,{LIMITUP_I1_DAILY_U}%)
|     前2天涨幅<2%，ma5_slope<{LIMITUP_I2_SLOPE_MA5_U}%，ma5_daily<{LIMITUP_I2_DAILY_U}%
+======================================================================+
|  [1] MA5>MA10>MA20 且 MA5>MA60, MA10>MA60 (多头排列)
|  [2-1_1] 均线斜率/ATR波动率 {SLOPE_MA5_ATR_MAX:.2f}>ma5_atr>={SLOPE_MA5_ATR:.2f} {SLOPE_MA10_ATR_MAX:.2f}>ma10_atr>={SLOPE_MA10_ATR:.2f} {SLOPE_MA20_ATR_MAX:.2f}>ma20_atr>={SLOPE_MA20_ATR:.2f}
|  [2-2_1] MA5日变>={MA5_DAILY:.2f}% and MA10日变>={MA10_DAILY:.2f}% and MA20日变>={MA20_DAILY:.2f}%
|          5日涨幅 {GAIN5_MIN:.1f}%~{GAIN5_MAX:.1f}%  20日涨幅 {GAIN20_MIN:.1f}%~{GAIN20_MAX:.1f}%
|  [2-1_2] 均线斜率/ATR波动率 {SLOPE_MA5_ATR_MAX2:.2f}>ma5_atr>={SLOPE_MA5_ATR2:.2f} {SLOPE_MA10_ATR_MAX2:.2f}>ma10_atr>={SLOPE_MA10_ATR2:.2f} {SLOPE_MA20_ATR_MAX2:.2f}>ma20_atr>={SLOPE_MA20_ATR2:.2f}
|  [2-2_2] MA5日变>={MA5_DAILY2:.2f}% and MA10日变>={MA10_DAILY2:.2f}% and MA20日变>={MA20_DAILY2:.2f}%
|          5日涨幅 {GAIN5_MIN2:.1f}%~{GAIN5_MAX2:.1f}%  20日涨幅 {GAIN20_MIN2:.1f}%~{GAIN20_MAX2:.1f}%
|  [2-1_3] 均线斜率/ATR波动率 {SLOPE_MA5_ATR_MAX3:.2f}>ma5_atr>={SLOPE_MA5_ATR3:.2f} {SLOPE_MA10_ATR_MAX3:.2f}>ma10_atr>={SLOPE_MA10_ATR3:.2f} {SLOPE_MA20_ATR_MAX3:.2f}>ma20_atr>={SLOPE_MA20_ATR3:.2f}
|  [2-2_3] MA5日变>={MA5_DAILY3:.2f}% and MA10日变>={MA10_DAILY3:.2f}% and MA20日变>={MA20_DAILY3:.2f}%
|          5日涨幅 {GAIN5_MIN3:.1f}%~{GAIN5_MAX3:.1f}%  20日涨幅 {GAIN20_MIN3:.1f}%~{GAIN20_MAX3:.1f}%
|  [3] 流通市值 >= {MKT_CAP_MIN:.0f}亿
|  [4] 近5日平均换手率 >= 自适应（幂律，市值越大要求越低）或 当日涨幅>=3% 并且 当日换手率 >= {TURN_BASE:.1f}% 并且量比前一天放大
|  [5] 当日涨幅 {GAIN_DAY_MIN:.1f}%~{GAIN_DAY_MAX:.1f}%
|  [6] 收盘 ≥ 20日最高 × {CLOSE_NEAR_HIGH_RATIO}
+======================================================================+""")

def print_results(results, signal_date=None, weekday_cn=None):
    if not results:
        print("未找到符合条件的股票")
        return
    sep = "=" * 120
    date_label = f"{signal_date} {weekday_cn}" if signal_date else ""
    print(f"\n{sep}")
    print(f"  信号日: {date_label}  找到 {len(results)} 只")
    print(sep)
    print("%3s  %-8s  %-8s  %-10s  %-5s  %-6s  %-5s  %-5s %-5s %-7s %-7s %-7s %-6s" % ("#","代码","名称","日期","收盘","20日涨","均换手","换手今","量比", "MA5/ATR", "MA10/ATR", "MA20/ATR", "市值(亿)"))
    print("-" * 120)
    for rank, r in enumerate(results, 1):
        tag = " 🔴特殊" if r.get("_limitup") else ""
        print("%3d  %-8s  %-8s  %-10s  %7.2f  %+7.1f%%  %7.2f%%  %6.2f%%  %5.2fx   %5.2f    %5.2f    %5.2f  %5.0f%s" % (
            rank, r["code"], r["name"], r["date"], r["close"],
            r["gain20"], r["turnover_avg"], r["turnover_today"], r["vol_ratio"], r["ma5_slope_atr"], r["ma10_slope_atr"], r["ma20_slope_atr"], r["mktcap"], tag))
    print()
    print_detail(results[:10])


def print_detail(results: list):
    if not results:
        return
    print("=== 逐条核验 ===\n")
    for rank, r in enumerate(results, 1):
        tag = " 🔴特殊通道" if r.get("_limitup") else ""
        print(f"【{rank}. {r['code']} {r['name']}】  收盘 {r['close']}{tag}")

        if r.get("_limitup"):
            # ── 特殊通道详情 ──
            print(f"   信号日(i)  close={r['close']} 为20日最高，ma5_slope_atr={r['ma5_slope_atr']:+.3f}% ma5_daily={r['ma5_daily']:+.3f}% ma10_slope_atr={r['ma10_slope_atr']:+.3f}% ma20_slope_atr={r['ma20_slope_atr']:+.3f}%")
            print(f"   前1天(i-1)  涨幅={r['gain_i1']:+.2f}%  ma5_slope_atr={r['ma5_slope_atr_i1']:+.3f}%  ma5_daily={r['ma5_daily_i1']:+.3f}%")
            print(f"   前2天(i-2)  涨幅={r['gain_i2']:+.2f}%  ma5_slope_atr={r['ma5_slope_atr_i2']:+.3f}%  ma5_daily={r['ma5_daily_i2']:+.3f}%")
            print(f"   20日涨幅={r['gain20']:+.1f}%  5日涨幅={r['gain5']:+.1f}%  换手={r['turnover_today']:.2f}%  市值={r['mktcap']:.0f}亿")
            print()
            continue

        ok1 = r['ok1']
        print(f"   ① MA5>{r['ma5']:.2f} > MA10>{r['ma10']:.2f} > MA20>{r['ma20']:.2f}  →  {GREEN_CHECK if ok1 else RED_CROSS}")

        ok2a = r['ok2a']
        ok2b = r['ok2b']
        ok2a_2 = r['ok2a_2']
        ok2b_2 = r['ok2b_2']
        ok2a_3 = r['ok2a_3']
        ok2b_3 = r['ok2b_3']

        if ok2a and ok2b:
            print(f"   ②-1_1 MA5_atr={r['ma5_slope_atr']:.3f} [{SLOPE_MA5_ATR:.3f},{SLOPE_MA5_ATR_MAX:.3f})  MA10_atr={r['ma10_slope_atr']:.3f} [{SLOPE_MA10_ATR:.3f},{SLOPE_MA10_ATR_MAX:.3f})  MA20_atr={r['ma20_slope_atr']:.3f} [{SLOPE_MA20_ATR:.3f},{SLOPE_MA20_ATR_MAX:.3f})  →  {GREEN_CHECK if ok2a else RED_CROSS}")
            print(f"   ②-2_1 MA5_daily={r['ma5_daily']:+.3f}%>={MA5_DAILY:.3f}%  MA10_daily={r['ma10_daily']:+.3f}%>={MA10_DAILY:.3f}%  MA20_daily={r['ma20_daily']:+.3f}%>={MA20_DAILY:.3f}%  →  {GREEN_CHECK if ok2b else RED_CROSS}")
            print(f"         5日涨幅 {r['gain5']:+.1f}% (需{GAIN5_MIN}%~{GAIN5_MAX}%)  →  {GREEN_CHECK if GAIN5_MAX>=r['gain5']>=GAIN5_MIN else RED_CROSS}")
            print(f"         20日涨幅 {r['gain20']:+.1f}% (需{GAIN20_MIN}%~{GAIN20_MAX}%)  →  {GREEN_CHECK if GAIN20_MAX>=r['gain20']>=GAIN20_MIN else RED_CROSS}")
        if ok2a_2 and ok2b_2:
            print(f"   ②-1_2 MA5_atr={r['ma5_slope_atr']:.3f} [{SLOPE_MA5_ATR2:.3f},{SLOPE_MA5_ATR_MAX2:.3f})  MA10_atr={r['ma10_slope_atr']:.3f} [{SLOPE_MA10_ATR2:.3f},{SLOPE_MA10_ATR_MAX2:.3f})  MA20_atr={r['ma20_slope_atr']:.3f} [{SLOPE_MA20_ATR2:.3f},{SLOPE_MA20_ATR_MAX2:.3f})  →  {GREEN_CHECK if ok2a_2 else RED_CROSS}")
            print(f"   ②-2_2 MA5_daily={r['ma5_daily']:+.3f}%>={MA5_DAILY2:.3f}%  MA10_daily={r['ma10_daily']:+.3f}%>={MA10_DAILY2:.3f}%  MA20_daily={r['ma20_daily']:+.3f}%>={MA20_DAILY2:.3f}%  →  {GREEN_CHECK if ok2b_2 else RED_CROSS}")
            print(f"         5日涨幅 {r['gain5']:+.1f}% (需{GAIN5_MIN2}%~{GAIN5_MAX2}%)  →  {GREEN_CHECK if GAIN5_MAX2>=r['gain5']>=GAIN5_MIN2 else RED_CROSS}")
            print(f"         20日涨幅 {r['gain20']:+.1f}% (需{GAIN20_MIN2}%~{GAIN20_MAX2}%)  →  {GREEN_CHECK if GAIN20_MAX2>=r['gain20']>=GAIN20_MIN2 else RED_CROSS}")
        if ok2a_3 and ok2b_3:
            print(f"   ②-1_3 MA5_atr={r['ma5_slope_atr']:.3f} [{SLOPE_MA5_ATR3:.3f},{SLOPE_MA5_ATR_MAX3:.3f})  MA10_atr={r['ma10_slope_atr']:.3f} [{SLOPE_MA10_ATR3:.3f},{SLOPE_MA10_ATR_MAX3:.3f})  MA20_atr={r['ma20_slope_atr']:.3f} [{SLOPE_MA20_ATR3:.3f},{SLOPE_MA20_ATR_MAX3:.3f})  →  {GREEN_CHECK if ok2a_3 else RED_CROSS}")
            print(f"   ②-2_3 MA5_daily={r['ma5_daily']:+.3f}%>={MA5_DAILY3:.3f}%  MA10_daily={r['ma10_daily']:+.3f}%>={MA10_DAILY3:.3f}%  MA20_daily={r['ma20_daily']:+.3f}%>={MA20_DAILY3:.3f}%  →  {GREEN_CHECK if ok2b_3 else RED_CROSS}")
            print(f"         5日涨幅 {r['gain5']:+.1f}% (需{GAIN5_MIN3}%~{GAIN5_MAX3}%)  →  {GREEN_CHECK if GAIN5_MAX3>=r['gain5']>=GAIN5_MIN3 else RED_CROSS}")
            print(f"         20日涨幅 {r['gain20']:+.1f}% (需{GAIN20_MIN3}%~{GAIN20_MAX3}%)  →  {GREEN_CHECK if GAIN20_MAX3>=r['gain20']>=GAIN20_MIN3 else RED_CROSS}")
        mc_d = r.get('mktcap', 0)
        thr_d = get_turnover_min(mc_d)
        print(f"   ③ 近5日均换手率 {r['turnover_avg']:.2f}% (需≥{thr_d:.1f}%，市值{mc_d:.0f}亿)  →  {GREEN_CHECK if r['turnover_avg'] >= thr_d else RED_CROSS}")
        print(f"   MA5近5日变化 {r['ma5_chg5d']:+.2f}%  MA10近5日 {r['ma10_chg5d']:+.2f}%  MA20近5日 {r['ma20_chg5d']:+.2f}%")
        print(f"   均线发散度 {r['spread']:.2f}%")
        print()


# ══════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════
def main():
    global _log_file
    parser = argparse.ArgumentParser(
        description=f"条件选股器（同时满足以下全部条件）\n\n"
                    f"条件①：MA5>MA10>MA20 且 MA5>MA60, MA10>MA60（多头排列，均线在MA60上方）\n"
                    f"条件②：均线斜率/ATR波动率 + MA日变>{MA5_DAILY}%\n"
                    f"条件：   5日涨幅 {GAIN5_MIN:.0f}%~{GAIN5_MAX:.0f}%\n"
                    f"条件：   20日涨幅 {int(GAIN20_MIN)}%~{int(GAIN20_MAX)}%\n"
                    f"条件③：流通市值 >= {int(MKT_CAP_MIN)}亿\n"
                    f"条件④：近5日平均换手率 ≥ 自适应换手率（幂律，市值200亿→4%）\n"
                    f"条件⑤：当日涨幅 {int(GAIN_DAY_MIN)}%~{int(GAIN_DAY_MAX)}%\n"
                    f"条件⑥：收盘 ≥ 20日最高 × {CLOSE_NEAR_HIGH_RATIO}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--top",          type=int,   default=300,      help="显示前N只（默认300）")
    parser.add_argument("--date",         type=str,   default=None,       help="信号日（格式YYYY-MM-DD，默认最新缓存日）")
    parser.add_argument("--output",     type=str,   default=None,       help="输出文件路径（默认 output/screen_trend_YYYY-MM-DD.txt）")
    parser.add_argument("--code",        type=str,   default=None,   help="单票分析")
    parser.add_argument("--qfq",         action="store_true", help="使用前复权数据（默认使用原始不复权数据）")
    parser.add_argument("--blk",         type=str, nargs='?',    const="/mnt/d/new_tdx/T0002/blocknew/RMG.blk",  default=None,  help="通达信板块文件路径（7位代码，第一位为市场标识：0/2=深圳，1=上海）")
    parser.add_argument("--start",       type=str, default=None, help="区间起始日（YYYY-MM-DD，配合--code单票模式使用）")
    parser.add_argument("--end",         type=str, default=None, help="区间结束日（YYYY-MM-DD，配合--code单票模式使用）")
    args = parser.parse_args()

    tag = args.date or "latest"
    default_output = str(Path("output") / ("screen_trend_" + tag + ".txt"))
    out_path = Path(args.output) if args.output else Path(default_output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    _orig = sys.stdout

    class _Dual:
        def __init__(self):
            pass
        def write(self, s):
            if s:
                _orig.write(s)
                if _log_file is not None:
                    _log_file.write(s)
        def flush(self):
            _orig.flush()
            if _log_file is not None:
                _log_file.flush()

    sys.stdout = _Dual()

    sys.__stdout__.write("加载股票名称 ... ")
    sys.__stdout__.flush()
    names = load_stock_names_akshare()
    sys.__stdout__.write("完成 (%d 只)\n\n" % len(names))
    sys.__stdout__.flush()

    weekday_cn = get_weekday_cn(args.date) if args.date else None

    # ── 单票区间扫描模式 ──
    if args.code and args.start and args.end:
        banner()
        import pandas as pd
        sys.__stdout__.write(f"=== 区间扫描: {args.code}  {args.start} ~ {args.end} ===\n")
        sys.__stdout__.flush()
        df = load_history(args.code, end_date=args.end, data_mode=args.qfq and "qfq" or "raw")
        if df is None or df.empty:
            sys.__stdout__.write(f"无法加载 {args.code} 数据\n")
            sys.__stdout__.flush()
            return
        df["date"] = pd.to_datetime(df["date"])
        df_sorted = df.sort_values("date").reset_index(drop=True)
        start_dt = pd.to_datetime(args.start)
        end_dt   = pd.to_datetime(args.end)
        mask = (df_sorted["date"] >= start_dt) & (df_sorted["date"] <= end_dt)
        range_dates = df_sorted.loc[mask, "date"].tolist()
        if not range_dates:
            sys.__stdout__.write(f"区间 {args.start}~{args.end} 内无数据\n")
            sys.__stdout__.flush()
            return
        sys.__stdout__.write(f"区间内共 {len(range_dates)} 个交易日\n\n")
        sys.__stdout__.flush()
        passed_list = []
        for dt in range_dates:
            sig = str(dt)[:10]
            weekday = get_weekday_cn(sig)
            reason_lim = _check_limitup_conditions(df_sorted, sig, args.code)
            if reason_lim is not None and reason_lim:
                sys.__stdout__.write(f"\n{'='*60}\n")
                sys.__stdout__.write(f"【{sig} {weekday}】 ✓ 通过特殊通道\n")
                sys.__stdout__.write(f"{'='*60}\n")
                for line in reason_lim:
                    sys.__stdout__.write("  " + line + "\n")
                sys.__stdout__.flush()
                passed_list.append(sig)
                continue
            reason = _check_ma_conditions(df_sorted, sig)
            if reason is None:
                reason = [f"  [{sig} {weekday}] 数据不足或日期不在范围"]
            passed = any("通过全部条件" in line and "✓" in line for line in reason)
            if passed:
                sys.__stdout__.write(f"\n{'='*60}\n")
                sys.__stdout__.write(f"【{sig} {weekday}】 ✓ 通过正常通道\n")
                sys.__stdout__.write(f"{'='*60}\n")
                for line in reason:
                    sys.__stdout__.write("  " + line + "\n")
                sys.__stdout__.flush()
                passed_list.append(sig)
            else:
                last_fail = [line for line in reason if "✗" in line]
                fail_line = last_fail[-1] if last_fail else reason[-1]
                sys.__stdout__.write(f"  [{sig} {weekday}] ✗  {fail_line.strip()}\n")
                sys.__stdout__.flush()
        sys.__stdout__.write(f"\n{'='*60}\n")
        sys.__stdout__.write(f"区间 {args.start}~{args.end} 扫描完毕：\n")
        sys.__stdout__.write(f"  通过 {len(passed_list)} 个交易日\n")
        if passed_list:
            sys.__stdout__.write(f"  通过日期：{', '.join(passed_list)}\n")
        sys.__stdout__.write("\n")
        sys.__stdout__.flush()
        sys.stdout.flush()
        sys.stdout = _orig
        sys.__stdout__.write("\n结果已写入: %s\n" % out_path)
        sys.__stdout__.flush()
        return

    if args.code:
        # ── 单票模式：输出逐级条件核对 ──
        banner()
        if weekday_cn is None and args.date:
            # 周末/节假日，直接打日期后退出
            sys.__stdout__.write(f"信号日 {args.date} 是周末/节假日，不进行选股分析。\n")
            sys.__stdout__.flush()
            sys.stdout.flush()
            sys.stdout = _orig
            sys.__stdout__.write("结果已写入: %s\n" % out_path)
            sys.__stdout__.flush()
            return

        sys.__stdout__.write(f"=== 单票分析: {args.date} {weekday_cn} ===\n")
        sys.__stdout__.flush()
        df = load_history(args.code, end_date=args.date, data_mode=args.qfq and "qfq" or "raw")
        # ──  特殊通道输出逐级核对 ── 
        reason_lim = _check_limitup_conditions(df, args.date, args.code)
        if reason_lim is not None:
            sys.__stdout__.write("\n  特殊通道条件核对：\n")
            for line in reason_lim:
                sys.__stdout__.write("  " + line + "\n")
            sys.__stdout__.flush()
        # ──  正常通道输出逐级条件核对 ──
        reason = _check_ma_conditions(df, args.date)
        sys.__stdout__.write("\n  逐级条件核对：\n")
        for line in reason:
            sys.__stdout__.write("  " + line + "\n")
        sys.__stdout__.flush()
        sys.stdout.flush()
        sys.stdout = _orig
        sys.__stdout__.write("\n结果已写入: %s\n" % out_path)
        sys.__stdout__.flush()
        return

    _log_file = open(out_path, "w", encoding="utf-8")

    banner()

    if weekday_cn is None and args.date:
        # 周末/节假日，直接打日期后退出
        sys.__stdout__.write(f"信号日 {args.date} 是周末/节假日，不进行选股分析。\n")
        sys.__stdout__.flush()
        sys.stdout.flush()
        sys.stdout = _orig
        _log_file.close()
        sys.__stdout__.write("结果已写入: %s\n" % out_path)
        sys.__stdout__.flush()
        return

    hint = f"（信号日 {args.date} {weekday_cn}）" if args.date else "（信号日 最新交易日）"
    sys.__stdout__.write(f"全市场扫描 {hint}（自适应幂律换手率）...\n")
    sys.__stdout__.flush()
    codes = get_all_stock_codes_akshare()
    if args.blk:
        blk_path = Path(args.blk)
        if not blk_path.exists():
            sys.__stdout__.write(f"文件不存在: {blk_path}\n")
            sys.__stdout__.flush()
            return
        blk_codes = []
        with open(blk_path) as f:
            for line in f:
                line = line.strip()
                if not line: continue
                if len(line) == 7 and line.isdigit():
                    p, num = line[0], line[1:]
                    if p == '0' or p == '2':
                        blk_codes.append('sz' + num)
                    elif p == '1':
                        blk_codes.append('sh' + num)
        codes = blk_codes
        sys.__stdout__.write(f"从板块文件加载 {len(codes)} 只: {blk_path.name}\n")
        sys.__stdout__.flush()
    sys.__stdout__.write("股票数量: %d\n\n" % len(codes))
    sys.__stdout__.flush()
    _log_file.write("股票数量: %d\n" % len(codes))
    _log_file.flush()

    data_mode = "qfq" if args.qfq else "raw"
    preload(args.date, data_mode=data_mode)
    results = scan(codes, names, top=args.top, signal_date=args.date, data_mode=data_mode)
    print_results(results, signal_date=args.date, weekday_cn=weekday_cn)

    sys.stdout.flush()
    sys.stdout = _orig
    _log_file.close()
    sys.__stdout__.write("\n结果已写入: %s\n" % out_path)
    sys.__stdout__.flush()

if __name__ == "__main__":
    main()
