#!/usr/bin/env python3
"""
screen_double_winner.py — WINNER + P2 固定路径强势股筛选器
=========================================================
硬编码规则（不可通过参数关闭）：
  模式：WINNER（条件3 = MA5>MA10>MA20>MA60 且四线方向向上 且收盘>MA5 且涨幅>2%）
  趋势过滤：P2
    · MA20方向向上（斜率>2%，10日前→今）
    · ret20 >= P2_RET20_MIN%
    · close < MA20 × P2_CLOSE_OVER_MA20（距MA20偏离由P2_CLOSE_OVER_MA20控制，默认1.20=20%）
    · 均线发散度：MA5>MA10>MA20 且 (MA5-MA10)/MA10>1% 且 (MA10-MA20)/MA20>1%

可调参数：
  --gain20      20日涨幅最低门槛（默认12%）
  --turnover    5日均换手率最低门槛（默认按市值规则）
  --no-hot      不区分热点板块

输出：120宽表格 + TXT/JSON存档
"""
from datetime import date

import sys
import json
import time
import argparse
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

WORKSPACE = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(WORKSPACE))
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

from stock_trend.gain_turnover import (
    normalize_symbol,
    load_stock_names,
    get_stock_name,
    compute_rsi_scalar as _gt_rsi_scalar,
)

_REAL_WORKSPACE = Path.home() / ".openclaw" / "workspace"
QFQ_DIR     = _REAL_WORKSPACE / ".cache" / "qfq_daily"
SECTOR_FILE = _REAL_WORKSPACE / ".cache" / "sector" / "sector_hotspot.json"

# ── 策略常量 ─────────────────────────────────────────────
L1_MIN_BARS = 65
L1_GAIN_STRICT  = 5.0
L1_GAIN_RELAXED = 2.0
L1_RELAX_MA5_DAYS = 3

# P2 偏离MA20阈值：收盘价 / MA20 必须小于此值
P2_CLOSE_OVER_MA20 = 1.15
P2_RET20_MIN        = 8.0   # P2：近20日涨幅最低门槛
P2_SPREAD_MIN_PCT  = 1.0    # P2：均线发散度最小间隔（%）
P2_MA20_AGO_BARS    = 10    # P2：MA20方向判断用N日前均线（斜率法）
P2_MA20_SLOPE_THRESH = 0.02  # P2：MA20斜率阈值（2% = 稳健向上）

# ── Layer1 共用阈值 ───────────────────────────────────
WINNER_GAIN_MIN       = 2.0    # WINNER当日涨幅最低门槛（%）
WINNER_GAIN_RELAX_C5  = 1.0    # WINNER宽松模式：5日涨幅门槛（%）
WINNER_GAIN_RELAX_C6  = 5.0    # WINNER宽松模式：近1日涨幅门槛（%）
WINNER_MA5_GT_MA20    = True   # WINNER严格模式：要求MA5>MA20
DROP_FILTER_MAX       = -3.0   # 当日跌幅过滤（涨幅<-3%则排除）

# ── 技术指标参数 ──────────────────────────────────────
MACD_FAST   = 12      # MACD 快线周期
MACD_SLOW   = 26      # MACD 慢线周期
MACD_SIGNAL = 9       # MACD 信号线周期
RSI_PERIOD  = 14      # RSI 周期
RSI_DEFAULT = 50.0    # RSI 数据不足时默认值

EMA_ALPHA_FACTOR      = 2.0    # EMA alpha = 2/(period+1) 的分子

# ── Layer2 评分常量 ───────────────────────────────────
# RSI 评分区间
L2_RSI_STRONG_MIN     = 55     # RSI 强区间下限
L2_RSI_STRONG_MAX     = 70     # RSI 强区间上限
L2_RSI_SECOND_MIN     = 48     # RSI 次强区间下限
L2_RSI_SECOND_MAX     = 75     # RSI 次强区间上限
L2_RSI_WEAK_MIN       = 40     # RSI 偏弱区间下限
L2_RSI_OVERBOUGHT     = 82     # RSI 超买阈值
L2_SCORE_RSI_STRONG   = 20     # RSI 强加分
L2_SCORE_RSI_SECOND   = 14     # RSI 次强加分
L2_SCORE_RSI_WEAK     = 8      # RSI 偏弱加分
L2_SCORE_SECTOR_HOT   = 5      # 热点板块加分

# 偏离MA20 距离区间
L2_DIST_HEALTHY_MAX   = 15     # 健康区上限（%）
L2_DIST_HIGH_MAX      = 25     # 略高区上限（%）
L2_SCORE_DIST_HEALTHY = 15     # 偏离MA20健康区加分
L2_SCORE_DIST_HIGH    = 8      # 偏离MA20略高加分
L2_SCORE_DIST_CLOSE   = 5      # 贴近MA20加分

# 换手率评分
L2_TURNOVER_HIGH      = 10     # 换手率高阈值（%）
L2_TURNOVER_MID       = 8      # 换手中阈值（%）
L2_TURNOVER_LOW       = 5      # 换手低阈值（%）
L2_SCORE_TURNOVER_HIGH = 20    # 高换手加分
L2_SCORE_TURNOVER_MID  = 10    # 中换手加分
L2_SCORE_TURNOVER_LOW  = 5     # 低换手加分

# 5日涨幅评分区间
L2_GAIN_STRONG_MIN     = 8      # 5日涨强区间下限（%）
L2_GAIN_STRONG_MAX     = 15     # 5日涨强区间上限（%）
L2_GAIN_SECOND_MIN     = 5      # 5日涨次强区间下限（%）
L2_GAIN_BIAS_STRONG_MAX = 25    # 5日涨偏强阈值（%）
L2_GAIN_HOT_MAX        = 35     # 5日涨偏热阈值（%）
L2_GAIN_COOL_MIN       = 2      # 5日涨偏冷区间下限（%）
L2_GAIN_COOL_MAX       = 5      # 5日涨偏冷区间上限（%）
L2_GAIN_ACCEL_MIN      = 1      # 5日涨ACCEL偏弱阈值（%）
L2_SCORE_GAIN5_STRONG  = 15     # 5日涨强加分
L2_SCORE_GAIN5_SECOND  = 9      # 5日涨次强加分
L2_SCORE_GAIN5_WARM    = 3      # 5日涨偏强/偏热/偏冷加分
L2_SCORE_GAIN5_ACCEL   = 1      # ACCEL偏弱加分

# 市值→换手率门槛（亿）
CAP_TURNOVER_500      = 500     # 市值门槛：≥500亿
CAP_TURNOVER_50       = 50      # 市值门槛：≥50亿
CAP_TURNOVER_30       = 30      # 市值门槛：≥30亿
CAP_TURNOVER_20       = 20      # 市值门槛：≥20亿
CAP_TURNOVER_DEF      = 10.0   # 默认换手率门槛（%）

# 均线方向阈值
MA_DIR_UP_THRESH      = 1.001  # 均线向上（5日前到当前涨幅>0.1%）
MA_DIR_DOWN_THRESH    = 0.999  # 均线向下（5日前到当前跌幅>0.1%）

RPS_POOL_NAME = "全市场有效股票池"
RPS_MAX_SCORE = 25.0
RPS_WEIGHTS   = (0.30, 0.30, 0.40)   # rps5, rps10, rps20

# ── 可配置参数（CLI覆盖）──────────────────────────────────
CLI_GAIN20_DEFAULT = 12.0   # --gain20 CLI默认值（%）
_GAIN20_MIN   = None   # 20日涨幅最低门槛（%），None=不设限
_TURNOVER_MIN = None   # 5日均换手率最低门槛（%），None=使用市值规则
_NO_HOT       = False  # True = 不区分热点板块

# ── 表格格式 ─────────────────────────────────────────────
def _vw(s):
    return sum(2 if unicodedata.east_asian_width(c) in ("W","F") else 1 for c in str(s))

def _pr(s, w):
    return str(s) + " " * max(0, w - _vw(s))

def _pl(s, w):
    return " " * max(0, w - _vw(s)) + str(s)

_COLS = [
    ("代码",   10, ">"),
    ("名称",    8, "<"),
    ("日期",   12, "<"),
    ("总分",    6, ">"),
    ("RPS综合",  7, ">"),
    ("收盘",    9, ">"),
    ("5日%",   7, ">"),
    ("换手%",  7, ">"),
    ("市值亿",  8, ">"),
    ("RSI",    6, ">"),
    ("MA5距%", 8, ">"),
    ("MA20",   8, ">"),
    ("MA60",   8, ">"),
    ("板块",   12, "<"),
    ("热点",    6, "<"),
]

def _make_row(values):
    parts = []
    for (label, w, align), v in zip(_COLS, values):
        parts.append(_pl(v, w) if align == ">" else _pr(v, w))
    return " ".join(parts)

def _header_row():
    return _make_row([l for l, _, _ in _COLS])


_price = {}

def preload():
    global _price
    print("📂 加载数据...", flush=True)
    for f in QFQ_DIR.glob("*_qfq.csv"):
        code = normalize_symbol(f.stem.replace("_qfq", ""))
        try:
            df = pd.read_csv(f)
            df = df.sort_values("date").reset_index(drop=True)
            _price[code] = df
        except:
            pass
    print(f"✅ {len(_price)}只已加载", flush=True)

# ── 热点板块 ─────────────────────────────────────────────
_hot_sectors = {}
_hot_sector_meta = {"status": "unknown", "mtime": None}

def _sector_cache_note():
    if _hot_sector_meta.get("mtime"):
        return f"热点板块缓存时间: {_hot_sector_meta['mtime']}"
    return "板块热点缓存缺失时，条件2按中性处理（不加分不扣分）"

def load_hot_sectors():
    global _hot_sectors, _hot_sector_meta
    if not _hot_sectors and SECTOR_FILE.exists():
        try:
            data = json.load(open(SECTOR_FILE))
            for name, chg in data.items():
                _hot_sectors[name] = chg
            _hot_sector_meta = {
                "status": "ok",
                "mtime": time.strftime("%Y-%m-%d %H:%M:%S",
                    time.localtime(SECTOR_FILE.stat().st_mtime)),
            }
        except Exception as e:
            _hot_sector_meta = {"status": "error", "mtime": None}
    return _hot_sectors

# ── 指标计算 ─────────────────────────────────────────────
def calc_ma(closes, period):
    if len(closes) < period: return None
    return float(np.mean(closes[-period:]))

def calc_ema(closes, period):
    n = len(closes)
    if n < period: return None
    alpha = EMA_ALPHA_FACTOR / (period + 1)
    ema = float(closes[0])
    for v in closes[1:]:
        ema = alpha * float(v) + (1 - alpha) * ema
    return ema

def calc_macd(closes, fast=12, slow=26, signal=9):
    """标准MACD：快线EMA - 慢线EMA = DIF，DIF的EMA = DEA。"""
    n = len(closes)
    if n < slow + signal: return None, None, None
    # 一次递推计算 EMA 系列（标准递归，O(n)）
    alpha_f = EMA_ALPHA_FACTOR / (fast + 1)
    alpha_s = EMA_ALPHA_FACTOR / (slow + 1)
    alpha_d = EMA_ALPHA_FACTOR / (signal + 1)
    ef_vals, es_vals = [float(closes[0])], [float(closes[0])]
    for i in range(1, n):
        ef_vals.append(alpha_f * float(closes[i]) + (1 - alpha_f) * ef_vals[-1])
        es_vals.append(alpha_s * float(closes[i]) + (1 - alpha_s) * es_vals[-1])
    dif_vals = [ef_vals[i] - es_vals[i] for i in range(n)]
    # DEA = EMA(DIF, signal) 标准递推
    dea = dif_vals[0]
    for i in range(1, n):
        dea = alpha_d * dif_vals[i] + (1 - alpha_d) * dea
    dif = dif_vals[-1]
    macd = (dif - dea) * 2
    return round(macd, 4), round(dif, 4), round(dea, 4)

def ma_direction(closes, period):
    """均线方向：5日前均线 vs 当前均线（>1%则向上，<-1%则向下，否则横盘）"""
    if len(closes) < period + 5: return 0
    now = calc_ma(closes, period)
    ago = calc_ma(closes[:-5], period)
    if now is None or ago is None: return 0
    return 1 if now > ago * MA_DIR_UP_THRESH else (-1 if now < ago * MA_DIR_DOWN_THRESH else 0)

def calc_rsi(closes, period=14):
    return _gt_rsi_scalar(np.asarray(closes), period)

# ── 板块 ─────────────────────────────────────────────────
_sector_map = {}

def load_sector_map():
    global _sector_map
    f = WORKSPACE / ".cache" / "sector" / "stock_sector_map.json"
    if f.exists():
        try:
            _sector_map = json.load(open(f))
        except:
            _sector_map = {}

def get_sector(code):
    return _sector_map.get(code, None)

def min_turnover_by_cap(market_cap):
    if market_cap is None: return CAP_TURNOVER_DEF
    if market_cap >= CAP_TURNOVER_500: return 1.0
    elif market_cap >= CAP_TURNOVER_50:  return 3.0
    elif market_cap >= CAP_TURNOVER_30:  return 5.0
    elif market_cap >= CAP_TURNOVER_20:  return 8.0
    else:                   return CAP_TURNOVER_DEF

# ── 诊断 ─────────────────────────────────────────────────
def debug_base(code, signal_date):
    """逐条件诊断 check_base，返回通过/失败详情列表。"""
    code = normalize_symbol(code)
    df = _price.get(code)
    results = []
    if df is None:
        results.append(("数据", False, "缓存中无此代码"))
        return results
    il = df["date"].tolist()
    try: idx = il.index(signal_date)
    except:
        results.append(("数据", False, f"日期 {signal_date} 不在K线中"))
        return results
    if idx < L1_MIN_BARS:
        results.append(("条件1 数据窗口", False, f"仅有 {idx+1} 根K线，需 >{L1_MIN_BARS}"))
        return results
    results.append(("条件1 数据窗口", True, f"{idx+1} 根K线 > {L1_MIN_BARS}"))

    window = df.iloc[idx - L1_MIN_BARS:idx + 1]
    closes = window["close"].values
    if "true_turnover" in window.columns and window["true_turnover"].notna().any():
        turnovers = window["true_turnover"].values
    elif "turnover" in window.columns:
        turnovers = window["turnover"].values
    else:
        turnovers = np.zeros(len(window))
    T = len(closes) - 1
    close_T = closes[T]

    ma5=calc_ma(closes,5); ma10=calc_ma(closes,10)
    ma20=calc_ma(closes,20); ma60=calc_ma(closes,60)
    if None in [ma5,ma10,ma20,ma60]:
        results.append(("条件2 均线位置", False, "无法计算均线"))
        return results

    d5=ma_direction(closes,5); d10=ma_direction(closes,10)
    d20=ma_direction(closes,20); d60=ma_direction(closes,60)
    gain_today = (close_T / closes[T-1] - 1) * 100 if T >= 1 else 0
    turnover_today = turnovers[T] if T < len(turnovers) else 0
    winner = (ma5>ma10>ma20>ma60 and d5==1 and d10==1 and d20==1 and d60==1 and close_T>ma5 and gain_today>WINNER_GAIN_MIN)

    results.append(("条件3 WINNER(MA5>MA10>MA20>MA60且四线↑且收盘>MA5且涨幅>2%)", winner,
        f"MA序={'✓' if ma5>ma10>ma20>ma60 else '✗'}, "
        f"d5={'↑' if d5==1 else '↓' if d5==-1 else '→'} "
        f"d10={'↑' if d10==1 else '↓' if d10==-1 else '→'} "
        f"d20={'↑' if d20==1 else '↓' if d20==-1 else '→'} "
        f"d60={'↑' if d60==1 else '↓' if d60==-1 else '→'}, "
        f"收盘({close_T:.2f})>MA5({ma5:.2f})={'✓' if close_T>ma5 else '✗'}, "
        f"涨幅({gain_today:+.1f}%)>2%={'✓' if gain_today>2.0 else '✗'}"))

    d5=ma_direction(closes,5); d10=ma_direction(closes,10)
    d20=ma_direction(closes,20); d60=ma_direction(closes,60)
    gain_today = (close_T / closes[T-1] - 1) * 100 if T >= 1 else 0
    turnover_today = turnovers[T] if T < len(turnovers) else 0

    # 停牌过滤（与check_base一致）
    c7 = turnover_today > 0
    results.append(("条件7 停牌过滤", c7,
        f"当日换手={turnover_today:.2f}% {'✓>0' if c7 else '✗=0停牌'}"))
    if not c7: return results

    # 跌幅过滤（与check_base一致）
    c8 = gain_today >= DROP_FILTER_MAX
    results.append(("条件8 当日跌幅过滤", c8,
        f"当日涨幅={gain_today:+.2f}%（需≥{DROP_FILTER_MAX}%）={'✓' if c8 else '✗'}"))
    if not c8: return results

    # 条件3 WINNER：与check_base顺序一致（拆解为独立判断）
    c3_ma = ma5 > ma10 > ma20 > ma60
    c3_dir = d5==1 and d10==1 and d20==1 and d60==1
    c3_close = close_T > ma5
    c3_gain = gain_today > WINNER_GAIN_MIN
    winner = c3_ma and c3_dir and c3_close and c3_gain
    results.append(("条件3 WINNER(MA5>MA10>MA20>MA60且四线↑且收盘>MA5且涨幅>2%)", winner,
        f"MA序={'✓' if c3_ma else '✗'}, "
        f"d5={'↑' if d5==1 else '↓' if d5==-1 else '→'} "
        f"d10={'↑' if d10==1 else '↓' if d10==-1 else '→'} "
        f"d20={'↑' if d20==1 else '↓' if d20==-1 else '→'} "
        f"d60={'↑' if d60==1 else '↓' if d60==-1 else '→'}, "
        f"收盘({close_T:.2f})>MA5({ma5:.2f})={'✓' if c3_close else '✗'}, "
        f"涨幅({gain_today:+.1f}%)>WINNER_GAIN_MIN%={'✓' if c3_gain else '✗'}"))
    if not winner: return results

    # 条件4：DIF>0 且 DEA>0 且 DIF上升（与check_base一致，失败返回None）
    macd,dif,dea = calc_macd(closes)
    if macd is None:
        results.append(("条件4 DIF/DEA", False, "无法计算MACD"))
        return results
    if T < 1:
        results.append(("条件4 DIF/DEA", False, "K线不足1根"))
        return results
    ef_y = calc_ema(closes[:T], MACD_FAST); es_y = calc_ema(closes[:T], MACD_SLOW)
    dif_y = ef_y - es_y
    dif_up = dif > dif_y
    c4 = dif>0 and dea>0 and dif_up
    results.append(("条件4 DIF>0且DEA>0且DIF↑", c4,
        f"DIF={dif:.3f}>0={'✓' if dif>0 else '✗'}, "
        f"DEA={dea:.3f}>0={'✓' if dea>0 else '✗'}, "
        f"DIF↑={'✓' if dif_up else '✗'}"))
    if not c4: return results

    # 条件5：5日涨幅（三选一，与check_base一致）
    if T < 5:
        results.append(("条件5 5日涨幅", False, "K线不足5根"))
        return results
    gain5d = (close_T / closes[T-5] - 1) * 100
    days_above = sum(1 for i in range(max(0,T-4),T+1)
                     if calc_ma(closes[:i+1],5) is not None and closes[i] > calc_ma(closes[:i+1],5))
    gain_t_c6 = (close_T / closes[T-1] - 1) * 100 if T>=1 else -999
    c6a = gain5d > L1_GAIN_STRICT
    c6b = gain5d > L1_GAIN_RELAXED and days_above >= L1_RELAX_MA5_DAYS
    c6c = (gain5d > WINNER_GAIN_RELAX_C5 and gain_t_c6 > WINNER_GAIN_RELAX_C6 and close_T > ma5 and (not WINNER_MA5_GT_MA20 or (ma5 > ma20 and ma10 > ma20)))
    c5_pass = c6a or c6b or c6c
    results.append(("条件5 5日涨幅", c5_pass,
        f"5日涨={gain5d:+.2f}%（>{L1_GAIN_STRICT}%=✓）"
        f"，宽松={c6b}（>{L1_GAIN_RELAXED}%且≥{L1_RELAX_MA5_DAYS}天>MA5={days_above}天）"
        f"，ACCEL={c6c}"))
    if not c5_pass: return results

    # 20日涨幅（与check_base一致）
    ret20 = float(close_T / closes[T-21] - 1) * 100 if T >= 21 else None
    if _GAIN20_MIN is not None and ret20 is not None:
        c_g20 = ret20 >= _GAIN20_MIN
        results.append(("--gain20 门槛", c_g20,
            f"20日涨={ret20:+.2f}%（需≥{_GAIN20_MIN}%）={'✓' if c_g20 else '✗'}"))
        if not c_g20: return results

    # 条件9：均换手（与check_base一致）
    avg_turnover = float(np.mean(turnovers[T-4:T+1])) if T>=4 else float(turnovers[T])
    outstanding = window["outstanding_share"].values if "outstanding_share" in window.columns else None
    market_cap = None
    if outstanding is not None and len(outstanding)>0:
        latest = float(outstanding[-1])
        if latest > 0: market_cap = latest * close_T / 1e8
    threshold = _TURNOVER_MIN if _TURNOVER_MIN is not None else (min_turnover_by_cap(market_cap) if market_cap else 10.0)
    c9 = avg_turnover >= threshold
    results.append(("条件9 均换手≥{threshold}%".format(threshold=threshold), c9,
        f"5日均换手={avg_turnover:.2f}%={'✓' if c9 else '✗'}"))
    if not c9: return results

    # ── P2 趋势过滤（与check_base完全一致）────────────────────
    # MA20方向（斜率法：10日前MA20到当前MA20，涨幅>2%为向上）
    ma20_now = calc_ma(closes, 20)
    ma20_ago = calc_ma(closes[:-P2_MA20_AGO_BARS], 20) if T >= P2_MA20_AGO_BARS else None
    if ma20_now is None or ma20_ago is None:
        results.append(("P2 MA20方向向上", False, "均线计算失败"))
        return results
    ma20_slope = (ma20_now - ma20_ago) / ma20_ago * 100
    ma20_up = ma20_slope > P2_MA20_SLOPE_THRESH * 100
    results.append(("P2 MA20方向向上(斜率法)", ma20_up,
        f"MA20_now={ma20_now:.2f}，{P2_MA20_AGO_BARS}日前={ma20_ago:.2f}，斜率={ma20_slope:+.2f}%（需>{P2_MA20_SLOPE_THRESH*100:.0f}%）={'↑' if ma20_up else '→或↓'}"))
    if not ma20_up: return results

    # ret20 >= P2_RET20_MIN%
    if ret20 is None or ret20 < P2_RET20_MIN:
        results.append(("P2 ret20>=P2_RET20_MIN%", False,
            f"ret20={ret20:+.2f}%={'✓' if ret20 is not None and ret20>=P2_RET20_MIN else '✗'}"))
        return results
    results.append(("P2 ret20>=P2_RET20_MIN%", True,
        f"ret20={ret20:+.2f}%={'✓'}"))

    # close < MA20 * P2_CLOSE_OVER_MA20
    c_close_ma20 = close_T < ma20 * P2_CLOSE_OVER_MA20
    results.append(("P2 close<MA20×P2_CLOSE_OVER_MA20", c_close_ma20,
        f"dist={(close_T/ma20-1)*100:.1f}%={'✓' if c_close_ma20 else '✗'}"))
    if not c_close_ma20: return results

    # 均线发散度 > P2_SPREAD_MIN_PCT%
    s1 = (ma5-ma10)/ma10*100; s2 = (ma10-ma20)/ma20*100
    spread_ok = ma5>ma10>ma20 and s1>P2_SPREAD_MIN_PCT and s2>P2_SPREAD_MIN_PCT
    results.append(("P2 均线发散度>P2_SPREAD_MIN_PCT%", spread_ok,
        f"MA5-10={s1:.2f}% MA10-20={s2:.2f}%={'✓' if spread_ok else '✗'}"))
    if not spread_ok: return results

    results.append(("P2 全部通过", True, "通过全部P2过滤"))
    return results


# ── 第一层过滤（WINNER + P2，固定路径）────────────────────
def check_base(code, signal_date):
    """
    WINNER 路径 + P2 趋势过滤（不可关闭）。
    可调参数：_GAIN20_MIN（默认12%）、_TURNOVER_MIN（默认市值规则）
    """
    code = normalize_symbol(code)
    df = _price.get(code)
    if df is None: return None
    il = df["date"].tolist()
    try: idx = il.index(signal_date)
    except: return None
    if idx < L1_MIN_BARS: return None

    window = df.iloc[idx - L1_MIN_BARS:idx + 1]
    closes = window["close"].values
    if "true_turnover" in window.columns and window["true_turnover"].notna().any():
        turnovers = window["true_turnover"].values
    elif "turnover" in window.columns:
        turnovers = window["turnover"].values
    else:
        turnovers = np.zeros(len(window))
    T = len(closes) - 1
    close_T = closes[T]

    # 市值
    outstanding = window["outstanding_share"].values if "outstanding_share" in window.columns else None
    market_cap = None
    if outstanding is not None and len(outstanding) > 0:
        latest = float(outstanding[-1])
        if latest > 0: market_cap = latest * close_T / 1e8

    ma5=calc_ma(closes,5); ma10=calc_ma(closes,10)
    ma20=calc_ma(closes,20); ma60=calc_ma(closes,60)
    if None in [ma5,ma10,ma20,ma60]: return None

    # 条件3 WINNER：MA空间有序 + 四线方向向上 + 收盘>MA5 + 涨幅>2%
    d5=ma_direction(closes,5); d10=ma_direction(closes,10)
    d20=ma_direction(closes,20); d60=ma_direction(closes,60)
    gain_today = (close_T / closes[T-1] - 1) * 100 if T >= 1 else 0
    turnover_today = turnovers[T] if T < len(turnovers) else 0

    # 停牌过滤
    if turnover_today <= 0: return None
    # 跌幅过滤
    if gain_today < -3.0: return None

    if not (ma5 > ma10 > ma20 > ma60): return None
    if not (d5==1 and d10==1 and d20==1 and d60==1): return None
    if not (close_T > ma5): return None
    if not (gain_today > 2.0): return None

    # 条件4：DIF>0 且 DEA>0 且 DIF上升
    macd,dif,dea = calc_macd(closes)
    if macd is None or not (dif>0 and dea>0): return None
    if T < 1: return None
    ef_y = calc_ema(closes[:T], MACD_FAST); es_y = calc_ema(closes[:T], MACD_SLOW)
    dif_y = ef_y - es_y
    if dif <= dif_y: return None

    # 条件5：5日涨幅（三选一）
    if T < 5: return None
    gain5d = (close_T / closes[T-5] - 1) * 100
    days_above = sum(1 for i in range(max(0,T-4),T+1)
                     if calc_ma(closes[:i+1],5) is not None and closes[i] > calc_ma(closes[:i+1],5))
    gain_t_c6 = (close_T / closes[T-1] - 1) * 100 if T>=1 else -999
    c6c = (gain5d > WINNER_GAIN_RELAX_C5 and gain_t_c6 > WINNER_GAIN_RELAX_C6 and close_T > ma5 and (not WINNER_MA5_GT_MA20 or (ma5 > ma20 and ma10 > ma20)))
    if gain5d <= L1_GAIN_STRICT:
        if not (gain5d > L1_GAIN_RELAXED and days_above >= L1_RELAX_MA5_DAYS):
            if not c6c: return None

    # 20日涨幅（用于P2 + _GAIN20_MIN）
    ret20 = float(close_T / closes[T-21] - 1) * 100 if T >= 21 else None
    ret10 = float(close_T / closes[T-11] - 1) * 100 if T >= 11 else None

    # --gain20 门槛
    if _GAIN20_MIN is not None and ret20 is not None:
        if ret20 < _GAIN20_MIN: return None

    # ── P2 趋势过滤（硬编码，不可关闭）────────────────────
    # MA20方向（斜率法）
    ma20_now = calc_ma(closes, 20)
    ma20_ago = calc_ma(closes[:-P2_MA20_AGO_BARS], 20) if T >= P2_MA20_AGO_BARS else None
    if ma20_now is None or ma20_ago is None: return None
    if not (ma20_now > ma20_ago * (1 + P2_MA20_SLOPE_THRESH)): return None
    # ret20 >= P2_RET20_MIN%
    if ret20 is None or ret20 < P2_RET20_MIN: return None
    # close < MA20 * 1.20
    if close_T >= ma20 * P2_CLOSE_OVER_MA20: return None
    # 均线发散度
    s1 = (ma5 - ma10) / ma10 * 100
    s2 = (ma10 - ma20) / ma20 * 100
    if not (ma5 > ma10 > ma20 and s1 > 1.0 and s2 > 1.0): return None

    # 5日均换手率
    avg_turnover = float(np.mean(turnovers[T-4:T+1])) if T>=4 else float(turnovers[T])
    threshold = _TURNOVER_MIN if _TURNOVER_MIN is not None else (min_turnover_by_cap(market_cap) if market_cap else 10.0)
    if avg_turnover < threshold: return None

    # RSI
    rsi = calc_rsi(closes[:T+1])
    if rsi is None: rsi = 50.0

    return {
        "code": code,
        "signal_date": signal_date,
        "close": round(close_T, 2),
        "ma5": round(ma5, 2), "ma10": round(ma10, 2),
        "ma20": round(ma20, 2), "ma60": round(ma60, 2),
        "macd": round(macd, 4), "dif": round(dif, 4), "dea": round(dea, 4),
        "gain5d": round(gain5d, 2),
        "avg_turnover_5": round(avg_turnover, 2),
        "ret20": round(ret20, 2) if ret20 is not None else None,
        "ret10": round(ret10, 2) if ret10 is not None else None,
        "market_cap": round(market_cap, 1) if market_cap is not None else None,
        "rsi": round(rsi, 1),
        "dist_ma20": round((close_T - ma20) / ma20 * 100, 1),
    }

# ── 第二层精筛 ────────────────────────────────────────────
def score_stock(sig, hot_sectors):
    score = 0; reasons = []

    # RSI
    rsi = sig["rsi"]
    if L2_RSI_STRONG_MIN <= rsi <= L2_RSI_STRONG_MAX:
        score += L2_SCORE_RSI_STRONG; reasons.append(f"RSI={rsi:.0f}(强)")
    elif L2_RSI_SECOND_MIN <= rsi < L2_RSI_STRONG_MIN or L2_RSI_STRONG_MAX < rsi <= L2_RSI_SECOND_MAX:
        score += L2_SCORE_RSI_SECOND; reasons.append(f"RSI={rsi:.0f}(次强)")
    elif L2_RSI_WEAK_MIN <= rsi < L2_RSI_SECOND_MIN or L2_RSI_SECOND_MAX < rsi <= L2_RSI_OVERBOUGHT:
        score += L2_SCORE_RSI_WEAK; reasons.append(f"RSI={rsi:.0f}(偏弱)")
    elif rsi > L2_RSI_OVERBOUGHT:
        penalty = int((rsi - L2_RSI_OVERBOUGHT) * 2)
        score += max(0, 4 - penalty); reasons.append(f"RSI={rsi:.0f}(超买扣{penalty}分)")
    else:
        reasons.append(f"RSI={rsi:.0f}(弱)")

    # 板块动量
    sector = get_sector(sig["code"])
    if _hot_sector_meta.get("status") == "ok":
        if sector and sector in hot_sectors:
            score += L2_SCORE_SECTOR_HOT; reasons.append(f"板块={sector}(热点)")
        elif sector:
            chg = _hot_sectors.get(sector, 0)
            if chg > 0:
                score += int(min(chg * 2, 6)); reasons.append(f"板块={sector}({chg:+.1f}%)")
            else:
                reasons.append(f"板块={sector}({chg:+.1f}%不加分)")
        else:
            reasons.append("无板块(中性)")
    else:
        if sector and sector in hot_sectors:
            score += L2_SCORE_SECTOR_HOT; reasons.append(f"板块={sector}(热点)")
        elif sector:
            reasons.append(f"板块={sector}(中性)")
        else:
            reasons.append("无板块(中性)")

    # 偏离MA20
    dist = sig["dist_ma20"]
    if 0 <= dist <= L2_DIST_HEALTHY_MAX:
        score += L2_SCORE_DIST_HEALTHY; reasons.append(f"偏离MA20={dist:+.1f}%(健康)")
    elif L2_DIST_HEALTHY_MAX < dist <= L2_DIST_HIGH_MAX:
        score += L2_SCORE_DIST_HIGH; reasons.append(f"偏离MA20={dist:+.1f}%(略高)")
    else:
        if dist > L2_DIST_HIGH_MAX:
            reasons.append(f"偏离MA20={dist:+.1f}%拒绝"); return False, 0, reasons
        score += L2_SCORE_DIST_CLOSE; reasons.append(f"偏离MA20={dist:+.1f}%贴线")

    # 换手率质量
    turnover = sig["avg_turnover_5"]
    if turnover >= L2_TURNOVER_HIGH: score += L2_SCORE_TURNOVER_HIGH; reasons.append(f"换手={turnover:.1f}%")
    elif turnover >= L2_TURNOVER_MID:  score += L2_SCORE_TURNOVER_MID;  reasons.append(f"换手={turnover:.1f}%")
    elif turnover >= L2_TURNOVER_LOW:  score += L2_SCORE_TURNOVER_LOW;  reasons.append(f"换手={turnover:.1f}%")
    else:                             reasons.append(f"换手={turnover:.1f}%(低)")

    # 5日涨幅健康
    gain = sig["gain5d"]
    is_accel = (L2_GAIN_ACCEL_MIN < gain <= L2_GAIN_COOL_MIN)
    if L2_GAIN_STRONG_MIN <= gain <= L2_GAIN_STRONG_MAX:
        score += L2_SCORE_GAIN5_STRONG; reasons.append(f"5日涨={gain:.1f}%(强)")
    elif L2_GAIN_SECOND_MIN < gain < L2_GAIN_STRONG_MIN:
        score += L2_SCORE_GAIN5_SECOND; reasons.append(f"5日涨={gain:.1f}%(次强)")
    elif L2_GAIN_STRONG_MAX < gain <= L2_GAIN_BIAS_STRONG_MAX:
        score += L2_SCORE_GAIN5_WARM; reasons.append(f"5日涨={gain:.1f}%(偏强)")
    elif L2_GAIN_BIAS_STRONG_MAX < gain <= L2_GAIN_HOT_MAX:
        score += L2_SCORE_GAIN5_WARM; reasons.append(f"5日涨={gain:.1f}%(偏热)")
    elif L2_GAIN_COOL_MIN < gain <= L2_GAIN_COOL_MAX:
        score += L2_SCORE_GAIN5_WARM; reasons.append(f"5日涨={gain:.1f}%(偏冷)")
    elif is_accel:
        score += L2_SCORE_GAIN5_ACCEL; reasons.append(f"5日涨={gain:.1f}%(ACCEL偏弱,+1)")
    else:
        reasons.append(f"5日涨={gain:.1f}%拒绝"); return False, 0, reasons

    # RPS综合
    rps = float(sig.get("rps_composite", 0))
    rps_score = round(min(rps, 100.0) / 100.0 * RPS_MAX_SCORE, 1)
    score += rps_score; reasons.append(f"RPS={rps:.0f}(+{rps_score:.1f})")

    return True, score, reasons


# ── RPS 计算 ─────────────────────────────────────────────
def _calc_market_metrics(code, signal_date):
    code = normalize_symbol(code)
    df = _price.get(code)
    if df is None or df.empty: return None
    il = df["date"].tolist()
    try: idx = il.index(signal_date)
    except: return None
    if idx < 21: return None
    closes = df.iloc[:idx+1]["close"].values.astype(float)
    t = len(closes) - 1
    close_t = closes[t]
    g5 = float(close_t / closes[t-5] - 1) * 100 if t>=5 else None
    r10 = float(close_t / closes[t-10] - 1) * 100 if t>=10 else None
    r20 = float(close_t / closes[t-20] - 1) * 100 if t>=20 else None
    if g5 is None or r10 is None or r20 is None: return None
    return {"code": code, "gain5d": g5, "ret10": r10, "ret20": r20}

def _build_market_pool(codes, signal_date):
    return [m for c in codes if (m := _calc_market_metrics(c, signal_date))]

def _inject_rps(signals, pool):
    if not signals or not pool:
        for s in signals:
            s["rps_composite"] = 0.0
        return
    vals5  = np.array([m["gain5d"] for m in pool], dtype=float)
    vals10 = np.array([m["ret10"]  for m in pool], dtype=float)
    vals20 = np.array([m["ret20"]  for m in pool], dtype=float)
    def pct(arr, v): return float(np.sum(arr < v)) / max(float(len(arr)), 1) * 100
    for s in signals:
        s["rps_composite"] = round(
            pct(vals5, s["gain5d"]) * RPS_WEIGHTS[0] +
            pct(vals10, s["ret10"])  * RPS_WEIGHTS[1] +
            pct(vals20, s["ret20"])  * RPS_WEIGHTS[2], 1)


# ── 主扫描 ───────────────────────────────────────────────
def screen_strategy(target_date, top_n=20, single_code=None):
    print(f"📊 WINNER+P2 扫描: {target_date}", flush=True)
    if not _price: preload()
    if not _sector_map: load_sector_map()

    hot = load_hot_sectors() if not _NO_HOT else {}
    if hot:
        print(f"   热点板块({len(hot)}个, {_sector_cache_note()}): "
              f"{', '.join(list(hot.keys())[:5])}...", flush=True)
    else:
        print(f"   热点板块(0个, {_sector_cache_note()})", flush=True)

    # ── 单票模式 ──────────────────────────────────────
    if single_code:
        single_code = normalize_symbol(single_code)
        sig = check_base(single_code, target_date)
        if sig is None:
            print(f"\n{'='*100}", flush=True)
            print(f"📊 单票诊断: {single_code}  日期: {target_date}", flush=True)
            print("="*100, flush=True)
            for label, passed, detail in debug_base(single_code, target_date):
                icon = "✅" if passed else "❌"
                print(f"  {icon} {label}", flush=True)
                if not passed: print(f"      └ {detail}", flush=True)
            print("="*100, flush=True)
            return
        import stock_trend.gain_turnover as gt
        pool = _build_market_pool(gt.get_all_stock_codes(), target_date)
        _inject_rps([sig], pool)
        passed, score, reasons = score_stock(sig, hot)
        names = load_stock_names()
        name = names.get(single_code, single_code)
        sector = get_sector(single_code) or "-"
        is_hot = "✅" if (hot and sector in hot) else "-"
        print(f"\n{'='*100}", flush=True)
        print(f"📊 单票分析: {single_code} {name}  日期: {target_date}", flush=True)
        print("="*100, flush=True)
        print(f"  收盘价: {sig['close']:.2f}", flush=True)
        print(f"  MA5/10/20/60: {sig['ma5']:.2f}/{sig['ma10']:.2f}/{sig['ma20']:.2f}/{sig['ma60']:.2f}", flush=True)
        print(f"  5日涨幅: {sig['gain5d']:+.2f}%  RSI: {sig['rsi']:.1f}", flush=True)
        print(f"  5日均换手: {sig['avg_turnover_5']:.2f}%  偏离MA20: {sig['dist_ma20']:+.1f}%", flush=True)
        print(f"  20日涨幅: {sig['ret20']:+.2f}%" if sig.get("ret20") else "  20日涨幅: N/A", flush=True)
        print(f"  市值: {sig['market_cap']:.0f}亿" if sig.get("market_cap") else "  市值: N/A", flush=True)
        print(f"  板块: {sector}  热点: {is_hot}", flush=True)
        print(f"\n  第一层: {'✅ 通过' if passed else '❌ 失败'}", flush=True)
        if passed:
            print(f"  WINNER+P2 评分: {score}分", flush=True)
            print(f"  详细原因: {' | '.join(reasons)}", flush=True)
        print("="*100, flush=True)
        return

    # ── 全市场扫描 ────────────────────────────────────
    codes = [f.stem.replace("_qfq","") for f in QFQ_DIR.glob("*_qfq.csv")]
    print(f"   全市场 {len(codes)} 只", flush=True)

    pool = _build_market_pool(codes, target_date)
    print(f"   {RPS_POOL_NAME}: {len(pool)} 只", flush=True)

    base = [s for c in codes if (s := check_base(c, target_date))]
    print(f"[Layer1] 第一层通过: {len(base)} 只", flush=True)

    _inject_rps(base, pool)
    print(f"   RPS注入完成: {len(base)} 只", flush=True)

    final = []
    for s in base:
        ok, sc, reasons = score_stock(s, hot)
        if ok and sc > 0:
            s["score"] = round(sc, 1)
            s["reasons"] = reasons
            final.append(s)

    print(f"🏆 第二层通过: {len(final)} 只", flush=True)
    final.sort(key=lambda x: (-x["score"], -x["rsi"], -x["gain5d"]))

    output_name = f"screen_double_winner_{target_date}"
    names = load_stock_names()
    hot_keys = list(hot.keys())

    print(f"\n{'='*120}", flush=True)
    print(f"📊 WINNER+P2 {target_date}（Layer1 {len(base)} / Layer2 {len(final)} 只）", flush=True)
    print("=" * 120, flush=True)
    print(_header_row(), flush=True)
    print("-" * 120, flush=True)

    txt_lines = [
        f"📊 WINNER+P2 {target_date}（Layer1 {len(base)} / Layer2 {len(final)} 只）",
        "=" * 120, _header_row(), "-" * 120,
    ]
    for s in final[:top_n]:
        nm = names.get(s["code"], s["code"])[:6]
        sector = get_sector(s["code"]) or "-"
        is_hot = "✅" if (hot_keys and sector in hot_keys) else "-"
        row = _make_row([
            s["code"], nm, target_date,
            f"{s.get('score', 0):.1f}",
            f"{s.get('rps_composite', 0):.0f}",
            f"{s['close']:.2f}",
            f"{s['gain5d']:+.1f}%",
            f"{s['avg_turnover_5']:.1f}%",
            "-" if s.get('market_cap') is None else f"{s['market_cap']:.0f}",
            f"{s['rsi']:.1f}",
            f"{s['dist_ma20']:+.1f}%",
            f"{s['ma20']:.2f}",
            f"{s['ma60']:.2f}",
            sector[:10],
            is_hot,
        ])
        print(row, flush=True)
        txt_lines.append(row)

    print("=" * 120, flush=True)
    txt_lines.append("=" * 120)

    # TXT
    txt_path = OUTPUT_DIR / f"{output_name}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(txt_lines))
    print(f"💾 TXT已保存: {txt_path}", flush=True)

    # JSON
    json_path = OUTPUT_DIR / f"{output_name}.json"
    lines = [json.dumps({"code": s["code"], "name": names.get(s["code"], s["code"]),
                          "signal_date": target_date}, ensure_ascii=False)
             for s in final[:top_n]]
    with open(json_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"💾 JSON已保存: {json_path}", flush=True)


# ── 帮助 ─────────────────────────────────────────────────
def print_screening_logic():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║       screen_double_winner.py    WINNER + P2 固定路径筛选          ║
╚══════════════════════════════════════════════════════════════════════╝

【硬编码规则（不可关闭）】
  WINNER 条件3：
    MA5>MA10>MA20>MA60（空间有序）
    四线方向全部向上（5日均线 > 5日前均线×1.001）
    当日收盘 > MA5
    当日涨幅 > 2%
  P2 趋势过滤：
    MA20方向向上（MA20斜率>2%，10日前→今）
    近20日涨幅 >= 12%
    收盘价 < MA20 × P2_CLOSE_OVER_MA20（偏离MA20不超过P2_CLOSE_OVER_MA20-1=10%，由常量控制）
    均线发散度：MA5>MA10>MA20 且 (MA5-MA10)/MA10>1% 且 (MA10-MA20)/MA20>1%

【Layer1 其余条件】（条件3 WINNER见上方专表）
  条件4：DIF>0 且 DEA>0 且 DIF上升
  条件5：5日涨幅>5%，或（>2%且近5日≥3天>MA5），或（>1%且当日>5%且收盘>MA5且MA5>MA20>MA10）
  条件6：当日换手>0（停牌过滤）
  条件7：当日跌幅<3%排除
  条件8：5日均换手率>=门槛值（按市值，默认10%）

【Layer2 精筛】（满分100分）
  RSI健康 +15、板块动量 +5、偏离MA20 +15、换手质量 +20、5日涨幅 +25、RPS综合 +25

【可调参数】
  --gain20    20日涨幅最低门槛（默认12%）
  --turnover   5日均换手率最低门槛（默认按市值规则）
  --no-hot     不区分热点板块

【排序规则】总分降序 → RSI降序 → 5日涨幅降序
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="screen_double WINNER+P2 强势股筛选")
    parser.add_argument("--date", default=str(date.today()),
                        help="信号日期（YYYY-MM-DD），默认今日")
    parser.add_argument("--top-n", type=int, default=500,
                        help="最多输出前N只（默认500）")
    parser.add_argument("--code", default=None,
                        help="单只股票代码，只分析指定股票")
    parser.add_argument("--gain20", type=float, default=CLI_GAIN20_DEFAULT,
                        help="20日涨幅最低门槛（%%，默认%.0f%%）" % CLI_GAIN20_DEFAULT)
    parser.add_argument("--turnover", type=float, default=None,
                        help="5日均换手率最低门槛（%%，默认按市值规则）")
    parser.add_argument("--no-hot", action="store_true",
                        help="不区分热点板块（默认显示热点加分）")
    args = parser.parse_args()

    if "--help" in sys.argv or "-h" in sys.argv:
        print_screening_logic()
        sys.exit(0)

    _GAIN20_MIN   = args.gain20
    _TURNOVER_MIN = args.turnover
    _NO_HOT       = args.no_hot

    screen_strategy(args.date, top_n=args.top_n, single_code=args.code)
