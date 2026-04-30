#!/usr/bin/env python3
"""
screen_double.py — 策略扫描器（双层过滤 + 全市场RPS综合排序）
========================================================
第一层（趋势基础）：7个条件过滤
  1. MA5 > MA10 > MA20 > MA60（均线多头排列）
  2. MA5/MA10/MA20/MA60 方向全部向上
  3. 收盘价 > MA10
  4. MACD > 0 且 DIF > 0 且 DEA > 0
  5. 5日涨幅 > 5%（或宽松条件）
  6. 5日均换手率达到市值门槛
  7. 信号日数据窗口 ≥ 66 根K线（距上市 > 65天）

第二层（精筛）：6个评分条件
  条件1：RSI健康（满分20）
  条件2：板块动量（满分15）
  条件3：偏离MA20（满分15）
  条件4：换手率质量（满分10）
  条件5：5日涨幅健康（满分15）
  条件6：全市场RPS综合（满分5）
  总分满分 85 分

输出：120宽表格 + TXT/JSON存档
"""
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

from stock_trend.gain_turnover import (
    normalize_symbol,
    load_stock_names,
    get_stock_name,
    compute_rsi_scalar as _gt_rsi_scalar,
)

# 修正：真实缓存路径在 workspace/ 而非 stock_trend/
_REAL_WORKSPACE = Path.home() / ".openclaw" / "workspace"
QFQ_DIR     = _REAL_WORKSPACE / ".cache" / "qfq_daily"
SECTOR_FILE = _REAL_WORKSPACE / ".cache" / "sector" / "sector_hotspot.json"

# ── 策略常量（代码/帮助共用，防止口径漂移）────────────────────
L1_MIN_BARS = 65
L1_GAIN_STRICT = 5.0
L1_GAIN_RELAXED = 2.0
L1_RELAX_MA5_DAYS = 3

L2_RSI_STRONG_MIN = 55
L2_RSI_STRONG_MAX = 70
L2_RSI_SECOND_MIN = 48
L2_RSI_SECOND_MAX = 75
L2_RSI_WEAK_MIN = 40
L2_RSI_OVERBOUGHT = 82

L2_DIST_HEALTHY_MAX = 15
L2_DIST_HIGH_MAX = 25

L2_TURNOVER_HIGH = 10
L2_TURNOVER_LOW = 5

L2_GAIN_STRONG_MIN = 8
L2_GAIN_STRONG_MAX = 15
L2_GAIN_SECOND_MIN = 5
L2_GAIN_BIAS_STRONG_MAX = 25
L2_GAIN_HOT_MAX = 35
L2_GAIN_COOL_MIN = 2
L2_GAIN_COOL_MAX = 5

RPS_POOL_NAME = "全市场有效股票池"
# RPS_MIN_ACCEPT = 10.0  # 已弃用：不再硬拒绝，由得分自动体现
RPS_MAX_SCORE = 5.0
RPS_WEIGHTS = (0.30, 0.30, 0.40)  # rps5, rps10, rps20

HELP_L1_GAIN_RULE = (
    f"5日涨幅 > {L1_GAIN_STRICT:.0f}%，或（5日涨幅 > {L1_GAIN_RELAXED:.0f}% "
    f"且近5日≥{L1_RELAX_MA5_DAYS}天收盘>MA5）"
)
HELP_L2_GAIN_RULES = [
    f"8%≤涨幅≤15%：+15分（强）",
    f"5%<涨幅<8% 或 15%<涨幅≤25%：+9分（次强/偏强）",
    f"25%<涨幅≤35%：+3分（偏热）",
    f"2%<涨幅≤5%：+3分（偏冷）",
    f"涨幅<{L2_GAIN_COOL_MIN:.0f}% 或 涨幅>{L2_GAIN_HOT_MAX:.0f}%：直接拒绝",
]
HELP_RPS_RULES = [
    f"{RPS_POOL_NAME}内计算RPS综合",
    f"RPS综合<5：得分<0.3（实际影响很小）",
    f"RPS综合得分 = RPS综合 / 100 × {RPS_MAX_SCORE:.0f}（满分{RPS_MAX_SCORE:.0f}）",
]
HELP_SORT_RULE = "总分降序 → RSI降序 → 5日涨幅降序"
HELP_SECTOR_CACHE_NOTE = "板块热点缓存缺失时，条件2按中性处理（不加分不扣分）"

# ── 表格格式（与screen_trend.py一致）──────────────────────────────
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

# ── 热点板块 ──────────────────────────────────────────────
_hot_sectors = {}  # 板块名 -> 涨跌幅
_hot_sector_meta = {"status": "unknown", "mtime": None}


def _sector_cache_note():
    if _hot_sector_meta.get("mtime"):
        return f"热点板块缓存时间: {_hot_sector_meta['mtime']}"
    return HELP_SECTOR_CACHE_NOTE


def load_hot_sectors():
    global _hot_sectors, _hot_sector_meta
    if not _hot_sectors and SECTOR_FILE.exists():
        try:
            data = json.load(open(SECTOR_FILE))
            for name, chg in data.items():
                _hot_sectors[name] = chg
            _hot_sector_meta = {
                "status": "ok",
                "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(SECTOR_FILE.stat().st_mtime)),
            }
        except Exception:
            _hot_sector_meta = {"status": "error", "mtime": None}
    elif not _hot_sectors:
        _hot_sector_meta = {"status": "missing", "mtime": None}

    sorted_sectors = sorted(_hot_sectors.items(), key=lambda x: -x[1])[:15]
    return dict(sorted_sectors)


def calc_ma(closes, period):
    if len(closes) < period: return None
    return float(np.mean(closes[-period:]))

def calc_ema(closes, period):
    if len(closes) < period: return None
    alpha = 2.0 / (period + 1)
    ema = float(closes[0])
    for c in closes[1:]: ema = alpha * c + (1 - alpha) * ema
    return ema

def calc_macd(closes, fast=12, slow=26, signal=9):
    n = len(closes)
    if n < slow + signal: return None, None, None
    ef = calc_ema(closes, fast); es = calc_ema(closes, slow)
    if ef is None or es is None: return None, None, None
    dif = ef - es
    dif_series = []
    for i in range(signal, n):
        ef_i = calc_ema(closes[:i+1], fast); es_i = calc_ema(closes[:i+1], slow)
        if ef_i is not None and es_i is not None:
            dif_series.append(ef_i - es_i)
    if len(dif_series) < signal: return None, None, None
    dea = calc_ema(dif_series, signal)
    macd = (dif - dea) * 2 if dea is not None else None
    return macd, dif, dea

def ma_direction(closes, period):
    if len(closes) < period + 5: return 0
    now = calc_ma(closes, period)
    ago = calc_ma(closes[:-5], period)
    if now is None or ago is None: return 0
    return 1 if now > ago * 1.001 else (-1 if now < ago * 0.999 else 0)

def calc_rsi(closes, period=14):
    """RSI（Wilder平滑，统一复用gain_turnover.compute_rsi_scalar）"""
    return _gt_rsi_scalar(np.asarray(closes), period)

# ── 板块判断 ─────────────────────────────────────────────
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

# ── 第一层：策略1基础条件 ────────────────────────────────
def min_turnover_by_cap(market_cap):
    """根据市值返回最低换手率门槛（%）。
    市值单位：亿元（outstanding_share * close / 1e8）

    门槛表：
      市值 ≥ 500亿 → ≥1%
      市值 ≥ 100亿 → ≥3%
      市值 ≥ 30亿  → ≥5%
      市值 ≥ 20亿  → ≥8%
      市值 < 20亿  → ≥10%
    """
    if market_cap >= 500:
        return 1.0
    elif market_cap >= 100:
        return 3.0
    elif market_cap >= 30:
        return 5.0
    elif market_cap >= 20:
        return 8.0
    else:
        return 10.0

def check_base(code, signal_date):
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
    T_pos = len(closes) - 1
    close_T = closes[T_pos]

    # ── 市值计算（单位：亿元）──────────────────────────
    # outstanding_share 单位为股，* close / 1e8 = 市值（亿元）
    outstanding = window["outstanding_share"].values if "outstanding_share" in window.columns else None
    market_cap = None  # None = 未知市值，使用最保守换手率门槛
    if outstanding is not None and len(outstanding) > 0:
        latest_outstanding = float(outstanding[-1])
        if latest_outstanding > 0:
            market_cap = latest_outstanding * close_T / 1e8

    ma5=calc_ma(closes,5); ma10=calc_ma(closes,10)
    ma20=calc_ma(closes,20); ma60=calc_ma(closes,60)
    if None in [ma5,ma10,ma20,ma60]: return None

    if not (close_T > ma10): return None
    d5=ma_direction(closes,5); d10=ma_direction(closes,10)
    d20=ma_direction(closes,20); d60=ma_direction(closes,60)
    if not (d5==1 and d10==1 and d20==1 and d60==1): return None
    if not (ma5>ma10>ma20>ma60): return None

    macd,dif,dea = calc_macd(closes)
    if macd is None or not (macd>0 and dif>0 and dea>0): return None

    if T_pos < 5: return None
    # ── 5日涨幅 + 辅助条件 ─────────────────────────────────
    # 第一层涨幅通过规则（二选一）：
    #   条件A（正常）：5日涨幅 > 5%
    #   条件B（宽松）：5日涨幅 > 2% 且 近5日中 ≥3天收盘 > 当日MA5
    # 解释：短期涨幅温和但持续在MA5上方的股票，仍有进入价值
    gain5d = (close_T / closes[T_pos-5] - 1) * 100
    days_above_ma5 = 0
    for i in range(max(0, T_pos-4), T_pos+1):
        ma5_i = calc_ma(closes[:i+1], 5)
        if ma5_i is not None and closes[i] > ma5_i:
            days_above_ma5 += 1
    if gain5d <= L1_GAIN_STRICT:
        if not (gain5d > L1_GAIN_RELAXED and days_above_ma5 >= L1_RELAX_MA5_DAYS):
            return None

    avg_turnover_5 = float(np.mean(turnovers[T_pos-4:T_pos+1])) if T_pos>=4 else float(turnovers[T_pos])
    # 市值未知/无效时使用最高换手率门槛（10%）
    threshold = min_turnover_by_cap(market_cap) if (market_cap is not None and market_cap > 0) else 10.0
    if avg_turnover_5 < threshold: return None

    # RSI 只计算到信号日（含），窗口为 L1_MIN_BARS=65 根K线
    rsi = calc_rsi(closes[:T_pos+1])
    if rsi is None: rsi = 50.0

    # ── 中期涨幅（用于RPS排序）────────────────────────────
    ret20 = float(close_T / closes[T_pos-21] - 1) * 100 if T_pos >= 21 else None
    ret10 = float(close_T / closes[T_pos-11] - 1) * 100 if T_pos >= 11 else None

    return {
        "code": code, "signal_date": signal_date,
        "close": round(close_T,2),
        "ma5": round(ma5,2), "ma10": round(ma10,2),
        "ma20": round(ma20,2), "ma60": round(ma60,2),
        "macd": round(macd,4), "dif": round(dif,4), "dea": round(dea,4),
        "gain5d": round(gain5d,2), "avg_turnover_5": round(avg_turnover_5,2),
        "ret20": round(ret20, 2) if ret20 is not None else None,
        "ret10": round(ret10, 2) if ret10 is not None else None,
        "market_cap": round(market_cap, 1) if market_cap is not None else None,
        "rsi": round(rsi,1),
        "dist_ma20": round((close_T - ma20) / ma20 * 100, 1),
    }

# ── 第二层：精筛条件 ─────────────────────────────────────
def score_stock(sig, hot_sectors):
    code = sig["code"]
    score = 0
    reasons = []

    # ── 条件1：RSI健康区间 ─────────────────────────
    rsi = sig["rsi"]
    if L2_RSI_STRONG_MIN <= rsi <= L2_RSI_STRONG_MAX:
        score += 20
        reasons.append(f"RSI={rsi:.0f}(强)")
    elif L2_RSI_SECOND_MIN <= rsi < L2_RSI_STRONG_MIN or L2_RSI_STRONG_MAX < rsi <= L2_RSI_SECOND_MAX:
        score += 14
        reasons.append(f"RSI={rsi:.0f}(次强)")
    elif L2_RSI_WEAK_MIN <= rsi < L2_RSI_SECOND_MIN or L2_RSI_SECOND_MAX < rsi <= L2_RSI_OVERBOUGHT:
        score += 8
        reasons.append(f"RSI={rsi:.0f}(偏弱)")
    elif rsi > L2_RSI_OVERBOUGHT:
        penalty = int((rsi - L2_RSI_OVERBOUGHT) * 2)
        score += max(0, 4 - penalty)
        reasons.append(f"RSI={rsi:.0f}(超买扣{penalty}分)")
    else:
        reasons.append(f"RSI={rsi:.0f}(弱)")

    # ── 条件2：板块动量 ──────────────────────────────
    # 缓存状态：ok（正常） / error（文件损坏，按中性） / missing（文件不存在，按中性）
    sector = get_sector(code)
    if _hot_sector_meta.get("status") == "ok":
        # 缓存正常：精确加分
        if sector and sector in hot_sectors:
            score += 15
            reasons.append(f"板块={sector}(热点)")
        elif sector:
            chg = _hot_sectors.get(sector, 0)
            if chg > 0:
                score += int(min(chg * 2, 6))
                reasons.append(f"板块={sector}({chg:+.1f}%)")
            else:
                reasons.append(f"板块={sector}({chg:+.1f}%不加分)")
        else:
            reasons.append("无板块数据(中性)")
    else:
        # 缓存异常（error / missing）：保守处理，sector in hot_sectors 仍加分，其余中性
        if sector and sector in hot_sectors:
            score += 15
            reasons.append(f"板块={sector}(热点缓存异常，按热点加分)")
        elif sector:
            reasons.append(f"板块={sector}(热点缓存异常，中性)")
        else:
            reasons.append("无板块数据(中性)")

    # ── 条件3：偏离MA20过滤 ──────────────────────────
    dist = sig["dist_ma20"]
    if 0 <= dist <= L2_DIST_HEALTHY_MAX:
        score += 15
        reasons.append(f"偏离MA20={dist:+.1f}%(健康)")
    elif L2_DIST_HEALTHY_MAX < dist <= L2_DIST_HIGH_MAX:
        score += 8
        reasons.append(f"偏离MA20={dist:+.1f}%(略高)")
    else:
        if dist > L2_DIST_HIGH_MAX:
            reasons.append(f"偏离MA20={dist:+.1f}%拒绝")
            return False, 0, reasons
        score += 5
        reasons.append(f"偏离MA20={dist:+.1f}%贴线")

    # ── 条件4：换手率质量 ────────────────────────────
    turnover = sig["avg_turnover_5"]
    gain = sig["gain5d"]
    if turnover >= L2_TURNOVER_HIGH:
        score += 10
        reasons.append(f"换手={turnover:.1f}%")
    elif turnover >= L2_TURNOVER_LOW:
        score += 5
        reasons.append(f"换手={turnover:.1f}%(低)")

    # ── 条件5：5日涨幅健康 ──────────────────────────
    if L2_GAIN_STRONG_MIN <= gain <= L2_GAIN_STRONG_MAX:
        score += 15
        reasons.append(f"5日涨={gain:.1f}%(强)")
    elif L2_GAIN_SECOND_MIN < gain < L2_GAIN_STRONG_MIN:
        score += 9
        reasons.append(f"5日涨={gain:.1f}%(次强)")
    elif L2_GAIN_STRONG_MAX < gain <= L2_GAIN_BIAS_STRONG_MAX:
        score += 9
        reasons.append(f"5日涨={gain:.1f}%(偏强)")
    elif L2_GAIN_BIAS_STRONG_MAX < gain <= L2_GAIN_HOT_MAX:
        score += 3
        reasons.append(f"5日涨={gain:.1f}%(偏热)")
    elif L2_GAIN_COOL_MIN < gain <= L2_GAIN_COOL_MAX:
        score += 3
        reasons.append(f"5日涨={gain:.1f}%(偏冷)")
    else:
        reasons.append(f"5日涨={gain:.1f}%拒绝")
        return False, 0, reasons

    # ── 条件6：全市场RPS综合（满分5）────────────────────
    rps = float(sig.get("rps_composite", 0))
    rps_score = round(min(rps, 100.0) / 100.0 * RPS_MAX_SCORE, 1)
    score += rps_score
    reasons.append(f"RPS={rps:.0f}(+{rps_score:.1f})")

    return True, score, reasons


def _calc_market_metrics(code, signal_date):
    """计算单只股票的全市场RPS基准指标，不做Layer1过滤。"""
    code = normalize_symbol(code)
    df = _price.get(code)
    if df is None or df.empty:
        return None
    il = df["date"].tolist()
    try:
        idx = il.index(signal_date)
    except Exception:
        return None
    if idx < 21:
        return None

    closes = df.iloc[:idx + 1]["close"].values.astype(float)
    t_pos = len(closes) - 1
    close_t = closes[t_pos]
    gain5d = float(close_t / closes[t_pos - 5] - 1) * 100 if t_pos >= 5 else None
    ret10 = float(close_t / closes[t_pos - 10] - 1) * 100 if t_pos >= 10 else None
    ret20 = float(close_t / closes[t_pos - 20] - 1) * 100 if t_pos >= 20 else None
    if gain5d is None or ret10 is None or ret20 is None:
        return None
    return {"code": code, "gain5d": gain5d, "ret10": ret10, "ret20": ret20}


def _build_market_rps_pool(codes, signal_date):
    pool = []
    for code in codes:
        item = _calc_market_metrics(code, signal_date)
        if item:
            pool.append(item)
    return pool


def _inject_rps(signals, market_pool):
    """显式在进入第二层前完成RPS注入，基于全市场有效股票池。"""
    if not signals or not market_pool:
        for s in signals:
            s["rps5"] = 0.0
            s["rps10"] = 0.0
            s["rps20"] = 0.0
            s["rps_composite"] = 0.0
        return

    vals5 = np.array([m["gain5d"] for m in market_pool if m.get("gain5d") is not None], dtype=float)
    vals10 = np.array([m["ret10"] for m in market_pool if m.get("ret10") is not None], dtype=float)
    vals20 = np.array([m["ret20"] for m in market_pool if m.get("ret20") is not None], dtype=float)

    def pct_rank(arr, val):
        return float(np.sum(arr < val)) / max(float(len(arr)), 1) * 100

    for s in signals:
        rps5 = pct_rank(vals5, s["gain5d"]) if s.get("gain5d") is not None else 0.0
        rps10 = pct_rank(vals10, s["ret10"]) if s.get("ret10") is not None else 0.0
        rps20 = pct_rank(vals20, s["ret20"]) if s.get("ret20") is not None else 0.0
        s["rps5"] = round(rps5, 1)
        s["rps10"] = round(rps10, 1)
        s["rps20"] = round(rps20, 1)
        s["rps_composite"] = round(rps5 * RPS_WEIGHTS[0] + rps10 * RPS_WEIGHTS[1] + rps20 * RPS_WEIGHTS[2], 1)


# ── 主扫描 ───────────────────────────────────────────────
def screen_strategy(target_date, top_n=20):
    print(f"📊 策略扫描: {target_date}", flush=True)
    t0 = time.time()
    if not _price: preload()
    if not _sector_map: load_sector_map()

    hot_sectors = load_hot_sectors()
    if hot_sectors:
        print(f"   热点板块({len(hot_sectors)}个, {_sector_cache_note()}): {', '.join(list(hot_sectors.keys())[:5])}...", flush=True)
    else:
        print(f"   热点板块(0个, {_sector_cache_note()})", flush=True)

    codes = [f.stem.replace("_qfq", "") for f in QFQ_DIR.glob("*_qfq.csv")]
    print(f"   全市场 {len(codes)} 只\n", flush=True)

    # ── 第一步：构建全市场RPS基准池 ─────────────────────
    market_rps_pool = _build_market_rps_pool(codes, target_date)
    print(f"   {RPS_POOL_NAME}: {len(market_rps_pool)} 只（数据不足26日的股票不进入RPS池）", flush=True)

    # ── 第二步：Layer1 过滤 ───────────────────────────
    base_signals = []
    for code in codes:
        sig = check_base(code, target_date)
        if sig:
            base_signals.append(sig)
    print(f"[Layer1] 第一层通过: {len(base_signals)} 只", flush=True)

    # ── 第三步：显式注入RPS后再进入第二层 ───────────────
    _inject_rps(base_signals, market_rps_pool)
    print(f"   RPS注入完成: {len(base_signals)} 只", flush=True)

    # ── 第四步：Layer2 精筛 ───────────────────────────
    final_signals = []
    for sig in base_signals:
        passed, score, reasons = score_stock(sig, hot_sectors)
        if passed and score > 0:
            sig["score"] = round(score, 1)
            sig["reasons"] = reasons
            final_signals.append(sig)
    print(f"🏆 第二层通过: {len(final_signals)} 只", flush=True)

    final_signals.sort(key=lambda x: (-x["score"], -x["rsi"], -x["gain5d"]))
    names = load_stock_names()
    hot_sector_names = list(hot_sectors.keys())

    # ── 表格输出（与screen_trend.py一致）───────────────────────
    print(f"\n{'='*120}", flush=True)
    print(f"📊 screen_double {target_date}（第一层 {len(base_signals)} / 第二层 {len(final_signals)} 只）", flush=True)
    print("=" * 120, flush=True)
    print(_header_row(), flush=True)
    print("-" * 120, flush=True)

    txt_lines = [
        f"📊 screen_double {target_date}（第一层 {len(base_signals)} / 第二层 {len(final_signals)} 只）",
        "=" * 120, _header_row(), "-" * 120,
    ]

    for s in final_signals[:top_n]:
        name = names.get(s["code"], s["code"])[:6]
        sector = get_sector(s["code"]) or "-"
        is_hot = "✅" if sector in hot_sector_names else "-"
        row = _make_row([
            s["code"], name, target_date,
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

    # ── 保存TXT ──────────────────────────────────────
    txt_path = Path.home() / "stock_reports" / f"screen_double_{target_date}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(txt_lines))
    print(f"💾 TXT已保存: {txt_path}", flush=True)


# ── 主入口 ───────────────────────────────────────────────
def print_screening_logic():
    gain_rules = "\n".join([f"    · {line}" for line in HELP_L2_GAIN_RULES])
    rps_rules = "\n".join([f"    · {line}" for line in HELP_RPS_RULES])
    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║              screen_double.py    筛选逻辑 & 评分条件                 ║
╚══════════════════════════════════════════════════════════════════════╝

【第一层：趋势基础条件】（全部通过才进入第二层）
  1. MA5 > MA10 > MA20 > MA60（均线多头排列）
  2. MA5/MA10/MA20/MA60 方向全部向上（5日内均值 > 5日前均值 × 1.001）
  3. 收盘价 > MA10
  4. MACD > 0 且 DIF > 0 且 DEA > 0（多头排列，非金叉）
  5. {HELP_L1_GAIN_RULE}
  6. 5日均换手率 ≥ 门槛值（市值≥500亿≥1%，≥100亿≥3%，≥30亿≥5%，≥20亿≥8%，＜20亿≥10%）
  7. 信号日数据窗口充足（距上市>{L1_MIN_BARS}天，排除新股）

【第二层：精筛评分条件】
  条件1：RSI 健康区间（满分20分）
    · 55≤RSI≤70：+20分（强）
    · 48≤RSI<55 或 70<RSI≤75：+14分（次强）
    · 40≤RSI<48 或 75<RSI≤82：+8分（偏弱）
    · RSI>82：max(0, 4 - (RSI-82)×2) 分（超买扣分）
    · RSI<40：+0分（弱）

  条件2：板块动量（满分15分）
    · 股票所在板块为当日热点（前15名涨幅板块）：+15分
    · 板块涨跌幅>0（非热点正涨幅）：+min(板块涨幅×2, 6)分
    · 板块涨跌幅≤0：+0分
    · {HELP_SECTOR_CACHE_NOTE}

  条件3：偏离MA20（满分15分）
    · 0%≤偏离≤15%：+15分（健康）
    · 15%<偏离≤25%：+8分（略高）
    · 偏离<0%（低于MA20）：+5分（贴线）
    · 偏离>25%：直接拒绝

  条件4：换手率质量（满分10分）
    · 5日均换手率≥10%：+10分
    · 5日均换手率≥5%：+5分

  条件5：5日涨幅健康（满分15分）
{gain_rules}

  条件6：RPS综合（满分5分）
{rps_rules}

【排序规则】{HELP_SORT_RULE}
""")


if __name__ == "__main__":
    import sys
    if "--help" in sys.argv or "-h" in sys.argv:
        print_screening_logic()
        print("可选参数：")
        parser = argparse.ArgumentParser()
        parser.add_argument("--date", default="2026-04-24")
        parser.add_argument("--top-n", type=int, default=80)
        parser.print_help()
        print()
        sys.exit(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="2026-04-24")
    parser.add_argument("--top-n", type=int, default=80)
    args = parser.parse_args()
    screen_strategy(args.date, args.top_n)
