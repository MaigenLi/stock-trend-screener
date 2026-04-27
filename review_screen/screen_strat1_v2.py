#!/usr/bin/env python3
"""
策略1 v2：两层过滤
==================
第一层（趋势基础）：策略1原始6个条件
第二层（精筛）：RSI健康 + 板块动量 + 偏离MA20 + 量价质量

输出样式与 screen.py 一致（120宽表格、中文宽度对齐）
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

from stock_trend.review_screen.screen import load_stock_names

QFQ_DIR     = WORKSPACE / ".cache" / "qfq_daily"
SECTOR_FILE = WORKSPACE / ".cache" / "sector" / "sector_hotspot.json"

# ── 表格格式（与screen.py一致）──────────────────────────────
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
    ("评分",    6, ">"),
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

# ── 数据加载 ──────────────────────────────────────────────
_price = {}

def preload():
    global _price
    print("📂 加载数据...", flush=True)
    for f in QFQ_DIR.glob("*_qfq.csv"):
        code = f.stem.replace("_qfq", "")
        try:
            df = pd.read_csv(f)
            df = df.sort_values("date").reset_index(drop=True)
            _price[code] = df
        except:
            pass
    print(f"✅ {len(_price)}只已加载", flush=True)

# ── 热点板块 ──────────────────────────────────────────────
_hot_sectors = {}  # 板块名 -> 涨跌幅

def load_hot_sectors():
    global _hot_sectors
    _hot_sectors = {}
    if SECTOR_FILE.exists():
        try:
            data = json.load(open(SECTOR_FILE))
            for name, chg in data.items():
                _hot_sectors[name] = chg
        except:
            pass
    sorted_sectors = sorted(_hot_sectors.items(), key=lambda x: -x[1])[:15]
    return dict(sorted_sectors)

# ── 指标计算 ─────────────────────────────────────────────
from stock_trend.gain_turnover import compute_rsi_scalar as _gt_rsi_scalar

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
    市值单位：亿元（outstanding_share * close / 1e8）"""
    if market_cap >= 500:
        return 1.0
    elif market_cap >= 100:
        return 3.0
    elif market_cap >= 30:
        return 5.0
    else:
        return 10.0

def check_base(code, signal_date):
    df = _price.get(code)
    if df is None: return None
    il = df["date"].tolist()
    try: idx = il.index(signal_date)
    except: return None
    if idx < 65: return None

    window = df.iloc[idx - 65:idx + 1]
    closes    = window["close"].values
    turnovers = window["turnover"].values if "turnover" in window.columns else np.zeros(len(window))
    T_pos = len(closes) - 1
    close_T = closes[T_pos]

    # ── 市值计算（单位：亿元）──────────────────────────
    outstanding = window["outstanding_share"].values if "outstanding_share" in window.columns else None
    market_cap = 0.0
    if outstanding is not None and len(outstanding) > 0:
        # outstanding_share 单位为股，* close / 1e8 = 市值（亿元）
        latest_outstanding = float(outstanding[-1])
        if latest_outstanding > 0:
            market_cap = latest_outstanding * close_T / 1e8

    ma5=calc_ma(closes,5); ma10=calc_ma(closes,10)
    ma20=calc_ma(closes,20); ma60=calc_ma(closes,60)
    if None in [ma5,ma10,ma20,ma60]: return None

    if not (close_T > ma5): return None
    d5=ma_direction(closes,5); d10=ma_direction(closes,10)
    d20=ma_direction(closes,20); d60=ma_direction(closes,60)
    if not (d5==1 and d10==1 and d20==1 and d60==1): return None
    if not (ma5>ma10>ma20>ma60): return None

    macd,dif,dea = calc_macd(closes)
    if macd is None or not (macd>0 and dif>0 and dea>0): return None

    if T_pos < 5: return None
    gain5d = (close_T / closes[T_pos-5] - 1) * 100
    if gain5d <= 5.0: return None

    avg_turnover_5 = float(np.mean(turnovers[T_pos-4:T_pos+1])) if T_pos>=4 else float(turnovers[T_pos])
    threshold = min_turnover_by_cap(market_cap)
    if avg_turnover_5 < threshold: return None

    rsi = calc_rsi(closes)
    if rsi is None: rsi = 50.0

    return {
        "code": code, "signal_date": signal_date,
        "close": round(close_T,2),
        "ma5": round(ma5,2), "ma10": round(ma10,2),
        "ma20": round(ma20,2), "ma60": round(ma60,2),
        "macd": round(macd,4), "dif": round(dif,4), "dea": round(dea,4),
        "gain5d": round(gain5d,2), "avg_turnover_5": round(avg_turnover_5,2),
        "market_cap": round(market_cap, 1),
        "rsi": round(rsi,1),
        "dist_ma20": round((close_T - ma20) / ma20 * 100, 1),
    }

# ── 第二层：精筛条件 ─────────────────────────────────────
def score_stock(sig, hot_sectors):
    code = sig["code"]
    score = 0
    reasons = []

    # ── 条件1：RSI健康区间（细分4档）──────────────
    rsi = sig["rsi"]
    if 55 <= rsi <= 70:
        score += 20
        reasons.append(f"RSI={rsi:.0f}(强)")
    elif 48 <= rsi < 55 or 70 < rsi <= 75:
        score += 14
        reasons.append(f"RSI={rsi:.0f}(次强)")
    elif 40 <= rsi < 48 or 75 < rsi <= 82:
        score += 8
        reasons.append(f"RSI={rsi:.0f}(偏弱)")
    elif rsi > 82:
        penalty = int((rsi - 82) * 2)
        score += max(0, 4 - penalty)
        reasons.append(f"RSI={rsi:.0f}(超买扣{penalty}分)")
    else:
        score += 0
        reasons.append(f"RSI={rsi:.0f}(弱)")

    # ── 条件2：板块动量 ──────────────────────────────
    sector = get_sector(code)
    if sector and sector in hot_sectors:
        score += 15
        reasons.append(f"板块={sector}(热点)")
    elif sector:
        chg = _hot_sectors.get(sector, 0)
        if chg > 0:
            score += int(min(chg * 3, 12))
            reasons.append(f"板块={sector}({chg:+.1f}%)")
        else:
            reasons.append(f"板块={sector}({chg:+.1f}%不加分)")
    else:
        reasons.append("无板块数据")

    # ── 条件3：偏离MA20过滤 ──────────────────────────
    dist = sig["dist_ma20"]
    if 0 <= dist <= 15:
        score += 15
        reasons.append(f"偏离MA20={dist:+.1f}%(健康)")
    elif 15 < dist <= 25:
        score += 8
        reasons.append(f"偏离MA20={dist:+.1f}%(略高)")
    else:
        if dist > 25:
            reasons.append(f"偏离MA20={dist:+.1f}%拒绝")
            return False, 0, reasons
        else:
            score += 5
            reasons.append(f"偏离MA20={dist:+.1f}%贴线")

    # ── 条件4：量价质量 ──────────────────────────────
    turnover = sig["avg_turnover_5"]
    gain = sig["gain5d"]
    if turnover >= 10:
        score += 10
        reasons.append(f"换手={turnover:.1f}%")
    elif turnover >= 5:
        score += 5
        reasons.append(f"换手={turnover:.1f}%(低)")

    # ── 条件5：5日涨幅健康（细分3档）─────────────
    if 8 <= gain <= 15:
        score += 15
        reasons.append(f"5日涨={gain:.1f}%(强)")
    elif 5 < gain < 8:
        score += 9
        reasons.append(f"5日涨={gain:.1f}%(次强)")
    elif 15 < gain <= 25:
        score += 9
        reasons.append(f"5日涨={gain:.1f}%(偏强)")
    elif 25 < gain <= 35:
        score += 3
        reasons.append(f"5日涨={gain:.1f}%(偏热)")
    else:
        reasons.append(f"5日涨={gain:.1f}%拒绝")
        return False, 0, reasons

    return True, score, reasons

# ── 主扫描 ───────────────────────────────────────────────
def screen_strat1_v2(target_date, top_n=20):
    print(f"📊 策略1v2扫描: {target_date}", flush=True)
    t0 = time.time()
    if not _price: preload()
    if not _sector_map: load_sector_map()

    hot_sectors = load_hot_sectors()
    print(f"   热点板块({len(hot_sectors)}个): {', '.join(list(hot_sectors.keys())[:5])}...", flush=True)

    codes = [f.stem.replace("_qfq", "") for f in QFQ_DIR.glob("*_qfq.csv")]
    print(f"   全市场 {len(codes)} 只\n", flush=True)

    # ── 第一层 ─────────────────────────────────────
    base_signals = []
    for code in codes:
        sig = check_base(code, target_date)
        if sig: base_signals.append(sig)
    print(f"�罒 第一层通过: {len(base_signals)} 只", flush=True)

    # ── 第二层 ─────────────────────────────────────
    final_signals = []
    for sig in base_signals:
        passed, score, reasons = score_stock(sig, hot_sectors)
        if passed:
            sig["score"] = score
            sig["reasons"] = reasons
            final_signals.append(sig)
    print(f"🏆 第二层通过: {len(final_signals)} 只", flush=True)

    final_signals.sort(key=lambda x: (-x["score"], -x["rsi"], -x["gain5d"]))
    names = load_stock_names()
    hot_sector_names = list(hot_sectors.keys())

    # ── 表格输出（与screen.py一致）───────────────────────
    print(f"\n{'='*120}", flush=True)
    print(f"📊 策略1-v2 {target_date}（第一层 {len(base_signals)} / 第二层 {len(final_signals)} 只）", flush=True)
    print("=" * 120, flush=True)
    print(_header_row(), flush=True)
    print("-" * 120, flush=True)

    txt_lines = [
        f"📊 策略1-v2 {target_date}（第一层 {len(base_signals)} / 第二层 {len(final_signals)} 只）",
        "=" * 120, _header_row(), "-" * 120,
    ]

    for s in final_signals[:top_n]:
        name = names.get(s["code"], s["code"])[:6]
        sector = get_sector(s["code"]) or "-"
        is_hot = "✅" if sector in hot_sector_names else "-"
        row = _make_row([
            s["code"], name, target_date,
            str(s["score"]),
            f"{s['close']:.2f}",
            f"{s['gain5d']:+.1f}%",
            f"{s['avg_turnover_5']:.1f}%",
            f"{s.get('market_cap', 0):.0f}",
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
    txt_path = Path.home() / "stock_reports" / f"strat1_v2_{target_date}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(txt_lines))
    print(f"💾 TXT已保存: {txt_path}", flush=True)

    # ── 保存JSON ──────────────────────────────────────
    out = {
        "date": target_date,
        "layer1_count": len(base_signals),
        "layer2_count": len(final_signals),
        "hot_sectors": hot_sectors,
        "signals": final_signals[:top_n],
    }
    out_path = Path.home() / "stock_reports" / f"strat1_v2_{target_date}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"💾 JSON已保存: {out_path}", flush=True)
    print(f"⏱  用时 {time.time()-t0:.1f}秒", flush=True)

# ── 主入口 ───────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="2026-04-24")
    parser.add_argument("--top-n", type=int, default=100)
    args = parser.parse_args()
    screen_strat1_v2(args.date, args.top_n)
