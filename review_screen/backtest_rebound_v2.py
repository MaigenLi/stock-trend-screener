#!/usr/bin/env python3
"""
8步超跌反弹策略回测
====================

严格无未来数据：
  T日收盘后判断 → T+1开盘买入 → T+5收盘卖出

验证区间：
  训练集：2025-01-02 ~ 2025-09-30
  验证集：2025-10-01 ~ 2026-04-23

使用方法：
    python backtest_rebound_v2.py
"""

import json
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

WORKSPACE = Path(__file__).parent.parent.parent.resolve()
CACHE_DIR = WORKSPACE / ".cache"
QFQ_DIR = CACHE_DIR / "qfq_daily"
MERGED_PATH = CACHE_DIR / "indicators_merged.json"

_price = {}
_merged = {}


def preload():
    global _price, _merged
    for f in QFQ_DIR.glob("*_qfq.csv"):
        code = f.stem.replace("_qfq", "")
        try:
            df = pd.read_csv(f, usecols=["date","open","close","high","low","volume","amount"])
            df = df.sort_values("date").reset_index(drop=True)
            _price[code] = df
        except:
            pass
    with open(MERGED_PATH) as f:
        _merged = json.load(f)
    print(f"✅ {len(_price)}只股票已加载", flush=True)


def get_dates():
    for df in list(_price.values())[:1]:
        return sorted(df["date"].tolist())
    return []


def get_price(code, date):
    df = _price.get(code)
    if df is None: return None
    r = df[df["date"] == date]
    if r.empty: return None
    r = r.iloc[0]
    return {"open":float(r["open"]),"close":float(r["close"]),
            "high":float(r["high"]),"low":float(r["low"])}


def next_date(date, offset=1):
    dates = get_dates()
    try:
        idx = dates.index(date)
        if 0 <= idx + offset < len(dates):
            return dates[idx + offset]
    except:
        pass
    return None


def calc_rsi(closes, period=9):
    if len(closes) < period + 1:
        return None
    deltas = np.diff(closes[-period - 1:])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = float(np.mean(gains))
    avg_loss = float(np.mean(losses))
    if avg_loss == 0:
        return 100.0
    return 100 - 100 / (1 + avg_gain / avg_loss)


def lower_shadow_ratio(row):
    low = float(row["low"])
    high = float(row["high"])
    close = float(row["close"])
    open_ = float(row["open"])
    total = high - low
    if total <= 0:
        return 0.0
    return max(0.0, min(1.0, (low - min(close, open_)) / total))


def analyze_stock_T(code, T, trend_filter=False):
    """完整8步检查"""
    df = _price.get(code)
    if df is None:
        return None
    il = df["date"].tolist()
    try:
        idx = il.index(T)
    except:
        return None
    if idx < 65:
        return None

    closes_all = df.iloc[idx - 65:idx + 1]["close"].values
    volumes_all = df.iloc[idx - 65:idx + 1]["volume"].values
    T_pos = len(closes_all) - 1
    close_T = closes_all[T_pos]

    # Step 1: 价格+流动性
    if not (3.0 <= close_T <= 150.0):
        return None
    avg_amount = float(np.mean(df.iloc[idx - 60:idx]["amount"].values))
    if avg_amount < 30_000_000:
        return None
    for off in [0, 1]:
        if idx - off < 1:
            continue
        pc = float(df.iloc[idx - off]["close"])
        ppc = float(df.iloc[idx - off - 1]["close"])
        if ppc > 0 and abs((pc - ppc) / ppc * 100) >= 9.7:
            return None

    # Step 2: 超卖（OR）
    rsi = calc_rsi(closes_all[:T_pos + 1], 9)
    cond_A = (rsi is not None and rsi < 40)
    consec = 0
    for i in range(T_pos - 1, T_pos - 4, -1):
        if i < 0:
            break
        if float(df.iloc[i]["close"]) < float(df.iloc[i - 1]["close"]):
            consec += 1
        else:
            break
    loss_3d = (close_T / closes_all[T_pos - 3] - 1) * 100 if T_pos >= 3 else 0.0
    cond_B = (consec >= 2 and loss_3d < -2.0)
    if not (cond_A or cond_B):
        return None

    vol_5d = float(np.mean(volumes_all[T_pos - 4:T_pos + 1]))
    vol_20d = float(np.mean(volumes_all[T_pos - 19:T_pos + 1]))
    if vol_20d <= 0 or vol_5d < vol_20d * 0.5:
        return None

    # Step 3: 横盘（AND）
    high_20d = float(np.max(closes_all[T_pos - 19:T_pos + 1]))
    low_20d = float(np.min(closes_all[T_pos - 19:T_pos + 1]))
    if low_20d <= 0:
        return None
    range_20d = (high_20d / low_20d - 1) * 100
    if range_20d >= 25.0:
        return None
    if vol_5d >= vol_20d * 0.90:
        return None

    # Step 4: 止跌（OR）
    ls_T1 = lower_shadow_ratio(df.iloc[idx - 1])
    cond_ls = ls_T1 > 0.40
    open_T = float(df.iloc[idx]["open"])
    cond_bullish = close_T > open_T
    low_5d_before = float(np.min(df.iloc[idx - 5:idx]["low"].values))
    cond_no_new_low = float(df.iloc[idx]["low"]) > low_5d_before
    if not (cond_ls or (cond_bullish and cond_no_new_low)):
        return None

    # Step 6: 趋势方向（可开关）
    if trend_filter:
        ma20 = float(np.mean(closes_all[T_pos - 19:T_pos + 1]))
        ma60 = float(np.mean(closes_all[T_pos - 59:T_pos + 1])) if T_pos >= 59 else None
        if ma60 is None or not (ma20 > ma60 > 0 and close_T > ma20):
            return None

    return {"code": code, "rsi": rsi, "consec": consec}


def run_backtest(signal_dates, hold_days=5, trend_filter=False):
    """
    T日收盘后信号 → T+1开盘买入 → T+hold_days收盘卖出
    """
    results = []
    for T in signal_dates:
        for code in _price.keys():
            if not analyze_stock_T(code, T, trend_filter):
                continue

            T1 = next_date(T, 1)
            if not T1:
                continue
            entry = get_price(code, T1)
            if not entry or entry["open"] <= 0:
                continue
            ep = entry["open"]

            exit_price = None
            for d in range(1, hold_days + 1):
                ed = next_date(T1, d - 1)
                if not ed:
                    break
                px = get_price(code, ed)
                if not px:
                    break
                if d == hold_days:
                    exit_price = px["close"]

            if exit_price is None:
                continue
            pnl = (exit_price - ep) / ep * 100
            results.append({
                "code": code, "signal_date": T,
                "entry_date": T1, "exit_date": next_date(T1, hold_days - 1),
                "entry_price": round(ep, 3), "exit_price": round(exit_price, 3),
                "pnl_pct": round(pnl, 3),
            })
    return results


def analyze(results):
    if not results:
        return {}
    pnls = [r["pnl_pct"] for r in results]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    return {
        "total": len(results),
        "win_rate": round(len(wins) / len(results) * 100, 1),
        "avg_pnl": round(float(np.mean(pnls)), 3),
        "median_pnl": round(float(np.median(pnls)), 3),
        "max_pnl": round(max(pnls), 2),
        "min_pnl": round(min(pnls), 2),
        "profit_factor": round(abs(sum(wins) / sum(losses)), 2) if losses else 99,
        "avg_win": round(float(np.mean(wins)), 3) if wins else 0,
        "avg_loss": round(float(np.mean(losses)), 3) if losses else 0,
    }


def print_stats(s, label=""):
    if not s:
        print(f"\n{'='*60}\n📊 {label} — 无数据\n{'='*60}")
        return
    print(f"\n{'='*60}")
    print(f"📊 {label}")
    print(f"{'='*60}")
    print(f"  总信号:   {s['total']}  胜率: {s['win_rate']}%  均值: {s['avg_pnl']:+.3f}%")
    print(f"  中位数:   {s['median_pnl']:+.3f}%  盈亏比: {s['profit_factor']}")
    print(f"  最大盈利: {s['max_pnl']:+.2f}%  最大亏损: {s['min_pnl']:+.2f}%")
    print(f"  均盈利:   {s['avg_win']:+.3f}%  均亏损: {s['avg_loss']:+.3f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hold", type=int, default=5, help="持有天数")
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--trend", dest="trend_filter", action="store_true", help="开启MA20>MA60趋势过滤")
    g.add_argument("--no-trend", dest="trend_filter", action="store_false", help="关闭趋势过滤（默认）")
    parser.set_defaults(trend_filter=False)
    args = parser.parse_args()

    preload()
    dates = get_dates()
    all_dates = dates[65:]

    train_dates = [d for d in all_dates if "2025-01-01" <= d <= "2025-09-30"][::4]
    val_dates = [d for d in all_dates if d >= "2025-10-01"][::4]

    print(f"\n📍 训练集: {train_dates[0]}~{train_dates[-1]} ({len(train_dates)}批)")
    print(f"📍 验证集: {val_dates[0]}~{val_dates[-1]} ({len(val_dates)}批)")
    print(f"🔧 趋势过滤: {'开启' if args.trend_filter else '关闭'}")

    train = run_backtest(train_dates, hold_days=args.hold, trend_filter=args.trend_filter)
    val = run_backtest(val_dates, hold_days=args.hold, trend_filter=args.trend_filter)

    ts = analyze(train)
    vs = analyze(val)

    print_stats(ts, "训练集")
    print_stats(vs, "验证集")

    wr = vs['win_rate']
    print(f"\n🏆 {'✅ 达成60%+目标！' if wr >= 60 else f'⚠️ 胜率{wr}%，未达成60%'}")

    out = {
        "train_stats": ts, "val_stats": vs,
        "train_results": train, "val_results": val,
    }
    out_path = Path.home() / "stock_reports" / "rebound8_backtest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n💾 已保存: {out_path}")
