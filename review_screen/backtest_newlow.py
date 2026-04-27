#!/usr/bin/env python3
"""
N日新低反弹策略回测
====================

严格无未来数据：
  T日收盘后判断 → T+1开盘买入 → T+5收盘卖出

验证区间：
  训练集：2025-01-02 ~ 2025-09-30
  验证集：2025-10-01 ~ 2026-04-23

使用方法：
    python backtest_newlow.py
    python backtest_newlow.py --n 20
    python backtest_newlow.py --hold 5
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

_price = {}

def preload():
    global _price
    for f in QFQ_DIR.glob("*_qfq.csv"):
        code = f.stem.replace("_qfq", "")
        try:
            df = pd.read_csv(f, usecols=["date","open","close","high","low","volume","amount"])
            df = df.sort_values("date").reset_index(drop=True)
            _price[code] = df
        except:
            pass
    print(f"✅ {len(_price)}只股票已加载", flush=True)

def get_dates():
    for df in list(_price.values())[:1]:
        return sorted(df["date"].tolist())

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
        if 0 <= idx+offset < len(dates): return dates[idx+offset]
    except: pass
    return None


def check_signal(code, T, n_day_low):
    """检查T日是否满足N日新低反弹条件"""
    df = _price.get(code)
    if df is None: return None
    il = df["date"].tolist()
    try: idx = il.index(T)
    except: return None
    if idx < n_day_low: return None

    window = df.iloc[idx-n_day_low:idx+1]
    if len(window) < n_day_low+1: return None

    closes = window["close"].values
    T_pos = len(closes) - 1
    close_T = closes[T_pos]

    # 条件1: 收盘 = N日最低收盘
    min_close_N = float(np.min(closes[:-1]))
    if close_T != min_close_N: return None

    # 条件2: 阳线
    open_T = float(df.iloc[idx]["open"])
    if close_T <= open_T: return None

    # 条件3: 价格
    if not (3.0 <= close_T <= 200.0): return None

    # 条件4: 流动性
    if idx < 60: return None
    avg_amt = float(np.mean(df.iloc[idx-60:idx]["amount"].values))
    if avg_amt < 3_000_000: return None

    # 条件5: 涨跌停
    for off in [0, 1]:
        if idx-off < 1: continue
        pc = float(df.iloc[idx-off]["close"])
        ppc = float(df.iloc[idx-off-1]["close"])
        if ppc > 0 and abs((pc-ppc)/ppc*100) >= 9.7: return None

    return {"code": code}


def run_backtest(signal_dates, n_day_low=20, hold_days=5):
    """
    T日收盘后信号 → T+1开盘买入 → T+hold_days收盘卖出
    """
    results = []
    for T in signal_dates:
        for code in _price.keys():
            if not check_signal(code, T, n_day_low): continue

            # T+1开盘买入
            T1 = next_date(T, 1)
            if not T1: continue
            entry = get_price(code, T1)
            if not entry or entry["open"] <= 0: continue
            ep = entry["open"]

            # T+hold_days收盘卖出
            exit_price = None
            exit_date = None
            for d in range(1, hold_days + 1):
                ed = next_date(T1, d - 1)
                if not ed: break
                px = get_price(code, ed)
                if not px: break
                if d == hold_days:
                    exit_price = px["close"]
                    exit_date = ed

            if exit_price is None: continue
            pnl = (exit_price - ep) / ep * 100
            results.append({
                "code": code,
                "signal_date": T,
                "entry_date": T1,
                "exit_date": exit_date,
                "entry_price": round(ep, 3),
                "exit_price": round(exit_price, 3),
                "pnl_pct": round(pnl, 3),
            })

    return results


def analyze(results):
    if not results: return {}
    pnls = [r["pnl_pct"] for r in results]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    return {
        "total": len(results),
        "win_rate": round(len(wins)/len(results)*100, 1),
        "avg_pnl": round(float(np.mean(pnls)), 3),
        "median_pnl": round(float(np.median(pnls)), 3),
        "max_pnl": round(max(pnls), 2),
        "min_pnl": round(min(pnls), 2),
        "profit_factor": round(abs(sum(wins)/sum(losses)), 2) if losses else 99,
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
    parser.add_argument("--n", type=int, default=20, help="N日新低（默认20）")
    parser.add_argument("--hold", type=int, default=5, help="持有天数（默认5）")
    args = parser.parse_args()

    preload()
    dates = get_dates()
    all_dates = dates[65:]

    train_dates = [d for d in all_dates if "2025-01-01" <= d <= "2025-09-30"][::4]
    val_dates = [d for d in all_dates if d >= "2025-10-01"][::4]

    print(f"\n📍 训练集: {train_dates[0]}~{train_dates[-1]} ({len(train_dates)}批)")
    print(f"📍 验证集: {val_dates[0]}~{val_dates[-1]} ({len(val_dates)}批)")
    print(f"🔧 参数: N日新低={args.n}, 持有天数={args.hold}", flush=True)

    train = run_backtest(train_dates, n_day_low=args.n, hold_days=args.hold)
    val = run_backtest(val_dates, n_day_low=args.n, hold_days=args.hold)

    ts = analyze(train)
    vs = analyze(val)

    print_stats(ts, "训练集")
    print_stats(vs, "验证集")

    verdict = f"验证集胜率{vs['win_rate']}%，{'✅ 接近60%目标' if vs['win_rate'] >= 55 else '⚠️ 未达成60%'}"
    print(f"\n🏆 {verdict}")

    # 保存
    out = {
        "params": {"n_day_low": args.n, "hold_days": args.hold},
        "train_stats": ts,
        "val_stats": vs,
        "train_results": train,
        "val_results": val,
    }
    out_path = Path.home() / "stock_reports" / "newlow_backtest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n💾 已保存: {out_path}")
