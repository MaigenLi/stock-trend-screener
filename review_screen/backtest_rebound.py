#!/usr/bin/env python3
"""
超跌反弹策略回测
================

完全无未来数据：
  T日收盘后扫描 → T+1开盘价买入 → T+5收盘卖出

验证区间：
  训练集：2025-01-02 ~ 2025-09-30
  验证集：2025-10-01 ~ 2026-04-23

使用方法：
    python backtest_rebound.py
    python backtest_rebound.py --params '{"rsi_max":30,"consec_down_min":4,"loss5d_min":5}'
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
    dates = sorted(_merged.keys())
    print(f"✅ {len(dates)}天 {dates[0]}~{dates[-1]} {len(_price)}只", flush=True)
    return dates

def get_dates():
    return sorted(_merged.keys())

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

def calc_rsi(closes, period=5):
    if len(closes) < period + 1: return None
    deltas = np.diff(closes[-period-1:])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = float(np.mean(gains))
    avg_loss = float(np.mean(losses))
    if avg_loss == 0: return 100.0
    return 100 - 100 / (1 + avg_gain / avg_loss)

def check_consecutive_down(code, T, n=4):
    df = _price.get(code)
    if df is None: return 0
    il = df["date"].tolist()
    try: idx = il.index(T)
    except: return 0
    if idx < n: return 0
    count = 0
    for i in range(idx-n+1, idx+1):
        c_prev = float(df.iloc[i-1]["close"])
        c_curr = float(df.iloc[i]["close"])
        if c_curr < c_prev: count += 1
        else: break
    return count

def lower_shadow_ratio(row):
    low = float(row["low"]); high = float(row["high"])
    close = float(row["close"]); open_ = float(row["open"])
    total = high - low
    if total <= 0: return 0.0
    lower = (low - min(close, open_)) / total
    return max(0.0, min(1.0, lower))


# ── 核心分析函数 ──────────────────────────────────────────
def analyze_stock_T(code, T, p):
    """
    全部只用T日及之前数据，返回是否发出信号
    不偷任何T+1的数据
    """
    df = _price.get(code)
    if df is None: return None
    il = df["date"].tolist()
    try: idx = il.index(T)
    except: return None
    if idx < 65: return None

    closes = df.iloc[idx-65:idx+1]["close"].values
    volumes = df.iloc[idx-65:idx+1]["volume"].values
    T_pos = len(closes) - 1

    close_T = closes[T_pos]
    if close_T < p.get("min_price", 3.0): return None
    if close_T > p.get("max_price", 150.0): return None

    # 60日均成交额
    avg_amt = float(np.mean(df.iloc[idx-60:idx]["amount"].values))
    if avg_amt < p.get("min_avg_amount", 5_000_000): return None

    # 涨跌停过滤（T-1和T）
    for off in [0, 1]:
        if idx - off < 1: continue
        pc = float(df.iloc[idx-off]["close"])
        ppc = float(df.iloc[idx-off-1]["close"])
        if ppc > 0:
            chg = (pc - ppc) / ppc * 100
            if abs(chg) >= 9.7: return None

    # RSI(5) < threshold
    rsi = calc_rsi(closes[:T_pos+1], p.get("rsi_period", 5))
    if rsi is None or rsi >= p.get("rsi_max", 30): return None

    # 连续下跌 >= n天
    consec = check_consecutive_down(code, T, p.get("consec_down_min", 4))
    if consec < p.get("consec_down_min", 4): return None

    # 近5日跌幅
    if T_pos < 5: return None
    loss5d = (close_T / closes[T_pos-5] - 1) * 100
    if loss5d >= 0 or abs(loss5d) < p.get("loss5d_min", 5.0): return None

    # 近5日均量 > 近20日均量的60%（有人承接）
    vol_5d = float(np.mean(volumes[T_pos-4:T_pos+1]))
    vol_20d = float(np.mean(volumes[T_pos-19:T_pos+1]))
    if vol_20d <= 0 or vol_5d < vol_20d * 0.6: return None

    # 20日振幅 < threshold
    high20 = float(np.max(closes[T_pos-19:T_pos+1]))
    low20 = float(np.min(closes[T_pos-19:T_pos+1]))
    if low20 <= 0: return None
    range20 = (high20 / low20 - 1) * 100
    if range20 >= p.get("range_20d_max", 20.0): return None

    # 近5日缩量（vs 20日）
    if vol_20d <= 0 or vol_5d >= vol_20d * p.get("vol缩量_max", 0.8): return None

    # T日温和放量 vs 近5日均量（不含T）
    vol_5d_before = float(np.mean(volumes[T_pos-4:T_pos]))
    vol_T = volumes[T_pos]
    if vol_5d_before <= 0: return None
    vol_T_ratio = vol_T / vol_5d_before
    if vol_T_ratio < p.get("vol_T_min", 1.5): return None

    # T-1下影线 > threshold
    if idx < 1: return None
    ls_T1 = lower_shadow_ratio(df.iloc[idx-1])
    if ls_T1 < p.get("lower_shadow_min", 0.6): return None

    # T日阳线
    open_T = float(df.iloc[idx]["open"])
    close_T_val = float(df.iloc[idx]["close"])
    if close_T_val <= open_T: return None

    # T日不创新低
    low_5d_before = float(np.min(df.iloc[idx-5:idx]["low"].values))
    if float(df.iloc[idx]["low"]) <= low_5d_before: return None

    # 趋势：MA20 > MA60
    ma20 = float(np.mean(closes[T_pos-19:T_pos+1]))
    ma60 = float(np.mean(closes[T_pos-59:T_pos+1])) if T_pos >= 59 else None
    if ma60 is None or not (ma20 > ma60 > 0): return None

    return {
        "code": code, "signal_date": T,
        "rsi": rsi, "consec_down": consec,
        "loss5d": loss5d, "range_20d": range20,
        "vol_T_ratio": vol_T_ratio,
    }


def run_backtest(signal_dates, p, hold_days=5):
    """
    T日收盘后信号 → T+1开盘买入 → T+hold_days收盘卖出
    """
    results = []
    for T in signal_dates:
        for code in _price.keys():
            sig = analyze_stock_T(code, T, p)
            if not sig: continue

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
                "code": code, "signal_date": T,
                "entry_date": T1, "entry_price": round(ep, 3),
                "exit_date": exit_date, "exit_price": round(exit_price, 3),
                "pnl_pct": round(pnl, 3),
                **{k: v for k, v in sig.items() if k not in ("code", "signal_date")},
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
        "avg": round(float(np.mean(pnls)), 3),
        "median": round(float(np.median(pnls)), 3),
        "max": round(max(pnls), 2),
        "min": round(min(pnls), 2),
        "pf": round(abs(sum(wins)/sum(losses)), 2) if losses else 99,
        "avg_win": round(float(np.mean(wins)), 3) if wins else 0,
        "avg_loss": round(float(np.mean(losses)), 3) if losses else 0,
    }


def print_stats(s, label=""):
    if not s:
        print(f"\n{'='*60}\n📊 {label} — 无数据\n{'='*60}")
        return
    print(f"\n{'='*60}\n📊 {label}\n{'='*60}")
    print(f"  总信号:   {s['total']}  胜率: {s['win_rate']}%  均值: {s['avg']:+.3f}%")
    print(f"  中位数:   {s['median']:+.3f}%  盈亏比: {s['pf']}")
    print(f"  最大盈利: {s['max']:+.2f}%  最大亏损: {s['min']:+.2f}%")
    print(f"  均盈利:   {s['avg_win']:+.3f}%  均亏损: {s['avg_loss']:+.3f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", type=str, default=None)
    args = parser.parse_args()

    default_p = {
        "min_price": 3.0, "max_price": 150.0,
        "min_avg_amount": 5_000_000,
        "rsi_period": 5, "rsi_max": 30,
        "consec_down_min": 4, "loss5d_min": 5.0,
        "range_20d_max": 20.0,
        "vol缩量_max": 0.8, "vol_T_min": 1.5,
        "lower_shadow_min": 0.6,
        "hold_days": 5,
    }

    if args.params:
        try:
            user = json.loads(args.params)
            default_p.update(user)
        except:
            print("❌ JSON格式错误"); sys.exit(1)

    p = default_p
    print(f"🔧 参数: {json.dumps(p, indent=2)}", flush=True)

    dates = preload()
    all_dates = dates[65:]

    train_dates = [d for d in all_dates if "2025-01-01" <= d <= "2025-09-30"][::4]
    val_dates = [d for d in all_dates if d >= "2025-10-01"][::4]

    print(f"\n📍 训练集: {train_dates[0]}~{train_dates[-1]} ({len(train_dates)}批)")
    print(f"📍 验证集: {val_dates[0]}~{val_dates[-1]} ({len(val_dates)}批)")

    train = run_backtest(train_dates, p, hold_days=p["hold_days"])
    val = run_backtest(val_dates, p, hold_days=p["hold_days"])

    ts = analyze(train); vs = analyze(val)
    print_stats(ts, "训练集")
    print_stats(vs, "验证集")

    print(f"\n🏆 {'✅ 达成60%+目标！' if vs['win_rate'] >= 60 else f'⚠️ 胜率{vs['win_rate']}%，未达成60%'}")

    # 保存
    out = {
        "params": p,
        "train_stats": ts, "val_stats": vs,
        "train_results": train, "val_results": val,
    }
    out_path = Path.home() / "stock_reports" / "rebound_backtest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n💾 已保存: {out_path}")
