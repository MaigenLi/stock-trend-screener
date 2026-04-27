#!/usr/bin/env python3
"""
横盘突破策略回测 v2（修正版）
==============================

核心逻辑：
  T日：横盘缩量整理
  T+1：放量突破（量>2x近5日均量，突破T日前20日高点）
  
验证区间（样本外）：
  训练集：2025-01-02 ~ 2025-09-30
  验证集：2025-10-01 ~ 2026-04-23

使用方法：
  python backtest_breakout.py
  python backtest_breakout.py --params '{"vol_ratio_th":2.0,"gain20_min":10,"ma_filter":true}'
"""

import json
import sys
import argparse
from pathlib import Path
from datetime import datetime

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
            df = pd.read_csv(f, usecols=["date","open","close","high","low","volume"])
            df = df.sort_values("date").reset_index(drop=True)
            _price[code] = df
        except:
            pass
    with open(MERGED_PATH) as f:
        _merged = json.load(f)
    dates = sorted(_merged.keys())
    print(f"✅ {len(dates)}天 {dates[0]}~{dates[-1]}  {len(_price)}只", flush=True)

def get_dates():
    return sorted(_merged.keys())

def get_price(code, date):
    df = _price.get(code)
    if df is None: return None
    r = df[df["date"] == date]
    if r.empty: return None
    r = r.iloc[0]
    return {"open":float(r["open"]),"close":float(r["close"]),"high":float(r["high"])}

def next_date(date, offset=1):
    dates = get_dates()
    try:
        idx = dates.index(date)
        if 0 <= idx+offset < len(dates): return dates[idx+offset]
    except: pass
    return None

def vol_ratio_vs_recent(code, T1, n=5):
    df = _price.get(code)
    if df is None: return None
    il = df["date"].tolist()
    try: idx = il.index(T1)
    except: return None
    if idx < n+1: return None
    vol_t1 = float(df.iloc[idx]["volume"])
    vol_recent = float(np.mean(df.iloc[idx-n:idx]["volume"].values))
    return vol_t1 / vol_recent if vol_recent > 0 else None

def prev_high_T(code, T):
    df = _price.get(code)
    if df is None: return None
    il = df["date"].tolist()
    try: idx = il.index(T)
    except: return None
    if idx < 21: return None
    return float(df.iloc[idx-20:idx]["high"].max())

def gain20_at_T(code, T):
    df = _price.get(code)
    if df is None: return None
    il = df["date"].tolist()
    try: idx = il.index(T)
    except: return None
    if idx < 20: return None
    c_now = float(df.iloc[idx]["close"])
    c_20 = float(df.iloc[idx-20]["close"])
    return (c_now/c_20 - 1)*100 if c_20 > 0 else 0

def ma20_above_ma60_at_T(code, T):
    df = _price.get(code)
    if df is None: return False
    il = df["date"].tolist()
    try: idx = il.index(T)
    except: return False
    if idx < 59: return False
    ma20 = float(df.iloc[idx-19:idx+1]["close"].mean())
    ma60 = float(df.iloc[idx-59:idx+1]["close"].mean())
    return bool(ma20 > ma60 > 0)


def run_backtest(signal_dates, params):
    """
    参数：
        vol_ratio_th: T+1量/近N日均量阈值
        gain20_min: 20日涨幅门槛
        ma_filter: 是否要求MA20>MA60
        hold_days: 持有天数
        break_th: 突破前高阈值（收盘/前高 > break_th）
        vol_avg_n: 计算均量的天数
    """
    vr_th = params.get("vol_ratio_th", 2.0)
    gain_min = params.get("gain20_min", 10.0)
    ma_filter = params.get("ma_filter", True)
    hold = params.get("hold_days", 5)
    break_th = params.get("break_th", 0.98)
    vol_n = params.get("vol_avg_n", 5)

    results = []
    for sd in signal_dates:
        for code in _price.keys():
            # T+1日期
            T1 = next_date(sd, 1)
            if not T1: continue

            # 趋势过滤
            gain20 = gain20_at_T(code, T1)
            if gain_min > 0 and (gain20 is None or gain20 < gain_min): continue
            if ma_filter and not ma20_above_ma60_at_T(code, T1): continue

            # 放量过滤
            vr = vol_ratio_vs_recent(code, T1, n=vol_n)
            if vr is None or vr < vr_th: continue

            # 突破前高
            ph = prev_high_T(code, T1)
            if ph is None or ph <= 0: continue

            p = get_price(code, T1)
            if not p: continue
            if p["close"] <= ph * break_th: continue
            if not (p["close"] > p["open"]): continue  # 阳线

            ep = p["open"]
            if ep <= 0: continue

            # 持有并卖出
            exit_price = None
            for d in range(1, hold + 1):
                ed = next_date(sd, d)
                if not ed: break
                px = get_price(code, ed)
                if not px: break
                if d == hold: exit_price = px["close"]

            if exit_price is None: continue
            pnl = (exit_price - ep) / ep * 100
            results.append({
                "code": code, "signal_date": sd,
                "entry_date": T1, "exit_date": next_date(sd, hold),
                "entry_price": round(ep, 3), "exit_price": round(exit_price, 3),
                "pnl_pct": round(pnl, 3),
                "vol_ratio": round(vr, 2), "gain20": round(gain20, 1),
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


def print_stats(stats, label=""):
    if not stats:
        print(f"\n{'='*60}\n📊 {label} — 无数据\n{'='*60}")
        return
    print(f"\n{'='*60}")
    print(f"📊 {label}")
    print(f"{'='*60}")
    print(f"  总信号:   {stats['total']}  胜率: {stats['win_rate']}%  均值: {stats['avg_pnl']:+.3f}%")
    print(f"  中位数:   {stats['median_pnl']:+.3f}%  盈亏比: {stats['profit_factor']}")
    print(f"  最大盈利: {stats['max_pnl']:+.2f}%  最大亏损: {stats['min_pnl']:+.2f}%")
    print(f"  均盈利:   {stats['avg_win']:+.3f}%  均亏损: {stats['avg_loss']:+.3f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", type=str, default=None, help='JSON参数如 \'{"vol_ratio_th":2.0}\'')
    args = parser.parse_args()

    default_params = {
        "vol_ratio_th": 2.0,
        "gain20_min": 10.0,
        "ma_filter": True,
        "hold_days": 5,
        "break_th": 0.98,
        "vol_avg_n": 5,
    }

    if args.params:
        try:
            user_params = json.loads(args.params)
            default_params.update(user_params)
        except:
            print("❌ 参数JSON格式错误")
            sys.exit(1)

    params = default_params
    print(f"🔧 参数: {params}", flush=True)

    preload()
    dates = get_dates()
    all_dates = dates[60:]

    train_dates = [d for d in all_dates if "2025-01-01" <= d <= "2025-09-30"][::4]
    val_dates = [d for d in all_dates if d >= "2025-10-01"][::4]

    print(f"\n📍 训练集: {train_dates[0]}~{train_dates[-1]} ({len(train_dates)}批)")
    print(f"📍 验证集: {val_dates[0]}~{val_dates[-1]} ({len(val_dates)}批)")

    train_results = run_backtest(train_dates, params)
    val_results = run_backtest(val_dates, params)

    train_stats = analyze(train_results)
    val_stats = analyze(val_results)

    print_stats(train_stats, "训练集")
    print_stats(val_stats, "验证集")

    verdict = "✅ 达成60%+目标！" if val_stats["win_rate"] >= 60 else "⚠️ 未达成60%"
    print(f"\n🏆 {verdict}")

    # 保存
    out = {
        "params": params,
        "train_stats": train_stats,
        "val_stats": val_stats,
        "train_results": train_results,
        "val_results": val_results,
    }
    out_path = Path.home() / "stock_reports" / "breakout_backtest_v2.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n💾 已保存: {out_path}")
