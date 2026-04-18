#!/usr/bin/env python3
"""T+1>2% 动量策略回测"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

CACHE_DIR = Path.home() / ".openclaw/workspace/.cache/qfq_daily"
REPORTS_DIR = Path.home() / "stock_reports"

def calc_rsi(closes, period=14):
    if len(closes) < period + 1:
        return np.nan
    deltas = np.diff(closes, prepend=closes[0])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.convolve(gains, np.ones(period)/period, mode='same')
    avg_loss = np.convolve(losses, np.ones(period)/period, mode='same')
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = np.where(avg_loss == 0, 100, avg_gain / avg_loss)
        rsi = 100 - (100 / (1 + rs))
    return rsi[-1] if len(rsi) > 0 else np.nan

print("加载 T+1 数据...")
with open(REPORTS_DIR / "backtest_t1_gains.json") as f:
    t1_gains = json.load(f)
sample_days = sorted(set(k.split("|")[0] for k in t1_gains.keys()))
print(f"T+1: {len(t1_gains)} 条 | {len(sample_days)} 天")

codes = []
for fp in CACHE_DIR.glob("*.csv"):
    pure = fp.stem.replace("_qfq", "")
    if len(pure) == 6 and pure.isdigit():
        prefix = "sh" if pure.startswith(("60", "68", "90")) else "sz"
        codes.append(f"{prefix}{pure}")

print("加载全量历史...")
code_dfs = {}
for code in codes:
    pure = code[-6:]
    fp = CACHE_DIR / f"{pure}_qfq.csv"
    try:
        df = pd.read_csv(fp)
        df["date"] = df["date"].astype(str).str[:10]
        df = df.sort_values("date").reset_index(drop=True)
        code_dfs[code] = df
    except:
        continue
print(f"已加载: {len(code_dfs)} 只")

PARAM_COMBOS = [
    (10.0, 100,  70, 80,  1.0, 1.3, 0.95),
    (8.0,  100,  70, 80,  0.8, 1.5, 0.93),
    (8.0,  100,  65, 82,  0.8, 2.0, 0.90),
    (5.0,  100,  60, 85,  0.8, 2.5, 0.85),
    (10.0, 100,  60, 85,  0.8, 2.0, 0.90),
    (8.0,  100,  50, 90,  0.5, 3.0, 0.85),
    (5.0,  100,  50, 80,  0.8, 2.0, 0.90),
    (3.0,  100,  45, 80,  0.8, 2.0, 0.90),
]

THRESHOLD = 2.0
print(f"\n{'='*70}")
print(f"T+1 涨幅 >{THRESHOLD}% 命中率回测")
print(f"{'='*70}")

results = []
for ret1_min, ret1_max, rsi_low, rsi_high, vol_min, vol_max, cp_min in PARAM_COMBOS:
    total = 0; total_hit = 0; daily_stats = []

    for day in sample_days:
        selected = []
        for code, df in code_dfs.items():
            idx_arr = np.where(df["date"] == day)[0]
            if len(idx_arr) == 0:
                continue
            idx = idx_arr[0]
            if idx < 20 or idx >= len(df) - 1:
                continue
            closes = df["close"].values.astype(float)
            highs  = df["high"].values.astype(float)
            lows   = df["low"].values.astype(float)
            vols   = df["volume"].values.astype(float)

            ret1 = (closes[idx] / closes[idx-1] - 1) * 100
            if not (ret1_min <= ret1 <= ret1_max):
                continue
            rsi = calc_rsi(closes[:idx+1], 14)
            if np.isnan(rsi) or not (rsi_low <= rsi <= rsi_high):
                continue
            vol_ratio = vols[idx] / vols[idx-1] if vols[idx-1] > 0 else 1.0
            if not (vol_min <= vol_ratio <= vol_max):
                continue
            pr = highs[idx] - lows[idx]
            if pr <= 0:
                continue
            close_pos = (closes[idx] - lows[idx]) / pr
            if close_pos < cp_min:
                continue
            selected.append(code.lower())

        if not selected:
            continue
        checked = 0; hits = 0
        for code in selected:
            key = f"{day}|{code}"
            g = t1_gains.get(key)
            if g is None:
                continue
            checked += 1; total += 1
            if g > THRESHOLD:
                hits += 1; total_hit += 1
        if checked > 0:
            daily_stats.append({"date": day, "n": len(selected), "checked": checked, "hits": hits})

    hit_rate = total_hit / total * 100 if total > 0 else 0
    results.append({
        "ret1_min": ret1_min, "ret1_max": ret1_max,
        "rsi_low": rsi_low, "rsi_high": rsi_high,
        "vol_min": vol_min, "vol_max": vol_max, "cp_min": cp_min,
        "hit_rate": hit_rate, "total": total, "total_hit": total_hit, "daily_stats": daily_stats
    })
    mark = "✅" if hit_rate >= 60 else "❌"
    print(f"\n{mark} ret1=[{ret1_min},{ret1_max}]% RSI=[{rsi_low},{rsi_high}] "
          f"vol=[{vol_min},{vol_max}]x cp>{cp_min:.0%}")
    print(f"   命中率: {hit_rate:.1f}% ({total_hit}/{total})")
    if daily_stats:
        rates = [d['hits']/d['checked']*100 for d in daily_stats if d['checked']>0]
        print(f"   日均: 均{sum(rates)/len(rates):.0f}% min={min(rates):.0f}% max={max(rates):.0f}%")

results.sort(key=lambda x: -x["hit_rate"])
best = results[0]
mark = "✅" if best["hit_rate"] >= 60 else "❌"
print(f"\n{'='*70}")
print(f"🏆 最优组合（T+1>{THRESHOLD}%）：")
print(f"   {mark} 命中率: {best['hit_rate']:.1f}% ({best['total_hit']}/{best['total']})")
print(f"   ret1: {best['ret1_min']:.0f}%~{best['ret1_max']:.0f}% | RSI: [{best['rsi_low']},{best['rsi_high']}]")
print(f"   量比: [{best['vol_min']},{best['vol_max']}]x | 收盘位: >{best['cp_min']:.0%}")
print(f"{'='*70}")
