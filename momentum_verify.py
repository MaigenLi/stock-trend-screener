#!/usr/bin/env python3
"""验证最优动量组合（完整样本）"""
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

with open(REPORTS_DIR / "backtest_t1_gains.json") as f:
    t1_gains = json.load(f)
sample_days = sorted(set(k.split("|")[0] for k in t1_gains.keys()))
print(f"T+1 数据: {len(t1_gains)} 条 | {len(sample_days)} 天")

codes = []
for f_path in CACHE_DIR.glob("*.csv"):
    pure = f_path.stem.replace("_qfq", "")
    if len(pure) == 6 and pure.isdigit():
        prefix = "sh" if pure.startswith(("60", "68", "90")) else "sz"
        codes.append(f"{prefix}{pure}")
print(f"全市场: {len(codes)} 只")

print("\n加载全量历史数据...")
code_dfs = {}
for code in codes:
    pure = code[-6:]
    fp = CACHE_DIR / f"{pure}_qfq.csv"
    if not fp.exists():
        continue
    try:
        df = pd.read_csv(fp)
        df["date"] = df["date"].astype(str).str[:10]
        df = df.sort_values("date").reset_index(drop=True)
        code_dfs[code] = df
    except:
        continue
print(f"已加载: {len(code_dfs)} 只")

PARAM_COMBOS = [
    (10.0, 100,  70, 80,  1.0, 1.3, 0.95),   # 最优组合
    (8.0,  100,  70, 80,  0.8, 1.5, 0.93),   # 放宽
    (8.0,  100,  65, 82,  0.8, 2.0, 0.90),   # 更宽
    (5.0,  100,  60, 85,  0.8, 2.5, 0.85),   # 最宽
    (10.0, 100,  60, 85,  0.8, 2.0, 0.90),   # 去掉RSI限制
    (8.0,  100,  50, 90,  0.5, 3.0, 0.85),   # 极端宽松
]

for params in PARAM_COMBOS:
    ret1_min, ret1_max, rsi_low, rsi_high, vol_min, vol_max, cp_min = params
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
            price_range = highs[idx] - lows[idx]
            if price_range <= 0:
                continue
            close_pos = (closes[idx] - lows[idx]) / price_range
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
            if g > 3.0:
                hits += 1; total_hit += 1
        if checked > 0:
            daily_stats.append({"date": day, "n": len(selected), "checked": checked, "hits": hits})

    hit_rate = total_hit / total * 100 if total > 0 else 0
    mark = "✅" if hit_rate >= 60 else "❌"
    print(f"\n{mark} ret1=[{ret1_min},{ret1_max}]% RSI=[{rsi_low},{rsi_high}] "
          f"vol=[{vol_min},{vol_max}]x cp>{cp_min:.0%}")
    print(f"   命中率: {hit_rate:.1f}% ({total_hit}/{total})")
    if daily_stats:
        rates = [d['hits']/d['checked']*100 for d in daily_stats if d['checked']>0]
        print(f"   日均命中: 均{sum(rates)/len(rates):.0f}% min={min(rates):.0f}% max={max(rates):.0f}%")
        print(f"   样本分布: {sum(d['n'] for d in daily_stats)}只/{len(daily_stats)}天")
