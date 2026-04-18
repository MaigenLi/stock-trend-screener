#!/usr/bin/env python3
"""
Momentum T+1 分析工具
直接分析已有缓存数据，找出哪些特征组合 → T+1>3%
"""
import json, sys, itertools
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from collections import defaultdict

CACHE_DIR = Path.home() / ".openclaw/workspace/.cache/qfq_daily"
REPORTS_DIR = Path.home() / "stock_reports"

def get_all_codes():
    codes = []
    for f in CACHE_DIR.glob("*.csv"):
        pure = f.stem.replace("_qfq", "")
        if len(pure) == 6 and pure.isdigit():
            prefix = "sh" if pure.startswith(("60", "68", "90")) else "sz"
            codes.append(f"{prefix}{pure}")
    return codes

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

def analyze_t1_characteristics():
    """
    对所有采样日，计算每只股票的 T+1 涨幅和各特征，
    然后分析特征组合的命中率。
    """
    # 加载已有的 backtest_t1_gains 缓存 (date|code → t1_gain)
    t1_path = REPORTS_DIR / "backtest_t1_gains.json"
    if not t1_path.exists():
        print("❌ backtest_t1_gains.json 不存在，需要先运行 backtest_fast.py 的 Phase 3")
        return

    with open(t1_path) as f:
        t1_gains = json.load(f)
    print(f"T+1 涨幅数据: {len(t1_gains)} 条")

    # 从 sample_days 获取采样日列表
    sample_days = sorted(set(k.split("|")[0] for k in t1_gains.keys()))
    print(f"采样日: {len(sample_days)} 天")

    # 加载 backtest_step1_cache 获取信号日数据
    step1_path = REPORTS_DIR / "backtest_step1_cache.json"
    with open(step1_path) as f:
        raw = json.load(f)
    step1 = {d: pd.DataFrame(v["data"], columns=v["cols"]) for d, v in raw.items()}
    print(f"Step1 数据: {len(step1)} 天")

    # 对每个采样日，加载每只股票的数据，计算特征
    # 由于要算 RSI，需要每只股票的历史数据
    # 策略：抽样分析，用 t1_gains 的 code
    all_codes = get_all_codes()
    code_set = set()
    for k in t1_gains:
        code = k.split("|")[1]
        code_set.add(code)

    print(f"T+1 数据涉及股票: {len(code_set)}")

    # 特征分析：对每只股票，计算信号日的特征
    records = []
    sample = list(code_set)[:2000]  # 限制分析范围

    for i, code in enumerate(sample):
        pure = code[-6:]
        f = CACHE_DIR / f"{pure}_qfq.csv"
        if not f.exists():
            continue
        try:
            df = pd.read_csv(f)
            df["date"] = df["date"].astype(str).str[:10]
            df = df.sort_values("date").reset_index(drop=True)
        except:
            continue

        # 找信号日
        for day in sample_days:
            key = f"{day}|{code}"
            t1 = t1_gains.get(key)
            if t1 is None:
                continue

            idx_arr = np.where(df["date"] == day)[0]
            if len(idx_arr) == 0:
                continue
            idx = idx_arr[0]
            if idx < 1 or idx >= len(df) - 1:
                continue

            closes = df["close"].values.astype(float)
            highs  = df["high"].values.astype(float)
            lows   = df["low"].values.astype(float)
            vols   = df["volume"].values.astype(float)

            # 特征计算
            ret1 = (closes[idx] / closes[idx-1] - 1) * 100  # 昨日涨幅
            if idx < 20:
                continue
            rsi = calc_rsi(closes[:idx+1], 14)
            if np.isnan(rsi):
                rsi = 50.0

            vol_ratio = vols[idx] / vols[idx-1] if vols[idx-1] > 0 else 1.0
            high_today = highs[idx]
            low_today  = lows[idx]
            price_range = high_today - low_today
            close_pos = (closes[idx] - low_today) / price_range if price_range > 0 else 0.5

            # MA20
            ma20_arr = pd.Series(closes[:idx+1]).rolling(20).mean().values
            ma20 = ma20_arr[-1] if not np.isnan(ma20_arr[-1]) else closes[idx]
            ma20_above = closes[idx] > ma20

            # MA5 上穿 MA20 确认趋势
            ma5_arr = pd.Series(closes[:idx+1]).rolling(5).mean().values
            ma5 = ma5_arr[-1]
            ma10_arr = pd.Series(closes[:idx+1]).rolling(10).mean().values
            ma10 = ma10_arr[-1]
            ma_trend = ma5 > ma10  # 短期多头

            records.append({
                "code": code,
                "date": day,
                "t1_gain": t1,
                "ret1": ret1,
                "rsi": rsi,
                "vol_ratio": vol_ratio,
                "close_position": close_pos,
                "ma20_above": ma20_above,
                "ma_trend": ma_trend,
            })

        if (i + 1) % 500 == 0:
            print(f"  已分析 {i+1}/{len(sample)} 只股票 ...")

    df_all = pd.DataFrame(records)
    print(f"\n总记录: {len(df_all)} 条")
    print(f"T+1>3% 数量: {(df_all['t1_gain'] > 3).sum()} ({(df_all['t1_gain'] > 3).mean()*100:.1f}%)")

    # ── 特征组合分析 ────────────────────────────────────
    print("\n" + "="*60)
    print("特征组合命中率分析")
    print("="*60)

    # ret1 分档
    bins_ret1 = [-100, 1, 2, 3, 5, 8, 10, 100]
    labels_ret1 = ["<-1%", "1~2%", "2~3%", "3~5%", "5~8%", "8~10%", ">10%"]
    df_all["ret1_bin"] = pd.cut(df_all["ret1"], bins=bins_ret1, labels=labels_ret1, right=False)

    # RSI 分档
    bins_rsi = [0, 40, 50, 60, 70, 80, 100]
    labels_rsi = ["<40", "40~50", "50~60", "60~70", "70~80", ">80"]
    df_all["rsi_bin"] = pd.cut(df_all["rsi"], bins=bins_rsi, labels=labels_rsi, right=False)

    # vol_ratio 分档
    bins_vol = [0, 1.0, 1.3, 1.5, 2.0, 100]
    labels_vol = ["<1.0x", "1.0~1.3x", "1.3~1.5x", "1.5~2.0x", ">2.0x"]
    df_all["vol_bin"] = pd.cut(df_all["vol_ratio"], bins=bins_vol, labels=labels_vol, right=False)

    # close_position 分档
    df_all["cp_bin"] = pd.cut(df_all["close_position"], bins=[0, 0.7, 0.85, 0.95, 1.01], labels=["<70%", "70~85%", "85~95%", ">95%"], right=False)

    # 分析每个特征的命中率
    for feature in ["ret1_bin", "rsi_bin", "vol_bin", "cp_bin", "ma20_above", "ma_trend"]:
        print(f"\n─── {feature} ───")
        grouped = df_all.groupby(feature, observed=True).agg(
            total=("t1_gain", "count"),
            hits=("t1_gain", lambda x: (x > 3).sum()),
            avg_t1=("t1_gain", "mean"),
        )
        grouped["hit_rate"] = grouped["hits"] / grouped["total"] * 100
        grouped = grouped.sort_values("hit_rate", ascending=False)
        for idx, row in grouped.iterrows():
            print(f"  {str(idx):<15} n={row['total']:5.0f} 命中率={row['hit_rate']:5.1f}%  均T+1={row['avg_t1']:+.2f}%")

    # ── 组合分析：ret1_bin × rsi_bin ────────────────────
    print("\n─── ret1_bin × rsi_bin 组合命中率 ───")
    pivot = df_all.pivot_table(
        values="t1_gain",
        index="ret1_bin",
        columns="rsi_bin",
        aggfunc=lambda x: f"{(x>3).mean()*100:.1f}%"
    )
    print(pivot.to_string())

    # 找最优组合
    print("\n─── 最优特征组合（命中率）───")
    combo_results = []
    for ret1_val in labels_ret1:
        for rsi_val in labels_rsi:
            for vol_val in labels_vol:
                for cp_val in ["<70%", "70~85%", "85~95%", ">95%"]:
                    subset = df_all[
                        (df_all["ret1_bin"] == ret1_val) &
                        (df_all["rsi_bin"] == rsi_val) &
                        (df_all["vol_bin"] == vol_val) &
                        (df_all["cp_bin"] == cp_val)
                    ]
                    if len(subset) < 20:
                        continue
                    hit_rate = (subset["t1_gain"] > 3).mean() * 100
                    combo_results.append({
                        "ret1": ret1_val,
                        "rsi": rsi_val,
                        "vol": vol_val,
                        "close_pos": cp_val,
                        "n": len(subset),
                        "hit_rate": hit_rate,
                    })

    combo_results.sort(key=lambda x: -x["hit_rate"])
    for r in combo_results[:15]:
        mark = "✅" if r["hit_rate"] >= 60 else "❌"
        print(f"  {mark} {r['hit_rate']:.1f}% ({r['n']}只) | ret1={r['ret1']} RSI={r['rsi']} vol={r['vol']} close_pos={r['close_pos']}")

if __name__ == "__main__":
    print("="*60)
    print("📊 T+1 涨幅特征分析")
    print("="*60)
    analyze_t1_characteristics()
