#!/usr/bin/env python3
"""
策略参数优化器
==============

在训练集上遍历参数组合，找到胜率≥60%的最优参数，
然后在验证集上验证。

优化目标：胜率 ≥ 60%

关键参数空间：
- RSI上限: 70/75/80
- 20日涨幅上限: 20/25/30/35（排除过涨）
- 止损线: 3/5/7%
- 趋势三条件阈值
- TOP N: 3
"""

import json, sys, itertools
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import numpy as np

WORKSPACE = Path(__file__).parent.parent.parent.resolve()
CACHE_DIR = WORKSPACE / ".cache"
MERGED_PATH = CACHE_DIR / "indicators_merged.json"
QFQ_DIR = CACHE_DIR / "qfq_daily"

# ── 数据加载（复用backtest_v2的逻辑）────────────────────────
_merged_index = {}
_price_cache = {}

def preload():
    global _merged_index, _price_cache
    print("  📂 加载合并指标索引...", flush=True)
    with open(MERGED_PATH) as f:
        _merged_index = json.load(f)
    dates = sorted(_merged_index.keys())
    print(f"   ✅ {len(dates)}天 {dates[0]}~{dates[-1]}", flush=True)
    print("  📂 加载价格数据...", flush=True)
    import pandas as pd
    for csv_path in QFQ_DIR.glob("*_qfq.csv"):
        code = csv_path.stem.replace("_qfq", "")
        try:
            df = pd.read_csv(csv_path, usecols=["date","open","high","low","close","volume"])
            df = df.sort_values("date").reset_index(drop=True)
            _price_cache[code] = df
        except:
            pass
    print(f"   ✅ {len(_price_cache)}只", flush=True)

def get_indicators(code, date):
    return _merged_index.get(date, {}).get(code)

def get_price_on_date(code, date):
    df = _price_cache.get(code)
    if df is None: return None
    row = df[df["date"] == date]
    if row.empty: return None
    r = row.iloc[0]
    return {"open":float(r["open"]),"high":float(r["high"]),"low":float(r["low"]),"close":float(r["close"])}

def get_next_date(date, offset=1):
    dates = sorted(_merged_index.keys())
    try:
        idx = dates.index(date)
        tidx = idx + offset
        if 0 <= tidx < len(dates): return dates[tidx]
    except: pass
    return None

def get_trading_dates(start, end):
    dates = sorted(_merged_index.keys())
    return [d for d in dates if start <= d <= end]

# ── 可调参数 ─────────────────────────────────────────────
DEFAULT_PARAMS = {
    "rsi_max": 80,
    "gain20_min": 13.8,
    "gain20_max": 50.0,     # 新增：排除过涨
    "wave_quality_min": 4.0,
    "ma20_60_sep_min": 3.5,
    "min_gain1": 0.3,
    "max_gain1": 8.0,
    "max_ma5_dist": 8.0,
    "stop_loss_pct": 8.0,
    "trend_score_min": 0,    # 新增：趋势评分门槛
    "use_rsi_filter": True,
    "use_gain20_max": False,
    "use_consecutive_up": False,  # 新增：近3日不能连续大涨
}

def apply_trend_filter(ind, params):
    if ind.get("gain20", 0) < params["gain20_min"]: return False
    if ind.get("wave_quality", 0) < params["wave_quality_min"]: return False
    if ind.get("ma_sep", 0) < params["ma20_60_sep_min"]: return False
    if params["use_gain20_max"] and ind.get("gain20", 0) > params["gain20_max"]: return False
    return True

def apply_buy_ready(ind, params):
    g = ind.get("gain1", 0)
    rsi = ind.get("rsi", 50)
    ma5d = ind.get("ma5_distance_pct", 0)
    if g >= 9.5: return False, "涨停"
    if g < params["min_gain1"]: return False, f"涨幅弱"
    if g > params["max_gain1"]: return False, f"涨幅过高"
    if params["use_rsi_filter"] and rsi >= params["rsi_max"]: return False, f"RSI过热"
    if abs(ma5d) > params["max_ma5_dist"]: return False, f"偏离过大"
    return True, "买入就绪"

def compute_score(ind):
    s = 50.0
    s += min(ind.get("gain20", 0) * 0.5, 20)
    s += min(ind.get("wave_quality", 0) * 2, 15)
    s += min(ind.get("ma_sep", 0) * 1.5, 10)
    rsi = ind.get("rsi", 50)
    if rsi > 85: s -= 5
    elif rsi > 80: s -= 2
    if ind.get("vol_ratio", 1) < 0.70: s -= 3
    if ind.get("vol_up_vs_down", 1) < 0.90: s -= 2
    return s

def run_backtest(start, end, params, hold_days=5, skip_interval=4, top_n=3):
    warmup = 60
    all_dates = sorted(_merged_index.keys())
    all_dates = all_dates[warmup:]
    signal_dates = [d for d in all_dates if d >= start][::skip_interval]
    if not signal_dates: return []

    all_results = []
    for sd in signal_dates:
        day_data = _merged_index.get(sd, {})
        candidates = []
        for code, ind in day_data.items():
            if not apply_trend_filter(ind, params): continue
            ready, _ = apply_buy_ready(ind, params)
            if not ready: continue
            entry_date = get_next_date(sd, 1)
            if not entry_date: continue
            entry = get_price_on_date(code, entry_date)
            if not entry or entry["open"] <= 0: continue

            entry_price = entry["open"]
            stop_ref = ind.get("stop_loss_ref", entry_price * 0.97)
            exit_price = None
            hold_actual = hold_days
            stopped = False

            for d in range(1, hold_days + 1):
                ex_dt = get_next_date(sd, d)
                if not ex_dt: break
                price = get_price_on_date(code, ex_dt)
                if not price: break
                sl = 1 - params["stop_loss_pct"] / 100
                if price["low"] <= stop_ref * sl:
                    exit_price = stop_ref * sl
                    hold_actual = d
                    stopped = True
                    break
                if d == hold_days:
                    exit_price = price["close"]

            if exit_price is None: continue

            pnl_pct = (exit_price - entry_price) / entry_price * 100
            score = compute_score(ind)
            candidates.append({
                "code": code, "signal_date": sd,
                "entry_price": round(entry_price, 3),
                "pnl_pct": round(pnl_pct, 3),
                "stopped": stopped,
                "signal_score": round(score, 1),
                "hold_days_actual": hold_actual,
                "gain20": ind.get("gain20", 0),
                "rsi": ind.get("rsi", 0),
            })

        # TOP N
        candidates.sort(key=lambda x: x["signal_score"], reverse=True)
        all_results.extend(candidates[:top_n])

    return all_results

def analyze(results):
    if not results: return {}
    pnls = [r["pnl_pct"] for r in results]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    stopped = [r for r in results if r["stopped"]]
    not_stopped = [r for r in results if not r["stopped"]]
    return {
        "total": len(results),
        "win_rate": round(len(wins)/len(results)*100, 1),
        "avg_pnl": round(float(np.mean(pnls)), 3),
        "median_pnl": round(float(np.median(pnls)), 3),
        "std_pnl": round(float(np.std(pnls)), 2),
        "max_pnl": round(max(pnls), 2),
        "min_pnl": round(min(pnls), 2),
        "profit_factor": round(abs(np.sum(wins)/np.sum(losses)), 2) if losses else 99.99,
        "stopped_n": len(stopped),
        "stopped_avg": round(float(np.mean([r["pnl_pct"] for r in stopped])), 3) if stopped else 0,
        "not_stopped_n": len(not_stopped),
        "not_stopped_avg": round(float(np.mean([r["pnl_pct"] for r in not_stopped])), 3) if not_stopped else 0,
        "win_n": len(wins),
        "loss_n": len(losses),
        "avg_win": round(float(np.mean(wins)), 3) if wins else 0,
        "avg_loss": round(float(np.mean(losses)), 3) if losses else 0,
    }

def print_stats(stats, label=""):
    print(f"\n{'='*60}")
    print(f"📊 {label}")
    print(f"{'='*60}")
    print(f"  总信号: {stats['total']}  胜率: {stats['win_rate']}%  均值: {stats['avg_pnl']:+.3f}%")
    print(f"  中位数: {stats['median_pnl']:+.3f}%  盈亏比: {stats['profit_factor']}")
    print(f"  最大盈利: {stats['max_pnl']:+.2f}%  最大亏损: {stats['min_pnl']:+.2f}%")
    print(f"  止损: {stats['stopped_n']}次({stats['stopped_avg']:+.3f}%)  非止损: {stats['not_stopped_n']}次({stats['not_stopped_avg']:+.3f}%)")

# ── 网格搜索 ─────────────────────────────────────────────
def grid_search():
    print("\n🚀 开始网格搜索...", flush=True)

    # 第一轮：粗搜索
    print("\n📍 第1轮：粗搜索", flush=True)
    candidates = []

    # 关键参数组合
    rsi_values = [65, 70, 75]
    gain20_max_values = [30, 40, 50]
    stop_loss_values = [5, 7]
    gain20_min_values = [10, 14, 18]
    wave_quality_values = [3, 5, 7]

    count = 0
    for rsi_max in rsi_values:
        for gain20_max in gain20_max_values:
            for stop_loss in stop_loss_values:
                for gain20_min in gain20_min_values:
                    for wave_q in wave_quality_values:
                        params = {
                            **DEFAULT_PARAMS,
                            "rsi_max": rsi_max,
                            "gain20_max": gain20_max,
                            "stop_loss_pct": stop_loss,
                            "gain20_min": gain20_min,
                            "wave_quality_min": wave_q,
                            "use_rsi_filter": True,
                            "use_gain20_max": True,
                        }
                        results = run_backtest("2025-01-02", "2025-09-30", params, skip_interval=4, top_n=3)
                        stats = analyze(results)
                        count += 1
                        if stats.get("win_rate", 0) >= 55 and stats["total"] >= 30:
                            candidates.append((params, stats))

                        if count % 200 == 0:
                            print(f"  已测{count}组合  当前最优{len(candidates)}个候选  best_wr={max([s['win_rate'] for _,s in candidates] or [0])}%", flush=True)

    print(f"\n✅ 第1轮完成: 测试了{count}组  候选{len(candidates)}个(胜率≥55%,样本≥30)")

    if not candidates:
        # 放宽条件重试
        print("\n📍 重新搜索（放宽条件）...", flush=True)
        for rsi_max in [70, 75, 80]:
            for gain20_max in [35, 50]:
                for stop_loss in [5, 7, 8]:
                    for gain20_min in [10, 14]:
                        for wave_q in [3, 4, 5]:
                            params = {
                                **DEFAULT_PARAMS,
                                "rsi_max": rsi_max,
                                "gain20_max": gain20_max,
                                "stop_loss_pct": stop_loss,
                                "gain20_min": gain20_min,
                                "wave_quality_min": wave_q,
                                "use_rsi_filter": rsi_max < 80,
                                "use_gain20_max": True,
                            }
                            results = run_backtest("2025-01-02", "2025-09-30", params, skip_interval=4, top_n=3)
                            stats = analyze(results)
                            if stats.get("win_rate", 0) >= 50 and stats["total"] >= 20:
                                candidates.append((params, stats))

    # 找最优
    best = max(candidates, key=lambda x: (x[1]["win_rate"], x[1]["total"]))
    best_params, best_train_stats = best
    print(f"\n🏆 最优参数（训练集）:")
    print(f"  RSI上限: {best_params['rsi_max']}")
    print(f"  20日涨幅范围: {best_params['gain20_min']}%~{best_params['gain20_max']}%")
    print(f"  波段质量≥: {best_params['wave_quality_min']}")
    print(f"  止损线: {best_params['stop_loss_pct']}%")
    print_stats(best_train_stats, "训练集")

    # 验证集
    print("\n🔍 验证集验证...", flush=True)
    val_results = run_backtest("2025-10-01", "2026-04-23", best_params, skip_interval=4, top_n=3)
    val_stats = analyze(val_results)
    print_stats(val_stats, "验证集")

    return best_params, best_train_stats, val_stats, val_results

# ── 精细搜索（围绕最优参数附近细调）────────────────────────
def fine_tune(best_params):
    print("\n📍 第2轮：精细搜索（围绕最优参数微调）...", flush=True)

    candidates = []
    base = best_params.copy()

    # 在最优RSI附近±5
    for rsi_adj in [-5, 0, 5, 10]:
        rsi = max(60, min(80, best_params["rsi_max"] + rsi_adj))
        for gain20_max_adj in [-5, 0, 10, 20]:
            gain20_max = max(gain20_min + 5, best_params["gain20_max"] + gain20_max_adj)
            for stop_adj in [-2, -1, 0, 1]:
                stop_loss = max(3, min(10, best_params["stop_loss_pct"] + stop_adj))
                for gain20_min_adj in [-3, 0, 3]:
                    gain20_min = max(5, best_params["gain20_min"] + gain20_min_adj)
                    for wave_adj in [-1, 0, 1, 2]:
                        wave_q = max(1, best_params["wave_quality_min"] + wave_adj)

                        params = {
                            **DEFAULT_PARAMS,
                            **base,
                            "rsi_max": rsi,
                            "gain20_max": gain20_max,
                            "stop_loss_pct": stop_loss,
                            "gain20_min": gain20_min,
                            "wave_quality_min": wave_q,
                            "use_rsi_filter": rsi < 80,
                            "use_gain20_max": True,
                        }
                        results = run_backtest("2025-01-02", "2025-09-30", params, skip_interval=4, top_n=3)
                        stats = analyze(results)
                        if stats.get("win_rate", 0) >= 55 and stats["total"] >= 25:
                            candidates.append((params, stats, stats))

    if candidates:
        best = max(candidates, key=lambda x: (x[1]["win_rate"], x[1]["total"]))
        return best[0], best[1]
    return best_params, analyze([])

if __name__ == "__main__":
    preload()
    best_params, train_stats, val_stats, val_results = grid_search()
    # 精细调优
    best_params2, train_stats2 = fine_tune(best_params)
    val_results2 = run_backtest("2025-10-01", "2026-04-23", best_params2, skip_interval=4, top_n=3)
    val_stats2 = analyze(val_results2)
    print(f"\n🏆 最终参数（精细调优后）:")
    print(f"  RSI上限: {best_params2['rsi_max']}")
    print(f"  20日涨幅: {best_params2['gain20_min']}%~{best_params2['gain20_max']}%")
    print(f"  波段质量≥: {best_params2['wave_quality_min']}")
    print(f"  MA分离≥: {best_params2['ma20_60_sep_min']}%")
    print(f"  止损: {best_params2['stop_loss_pct']}%")
    print_stats(train_stats2, "训练集(精细)")
    print_stats(val_stats2, "验证集(精细)")

    # 保存结果
    import json
    out = {
        "best_params": best_params2,
        "train_stats": train_stats2,
        "val_stats": val_stats2,
        "val_results": val_results2[:50],  # 只保存前50条
    }
    Path("/home/lyc/stock_reports/optimized_result.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n💾 已保存: /home/lyc/stock_reports/optimized_result.json")
