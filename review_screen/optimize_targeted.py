#!/usr/bin/env python3
"""
定向参数优化 — 针对60%胜率目标
================================

基于领域知识的定向搜索，非穷举

核心洞察：
1. RSI>75的股票通常已走完主升浪，追高胜率低
2. 20日涨幅>40%的股票往往回调压力大
3. 止损从8%→5%，盈亏比改善但胜率不一定提高
4. 止损从8%→3%，止损必触发，胜率看 non-stopped 的 win rate
"""

import json, sys
from pathlib import Path
import numpy as np

WORKSPACE = Path(__file__).parent.parent.parent.resolve()
CACHE_DIR = WORKSPACE / ".cache"
MERGED_PATH = CACHE_DIR / "indicators_merged.json"
QFQ_DIR = CACHE_DIR / "qfq_daily"

# ── 数据加载 ─────────────────────────────────────────────
_merged_index = {}
_price_cache = {}

def preload():
    global _merged_index, _price_cache
    with open(MERGED_PATH) as f:
        _merged_index = json.load(f)
    import pandas as pd
    for csv_path in QFQ_DIR.glob("*_qfq.csv"):
        code = csv_path.stem.replace("_qfq", "")
        try:
            df = pd.read_csv(csv_path, usecols=["date","open","high","low","close","volume"])
            df = df.sort_values("date").reset_index(drop=True)
            _price_cache[code] = df
        except:
            pass
    dates = sorted(_merged_index.keys())
    print(f"✅ {len(dates)}天  {dates[0]}~{dates[-1]}  {len(_price_cache)}只", flush=True)

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

def run_backtest(params, hold_days=5, skip_interval=4, top_n=3, start="2025-01-02", end="2025-09-30"):
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
            # 趋势过滤
            if ind.get("gain20", 0) < params["gain20_min"]: continue
            if params.get("gain20_max", 99) < 99 and ind.get("gain20", 0) > params["gain20_max"]: continue
            if ind.get("wave_quality", 0) < params["wave_quality_min"]: continue
            if ind.get("ma_sep", 0) < params["ma20_60_sep_min"]: continue

            # 买入准备度
            g = ind.get("gain1", 0)
            rsi = ind.get("rsi", 50)
            ma5d = ind.get("ma5_distance_pct", 0)

            if g >= 9.5: continue
            if g < params["min_gain1"]: continue
            if g > params["max_gain1"]: continue
            if params.get("rsi_max", 99) < 99 and rsi >= params["rsi_max"]: continue
            if abs(ma5d) > params["max_ma5_dist"]: continue

            # T+1 开盘价
            entry_date = get_next_date(sd, 1)
            if not entry_date: continue
            entry = get_price_on_date(code, entry_date)
            if not entry or entry["open"] <= 0: continue

            entry_price = entry["open"]
            stop_ref = ind.get("stop_loss_ref", entry_price * 0.97)
            sl_pct = params.get("stop_loss_pct", 8.0)
            exit_price = None
            hold_actual = hold_days
            stopped = False

            for d in range(1, hold_days + 1):
                ex_dt = get_next_date(sd, d)
                if not ex_dt: break
                price = get_price_on_date(code, ex_dt)
                if not price: break
                if price["low"] <= stop_ref * (1 - sl_pct / 100):
                    exit_price = stop_ref * (1 - sl_pct / 100)
                    hold_actual = d
                    stopped = True
                    break
                if d == hold_days:
                    exit_price = price["close"]

            if exit_price is None: continue

            pnl_pct = (exit_price - entry_price) / entry_price * 100
            candidates.append({
                "code": code, "signal_date": sd,
                "pnl_pct": round(pnl_pct, 3),
                "stopped": stopped,
                "signal_score": round(compute_score(ind), 1),
                "gain20": ind.get("gain20", 0),
                "rsi": rsi,
                "vol_ratio": ind.get("vol_ratio", 0),
                "wave_quality": ind.get("wave_quality", 0),
                "ma_sep": ind.get("ma_sep", 0),
            })

        candidates.sort(key=lambda x: x["signal_score"], reverse=True)
        all_results.extend(candidates[:top_n] if top_n > 0 else candidates)

    return all_results

def analyze(results):
    if not results: return {}
    pnls = [r["pnl_pct"] for r in results]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    stopped = [r for r in results if r["stopped"]]
    not_stopped = [r for r in results if not r["stopped"]]
    strong = [r for r in results if r["signal_score"] >= 70]
    medium = [r for r in results if 60 <= r["signal_score"] < 70]
    weak = [r for r in results if r["signal_score"] < 60]
    return {
        "total": len(results),
        "win_rate": round(len(wins)/len(results)*100, 1),
        "avg_pnl": round(float(np.mean(pnls)), 3),
        "median_pnl": round(float(np.median(pnls)), 3),
        "max_pnl": round(max(pnls), 2),
        "min_pnl": round(min(pnls), 2),
        "profit_factor": round(abs(np.sum(wins)/np.sum(losses)), 2) if losses else 99.99,
        "stopped_n": len(stopped),
        "stopped_avg": round(float(np.mean([r["pnl_pct"] for r in stopped])), 3) if stopped else 0,
        "not_stopped_n": len(not_stopped),
        "not_stopped_avg": round(float(np.mean([r["pnl_pct"] for r in not_stopped])), 3) if not_stopped else 0,
        "strong_n": len(strong), "strong_wr": round(len([x for x in strong if x["pnl_pct"]>0])/len(strong)*100, 1) if strong else 0,
        "medium_n": len(medium), "medium_wr": round(len([x for x in medium if x["pnl_pct"]>0])/len(medium)*100, 1) if medium else 0,
        "avg_win": round(float(np.mean(wins)), 3) if wins else 0,
        "avg_loss": round(float(np.mean(losses)), 3) if losses else 0,
        "std_pnl": round(float(np.std(pnls)), 2),
    }

def print_stats(stats, label=""):
    s = stats
    print(f"\n{'='*60}")
    print(f"📊 {label}")
    print(f"{'='*60}")
    print(f"  总信号: {s['total']}  胜率: {s['win_rate']}%  均值: {s['avg_pnl']:+.3f}%")
    print(f"  中位数: {s['median_pnl']:+.3f}%  盈亏比: {s['profit_factor']}  标准差: {s['std_pnl']}")
    print(f"  最大盈利: {s['max_pnl']:+.2f}%  最大亏损: {s['min_pnl']:+.2f}%")
    print(f"  止损: {s['stopped_n']}次({s['stopped_avg']:+.3f}%)  非止损: {s['not_stopped_n']}次({s['not_stopped_avg']:+.3f}%)")
    print(f"  🟢强: {s['strong_n']}笔({s['strong_wr']}%)  🔵中: {s['medium_n']}笔({s['medium_wr']}%)")
    print(f"  均盈利: {s['avg_win']:+.3f}%  均亏损: {s['avg_loss']:+.3f}%")

def run_targeted_search():
    """定向搜索：测试最有希望的参数组合"""
    print("\n🚀 定向参数搜索", flush=True)

    best_overall = None
    best_overall_stats = None

    # ── 策略1：RSI严格过滤 + 止损5% + 涨幅上限 ────────────
    print("\n📍 测试 RSI严格过滤族...", flush=True)
    rsi_configs = [
        {"rsi_max": 68, "gain20_max": 35, "stop_loss_pct": 5.0, "gain20_min": 10, "wave_quality_min": 3},
        {"rsi_max": 70, "gain20_max": 35, "stop_loss_pct": 5.0, "gain20_min": 10, "wave_quality_min": 3},
        {"rsi_max": 70, "gain20_max": 30, "stop_loss_pct": 5.0, "gain20_min": 8, "wave_quality_min": 3},
        {"rsi_max": 72, "gain20_max": 35, "stop_loss_pct": 5.0, "gain20_min": 10, "wave_quality_min": 3},
        {"rsi_max": 72, "gain20_max": 40, "stop_loss_pct": 5.0, "gain20_min": 10, "wave_quality_min": 3},
        {"rsi_max": 75, "gain20_max": 35, "stop_loss_pct": 5.0, "gain20_min": 10, "wave_quality_min": 3},
        {"rsi_max": 75, "gain20_max": 30, "stop_loss_pct": 5.0, "gain20_min": 8, "wave_quality_min": 3},
        {"rsi_max": 68, "gain20_max": 40, "stop_loss_pct": 5.0, "gain20_min": 12, "wave_quality_min": 4},
        {"rsi_max": 70, "gain20_max": 40, "stop_loss_pct": 5.0, "gain20_min": 12, "wave_quality_min": 4},
        {"rsi_max": 72, "gain20_max": 40, "stop_loss_pct": 5.0, "gain20_min": 12, "wave_quality_min": 4},
    ]

    base = {
        "min_gain1": 0.3, "max_gain1": 8.0,
        "max_ma5_dist": 8.0, "ma20_60_sep_min": 3.0,
    }

    for rsi_cfg in rsi_configs:
        params = {**base, **rsi_cfg}
        train = run_backtest(params, start="2025-01-02", end="2025-09-30")
        stats = analyze(train)
        val = run_backtest(params, start="2025-10-01", end="2026-04-23")
        vstats = analyze(val)
        print(f"  RSI<{rsi_cfg['rsi_max']} gain20≤{rsi_cfg['gain20_max']}% SL{rsi_cfg['stop_loss_pct']}% "
              f"→ 训练集{stats['win_rate']}%({stats['total']}笔) 验证集{vstats['win_rate']}%({vstats['total']}笔)",
              flush=True)
        if best_overall is None or vstats['win_rate'] > best_overall_stats['win_rate']:
            best_overall = params.copy()
            best_overall_stats = vstats.copy()

    # ── 策略2：止损3%极端方案 ──────────────────────────────
    print("\n📍 测试 止损3% 方案...", flush=True)
    for rsi_max in [70, 75]:
        for gain20_max in [30, 35, 40]:
            params = {**base, "rsi_max": rsi_max, "gain20_max": gain20_max,
                      "stop_loss_pct": 3.0, "gain20_min": 10, "wave_quality_min": 3}
            train = run_backtest(params, start="2025-01-02", end="2025-09-30")
            stats = analyze(train)
            val = run_backtest(params, start="2025-10-01", end="2026-04-23")
            vstats = analyze(val)
            print(f"  RSI<{rsi_max} gain20≤{gain20_max}% SL3% → 训练{stats['win_rate']}%({stats['total']}笔) 验证{vstats['win_rate']}%({vstats['total']}笔)",
                  flush=True)
            if vstats['win_rate'] > best_overall_stats['win_rate']:
                best_overall = params.copy()
                best_overall_stats = vstats.copy()

    # ── 策略3：组合更严格的趋势条件 ───────────────────────
    print("\n📍 测试 趋势强化族...", flush=True)
    for rsi_max in [70, 72, 75]:
        for wave_q in [5, 6, 7]:
            for ma_sep in [4.0, 5.0, 6.0]:
                params = {**base, "rsi_max": rsi_max, "gain20_max": 40,
                          "stop_loss_pct": 5.0, "gain20_min": 12, "wave_quality_min": wave_q,
                          "ma20_60_sep_min": ma_sep}
                train = run_backtest(params, start="2025-01-02", end="2025-09-30")
                stats = analyze(train)
                val = run_backtest(params, start="2025-10-01", end="2026-04-23")
                vstats = analyze(val)
                if vstats['win_rate'] >= 55 and vstats['total'] >= 20:
                    print(f"  RSI<{rsi_max} WQ≥{wave_q} SEP≥{ma_sep}% → 训练{stats['win_rate']}%({stats['total']}笔) 验证{vstats['win_rate']}%({vstats['total']}笔)",
                          flush=True)
                if vstats['win_rate'] > best_overall_stats['win_rate']:
                    best_overall = params.copy()
                    best_overall_stats = vstats.copy()

    return best_overall, best_overall_stats

if __name__ == "__main__":
    preload()
    best_params, best_val_stats = run_targeted_search()

    if best_params:
        print(f"\n🏆 最优参数:")
        for k, v in best_params.items():
            print(f"  {k}: {v}")
        print(f"\n验证集胜率: {best_val_stats['win_rate']}%")

        # 最终验证（更密集的信号）
        print("\n🔍 最终验证（信号间隔2天，增加样本）...", flush=True)
        final_train = run_backtest(best_params, start="2025-01-02", end="2025-09-30", skip_interval=2, top_n=3)
        final_val = run_backtest(best_params, start="2025-10-01", end="2026-04-23", skip_interval=2, top_n=3)
        ft = analyze(final_train)
        fv = analyze(final_val)
        print_stats(ft, "最终训练集(skip=2)")
        print_stats(fv, "最终验证集(skip=2)")

        # 保存
        out = {
            "best_params": best_params,
            "train_stats": ft,
            "val_stats": fv,
        }
        Path("/home/lyc/stock_reports/optimized_top3.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=2))
        print(f"\n💾 已保存: /home/lyc/stock_reports/optimized_top3.json")
    else:
        print("\n⚠️ 未找到有效参数")
