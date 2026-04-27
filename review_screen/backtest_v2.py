#!/usr/bin/env python3
"""
高效回测引擎 v2 — 使用合并日期索引
=====================================

使用 .cache/indicators_merged.json（1.21GB）
O(1) 获取任意股票在任意日期的指标

用法：
    python backtest_v2.py --start 2025-01-01 --end 2026-04-23
"""

import json
import sys
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

WORKSPACE = Path(__file__).parent.parent.parent.resolve()
CACHE_DIR = WORKSPACE / ".cache"
MERGED_PATH = CACHE_DIR / "indicators_merged.json"
QFQ_DIR = CACHE_DIR / "qfq_daily"

# ── 全局数据 ──────────────────────────────────────────────
_merged_index = {}  # date -> {code -> row}
_price_cache = {}  # code -> DataFrame


def preload():
    """预加载所有数据"""
    global _merged_index, _price_cache

    # 加载合并指标索引
    print("  📂 加载合并指标索引...", flush=True)
    with open(MERGED_PATH, "r") as f:
        _merged_index = json.load(f)
    dates = sorted(_merged_index.keys())
    print(f"   ✅ 指标索引: {len(dates)} 天 ({dates[0]} ~ {dates[-1]})", flush=True)

    # 加载价格数据
    print("  📂 加载价格数据...", flush=True)
    for csv_path in QFQ_DIR.glob("*_qfq.csv"):
        code = csv_path.stem.replace("_qfq", "")
        try:
            df = pd.read_csv(csv_path, usecols=["date", "open", "high", "low", "close", "volume"])
            df = df.sort_values("date").reset_index(drop=True)
            _price_cache[code] = df
        except Exception:
            pass
    print(f"   ✅ 价格缓存: {len(_price_cache)} 只", flush=True)


def get_indicators(code: str, date: str):
    """O(1) 指标查询"""
    return _merged_index.get(date, {}).get(code)


def get_price_on_date(code: str, date: str):
    """价格查询"""
    df = _price_cache.get(code)
    if df is None:
        return None
    row = df[df["date"] == date]
    if row.empty:
        return None
    r = row.iloc[0]
    return {"open": float(r["open"]), "high": float(r["high"]),
            "low": float(r["low"]), "close": float(r["close"]), "volume": float(r["volume"])}


def get_trading_dates(start: str, end: str) -> list[str]:
    dates = sorted(_merged_index.keys())
    return [d for d in dates if start <= d <= end]


def get_next_date(date: str, offset: int = 1) -> str | None:
    dates = sorted(_merged_index.keys())
    try:
        idx = dates.index(date)
        tidx = idx + offset
        if 0 <= tidx < len(dates):
            return dates[tidx]
    except ValueError:
        pass
    return None


# ── 筛选阈值 ─────────────────────────────────────────────
TREND_FILTER = {"gain20_min": 13.8, "wave_quality_min": 4.0, "ma20_60_sep_min": 3.5}
BUY_READY = {"min_gain1": 0.3, "max_gain1": 8.0, "max_rsi": 80.0, "max_ma5_dist": 8.0, "limit_pct": 9.5}


def apply_trend_filter(ind: dict) -> bool:
    return (ind.get("gain20", 0) >= TREND_FILTER["gain20_min"]
            and ind.get("wave_quality", 0) >= TREND_FILTER["wave_quality_min"]
            and ind.get("ma_sep", 0) >= TREND_FILTER["ma20_60_sep_min"])


def apply_buy_ready(ind: dict) -> tuple[bool, str]:
    g = ind.get("gain1", 0)
    rsi = ind.get("rsi", 50)
    ma5d = ind.get("ma5_distance_pct", 0)
    if g >= BUY_READY["limit_pct"]:
        return False, "涨停"
    if g < BUY_READY["min_gain1"]:
        return False, f"涨幅弱"
    if g > BUY_READY["max_gain1"]:
        return False, f"涨幅过高"
    if rsi >= BUY_READY["max_rsi"]:
        return False, f"RSI过热"
    if abs(ma5d) > BUY_READY["max_ma5_dist"]:
        return False, f"偏离过大"
    return True, "买入就绪"


def compute_score(ind: dict) -> float:
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


# ── 单日回测 ─────────────────────────────────────────────
def run_backtest_single_day(signal_date: str, hold_days: int, stop_loss_pct: float = 8.0, top_n: int = 0) -> list[dict]:
    day_data = _merged_index.get(signal_date, {})
    results = []

    for code, ind in day_data.items():
        if not apply_trend_filter(ind):
            continue
        ready, reason = apply_buy_ready(ind)
        if not ready:
            continue

        entry_date = get_next_date(signal_date, 1)
        if not entry_date:
            continue

        entry = get_price_on_date(code, entry_date)
        if not entry or entry["open"] <= 0:
            continue

        entry_price = entry["open"]
        stop_ref = ind.get("stop_loss_ref", entry_price * 0.97)
        exit_price = None
        hold_actual = hold_days
        stopped = False

        for d in range(1, hold_days + 1):
            exit_date = get_next_date(signal_date, d)
            if not exit_date:
                break
            price = get_price_on_date(code, exit_date)
            if not price:
                break
            if price["low"] <= stop_ref * (1 - stop_loss_pct / 100):
                exit_price = stop_ref * (1 - stop_loss_pct / 100)
                hold_actual = d
                stopped = True
                break
            if d == hold_days:
                exit_price = price["close"]

        if exit_price is None:
            continue

        pnl_pct = (exit_price - entry_price) / entry_price * 100
        results.append({
            "code": code,
            "signal_date": signal_date,
            "entry_price": round(entry_price, 3),
            "exit_price": round(exit_price, 3),
            "hold_days_actual": hold_actual,
            "pnl_pct": round(pnl_pct, 3),
            "stopped": stopped,
            "signal_score": round(compute_score(ind), 1),
            "reason": reason,
            "gain20": ind.get("gain20", 0),
            "wave_quality": ind.get("wave_quality", 0),
            "ma_sep": ind.get("ma_sep", 0),
            "rsi": ind.get("rsi", 0),
            "vol_ratio": ind.get("vol_ratio", 0),
            "hold_days_requested": hold_days,
        })

    # 只取 TOP-N（按信号评分排序）
    if top_n > 0:
        results.sort(key=lambda x: x["signal_score"], reverse=True)
        results = results[:top_n]

    return results


# ── 全量回测 ─────────────────────────────────────────────
def run_full_backtest(start: str, end: str, hold_days: int = 5, skip_interval: int = 4, top_n: int = 0) -> list[dict]:
    warmup = 60
    all_dates = sorted(_merged_index.keys())
    all_dates = all_dates[warmup:]  # 去掉最古老的warmup天
    signal_dates = [d for d in all_dates if d >= start][::skip_interval]

    if not signal_dates:
        print(f"⚠️  无有效信号日", flush=True)
        return []

    top_str = f" TOP{top_n}" if top_n > 0 else ""
    print(f"📊 回测: {signal_dates[0]}→{signal_dates[-1]}  信号日{len(signal_dates)}批  持有{hold_days}天{top_str}", flush=True)

    all_results = []
    for i, sd in enumerate(signal_dates):
        batch = run_backtest_single_day(sd, hold_days, top_n=top_n)
        all_results.extend(batch)
        if (i + 1) % 10 == 0 or (i + 1) == len(signal_dates):
            print(f"  {i+1}/{len(signal_dates)}  累计信号: {len(all_results)}", flush=True)

    return all_results


# ── 统计分析 ─────────────────────────────────────────────
def analyze_results(results: list[dict]) -> dict:
    if not results:
        return {"error": "无结果"}

    pnls = [r["pnl_pct"] for r in results]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    by_hold = defaultdict(list)
    for r in results:
        by_hold[r["hold_days_actual"]].append(r["pnl_pct"])

    stopped = [r["pnl_pct"] for r in results if r["stopped"]]
    not_stopped = [r["pnl_pct"] for r in results if not r["stopped"]]
    strong = [r for r in results if r["signal_score"] >= 70]
    medium = [r for r in results if 60 <= r["signal_score"] < 70]
    weak = [r for r in results if r["signal_score"] < 60]

    return {
        "total_signals": len(results),
        "win_count": len(wins),
        "win_rate": round(len(wins) / len(results) * 100, 1),
        "avg_pnl": round(np.mean(pnls), 3),
        "median_pnl": round(np.median(pnls), 3),
        "std_pnl": round(np.std(pnls), 3),
        "max_pnl": round(max(pnls), 2),
        "min_pnl": round(min(pnls), 2),
        "avg_win": round(np.mean(wins), 3) if wins else 0,
        "avg_loss": round(np.mean(losses), 3) if losses else 0,
        "profit_factor": round(abs(np.sum(wins) / np.sum(losses)), 2) if losses else float("inf"),
        "by_hold_days": {
            f"T+{k}": {"count": len(v),
                       "win_rate": round(len([x for x in v if x > 0]) / len(v) * 100, 1),
                       "avg_pnl": round(np.mean(v), 3) if v else 0}
            for k, v in sorted(by_hold.items())
        },
        "stopped_count": len(stopped),
        "stopped_avg_pnl": round(np.mean(stopped), 3) if stopped else 0,
        "not_stopped_count": len(not_stopped),
        "not_stopped_avg_pnl": round(np.mean(not_stopped), 3) if not_stopped else 0,
        "strong_signals": {"count": len(strong),
                           "win_rate": round(len([x for x in strong if x["pnl_pct"] > 0]) / len(strong) * 100, 1) if strong else 0,
                           "avg_pnl": round(np.mean([x["pnl_pct"] for x in strong]), 3) if strong else 0},
        "medium_signals": {"count": len(medium),
                           "win_rate": round(len([x for x in medium if x["pnl_pct"] > 0]) / len(medium) * 100, 1) if medium else 0,
                           "avg_pnl": round(np.mean([x["pnl_pct"] for x in medium]), 3) if medium else 0},
        "weak_signals": {"count": len(weak),
                         "win_rate": round(len([x for x in weak if x["pnl_pct"] > 0]) / len(weak) * 100, 1) if weak else 0,
                         "avg_pnl": round(np.mean([x["pnl_pct"] for x in weak]), 3) if weak else 0},
        "top5_avg": round(np.mean(sorted(pnls, reverse=True)[:5]), 2),
        "bottom5_avg": round(np.mean(sorted(pnls)[:5]), 2),
    }


def print_report(stats: dict, label: str = ""):
    print(f"\n{'=' * 60}")
    print(f"📊 回测统计 {label}")
    print(f"{'=' * 60}")
    print(f"  总信号数:   {stats['total_signals']}")
    print(f"  胜率:       {stats['win_rate']}%")
    print(f"  平均收益:   {stats['avg_pnl']:+.3f}%")
    print(f"  中位数:     {stats['median_pnl']:+.3f}%")
    print(f"  标准差:     {stats['std_pnl']}")
    print(f"  最大盈利:   {stats['max_pnl']:+.2f}%")
    print(f"  最大亏损:   {stats['min_pnl']:+.2f}%")
    print(f"  盈亏比:     {stats['profit_factor']}")
    print(f"\n📅 分持仓天数:")
    for k, v in stats["by_hold_days"].items():
        print(f"  {k}: {v['count']}笔  胜率{v['win_rate']}%  均值{v['avg_pnl']:+.3f}%")
    print(f"\n🛡️  止损: {stats['stopped_count']}次({stats['stopped_avg_pnl']:+.3f}%)  "
          f"非止损: {stats['not_stopped_count']}次({stats['not_stopped_avg_pnl']:+.3f}%)")
    print(f"\n🏆 信号强度:")
    print(f"  🟢强: {stats['strong_signals']['count']}笔  {stats['strong_signals']['win_rate']}%  {stats['strong_signals']['avg_pnl']:+.3f}%")
    print(f"  🔵中: {stats['medium_signals']['count']}笔  {stats['medium_signals']['win_rate']}%  {stats['medium_signals']['avg_pnl']:+.3f}%")
    print(f"  🟡弱: {stats['weak_signals']['count']}笔  {stats['weak_signals']['win_rate']}%  {stats['weak_signals']['avg_pnl']:+.3f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-04-23")
    parser.add_argument("--hold", type=int, default=5)
    parser.add_argument("--interval", type=int, default=4)
    parser.add_argument("--top-n", type=int, default=0, help="每批只买TOP N（0=全部，默认0）")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    print(f"\n🚀 加载数据...", flush=True)
    preload()

    print(f"\n🚀 回测: {args.start} → {args.end}", flush=True)
    results = run_full_backtest(args.start, args.end, hold_days=args.hold, skip_interval=args.interval, top_n=args.top_n)

    if not results:
        print("⚠️  无回测结果")
        sys.exit(0)

    stats = analyze_results(results)
    print_report(stats)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n💾 已写入: {args.out}")
