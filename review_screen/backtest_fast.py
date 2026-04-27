#!/usr/bin/env python3
"""
高效回测引擎 — screen_trend_filter 真实胜率验证
================================================

优化策略：
1. 价格数据全量预加载到内存（.cache/qfq_daily/*.csv → memory）
2. 指标数据惰性加载 + LRU 缓存
3. 每个信号日只遍历通过预筛选的股票（避免全量5196只扫描）

用法：
    python backtest_fast.py --start 2025-01-01 --end 2026-04-23
"""

import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

WORKSPACE = Path(__file__).parent.parent.parent.resolve()
CACHE_DIR = WORKSPACE / ".cache"
INDICATORS_DIR = CACHE_DIR / "indicators"
QFQ_DIR = CACHE_DIR / "qfq_daily"

COLUMNS = ["date", "open", "high", "low", "close", "volume", "amount",
           "turnover", "outstanding_share", "true_turnover"]

# ── 全局价格数据（预加载）────────────────────────────────────
_price_index = {}  # (code, date) -> {open, high, low, close, volume}
_price_cache = {}  # code -> DataFrame


def preload_all_prices():
    """预加载所有股票的价格数据到内存（只加载需要的列）"""
    global _price_cache, _price_index
    print("  📂 预加载价格数据（一次性）...", flush=True)
    count = 0
    for csv_path in QFQ_DIR.glob("*_qfq.csv"):
        code = csv_path.stem.replace("_qfq", "")
        try:
            df = pd.read_csv(csv_path, usecols=["date", "open", "high", "low", "close", "volume"])
            df = df.sort_values("date").reset_index(drop=True)
            _price_cache[code] = df
            # 建立 date -> row 索引
            for _, row in df.iterrows():
                _price_index[(code, row["date"])] = (
                    float(row["open"]), float(row["high"]),
                    float(row["low"]), float(row["close"]), float(row["volume"])
                )
            count += 1
        except Exception:
            pass
    print(f"  ✅ 已加载 {count} 只股票的价格数据（{len(_price_index)} 条记录）", flush=True)


def get_price_on_date(code: str, date: str):
    """从内存获取价格（O(1)）"""
    key = (code, date)
    if key not in _price_index:
        return None
    o, h, l, c, v = _price_index[key]
    return {"open": o, "high": h, "low": l, "close": c, "volume": v}


# ── 指标缓存（LRU）─────────────────────────────────────────
_MAX_IND_CACHE = 300
_ind_cache = {}
_ind_order = []


def load_indicators_for_stock(code: str):
    """惰性加载单只股票指标"""
    global _ind_cache, _ind_order
    if code in _ind_cache:
        _ind_order.remove(code)
        _ind_order.append(code)
        return _ind_cache[code]
    path = INDICATORS_DIR / f"{code}_indicators.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        index = {row["date"]: row for row in data}
        if len(_ind_cache) >= _MAX_IND_CACHE:
            oldest = _ind_order.pop(0)
            del _ind_cache[oldest]
        _ind_cache[code] = index
        _ind_order.append(code)
        return index
    except Exception:
        return None


def get_indicators(code: str, date: str):
    index = load_indicators_for_stock(code)
    return index.get(date) if index else None


# ── 日期工具 ──────────────────────────────────────────────
def get_trading_dates_merged(start: str, end: str) -> list[str]:
    """获取交易日列表（用000001作基准）"""
    df = _price_cache.get("000001")
    if df is None:
        return []
    mask = (df["date"] >= start) & (df["date"] <= end)
    return sorted(df[mask]["date"].tolist())


# ── 筛选阈值 ─────────────────────────────────────────────
TREND_FILTER = {"gain20_min": 13.8, "wave_quality_min": 4.0, "ma20_60_sep_min": 3.5}
BUY_READY = {"min_gain1": 0.3, "max_gain1": 8.0, "max_rsi": 80.0, "max_ma5_dist": 8.0, "limit_pct": 9.5}


def apply_trend_filter(ind: dict) -> bool:
    return (
        ind.get("gain20", 0) >= TREND_FILTER["gain20_min"]
        and ind.get("wave_quality", 0) >= TREND_FILTER["wave_quality_min"]
        and ind.get("ma_sep", 0) >= TREND_FILTER["ma20_60_sep_min"]
    )


def apply_buy_ready(ind: dict) -> tuple[bool, str]:
    g = ind.get("gain1", 0)
    rsi = ind.get("rsi", 50)
    ma5d = ind.get("ma5_distance_pct", 0)
    if g >= BUY_READY["limit_pct"]:
        return False, "涨停"
    if g < BUY_READY["min_gain1"]:
        return False, f"涨幅{g:.1f}%过弱"
    if g > BUY_READY["max_gain1"]:
        return False, f"涨幅{g:.1f}%过高"
    if rsi >= BUY_READY["max_rsi"]:
        return False, f"RSI={rsi:.0f}过热"
    if abs(ma5d) > BUY_READY["max_ma5_dist"]:
        return False, f"距MA5±{ma5d:.0f}%过远"
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


def get_next_date(date: str, offset: int = 1) -> str | None:
    df = _price_cache.get("000001")
    if df is None:
        return None
    dates = df["date"].tolist()
    try:
        idx = dates.index(date)
        tidx = idx + offset
        if 0 <= tidx < len(dates):
            return dates[tidx]
    except ValueError:
        pass
    return None


# ── 单日回测 ──────────────────────────────────────────────
def run_backtest_single_day(signal_date: str, hold_days: int, stop_loss_pct: float = 8.0) -> list[dict]:
    results = []
    # 找出在 signal_date 有指标的股票（在预筛选阶段用指标文件）
    all_codes = [f.stem.replace("_indicators", "") for f in INDICATORS_DIR.glob("*_indicators.json")]
    passed = []
    for code in all_codes:
        ind = get_indicators(code, signal_date)
        if not ind:
            continue
        if not apply_trend_filter(ind):
            continue
        ready, reason = apply_buy_ready(ind)
        if not ready:
            continue
        passed.append((code, ind, reason))

    # 获取这些股票的 T+1 价格
    entry_date = get_next_date(signal_date, 1)
    if not entry_date:
        return results

    for code, ind, reason in passed:
        entry_price_data = get_price_on_date(code, entry_date)
        if not entry_price_data or entry_price_data["open"] <= 0:
            continue

        entry_price = entry_price_data["open"]
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
    return results


# ── 全量回测 ─────────────────────────────────────────────
def run_full_backtest(start: str, end: str, hold_days: int = 5, skip_interval: int = 4) -> list[dict]:
    print(f"  指标惰性加载 + 价格内存索引", flush=True)
    warmup = 60
    all_dates = get_trading_dates_merged("1990-01-01", end)
    if len(all_dates) <= warmup:
        return []
    all_dates = all_dates[warmup:]  # 去掉最古老的warmup天
    signal_dates = [d for d in all_dates if d >= start][::skip_interval]
    if not signal_dates:
        print(f"⚠️  无有效信号日", flush=True)
        return []

    print(f"📊 回测: {signal_dates[0]}→{signal_dates[-1]}  信号日{len(signal_dates)}批  持有{hold_days}天", flush=True)

    all_results = []
    for i, sd in enumerate(signal_dates):
        batch = run_backtest_single_day(sd, hold_days)
        all_results.extend(batch)
        if (i + 1) % 10 == 0 or (i + 1) == len(signal_dates):
            print(f"  进度: {i+1}/{len(signal_dates)}  累计信号: {len(all_results)}", flush=True)

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
            f"T+{k}": {"count": len(v), "win_rate": round(len([x for x in v if x > 0]) / len(v) * 100, 1),
                       "avg_pnl": round(np.mean(v), 3) if v else 0}
            for k, v in sorted(by_hold.items())
        },
        "stopped_count": len(stopped),
        "stopped_avg_pnl": round(np.mean(stopped), 3) if stopped else 0,
        "not_stopped_count": len(not_stopped),
        "not_stopped_avg_pnl": round(np.mean(not_stopped), 3) if not_stopped else 0,
        "strong_signals": {"count": len(strong), "win_rate": round(len([x for x in strong if x["pnl_pct"] > 0]) / len(strong) * 100, 1) if strong else 0,
                           "avg_pnl": round(np.mean([x["pnl_pct"] for x in strong]), 3) if strong else 0},
        "medium_signals": {"count": len(medium), "win_rate": round(len([x for x in medium if x["pnl_pct"] > 0]) / len(medium) * 100, 1) if medium else 0,
                           "avg_pnl": round(np.mean([x["pnl_pct"] for x in medium]), 3) if medium else 0},
        "weak_signals": {"count": len(weak), "win_rate": round(len([x for x in weak if x["pnl_pct"] > 0]) / len(weak) * 100, 1) if weak else 0,
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
    print(f"\n🛡️  止损: {stats['stopped_count']}次({stats['stopped_avg_pnl']:+.3f}%)  非止损: {stats['not_stopped_count']}次({stats['not_stopped_avg_pnl']:+.3f}%)")
    print(f"\n🏆 信号强度:")
    print(f"  🟢强: {stats['strong_signals']['count']}笔  {stats['strong_signals']['win_rate']}%  {stats['strong_signals']['avg_pnl']:+.3f}%")
    print(f"  🔵中: {stats['medium_signals']['count']}笔  {stats['medium_signals']['win_rate']}%  {stats['medium_signals']['avg_pnl']:+.3f}%")
    print(f"  🟡弱: {stats['weak_signals']['count']}笔  {stats['weak_signals']['win_rate']}%  {stats['weak_signals']['avg_pnl']:+.3f}%")


# ── 主程序 ───────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-04-23")
    parser.add_argument("--hold", type=int, default=5)
    parser.add_argument("--interval", type=int, default=4)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    print(f"\n🚀 加载数据...", flush=True)
    preload_all_prices()

    print(f"\n🚀 回测: {args.start} → {args.end}", flush=True)
    results = run_full_backtest(args.start, args.end, hold_days=args.hold, skip_interval=args.interval)

    if not results:
        print("⚠️  无回测结果")
        sys.exit(0)

    stats = analyze_results(results)
    print_report(stats)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n💾 已写入: {args.out}")
