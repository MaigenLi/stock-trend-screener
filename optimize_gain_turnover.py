#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
参数寻优（升级版，前复权）
=========================

默认目标：在满足最小交易数约束下，按 Sharpe 优先，其次平均收益、胜率。

示例：
  python optimize_gain_turnover.py \
    --start 2024-01-01 --end 2025-12-31 \
    --days 2,3 --min-gain 1.5,2.0 --max-gain 5,6 \
    --quality-days 15,20,30 --turnover 0,1.5,3 \
    --score-threshold 50,60,70 --hold 3,5 \
    --max-extension 8,10,12 --max-picks-per-day 3
"""

from __future__ import annotations

import itertools
import json
import math
import sys
import time
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd

WORKSPACE = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(WORKSPACE))

from stock_trend.gain_turnover_strategy import StrategyConfig, get_all_stock_codes, normalize_prefixed
from stock_trend.backtest_gain_turnover import load_all_stock_data, run_backtest

DEFAULT_START = "2024-01-01"
DEFAULT_END = "2025-12-31"
DEFAULT_OUTPUT_DIR = Path.home() / "stock_reports"


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def summarize_trades(trades: list[dict], hold_days: int) -> dict:
    if not trades:
        return {
            "trades": 0,
            "win_rate": 0.0,
            "avg_ret": -999.0,
            "med_ret": -999.0,
            "std_ret": 0.0,
            "max_loss": 0.0,
            "sharpe": -999.0,
            "expected_annual": -999.0,
        }
    returns = np.array([t["ret_pct"] for t in trades], dtype=float)
    wins = returns[returns > 0]
    losses = returns[returns <= 0]
    avg_ret = float(np.mean(returns))
    med_ret = float(np.median(returns))
    std_ret = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
    win_rate = float(np.mean(returns > 0) * 100.0)
    max_loss = float(np.min(returns))
    expected_annual = avg_ret / hold_days * 252.0
    if std_ret > 1e-9:
        sharpe = ((avg_ret / 100.0) / hold_days * 252.0 - 0.03) / ((std_ret / 100.0) * math.sqrt(252.0 / hold_days))
    else:
        sharpe = 0.0
    return {
        "trades": int(len(trades)),
        "win_rate": round(win_rate, 2),
        "avg_ret": round(avg_ret, 4),
        "med_ret": round(med_ret, 4),
        "std_ret": round(std_ret, 4),
        "max_loss": round(max_loss, 4),
        "sharpe": round(float(sharpe), 4),
        "expected_annual": round(expected_annual, 2),
    }


def build_grid(args) -> list[dict]:
    combos = []
    for days, min_gain, max_gain, quality_days, turnover, score_threshold, hold, max_extension in itertools.product(
        args.days_list,
        args.min_gain_list,
        args.max_gain_list,
        args.quality_days_list,
        args.turnover_list,
        args.score_threshold_list,
        args.hold_list,
        args.max_extension_list,
    ):
        if min_gain > max_gain:
            continue
        combos.append(
            {
                "days": days,
                "min_gain": min_gain,
                "max_gain": max_gain,
                "quality_days": quality_days,
                "turnover": turnover,
                "score_threshold": score_threshold,
                "hold": hold,
                "max_extension": max_extension,
            }
        )
    return combos


def sort_key(row: dict):
    # 先保证有足够样本，再按 Sharpe > 平均收益 > 胜率 > 交易数
    penalty = 0 if row["trades"] >= row["min_trades"] else 1
    return (
        penalty,
        -row["sharpe"],
        -row["avg_ret"],
        -row["win_rate"],
        -row["trades"],
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="升级版策略参数寻优（前复权）")
    parser.add_argument("--start", type=str, default=DEFAULT_START)
    parser.add_argument("--end", type=str, default=DEFAULT_END)
    parser.add_argument("--days", type=str, default="2,3")
    parser.add_argument("--min-gain", type=str, default="1.5,2.0")
    parser.add_argument("--max-gain", type=str, default="5,6")
    parser.add_argument("--quality-days", type=str, default="15,20")
    parser.add_argument("--turnover", type=str, default="0,1.5,3")
    parser.add_argument("--score-threshold", type=str, default="50,60,70")
    parser.add_argument("--hold", type=str, default="3,5")
    parser.add_argument("--max-extension", type=str, default="8,10,12")
    parser.add_argument("--max-picks-per-day", type=int, default=3)
    parser.add_argument("--min-volume", type=float, default=1e8)
    parser.add_argument("--adjust", type=str, default="qfq", choices=["qfq", "", "hfq"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--min-trades", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--codes", nargs="+", default=None)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--output", "-o", type=str, default=None)
    args = parser.parse_args()

    args.days_list = parse_int_list(args.days)
    args.min_gain_list = parse_float_list(args.min_gain)
    args.max_gain_list = parse_float_list(args.max_gain)
    args.quality_days_list = parse_int_list(args.quality_days)
    args.turnover_list = parse_float_list(args.turnover)
    args.score_threshold_list = parse_float_list(args.score_threshold)
    args.hold_list = parse_int_list(args.hold)
    args.max_extension_list = parse_float_list(args.max_extension)

    combos = build_grid(args)
    if not combos:
        print("❌ 没有有效参数组合")
        sys.exit(1)

    codes = [normalize_prefixed(c) for c in args.codes] if args.codes else get_all_stock_codes()

    print("=" * 90)
    print(f"🔬 参数寻优 | 前复权={args.adjust or 'none'} | 区间 {args.start} ~ {args.end}")
    print(f"   股票数: {len(codes)} | 组合数: {len(combos)} | 最小交易数门槛: {args.min_trades}")
    print(f"   默认排序: 先满足交易数，再按 Sharpe > 平均收益 > 胜率")
    print("=" * 90)

    # 只需加载一次底层数据，后续复用
    max_quality = max(args.quality_days_list)
    stock_data, global_dates = load_all_stock_data(
        codes=codes,
        start_date=args.start,
        end_date=args.end,
        adjust=args.adjust,
        refresh_cache=args.refresh_cache,
        max_workers=args.workers,
        lookback_days=max(120, max_quality + 60),
    )
    if not stock_data:
        print("❌ 无有效股票数据")
        sys.exit(1)

    rows = []
    t0 = time.time()
    for idx, combo in enumerate(combos, 1):
        config = StrategyConfig(
            signal_days=combo["days"],
            min_gain=combo["min_gain"],
            max_gain=combo["max_gain"],
            quality_days=combo["quality_days"],
            min_turnover=combo["turnover"],
            min_amount=args.min_volume,
            score_threshold=combo["score_threshold"],
            adjust=args.adjust,
            max_extension_pct=combo["max_extension"],
        )
        trades = run_backtest(
            stock_data=stock_data,
            global_dates=global_dates,
            config=config,
            start_date=args.start,
            end_date=args.end,
            hold_days=combo["hold"],
            max_picks_per_day=args.max_picks_per_day,
            buy_slip=0.001,
            sell_slip=0.001,
            commission=0.0003,
            tax=0.001,
        )
        metrics = summarize_trades(trades, combo["hold"])
        row = {
            **combo,
            **metrics,
            "min_trades": args.min_trades,
        }
        rows.append(row)
        elapsed = time.time() - t0
        eta = elapsed / idx * (len(combos) - idx) if idx else 0
        print(
            f"\n[{idx}/{len(combos)}] days={combo['days']} gain=[{combo['min_gain']},{combo['max_gain']}] "
            f"q={combo['quality_days']} to={combo['turnover']} score={combo['score_threshold']} hold={combo['hold']} ext={combo['max_extension']}"
        )
        print(
            f"   trades={row['trades']} win={row['win_rate']:.1f}% avg={row['avg_ret']:+.3f}% "
            f"sharpe={row['sharpe']:.2f} annual={row['expected_annual']:+.1f}% ETA={eta/60:.1f}min"
        )

    rows.sort(key=sort_key)
    top = rows[: args.top_k]

    print("\n" + "=" * 140)
    print(f"🏆 Top {len(top)} 参数组合")
    print("=" * 140)
    header = (
        f"{'days':>4} {'min':>5} {'max':>5} {'qdays':>6} {'turn':>6} {'score':>6} {'hold':>5} {'ext':>5} "
        f"{'trades':>7} {'win':>7} {'avg':>8} {'sharp':>7} {'annual':>9} {'maxloss':>8}"
    )
    print(header)
    print("-" * 140)
    for r in top:
        print(
            f"{r['days']:>4} {r['min_gain']:>5.1f} {r['max_gain']:>5.1f} {r['quality_days']:>6} {r['turnover']:>6.1f} "
            f"{r['score_threshold']:>6.1f} {r['hold']:>5} {r['max_extension']:>5.1f} {r['trades']:>7} "
            f"{r['win_rate']:>6.1f}% {r['avg_ret']:>+7.3f}% {r['sharpe']:>7.2f} {r['expected_annual']:>+8.1f}% {r['max_loss']:>+7.2f}%"
        )
    print("=" * 140)

    out = Path(args.output) if args.output else DEFAULT_OUTPUT_DIR / f"gain_turnover_optimize_{args.start}_{args.end}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "meta": {
                "start": args.start,
                "end": args.end,
                "adjust": args.adjust,
                "codes": len(codes),
                "combos": len(combos),
                "min_trades": args.min_trades,
                "max_picks_per_day": args.max_picks_per_day,
            },
            "top": top,
            "all": rows,
        }, f, ensure_ascii=False, indent=2)
    print(f"💾 寻优结果已写入: {out.resolve()}")
