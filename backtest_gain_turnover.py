#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
每日涨幅区间策略回测（升级版，前复权）
================================================

升级点：
1. 默认使用前复权日线（AkShare + 本地缓存）
2. 与筛选脚本共用统一评分与过滤口径
3. T 日出信号，T+1 开盘买入，持有 N 个交易日后收盘卖出
4. 使用全市场交易日日历对齐，停牌股票自动跳过
5. 加入滑点、佣金、印花税
6. 支持“每个信号日最多买 K 只”的组合约束
"""

from __future__ import annotations

import math
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

WORKSPACE = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(WORKSPACE))

from stock_trend.gain_turnover_strategy import (
    PreparedData,
    StrategyConfig,
    evaluate_signal,
    get_all_stock_codes,
    load_qfq_history,
    normalize_prefixed,
    prepare_data,
)

DEFAULT_START = "2020-01-01"
DEFAULT_END = "2026-04-08"
DEFAULT_WORKERS = 8
DEFAULT_MAX_PICKS_PER_DAY = 5
DEFAULT_BUY_SLIP = 0.001
DEFAULT_SELL_SLIP = 0.001
DEFAULT_COMMISSION = 0.0003
DEFAULT_TAX = 0.001


def load_all_stock_data(
    codes: List[str],
    start_date: str,
    end_date: str,
    adjust: str = "qfq",
    refresh_cache: bool = False,
    max_workers: int = DEFAULT_WORKERS,
    lookback_days: int = 120,
) -> tuple[Dict[str, PreparedData], List[str]]:
    start_ts = pd.Timestamp(start_date) - pd.Timedelta(days=lookback_days * 2)
    end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=20)
    stock_data: Dict[str, PreparedData] = {}
    date_set = set()
    total = len(codes)
    t0 = time.time()

    print(f"\n📦 加载前复权数据: {total} 只股票 | adjust={adjust} | workers={max_workers}")

    def work(code: str):
        code = normalize_prefixed(code)
        df = load_qfq_history(
            code,
            start_date=start_ts,
            end_date=end_ts,
            adjust=adjust,
            refresh=refresh_cache,
            max_age_hours=0 if refresh_cache else 12,
        )
        if df is None or df.empty or len(df) < 80:
            return code, None
        prepared = prepare_data(df)
        return code, prepared

    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(work, code): code for code in codes}
        for fut in as_completed(futures):
            done += 1
            try:
                code, prepared = fut.result()
                if prepared is not None:
                    stock_data[code] = prepared
                    for d in prepared.dates:
                        date_set.add(str(d))
            except Exception:
                pass
            if done % 300 == 0 or done == total:
                elapsed = time.time() - t0
                eta = elapsed / done * (total - done) if done else 0
                print(f"   进度: {done}/{total} ({done/total*100:.1f}%) ETA={eta:.0f}s", end="\r")

    global_dates = sorted(date_set)
    print()
    print(f"✅ 数据加载完成: {len(stock_data)}/{len(codes)} 只有效, 交易日 {len(global_dates)} 个, 用时 {time.time()-t0:.1f}s")
    return stock_data, global_dates


def run_backtest(
    stock_data: Dict[str, PreparedData],
    global_dates: List[str],
    config: StrategyConfig,
    start_date: str,
    end_date: str,
    hold_days: int,
    max_picks_per_day: int,
    buy_slip: float,
    sell_slip: float,
    commission: float,
    tax: float,
) -> List[Dict]:
    date_to_global_idx = {d: i for i, d in enumerate(global_dates)}
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    all_candidates: List[Dict] = []
    total = len(stock_data)
    done = 0
    t0 = time.time()

    print("\n🔍 回测计算中...")

    for code, prepared in stock_data.items():
        done += 1
        dates = prepared.dates
        local_idx_map = {d: i for i, d in enumerate(dates)}
        for i in range(len(dates)):
            signal_date = pd.Timestamp(dates[i])
            if signal_date < start_ts or signal_date > end_ts:
                continue
            signal = evaluate_signal(prepared, i, config)
            if signal is None:
                continue

            global_signal_idx = date_to_global_idx.get(signal["signal_date"])
            if global_signal_idx is None:
                continue
            buy_global_idx = global_signal_idx + 1
            sell_global_idx = buy_global_idx + hold_days - 1
            if sell_global_idx >= len(global_dates):
                continue
            buy_date = global_dates[buy_global_idx]
            sell_date = global_dates[sell_global_idx]
            buy_local_idx = local_idx_map.get(buy_date)
            sell_local_idx = local_idx_map.get(sell_date)
            if buy_local_idx is None or sell_local_idx is None:
                continue

            buy_open = float(prepared.open_[buy_local_idx])
            sell_close = float(prepared.close[sell_local_idx])
            if buy_open <= 0 or sell_close <= 0:
                continue

            # 涨停一字/跌停一字等极端情况简单过滤
            buy_high = float(prepared.high[buy_local_idx])
            buy_low = float(prepared.low[buy_local_idx])
            prev_close = float(prepared.close[buy_local_idx - 1]) if buy_local_idx > 0 else buy_open
            if prev_close > 0:
                open_gap_pct = (buy_open / prev_close - 1.0) * 100.0
                if open_gap_pct >= 9.5 and abs(buy_open - buy_low) < 1e-6 and abs(buy_open - buy_high) < 1e-6:
                    continue
                if open_gap_pct <= -9.5 and abs(buy_open - buy_low) < 1e-6 and abs(buy_open - buy_high) < 1e-6:
                    continue

            buy_exec = buy_open * (1.0 + buy_slip + commission)
            sell_exec = sell_close * (1.0 - sell_slip - commission - tax)
            ret_pct = (sell_exec / buy_exec - 1.0) * 100.0

            all_candidates.append({
                "code": code,
                "signal_date": signal["signal_date"],
                "buy_date": buy_date,
                "sell_date": sell_date,
                "score": signal["score"],
                "buy_price": round(buy_open, 2),
                "sell_price": round(sell_close, 2),
                "buy_exec": round(buy_exec, 4),
                "sell_exec": round(sell_exec, 4),
                "ret_pct": round(ret_pct, 4),
                "rsi14": signal["rsi14"],
                "total_gain_window": signal["total_gain_window"],
                "avg_amount_20": signal["avg_amount_20"],
                "avg_turnover_5": signal["avg_turnover_5"],
                "extension_pct": signal["extension_pct"],
            })

        if done % 200 == 0 or done == total:
            elapsed = time.time() - t0
            eta = elapsed / done * (total - done) if done else 0
            print(f"   进度: {done}/{total} ({done/total*100:.1f}%) ETA={eta/60:.1f}min", end="\r")

    print()
    print(f"   原始信号交易: {len(all_candidates)} 笔")

    # 每天最多买 K 只，按分数/窗口涨幅排序
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for t in all_candidates:
        grouped[t["signal_date"]].append(t)

    trades: List[Dict] = []
    for signal_date, lst in grouped.items():
        lst.sort(key=lambda x: (x["score"], x["total_gain_window"]), reverse=True)
        trades.extend(lst[:max_picks_per_day])

    trades.sort(key=lambda x: (x["buy_date"], -x["score"], x["code"]))
    print(f"✅ 组合约束后交易: {len(trades)} 笔（每信号日最多 {max_picks_per_day} 只）")
    return trades


def report(trades: List[Dict], hold_days: int):
    if not trades:
        print("\n⚠️ 无有效交易记录")
        return

    returns = np.array([t["ret_pct"] for t in trades], dtype=float)
    wins = returns[returns > 0]
    losses = returns[returns <= 0]

    total = len(returns)
    win_rate = len(wins) / total * 100.0
    avg_ret = float(np.mean(returns))
    med_ret = float(np.median(returns))
    std_ret = float(np.std(returns, ddof=1)) if total > 1 else 0.0
    max_win = float(np.max(returns))
    max_loss = float(np.min(returns))
    avg_win = float(np.mean(wins)) if len(wins) else 0.0
    avg_loss = abs(float(np.mean(losses))) if len(losses) else 0.0
    pl_ratio = avg_win / avg_loss if avg_loss > 1e-12 else 0.0
    expected_annual = avg_ret / hold_days * 252.0
    sharpe = 0.0
    if std_ret > 1e-9:
        sharpe = ((avg_ret / 100.0) / hold_days * 252.0 - 0.03) / ((std_ret / 100.0) * math.sqrt(252.0 / hold_days))

    print("\n" + "=" * 80)
    print(f"📊 升级版回测结果（{total} 笔交易）")
    print("=" * 80)
    print(f"  胜率:       {win_rate:.1f}%  ({len(wins)} 胜 / {len(losses)} 负)")
    print(f"  平均收益:   {avg_ret:+.3f}%/笔")
    print(f"  中位收益:   {med_ret:+.3f}%")
    print(f"  标准差:     {std_ret:.3f}%")
    print(f"  最大盈利:   {max_win:+.2f}%")
    print(f"  最大亏损:   {max_loss:+.2f}%")
    print(f"  期望年化:   {expected_annual:+.1f}%/年")
    print(f"  夏普比率:   {sharpe:.2f}")
    print(f"  盈亏比:     {pl_ratio:.2f}")
    print()

    score_bins = [(0, 60), (60, 70), (70, 80), (80, 200)]
    print("  📈 按评分分组:")
    for lo, hi in score_bins:
        grp = np.array([t["ret_pct"] for t in trades if lo <= t["score"] < hi], dtype=float)
        if len(grp) >= 5:
            wr = float(np.mean(grp > 0) * 100)
            print(f"    {lo:>2}-{hi:<3}: {len(grp):>4}笔  胜率{wr:>5.1f}%  均值{np.mean(grp):+6.2f}%")
    print("=" * 80)


def save_trades(trades: List[Dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trades).to_csv(path, index=False, encoding="utf-8")
    print(f"💾 交易明细已保存: {path.resolve()}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="每日涨幅区间策略回测（升级版，前复权）")
    parser.add_argument("--days", type=int, default=2, help="信号窗口天数")
    parser.add_argument("--min-gain", type=float, default=2.0, help="每日涨幅最小值%%")
    parser.add_argument("--max-gain", type=float, default=7.0, help="每日涨幅最大值%%")
    parser.add_argument("--quality-days", type=int, default=10, help="质量窗口天数")
    parser.add_argument("--turnover", type=float, default=1.5, help="5日平均换手率下限%%")
    parser.add_argument("--min-volume", type=float, default=1e8, help="20日平均成交额下限")
    parser.add_argument("--score-threshold", type=float, default=60.0, help="评分门槛")
    parser.add_argument("--max-extension", type=float, default=10.0, help="距MA20最大偏离%%")
    parser.add_argument("--adjust", type=str, default="qfq", choices=["qfq", "", "hfq"], help="复权方式")
    parser.add_argument("--hold", type=int, default=3, help="持有交易日数")
    parser.add_argument("--start", type=str, default=DEFAULT_START, help="开始日期")
    parser.add_argument("--end", type=str, default=DEFAULT_END, help="结束日期")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="并行线程数")
    parser.add_argument("--max-picks-per-day", type=int, default=DEFAULT_MAX_PICKS_PER_DAY, help="每个信号日最多买几只")
    parser.add_argument("--buy-slip", type=float, default=DEFAULT_BUY_SLIP, help="买入滑点")
    parser.add_argument("--sell-slip", type=float, default=DEFAULT_SELL_SLIP, help="卖出滑点")
    parser.add_argument("--commission", type=float, default=DEFAULT_COMMISSION, help="单边佣金")
    parser.add_argument("--tax", type=float, default=DEFAULT_TAX, help="卖出印花税")
    parser.add_argument("--codes", nargs="+", default=None, help="指定股票代码")
    parser.add_argument("--refresh-cache", action="store_true", help="强制刷新前复权缓存")
    parser.add_argument("--output", "-o", type=str, default=None, help="交易明细输出文件")
    args = parser.parse_args()

    if args.min_gain > args.max_gain:
        print("❌ min-gain 不能大于 max-gain")
        sys.exit(1)

    config = StrategyConfig(
        signal_days=args.days,
        min_gain=args.min_gain,
        max_gain=args.max_gain,
        quality_days=args.quality_days,
        min_turnover=args.turnover,
        min_amount=args.min_volume,
        score_threshold=args.score_threshold,
        adjust=args.adjust,
        max_extension_pct=args.max_extension,
    )

    codes = [normalize_prefixed(c) for c in args.codes] if args.codes else get_all_stock_codes()
    print("=" * 80)
    print(
        f"📊 升级版回测 | 前复权={config.adjust or 'none'} | 信号{config.signal_days}天[{config.min_gain},{config.max_gain}] | "
        f"质量{config.quality_days}天 | 持有{args.hold}天"
    )
    print(f"   区间: {args.start} ~ {args.end} | 股票数: {len(codes)}")
    print(f"   成本: 买滑点{args.buy_slip:.4f} 卖滑点{args.sell_slip:.4f} 佣金{args.commission:.4f} 印花税{args.tax:.4f}")
    print("=" * 80)

    stock_data, global_dates = load_all_stock_data(
        codes=codes,
        start_date=args.start,
        end_date=args.end,
        adjust=args.adjust,
        refresh_cache=args.refresh_cache,
        max_workers=args.workers,
    )
    if not stock_data:
        print("⚠️ 无有效股票数据")
        sys.exit(1)

    trades = run_backtest(
        stock_data=stock_data,
        global_dates=global_dates,
        config=config,
        start_date=args.start,
        end_date=args.end,
        hold_days=args.hold,
        max_picks_per_day=args.max_picks_per_day,
        buy_slip=args.buy_slip,
        sell_slip=args.sell_slip,
        commission=args.commission,
        tax=args.tax,
    )

    report(trades, args.hold)

    output_path = Path(args.output) if args.output else (Path.home() / "stock_reports" / f"gain_turnover_backtest_upgrade_{args.start}_{args.end}.csv")
    save_trades(trades, output_path)
