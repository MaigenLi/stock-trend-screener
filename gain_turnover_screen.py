#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按每日涨幅区间筛选股票（策略升级版，前复权）
================================================

升级点：
1. 默认使用前复权日线（AkShare + 本地缓存）
2. 拆分“信号窗口”和“质量窗口”
3. 使用历史换手率，不再混用实时 turnover API
4. 与回测脚本共用同一套评分/过滤口径

筛选逻辑：
- 信号窗口：最近 N 个交易日，每天涨幅都在 [min_gain, max_gain]
- 质量窗口：最近 quality_days 个交易日，要求：
  * close > MA20 > MA60
  * MA20 向上
  * 20日涨幅 > 0
  * 不过度偏离 MA20
  * 20日均成交额、5日均换手满足要求
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd

WORKSPACE = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(WORKSPACE))

from stock_trend.gain_turnover import (
    StrategyConfig,
    evaluate_latest_signal,
    format_signal_results,
    get_all_stock_codes,
    get_stock_name,
    load_qfq_history,
    load_stock_names,
    normalize_prefixed,
)

DEFAULT_TOP_N = 50
DEFAULT_WORKERS = 8
DEFAULT_OUTPUT_DIR = Path.home() / "stock_reports"


def screen_market(
    codes: List[str],
    config: StrategyConfig,
    target_date: Optional[datetime] = None,
    top_n: int = DEFAULT_TOP_N,
    max_workers: int = DEFAULT_WORKERS,
    refresh_cache: bool = False,
) -> list:
    names = load_stock_names()
    total = len(codes)
    results = []
    t0 = time.time()

    end_date = target_date.strftime("%Y-%m-%d") if target_date else None
    print(f"\n🔍 升级版筛选: {total} 只股票 | 复权={config.adjust} | workers={max_workers}")
    print(
        f"   信号窗口={config.signal_days}天, 涨幅={config.min_gain}%~{config.max_gain}% | "
        f"质量窗口={config.quality_days}天"
    )
    print(
        f"   成交额≥{config.min_amount/1e8:.1f}亿, 5日换手≥{config.min_turnover:.2f}% | "
        f"评分门槛={config.score_threshold}"
    )

    def work(code: str):
        code = normalize_prefixed(code)
        df = load_qfq_history(code, end_date=end_date, adjust=config.adjust, refresh=refresh_cache)
        if df is None or df.empty:
            return None
        if target_date is not None:
            df = df[df["date"] <= pd.Timestamp(target_date.date())].reset_index(drop=True)
        if df.empty:
            return None
        return evaluate_latest_signal(code, get_stock_name(code, names), df, config)

    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(work, code): code for code in codes}
        for fut in as_completed(futures):
            done += 1
            try:
                r = fut.result()
                if r is not None:
                    results.append(r)
            except Exception:
                pass
            if done % 300 == 0 or done == total:
                elapsed = time.time() - t0
                eta = elapsed / done * (total - done) if done else 0
                print(f"   进度: {done}/{total} ({done/total*100:.1f}%) ETA={eta:.0f}s", end="\r")

    print()
    results.sort(key=lambda x: (x.score, x.total_gain_window), reverse=True)
    print(f"✅ 筛选完成: {len(results)} 只通过, 用时 {time.time()-t0:.1f}s")
    return results[:top_n]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="按每日涨幅区间筛选股票（升级版，前复权）")
    parser.add_argument("--days", type=int, default=2, help="信号窗口天数")
    parser.add_argument("--min-gain", type=float, default=2.0, help="每日涨幅最小值%%")
    parser.add_argument("--max-gain", type=float, default=7.0, help="每日涨幅最大值%%")
    parser.add_argument("--quality-days", type=int, default=10, help="质量窗口天数")
    parser.add_argument("--turnover", type=float, default=1.5, help="5日平均换手率下限%%")
    parser.add_argument("--min-volume", type=float, default=1e8, help="20日平均成交额下限")
    parser.add_argument("--score-threshold", type=float, default=60.0, help="评分门槛")
    parser.add_argument("--max-extension", type=float, default=10.0, help="距MA20最大偏离%%")
    parser.add_argument("--adjust", type=str, default="qfq", choices=["qfq", "", "hfq"], help="复权方式")
    parser.add_argument("--top-n", type=int, default=50, help="返回前N只")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="并行线程数")
    parser.add_argument("--codes", nargs="+", default=None, help="指定股票代码")
    parser.add_argument("--date", type=str, default=None, help="截止日期 YYYY-MM-DD")
    parser.add_argument("--refresh-cache", action="store_true", help="强制刷新前复权缓存")
    parser.add_argument("--output", "-o", type=str, default=None, help="输出文件路径")
    args = parser.parse_args()

    if args.min_gain > args.max_gain:
        print("❌ min-gain 不能大于 max-gain")
        sys.exit(1)

    target_date = None
    if args.date:
        target_date = datetime.strptime(args.date, "%Y-%m-%d")

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
    if args.codes:
        print(f"📋 指定股票: {codes}")
    else:
        print(f"📋 全市场股票: {len(codes)} 只")

    results = screen_market(
        codes=codes,
        config=config,
        target_date=target_date,
        top_n=args.top_n,
        max_workers=args.workers,
        refresh_cache=args.refresh_cache,
    )

    title_date = target_date.strftime("%Y-%m-%d") if target_date else datetime.now().strftime("%Y-%m-%d")
    title = (
        f"升级版筛选 前复权={config.adjust or 'none'} | 信号{config.signal_days}天[{config.min_gain},{config.max_gain}] "
        f"| 质量{config.quality_days}天 | {title_date}"
    )
    output_text = format_signal_results(results, title)
    print("\n" + output_text)

    output_path = Path(args.output) if args.output else DEFAULT_OUTPUT_DIR / f"gain_screen_upgrade_{title_date}.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output_text)
    print(f"\n💾 结果已写入: {output_path.resolve()}")
