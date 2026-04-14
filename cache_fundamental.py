#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基本面数据预缓存脚本
====================
每日 16:00 定时执行，将全市场股票基本面数据预加载到本地缓存，
供次日筛选器直接读取，无需盘中重复请求 AkShare。

用法：
  python cache_fundamental.py                    # 全市场增量预热（30天未更新才刷新）
  python cache_fundamental.py --max-age-days 10  # 自定义刷新阈值
  python cache_fundamental.py --codes 300568     # 指定股票
  python cache_fundamental.py --refresh          # 强制刷新（包括未过期）
  python cache_fundamental.py --dry-run          # 预览哪些会被请求
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

WORKSPACE = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(WORKSPACE))

from stock_trend.gain_turnover import (
    FundamentalData,
    _fundamental_cache_path,
    get_all_stock_codes,
    load_fundamental_data,
    normalize_prefixed,
)

DEFAULT_WORKERS = 8
DEFAULT_MAX_AGE_DAYS = 30  # 默认30天未更新则刷新


def _needs_refresh(path: Path, max_age_days: int) -> bool:
    if not path.exists():
        return True
    age_days = (time.time() - path.stat().st_mtime) / 86400.0
    return age_days >= max_age_days


def prewarm_fundamental(
    codes: list[str],
    workers: int = DEFAULT_WORKERS,
    force_refresh: bool = False,
    dry_run: bool = False,
    verbose: bool = False,
    max_age_days: int = DEFAULT_MAX_AGE_DAYS,
) -> dict:
    """
    预热基本面缓存。

    Returns:
        dict with keys: success, failed, skipped, elapsed_seconds
    """
    t0 = time.time()
    results = {"success": [], "failed": [], "skipped": [], "elapsed_seconds": 0.0}

    # 分组：需要刷新 / 跳过
    to_refresh = []
    for code in codes:
        c = normalize_prefixed(code)
        path = _fundamental_cache_path(c)
        if force_refresh or _needs_refresh(path, max_age_days):
            to_refresh.append(c)
        else:
            results["skipped"].append(c)

    total = len(to_refresh)
    print(
        f"\n{'[Dry Run] ' if dry_run else ''}基本面预缓存: "
        f"{len(to_refresh)} 只要刷新 / {len(results['skipped'])} 只跳过（缓存有效）"
    )
    if dry_run:
        print("以下股票将被请求 AkShare：")
        for c in to_refresh:
            print(f"  {c}")
        return results

    if not to_refresh:
        print("全部缓存有效，无需请求。")
        return results

    done = 0

    def work(code: str):
        try:
            fd = load_fundamental_data(code, refresh=True)
            if fd is not None:
                return code, fd, None
            return code, None, "fetch returned None"
        except Exception as e:
            return code, None, str(e)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(work, code): code for code in to_refresh}
        for fut in as_completed(futures):
            code, fd, err = fut.result()
            done += 1
            if err is None and fd is not None:
                results["success"].append(code)
                if verbose:
                    profit = "✓" if fd.is_profitable else "✗"
                    print(
                        f"  {normalize_prefixed(code):<12} | EPS={fd.eps:>8.3f} "
                        f"| ROE={fd.roe:>6.2f}% | 盈利={profit} | {fd.report_date}"
                    )
            else:
                results["failed"].append((code, err))
                print(f"  ✗ {code}: {err}")
            if done % 200 == 0 or done == total:
                elapsed = time.time() - t0
                eta = elapsed / done * (total - done) if done else 0
                print(f"   进度: {done}/{total} ({done/total*100:.1f}%) ETA={eta:.0f}s", end="\r")

    results["elapsed_seconds"] = time.time() - t0
    print()
    return results


def print_summary(results: dict):
    print("\n" + "=" * 60)
    print(f"✅ 成功: {len(results['success'])} 只")
    if results["failed"]:
        print(f"❌ 失败: {len(results['failed'])} 只")
        for code, err in results["failed"][:10]:
            print(f"   {code}: {err}")
        if len(results["failed"]) > 10:
            print(f"   ... 还有 {len(results['failed'])-10} 只失败")
    print(f"⏭️  跳过: {len(results['skipped'])} 只（缓存有效）")
    print(f"⏱️  总耗时: {results['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="基本面数据预缓存（每日 16:00 定时执行）")
    parser.add_argument("--codes", nargs="+", default=None, help="指定股票代码（默认全市场）")
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS, help=f"并行线程数（默认{DEFAULT_WORKERS}）"
    )
    parser.add_argument(
        "--max-age-days",
        type=int,
        default=DEFAULT_MAX_AGE_DAYS,
        help=f"缓存超过多少天视为过期需刷新（默认{DEFAULT_MAX_AGE_DAYS}天）",
    )
    parser.add_argument("--refresh", action="store_true", help="强制刷新所有股票（含未过期）")
    parser.add_argument("--dry-run", action="store_true", help="只显示哪些会更新，不实际请求")
    parser.add_argument("-v", "--verbose", action="store_true", help="详细输出（显示每只股票数据）")
    args = parser.parse_args()

    codes = args.codes if args.codes else get_all_stock_codes()
    print(f"📋 待处理: {len(codes)} 只股票")
    print(f"⏱️  增量模式：缓存超过 {args.max_age_days} 天未更新才刷新")

    results = prewarm_fundamental(
        codes=codes,
        workers=args.workers,
        force_refresh=args.refresh,
        dry_run=args.dry_run,
        verbose=args.verbose,
        max_age_days=args.max_age_days,
    )
    print_summary(results)
