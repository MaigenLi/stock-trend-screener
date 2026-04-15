#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
前复权日线数据预缓存脚本
========================
每个交易日 16:10 定时执行，将全市场股票前复权日线数据刷新到本地缓存，
确保本地 K 线数据为最新（覆盖当日实时数据）。

用法：
  python cache_qfq_daily.py                     # 增量刷新（只刷新超过6小时的缓存）
  python cache_qfq_daily.py --max-age-hours 4  # 自定义阈值
  python cache_qfq_daily.py --codes 300568     # 指定股票
  python cache_qfq_daily.py --refresh          # 强制刷新全部
  python cache_qfq_daily.py --dry-run          # 预览哪些会被请求
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

WORKSPACE = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(WORKSPACE))

from stock_trend.gain_turnover import (
    _cache_path,
    get_all_stock_codes,
    load_qfq_history,
    normalize_prefixed,
)

DEFAULT_WORKERS = 16
DEFAULT_MAX_AGE_HOURS = 6  # 超过6小时未更新则重新拉取


def _needs_refresh(path: Path, max_age_hours: float) -> bool:
    if not path.exists():
        return True
    age_hours = (time.time() - path.stat().st_mtime) / 3600.0
    return age_hours >= max_age_hours


def prewarm_qfq_daily(
    codes: list[str],
    workers: int = DEFAULT_WORKERS,
    force_refresh: bool = False,
    dry_run: bool = False,
    verbose: bool = False,
    max_age_hours: float = DEFAULT_MAX_AGE_HOURS,
) -> dict:
    """
    预热前复权日线缓存。

    Returns:
        dict with keys: success, failed, skipped, elapsed_seconds
    """
    t0 = time.time()
    results = {"success": [], "failed": [], "skipped": [], "elapsed_seconds": 0.0}

    to_refresh = []
    for code in codes:
        c = normalize_prefixed(code)
        path = _cache_path(c, adjust="qfq")
        if force_refresh or _needs_refresh(path, max_age_hours):
            to_refresh.append(c)
        else:
            results["skipped"].append(c)

    total = len(to_refresh)
    print(
        f"\n{'[Dry Run] ' if dry_run else ''}前复权日线缓存: "
        f"{len(to_refresh)} 只要刷新 / {len(results['skipped'])} 只跳过（缓存有效）"
    )
    if dry_run:
        print("以下股票将被请求 AkShare：")
        for c in to_refresh[:50]:
            print(f"  {c}")
        if len(to_refresh) > 50:
            print(f"  ... 还有 {len(to_refresh)-50} 只")
        return results

    if not to_refresh:
        print("全部缓存有效，无需请求。")
        return results

    done = 0

    def work(code: str):
        try:
            df = load_qfq_history(code, adjust="qfq", refresh=True, max_age_hours=0)
            if df is not None and not df.empty:
                return code, len(df), None
            return code, 0, "empty dataframe"
        except Exception as e:
            return code, 0, str(e)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(work, code): code for code in to_refresh}
        for fut in as_completed(futures):
            code, n_rows, err = fut.result()
            done += 1
            if err is None:
                results["success"].append(code)
                if verbose:
                    print(f"  ✓ {code}: {n_rows} 条")
            else:
                results["failed"].append((code, err))
                print(f"  ✗ {code}: {err}")
            if done % 300 == 0 or done == total:
                elapsed = time.time() - t0
                eta = elapsed / done * (total - done) if done else 0
                print(f"   进度: {done}/{total} ({done/total*100:.1f}%) ETA={eta:.0f}s", end="\r")

    results["elapsed_seconds"] = time.time() - t0
    print()
    return results


def _get_cache_latest_date(code: str) -> str | None:
    """返回缓存文件最新日期字符串（如 '2026-04-15'），不存在返回 None。"""
    path = _cache_path(code, adjust="qfq")
    if not path.exists():
        return None
    try:
        import pandas as pd
        df = pd.read_csv(path, parse_dates=["date"])
        if df is None or df.empty:
            return None
        return df["date"].max().strftime("%Y-%m-%d")
    except Exception:
        return None


def verify_and_retry_stale(
    codes: list[str],
    today_str: str,
    workers: int = DEFAULT_WORKERS,
    verbose: bool = False,
) -> dict:
    """
    检查 codes 中每只股票的缓存是否已更新到 today_str。
    未更新的自动重试一次，仍未更新则报告（可能是停牌）。
    返回 dict: verified_ok, retried_ok, still_stale
    """
    verified_ok = []
    retried_ok = []
    still_stale = []

    # 第一轮：快速检查所有缓存日期
    stale_codes = []
    for code in codes:
        c = normalize_prefixed(code)
        latest = _get_cache_latest_date(c)
        if latest is None or latest < today_str:
            stale_codes.append(c)
        else:
            verified_ok.append(c)

    if not stale_codes:
        return {"verified_ok": verified_ok, "retried_ok": retried_ok, "still_stale": still_stale}

    print(f"\n🔍 验证完成: {len(verified_ok)} 只已有今日数据，{len(stale_codes)} 只缓存较旧，进行重试...")

    # 重试拉取
    def work(code: str):
        try:
            df = load_qfq_history(code, adjust="qfq", refresh=True, max_age_hours=0)
            if df is not None and not df.empty:
                latest = df["date"].max().strftime("%Y-%m-%d")
                return code, latest, None
            return code, None, "empty"
        except Exception as e:
            return code, None, str(e)

    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(work, code): code for code in stale_codes}
        for fut in as_completed(futures):
            code, latest, err = fut.result()
            done += 1
            if latest is not None and latest >= today_str:
                retried_ok.append(code)
                if verbose:
                    print(f"  ✓ {code}: 重试成功，最新 {latest}")
            else:
                still_stale.append((code, latest, err))
                if verbose:
                    print(f"  ✗ {code}: 仍未更新到今日（最新 {latest}），可能停牌")
            if done % 300 == 0 or done == len(stale_codes):
                print(f"   重试进度: {done}/{len(stale_codes)}", end="\r")

    print()
    return {"verified_ok": verified_ok, "retried_ok": retried_ok, "still_stale": still_stale}


def print_summary(results: dict, verify_result: dict | None = None):
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

    if verify_result:
        vr = verify_result
        total_ok = len(vr["verified_ok"]) + len(vr["retried_ok"])
        print("\n" + "=" * 60)
        print(f"📊 数据新鲜度验证（今日= {datetime.now().strftime('%Y-%m-%d')}）")
        print(f"   ✅ 已有今日数据: {total_ok} 只（首次成功 {len(vr['verified_ok'])} + 重试成功 {len(vr['retried_ok'])})")
        if vr["still_stale"]:
            print(f"   ⚠️  仍未更新（疑似停牌）: {len(vr['still_stale'])} 只")
            for code, latest, err in vr["still_stale"][:10]:
                print(f"      {code}: 最新 {latest} | {err}")
            if len(vr["still_stale"]) > 10:
                print(f"      ... 还有 {len(vr['still_stale'])-10} 只")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="前复权日线数据预缓存（每个交易日 16:10 定时执行）")
    parser.add_argument("--codes", nargs="+", default=None, help="指定股票代码（默认全市场）")
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS, help=f"并行线程数（默认{DEFAULT_WORKERS}）"
    )
    parser.add_argument(
        "--max-age-hours",
        type=float,
        default=DEFAULT_MAX_AGE_HOURS,
        help=f"缓存超过多少小时视为过期需刷新（默认{DEFAULT_MAX_AGE_HOURS}小时）",
    )
    parser.add_argument("--refresh", action="store_true", help="强制刷新所有股票（含未过期）")
    parser.add_argument("--dry-run", action="store_true", help="只显示哪些会更新，不实际请求")
    parser.add_argument("-v", "--verbose", action="store_true", help="详细输出（显示每只股票数据条数）")
    parser.add_argument("--verify", action="store_true", default=None, help="刷新后验证所有缓存是否更新到今日（自动重试失败项，--refresh 时默认开启）")
    parser.add_argument("--no-verify", action="store_true", help="禁用刷新后验证")
    args = parser.parse_args()

    # --refresh 时默认开启验证，除非明确用 --no-verify 禁用
    do_verify = args.verify if args.verify is not None else args.refresh
    do_verify = do_verify and not args.no_verify

    codes = args.codes if args.codes else get_all_stock_codes()
    print(f"📋 待处理: {len(codes)} 只股票")
    if args.refresh:
        mode_str = f"🔄 强制刷新模式（{'✅验证开启' if do_verify else '⏭️ 验证关闭'}）"
    else:
        mode_str = f"⏱️  增量模式：缓存超过 {args.max_age_hours} 小时未更新才刷新"
    print(mode_str)

    results = prewarm_qfq_daily(
        codes=codes,
        workers=args.workers,
        force_refresh=args.refresh,
        dry_run=args.dry_run,
        verbose=args.verbose,
        max_age_hours=args.max_age_hours,
    )

    verify_result = None
    if do_verify and not args.dry_run:
        verify_result = verify_and_retry_stale(
            codes=codes,
            today_str=datetime.now().strftime("%Y-%m-%d"),
            workers=args.workers,
            verbose=args.verbose,
        )

    print_summary(results, verify_result)
