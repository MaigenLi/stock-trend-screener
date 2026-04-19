#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
前复权日线数据预缓存脚本
========================
每个交易日 16:40 定时执行，将全市场股票前复权日线数据刷新到本地缓存，
确保本地 K 线数据为最新（覆盖当日实时数据）。

用法：
  python cache_qfq_daily.py --date 2026-04-17        # 刷新目标日期数据
  python cache_qfq_daily.py --date 2026-04-17 --refresh  # 强制刷新并验证≥97%覆盖
  python cache_qfq_daily.py --date 2026-04-17 --codes 300568  # 指定股票
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
DEFAULT_COVERAGE_TARGET = 0.97  # 刷新验证覆盖率目标


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
    target_date: str | None = None,
) -> dict:
    """
    预热前复权日线缓存。

    Args:
        target_date: 若指定，--refresh 时只刷新尚未包含该日期的股票（已有该日期数据的跳过）。

    Returns:
        dict with keys: success, failed, skipped, elapsed_seconds
    """
    t0 = time.time()
    results = {"success": [], "failed": [], "skipped": [], "elapsed_seconds": 0.0}

    to_refresh = []
    for code in codes:
        c = normalize_prefixed(code)
        path = _cache_path(c, adjust="qfq")
        if force_refresh:
            # --refresh：检查缓存是否已有 target_date 数据，有则跳过
            if target_date:
                latest = _get_cache_latest_date(c)
                if latest == target_date:
                    results["skipped"].append(c)
                    continue
            to_refresh.append(c)
        elif _needs_refresh(path, max_age_hours):
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
    target_date: str,
    workers: int = DEFAULT_WORKERS,
    verbose: bool = False,
    coverage_target: float = DEFAULT_COVERAGE_TARGET,
) -> dict:
    """
    检查 codes 中每只股票的缓存是否已更新到 target_date。
    未更新的自动重试一次，仍未更新则报告（可能是停牌）。
    循环重试直到覆盖率 ≥ coverage_target 或无更多可更新股票。
    返回 dict: verified_ok, retried_ok, still_stale, final_coverage
    """
    verified_ok = []
    retried_ok = []
    still_stale = []
    total = len(codes)
    target_coverage = int(total * coverage_target)

    round_num = 0

    while True:
        round_num += 1
        stale_codes = []

        # 第一轮：快速检查所有缓存日期
        for code in codes:
            c = normalize_prefixed(code)
            latest = _get_cache_latest_date(c)
            if latest is None or latest < target_date:
                stale_codes.append(c)
            else:
                if c not in verified_ok and c not in retried_ok:
                    verified_ok.append(c)

        if not stale_codes:
            break

        current_coverage = len(verified_ok) + len(retried_ok)
        coverage_pct = current_coverage / total * 100 if total > 0 else 0

        # 已达到覆盖率目标，停止重试
        if current_coverage >= target_coverage:
            print(f"\n✅ 第{round_num-1}轮: 覆盖率 {coverage_pct:.1f}% ({current_coverage}/{total}) ≥ {coverage_target*100:.0f}% 目标，停止重试")
            break

        # 无更多可更新股票
        if round_num > 1 and not retried_ok:
            break

        print(f"\n🔍 第{round_num}轮: 缓存较旧 {len(stale_codes)} 只，已有 {current_coverage}/{total} ({coverage_pct:.1f}%)，进行重试...")

        retried_this_round = []
        still_stale_this_round = []

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
                if latest is not None and latest >= target_date:
                    retried_this_round.append(code)
                    retried_ok.append(code)
                    if verbose:
                        print(f"  ✓ {code}: 重试成功，最新 {latest}")
                else:
                    still_stale_this_round.append((code, latest, err))
                    if verbose:
                        print(f"  ✗ {code}: 仍未更新到 {target_date}（最新 {latest}），可能停牌")
                if done % 300 == 0 or done == len(stale_codes):
                    print(f"   重试进度: {done}/{len(stale_codes)}", end="\r")

        print()
        # 如果本轮没有任何重试成功，退出循环
        if not retried_this_round:
            print(f"⚠️  本轮重试无任何成功，停止重试（{len(stale_codes)} 只持续较旧，疑似停牌）")
            break

    final_coverage = len(verified_ok) + len(retried_ok)
    return {
        "verified_ok": verified_ok,
        "retried_ok": retried_ok,
        "still_stale": still_stale,
        "final_coverage": final_coverage,
    }


def print_summary(results: dict, verify_result: dict | None = None, target_date: str | None = None):
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
        total = len(vr["verified_ok"]) + len(vr["retried_ok"]) + len(vr["still_stale"])
        coverage_pct = total_ok / total * 100 if total > 0 else 0
        print("\n" + "=" * 60)
        print(f"📊 数据新鲜度验证（目标日期= {target_date}）")
        print(f"   ✅ 已有目标日期数据: {total_ok} 只 / {total} 只（覆盖率 {coverage_pct:.1f}%）")
        print(f"      首次成功: {len(vr['verified_ok'])} 只")
        print(f"      重试成功: {len(vr['retried_ok'])} 只")
        if vr["still_stale"]:
            print(f"   ⚠️  仍未更新（疑似停牌）: {len(vr['still_stale'])} 只")
            for code, latest, err in vr["still_stale"][:10]:
                print(f"      {code}: 最新 {latest} | {err}")
            if len(vr["still_stale"]) > 10:
                print(f"      ... 还有 {len(vr['still_stale'])-10} 只")
    elif target_date:
        # 没有 verify 时手动统计一次
        from stock_trend.gain_turnover import get_all_stock_codes
        codes = get_all_stock_codes()
        ok_count = sum(1 for c in codes if _get_cache_latest_date(normalize_prefixed(c)) == target_date)
        total = len(codes)
        print(f"\n📊 目标日期数据统计（{target_date}）: {ok_count} / {total} 只（{ok_count/total*100:.1f}%）")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="前复权日线数据预缓存（每个交易日收盘后执行）"
    )
    parser.add_argument(
        "--date", "-d", required=True,
        help="目标日期（YYYY-MM-DD），如 --date 2026-04-17"
    )
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
    parser.add_argument("--refresh", action="store_true",
        help="强制刷新所有股票（含未过期），并循环验证直到覆盖率≥97%%")
    parser.add_argument("--dry-run", action="store_true", help="只显示哪些会更新，不实际请求")
    parser.add_argument("-v", "--verbose", action="store_true", help="详细输出（显示每只股票数据条数）")
    parser.add_argument("--no-verify", action="store_true", help="禁用刷新后验证（仅 --refresh 时有效）")
    parser.add_argument("--coverage-target", type=float, default=DEFAULT_COVERAGE_TARGET,
        help=f"覆盖率目标（默认{DEFAULT_COVERAGE_TARGET*100:.0f}%%，仅 --refresh 时有效）")
    args = parser.parse_args()

    # 解析目标日期
    try:
        target_date = datetime.strptime(args.date, "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError:
        print(f"❌ 日期格式错误，请使用 YYYY-MM-DD，如 --date 2026-04-17")
        sys.exit(1)

    codes = args.codes if args.codes else get_all_stock_codes()
    print(f"📋 待处理: {len(codes)} 只股票 | 目标日期: {target_date}")

    mode_str = f"🔄 强制刷新模式" if args.refresh else f"⏱️ 增量模式：缓存超过 {args.max_age_hours} 小时未更新才刷新"
    if args.refresh:
        mode_str += f" | 覆盖率目标: {args.coverage_target*100:.0f}%"
        if not args.no_verify:
            mode_str += " | ✅验证开启"
        else:
            mode_str += " | ⏭️ 验证关闭"
    print(mode_str)

    # ── 刷新 ────────────────────────────────────────────
    results = prewarm_qfq_daily(
        codes=codes,
        workers=args.workers,
        force_refresh=args.refresh,
        dry_run=args.dry_run,
        verbose=args.verbose,
        max_age_hours=args.max_age_hours,
        target_date=target_date,
    )

    # ── 验证 & 循环重试（仅 --refresh 时）─────────────────
    verify_result = None
    if args.refresh and not args.dry_run and not args.no_verify:
        verify_result = verify_and_retry_stale(
            codes=codes,
            target_date=target_date,
            workers=args.workers,
            verbose=args.verbose,
            coverage_target=args.coverage_target,
        )

    print_summary(results, verify_result, target_date)

    # ── 目标日期数据为 0 → 非交易日 ──────────────────────
    if not args.dry_run:
        from stock_trend.gain_turnover import get_all_stock_codes
        all_codes = get_all_stock_codes()
        ok_count = sum(
            1 for c in all_codes
            if _get_cache_latest_date(normalize_prefixed(c)) == target_date
        )
        if ok_count == 0:
            print(f"\n❌ 错误: 目标日期 {target_date} 数据为 0，可能不是交易日")
            sys.exit(1)
        total = len(all_codes)
        coverage_pct = ok_count / total * 100 if total > 0 else 0
        print(f"\n✅ 目标日期 {target_date} 数据完整度: {ok_count}/{total} ({coverage_pct:.1f}%)")
        if args.refresh and not args.no_verify:
            target_num = int(total * args.coverage_target)
            if ok_count < target_num:
                print(f"⚠️  覆盖率 {coverage_pct:.1f}% < 目标 {args.coverage_target*100:.0f}%（{ok_count} < {target_num}），建议检查网络或重试")
