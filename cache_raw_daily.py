#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
原始（不复权）日线数据预缓存脚本
================================
每个交易日 16:40 定时执行，将全市场股票原始日线数据刷新到本地缓存，
数据为未复权的真实交易价格（前复权数据用于技术指标，原始数据用于看盘参考）。

用法：
  python cache_raw_daily.py --date 2026-04-17        # 刷新目标日期数据
  python cache_raw_daily.py --date 2026-04-17 --refresh  # 强制刷新并验证
  python cache_raw_daily.py --date 2026-04-17 --codes 000062  # 指定股票
  python cache_raw_daily.py --date 2026-04-17 --refresh-list  # 强制刷新股票列表后再刷新行情
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

WORKSPACE = Path.home() / ".openclaw/workspace"
CACHE_DIR = WORKSPACE / ".cache/raw_daily"

DEFAULT_WORKERS = 16
DEFAULT_MAX_AGE_HOURS = 6
DEFAULT_COVERAGE_TARGET = 0.97


def _cache_path(code: str) -> Path:
    """原始日线缓存路径"""
    return CACHE_DIR / f"{code}.csv"


def _needs_refresh(path: Path, max_age_hours: float) -> bool:
    if not path.exists():
        return True
    return (time.time() - path.stat().st_mtime) / 3600.0 >= max_age_hours


def _normalize(code: str) -> str:
    """统一格式：去掉 sh/sz/bj 前缀得到纯码"""
    return code.replace("sh", "").replace("sz", "").replace("bj", "")


def _fetch_raw(code: str) -> tuple[str, int | None, str | None]:
    """
    获取单只股票原始日线数据。
    返回 (code, n_rows, error)
    """
    try:
        import akshare as ak
        import pandas as pd

        # 确定交易所前缀
        c = _normalize(code)
        if c.startswith(("6", "5", "9")):
            sym = f"sh{c}"
        elif c.startswith(("0", "1", "2", "3")):
            sym = f"sz{c}"
        else:
            sym = code if not code.startswith(("sh", "sz", "bj")) else code

        df = ak.stock_zh_a_daily(
            symbol=sym,
            start_date="1990-01-01",
            end_date="2099-12-31",
            adjust="",  # 不复权
        )
        if df is None or df.empty:
            return code, None, "empty"

        # 统一列名（与前复权缓存格式完全一致）
        rename = {
            "日期": "date",
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "收盘": "close",
            "成交量": "volume",
            "成交额": "amount",
            "换手率": "turnover",
            "流通股本": "outstanding_share",
            "总市值": "total_market_cap",
            "流通市值": "float_market_cap",
        }
        df.rename(columns=rename, inplace=True)

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

        # 确保列顺序与前复权缓存一致
        cols_ordered = ["date", "open", "high", "low", "close", "volume", "amount"]
        for c in ["turnover", "outstanding_share"]:
            if c in df.columns:
                cols_ordered.append(c)
        # AkShare 原始换手率为小数（如 0.05727 表示 5.727%），转为百分比数值保留两位小数
        if "turnover" in df.columns:
            df["turnover"] = (df["turnover"] * 100).round(2)
            df["true_turnover"] = df["turnover"]
            cols_ordered.append("true_turnover")
        # 只保留已知列（防 akshare 新增列导致列顺序不一致）
        keep_cols = [c for c in cols_ordered if c in df.columns]
        df = df[keep_cols]

        cache_path = _cache_path(_normalize(code))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False, encoding="utf-8")
        return code, len(df), None
    except Exception as e:
        return code, None, str(e)


def prewarm_raw_daily(
    codes: list[str],
    workers: int = DEFAULT_WORKERS,
    force_refresh: bool = False,
    dry_run: bool = False,
    verbose: bool = False,
    max_age_hours: float = DEFAULT_MAX_AGE_HOURS,
    target_date: str | None = None,
) -> dict:
    """预热原始日线缓存"""
    t0 = time.time()
    results = {"success": [], "failed": [], "skipped": [], "elapsed_seconds": 0.0}

    to_refresh = []
    for code in codes:
        c = _normalize(code)
        path = _cache_path(c)
        if force_refresh:
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
        f"\n{'[Dry Run] ' if dry_run else ''}原始日线缓存: "
        f"{len(to_refresh)} 只要刷新 / {len(results['skipped'])} 只跳过"
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
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_fetch_raw, code): code for code in to_refresh}
        for fut in as_completed(futures):
            code, n, err = fut.result()
            done += 1
            if err is None:
                results["success"].append(code)
                if verbose:
                    print(f"  ✓ {code}: {n} 条")
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
    """返回缓存文件最新日期字符串，不存在返回 None"""
    path = _cache_path(code)
    if not path.exists():
        return None
    try:
        import pandas as pd
        df = pd.read_csv(path, parse_dates=["date"])
        return df["date"].max().strftime("%Y-%m-%d") if not df.empty else None
    except Exception:
        return None


def load_raw_history(code: str, start_date: str | None = None, end_date: str | None = None):
    """
    加载原始（不复权）日线数据。

    参数:
        code: 股票代码（如 '000062' 或 'sh600000'）
        start_date: 开始日期（YYYY-MM-DD）
        end_date: 结束日期（YYYY-MM-DD）

    返回:
        pd.DataFrame 或 None
    """
    import pandas as pd

    c = _normalize(code)
    path = _cache_path(c)
    if not path.exists():
        # 缓存不存在，先获取
        _, n, err = _fetch_raw(code)
        if err:
            return None

    try:
        df = pd.read_csv(path, parse_dates=["date"])
        if df is None or df.empty:
            return None
        if end_date:
            df = df[df["date"] <= end_date]
        if start_date:
            df = df[df["date"] >= start_date]
        return df
    except Exception:
        return None


def verify_and_retry_stale(
    codes: list[str],
    target_date: str,
    workers: int = DEFAULT_WORKERS,
    verbose: bool = False,
    coverage_target: float = DEFAULT_COVERAGE_TARGET,
) -> dict:
    """检查并重试未更新到 target_date 的缓存"""
    import pandas as pd

    verified_ok, retried_ok, still_stale = [], [], []
    total = len(codes)
    target_coverage = int(total * coverage_target)

    for round_num in range(1, 99):
        stale_codes = [
            c for c in codes
            if _get_cache_latest_date(c) is None or _get_cache_latest_date(c) < target_date
        ]
        if not stale_codes:
            break

        current = len(verified_ok) + len(retried_ok)
        if current >= target_coverage:
            print(f"\n✅ 第{round_num-1}轮: 覆盖率 {current/total*100:.1f}% ≥ {coverage_target*100:.0f}% 目标，停止")
            break

        print(f"\n🔍 第{round_num}轮: {len(stale_codes)} 只较旧，重试中...")
        this_round_ok, this_round_fail = [], []

        def work(code):
            return _fetch_raw(code)

        done = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(work, c): c for c in stale_codes}
            for fut in as_completed(futures):
                code, n, err = fut.result()
                done += 1
                if n is not None:
                    latest = _get_cache_latest_date(code)
                    if latest is not None and latest >= target_date:
                        this_round_ok.append(code)
                        retried_ok.append(code)
                        if verbose:
                            print(f"  ✓ {code}: 重试成功 {latest}")
                    else:
                        this_round_fail.append((code, latest, err))
                        if verbose:
                            print(f"  ✗ {code}: 最新 {latest}")
                else:
                    this_round_fail.append((code, None, err))
                if done % 300 == 0:
                    print(f"   {done}/{len(stale_codes)}", end="\r")

        print()
        if not this_round_ok:
            print(f"⚠️  重试无成功，停止（{len(stale_codes)}只疑似停牌）")
            still_stale.extend(this_round_fail)
            break

    final = len(verified_ok) + len(retried_ok)
    return {
        "verified_ok": verified_ok,
        "retried_ok": retried_ok,
        "still_stale": still_stale,
        "final_coverage": final,
    }


def print_summary(results: dict, verify_result: dict | None = None, target_date: str | None = None):
    print("\n" + "=" * 60)
    print(f"✅ 成功: {len(results['success'])} 只")
    if results["failed"]:
        print(f"❌ 失败: {len(results['failed'])} 只")
        for code, err in results["failed"][:10]:
            print(f"   {code}: {err}")
        if len(results["failed"]) > 10:
            print(f"   ... 还有 {len(results['failed'])-10} 只")
    print(f"⏭️  跳过: {len(results['skipped'])} 只（缓存有效）")
    print(f"⏱️  总耗时: {results['elapsed_seconds']:.1f}s")

    if verify_result:
        vr = verify_result
        total_ok = len(vr["verified_ok"]) + len(vr["retried_ok"])
        total = total_ok + len(vr["still_stale"])
        pct = total_ok / total * 100 if total > 0 else 0
        print(f"\n📊 数据新鲜度（目标={target_date}）")
        print(f"   ✅ 已更新: {total_ok}/{total} ({pct:.1f}%)")
        print(f"      首次: {len(vr['verified_ok'])}  重试: {len(vr['retried_ok'])}")
        if vr["still_stale"]:
            print(f"   ⚠️  仍未更新: {len(vr['still_stale'])} 只（疑似停牌）")
            for code, latest, err in vr["still_stale"][:10]:
                print(f"      {code}: {latest} | {err}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="原始（不复权）日线数据缓存")
    parser.add_argument("--date", "-d", required=True, help="目标日期 YYYY-MM-DD")
    parser.add_argument("--codes", nargs="+", default=None, help="指定股票代码")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="并行线程数")
    parser.add_argument("--max-age-hours", type=float, default=DEFAULT_MAX_AGE_HOURS)
    parser.add_argument("--refresh", action="store_true", help="强制刷新全部")
    parser.add_argument("--refresh-list", action="store_true", help="强制刷新股票列表后再刷新行情（联网获取最新全市场代码和名称）")
    parser.add_argument("--dry-run", action="store_true", help="仅显示不请求")
    parser.add_argument("-v", "--verbose", action="store_true", help="详细输出")
    parser.add_argument("--no-verify", action="store_true", help="禁用验证")
    parser.add_argument("--coverage-target", type=float, default=DEFAULT_COVERAGE_TARGET)
    args = parser.parse_args()

    try:
        target_date = datetime.strptime(args.date, "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError:
        print("❌ 日期格式错误，请使用 YYYY-MM-DD")
        sys.exit(1)

    # 如果没指定 codes，从 gain_turnover 拿全市场列表
    if args.codes:
        codes = [_normalize(c) for c in args.codes]
    else:
        sys.path.insert(0, str(WORKSPACE / "stock_trend"))
        from gain_turnover import get_all_stock_codes_akshare as get_all_stock_codes
        codes = [_normalize(c) for c in get_all_stock_codes()]

    print(f"📋 待处理: {len(codes)} 只 | 目标日期: {target_date}")
    mode = "🔄 强制刷新" if args.refresh else f"⏱️ 增量（>{args.max_age_hours}h）"
    if args.refresh_list:
        mode += " | 📋 强制刷新股票列表（联网更新全市场代码和名称）"
    print(mode)

    # ── 强制刷新股票列表（--refresh-list）──────────────────
    if args.refresh_list and not args.dry_run:
        sys.path.insert(0, str(WORKSPACE / "stock_trend"))
        from gain_turnover import (
            get_all_stock_codes_akshare, load_stock_names_akshare,
            AKSHARE_STOCK_CODES_FILE, AKSHARE_STOCK_NAMES_FILE,
        )
        for f in [AKSHARE_STOCK_CODES_FILE, AKSHARE_STOCK_NAMES_FILE]:
            if f.exists():
                f.unlink()
        print("📋 正在从 AkShare 获取最新全市场股票列表...")
        fresh_codes = get_all_stock_codes_akshare(force=True)
        fresh_names = load_stock_names_akshare(force=True)
        print(f"✅ 股票列表已更新: {len(fresh_codes)} 只代码, {len(fresh_names)} 个名称映射")
        codes = [_normalize(c) for c in fresh_codes]
        print(f"📋 本次待处理: {len(codes)} 只 | 目标日期: {target_date}")

    results = prewarm_raw_daily(
        codes=codes,
        workers=args.workers,
        force_refresh=args.refresh,
        dry_run=args.dry_run,
        verbose=args.verbose,
        max_age_hours=args.max_age_hours,
        target_date=target_date,
    )

    if args.refresh and not args.dry_run and not args.no_verify:
        verify_result = verify_and_retry_stale(
            codes=codes,
            target_date=target_date,
            workers=args.workers,
            verbose=args.verbose,
            coverage_target=args.coverage_target,
        )
    else:
        verify_result = None

    print_summary(results, verify_result, target_date)

    if not args.dry_run:
        ok_count = sum(
            1 for c in codes
            if _get_cache_latest_date(c) == target_date
        )
        total = len(codes)
        pct = ok_count / total * 100 if total > 0 else 0
        print(f"\n✅ 目标日期 {target_date} 完整度: {ok_count}/{total} ({pct:.1f}%)")