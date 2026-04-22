#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RPS 热力图筛选器
===============
基于欧奈尔 RPS（相对价格强度）理论，复用本地缓存极速全市场扫描。

数据来源：本地缓存 ~/.openclaw/workspace/.cache/qfq_daily/（无需联网）

使用方法：
  python rps_strong_screen.py                        # 全市场 RPS 排序
  python rps_strong_screen.py --top 50               # Top50
  python rps_strong_screen.py --rps20 90 --rps60 90  # RPS 门槛
  python rps_strong_screen.py --date 2026-04-15      # 复盘
  python rps_strong_screen.py --codes sz000967       # 指定股票
"""

import re
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

WORKSPACE = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(WORKSPACE))

from stock_trend.gain_turnover import (
    load_qfq_history,
    get_stock_name,
    load_stock_names,
    normalize_prefixed,
)

DEFAULT_WORKERS = 8
DEFAULT_TOP_N = 50
DEFAULT_MIN_VOLUME = 0.5e8  # 0.5亿
DEFAULT_MIN_TURNOVER = 2.0  # %%（5日均换手率下限，市值相对）


def _pad(s: str, width: int) -> str:
    """按显示宽度填充：中文2字符，ASCII 1字符"""
    import wcwidth
    dwidth = sum(wcwidth.wcwidth(c) for c in s)
    return s + " " * (width - dwidth)


def _f(ls: list, widths: list[int]) -> str:
    parts = [_pad(str(ls[i]), widths[i]) for i in range(len(ls))]
    return "  ".join(parts)


def get_all_stock_codes() -> List[str]:
    """从 stock_codes.txt 读取全市场股票代码"""
    code_file = Path.home() / "stock_code/results/stock_codes.txt"
    if code_file.exists():
        with open(code_file, "r") as f:
            return [normalize_prefixed(line.strip()) for line in f if line.strip()]
    return []


def calc_stock_rps(
    code: str,
    names_cache: Optional[Dict[str, str]] = None,
    target_date: Optional[datetime] = None,
) -> Optional[Dict]:
    """
    计算单只股票的 RPS（20/60/120日）和基础指标。
    返回 None 表示被过滤（数据不足等）。
    """
    try:
        end_date = target_date.strftime("%Y-%m-%d") if target_date else None
        df = load_qfq_history(code, end_date=end_date, adjust="qfq")
        if df is None or df.empty:
            return None

        actual_last = str(df["date"].iloc[-1])[:10]
        # 校验：指定了目标日期，数据必须包含该日期
        if target_date is not None and actual_last < end_date:
            # 数据实际最新日期 < 目标日期，说明目标日期无交易或数据未更新
            return None

        if len(df) < 130:  # 至少需要120日数据
            return None

        close = df["close"].astype(float)
        amount_vals = df["amount"].astype(float)
        # 优先用真实换手率（volume / outstanding / 10000 * 100），无则降级
        if "true_turnover" in df.columns and df["true_turnover"].notna().any():
            turnover_vals = df["true_turnover"].astype(float)
        else:
            turnover_vals = df["turnover"].astype(float)

        name = ""
        if names_cache is not None:
            name = get_stock_name(code, names_cache) or ""

        # ST/*ST 过滤（仅匹配风险警示股票）
        if re.search(r'S[T\*]|^\*ST', name):
            return None

        # 换手率过滤（5日均，替代绝对成交额，市值相对）
        avg_turnover_5 = float(turnover_vals.iloc[-5:].mean())
        if avg_turnover_5 < DEFAULT_MIN_TURNOVER:
            return None

        # 计算各周期涨幅
        def _ret(days: int) -> Optional[float]:
            if len(close) < days + 1:
                return None
            r = float(close.iloc[-1]) / float(close.iloc[-days - 1]) - 1
            return r * 100  # 转为百分比

        ret20 = _ret(20)
        ret60 = _ret(60)
        ret120 = _ret(120)
        ret5 = _ret(5)
        ret3 = _ret(3)

        # 3日收盘都在MA5上方（蓄势确认）
        ma5_series = close.rolling(5).mean()
        ma5_3day_above = all(
            float(close.iloc[-(d)]) > float(ma5_series.iloc[-(d)])
            for d in [1, 2, 3]
        ) if len(close) >= 6 else False

        if ret20 is None or ret60 is None or ret120 is None:
            return None

        ret10 = _ret(10)

        # 10日涨幅 > 40% 过滤
        if ret10 is not None and ret10 > 40:
            return None

        # RSI-14
        rsi_val = 50.0
        if len(close) >= 15:
            delta = close.diff()
            gain = delta.clip(lower=0)
            loss = (-delta).clip(lower=0)
            avg_gain = gain.rolling(14).mean().iloc[-1]
            avg_loss = loss.rolling(14).mean().iloc[-1]
            if avg_loss > 0:
                rs = avg_gain / avg_loss
                rsi_val = float(100 - 100 / (1 + rs))
            else:
                rsi_val = 100.0

        # RSI > 88 过滤
        if rsi_val > 88:
            return None

        # 5日均成交额（亿元）
        avg_amount_5 = float(amount_vals.iloc[-5:].mean()) / 1e8

        return {
            "code": code,
            "name": name,
            "ret20": round(ret20, 2),
            "ret60": round(ret60, 2),
            "ret120": round(ret120, 2),
            "ret5": round(ret5, 2) if ret5 else 0.0,
            "ret3": round(ret3, 2) if ret3 else 0.0,
            "rsi": round(rsi_val, 1),
            "avg_turnover_5": round(avg_turnover_5, 2),  # %
            "avg_amount": round(avg_amount_5, 2),        # 亿元（5日均）
            "ma5_3day_above": ma5_3day_above,
            "data_date": actual_last,
        }

    except Exception:
        return None


def scan_rps(
    codes: List[str],
    top_n: int = DEFAULT_TOP_N,
    max_workers: int = DEFAULT_WORKERS,
    target_date: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    全市场 RPS 扫描。
    返回包含 RPS20/RPS60/RPS120 排名百分比的 DataFrame。
    """
    t0 = time.time()
    names_cache = load_stock_names()
    results = []

    total = len(codes)
    done = [0]

    print(f"📋 {'复盘模式: ' + target_date.strftime('%Y-%m-%d') if target_date else '全市场股票'}: {total} 只")
    print(f"🚀 开始扫描（workers={max_workers}）...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(calc_stock_rps, code, names_cache, target_date): code
            for code in codes
        }

        for future in as_completed(futures):
            done[0] += 1
            if done[0] % 500 == 0 or done[0] == total:
                eta = (time.time() - t0) / done[0] * (total - done[0])
                print(f"  进度: {done[0]}/{total} ({done[0]/total*100:.1f}%) ETA={eta:.0f}s", end="\r")

            result = future.result()
            if result is not None:
                results.append(result)

    print(f"\n✅ 扫描完成: {len(results)} 只有效股票, 用时 {time.time()-t0:.1f}s")

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # 计算 RPS（排名百分比，0~100）
    for col in ["ret20", "ret60", "ret120"]:
        df[col + "_rank"] = df[col].rank(ascending=True)
        df[col + "_rps"] = (df[col + "_rank"] / len(df) * 100).round(2)

    # 综合 RPS（等权平均）
    df["rps_avg"] = ((df["ret20_rps"] + df["ret60_rps"] + df["ret120_rps"]) / 3).round(2)

    # 综合排名分（考虑绝对涨幅 + RPS）
    df["composite"] = (
        df["ret20_rps"] * 0.4 +
        df["ret60_rps"] * 0.4 +
        df["ret120_rps"] * 0.2
    ).round(2)

    return df


def print_rps_table(df: pd.DataFrame, title: str = "RPS 热力图", top_n: int = 50):
    """格式化打印 RPS 排名表"""
    if df.empty:
        print("\n⚠️  无数据")
        return

    df = df.sort_values("composite", ascending=False).head(top_n)

    col_widths = [10, 8, 7, 7, 7, 6, 6, 7, 6, 6, 12]
    headers = ["代码", "名称", "RPS20", "RPS60", "RPS120", "均RPS", "综合分", "20日%",
               "RSI", "额(亿)", "数据日"]

    sep = "=" * 110
    print(f"\n{sep}")
    print(f"📊 {title}（共 {len(df)} 只）")
    print(sep)
    print(_f(headers, col_widths))
    print("-" * 110)

    for _, row in df.iterrows():
        rsi_penalty = ""
        if row["rsi"] > 82:
            rsi_penalty = "-5"
        elif row["rsi"] > 75:
            rsi_penalty = "-2"
        rsi_str = f"{row['rsi']:.1f}{rsi_penalty}"
        print(_f([
            row["code"],
            row["name"],
            f"{row['ret20_rps']:.1f}",
            f"{row['ret60_rps']:.1f}",
            f"{row['ret120_rps']:.1f}",
            f"{row['rps_avg']:.1f}",
            f"{row['composite']:.1f}",
            f"{row['ret20']:+.2f}%",
            rsi_str,
            f"{row['avg_amount']:.2f}",
            row["data_date"],
        ], col_widths))

    print("-" * 110)
    print("RPS 说明：")
    print("  RPS20/RPS60/RPS120 = 该周期涨幅在全场排名百分比（0~100）")
    print("  均RPS = 三周期 RPS 等权平均")
    print("  综合分 = RPS20×40% + RPS60×40% + RPS120×20%（相对强弱加权）")
    print("  20日% = 实际 20 日涨幅")
    print("过滤条件：RSI>88 过滤，10日涨幅>40% 过滤，成交额<0.5亿 过滤")


def print_sector_rps(df: pd.DataFrame, sector_map: Dict[str, str]):
    """按板块汇总 RPS 均值，展示板块热力"""
    if df.empty or not sector_map:
        return

    df = df.copy()
    df["sector"] = df["code"].map(sector_map)
    sector_group = df.groupby("sector").agg(
        count=("code", "count"),
        rps20_avg=("ret20_rps", "mean"),
        rps60_avg=("ret60_rps", "mean"),
        rps120_avg=("ret120_rps", "mean"),
        composite_avg=("composite", "mean"),
    ).sort_values("composite_avg", ascending=False)

    print(f"\n🔥 板块 RPS 热力（Top20，按综合分排序）")
    print("=" * 80)
    print(f"{'板块':<14} {'数量':>6} {'RPS20':>8} {'RPS60':>8} {'RPS120':>8} {'综合':>8}")
    print("-" * 80)
    for sector, row in sector_group.head(20).iterrows():
        print(f"{sector:<14} {row['count']:>6} "
              f"{row['rps20_avg']:>7.1f}  {row['rps60_avg']:>7.1f}  "
              f"{row['rps120_avg']:>7.1f}  {row['composite_avg']:>7.1f}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="RPS 热力图筛选器（本地缓存版）")
    parser.add_argument("--top", type=int, default=DEFAULT_TOP_N, help=f"输出前N只（默认{DEFAULT_TOP_N}）")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help=f"并行线程数（默认{DEFAULT_WORKERS}）")
    parser.add_argument("--codes", nargs="+", default=None, help="指定股票代码")
    parser.add_argument("--date", type=str, default=None, help="截止日期 YYYY-MM-DD（复盘用）")
    parser.add_argument("--rps20", type=float, default=0, help="RPS20 最低门槛（0=不限制）")
    parser.add_argument("--rps60", type=float, default=0, help="RPS60 最低门槛（0=不限制）")
    parser.add_argument("--rps120", type=float, default=0, help="RPS120 最低门槛（0=不限制）")
    parser.add_argument("--output", "-o", type=str, default=None, help="输出 CSV 路径")
    args = parser.parse_args()

    target_date = None
    if args.date:
        target_date = datetime.strptime(args.date, "%Y-%m-%d")
        print(f"📅 复盘模式: {args.date}")

    codes = [normalize_prefixed(c) for c in args.codes] if args.codes else get_all_stock_codes()
    if args.codes:
        print(f"📋 指定股票: {args.codes}")

    df = scan_rps(codes, top_n=args.top, max_workers=args.workers, target_date=target_date)

    if df.empty:
        print("⚠️  无有效数据")
        return

    # 过滤
    if args.rps20 > 0:
        df = df[df["ret20_rps"] >= args.rps20]
        print(f"  RPS20 ≥ {args.rps20}: {len(df)} 只")
    if args.rps60 > 0:
        df = df[df["ret60_rps"] >= args.rps60]
        print(f"  RPS60 ≥ {args.rps60}: {len(df)} 只")
    if args.rps120 > 0:
        df = df[df["ret120_rps"] >= args.rps120]
        print(f"  RPS120 ≥ {args.rps120}: {len(df)} 只")

    title_suffix = f"（RPS20≥{args.rps20} RPS60≥{args.rps60} RPS120≥{args.rps120})" if (args.rps20 or args.rps60 or args.rps120) else ""
    date_str = target_date.strftime("%Y-%m-%d") if target_date else datetime.now().strftime("%Y-%m-%d")
    print_rps_table(df, f"RPS 热力图 {date_str}{title_suffix}", top_n=args.top)

    if args.output:
        out_cols = ["code", "name", "ret20", "ret60", "ret120", "ret5", "rsi",
                    "ret20_rps", "ret60_rps", "ret120_rps", "rps_avg", "composite",
                    "avg_amount", "data_date"]
        df[out_cols].to_csv(args.output, index=False, encoding="utf-8")
        print(f"\n💾 结果已保存: {args.output}")


if __name__ == "__main__":
    main()
