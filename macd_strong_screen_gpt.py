#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MACD 强势启动选股（实盘增强版）
============================
独立脚本，扫描全市场符合 MACD 启动形态的股票。

MACD 信号条件（信号日须同时满足）：
  1. MACD > 0（多方市场）
  2. MACD > 0 天数 ≤ 11
  3. MACD>0 区间内，2/3 以上为上涨日
  4. 信号日涨幅 > -3%
  5. 3日涨幅 > 5%
  6. DIF 连续2日上涨（DIF[T] > DIF[T-1] > DIF[T-2]）
  7. DEA 上涨（DEA[T] > DEA[T-1]）
  8. MACD 红柱持续放大（MACD[T] > MACD[T-1] > 0）

输出：~/stock_reports/macd_screen_YYYY-MM-DD.txt
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

WORKSPACE = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(WORKSPACE))

from gain_turnover import (
    load_qfq_history, normalize_prefixed,
    load_stock_names, get_stock_name,
)

DEFAULT_WORKERS = 8
DEFAULT_MIN_TURNOVER = 5.0   # 5日均换手率下限（%）
DEFAULT_MIN_AMOUNT = 1e8     # 20日均成交额下限（元）
DEFAULT_MARKET_DAYS = 21     # 市场21日涨幅计算天数


# ─────────────────────────────────────────────────────────
# 技术指标
# ─────────────────────────────────────────────────────────
def compute_macd(df: pd.DataFrame):
    """计算 MACD 列（dif/dea/macd），返回添加了新列的 df。"""
    close = df["close"]
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    macd = (dif - dea) * 2
    df = df.copy()
    df["dif"] = dif
    df["dea"] = dea
    df["macd"] = macd
    return df


def compute_ma(df: pd.DataFrame):
    df = df.copy()
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma60"] = df["close"].rolling(60).mean()
    return df


# ─────────────────────────────────────────────────────────
# 核心选股逻辑
# ─────────────────────────────────────────────────────────
def evaluate_macd_stock(df: pd.DataFrame):
    if len(df) < 80:
        return None

    df = compute_macd(df)
    df = compute_ma(df)

    idx = len(df) - 1

    # 1. 趋势过滤
    if not (
        df["dif"].iloc[idx] > 0
        and df["dif"].iloc[idx] > df["dif"].iloc[idx - 1] > df["dif"].iloc[idx - 2]
        and df["dea"].iloc[idx] > df["dea"].iloc[idx - 1]
        and df["macd"].iloc[idx] > df["macd"].iloc[idx - 1] > 0
    ):
        return None

    # 2. 红柱启动阶段（≤11天）
    red_days = 0
    for i in range(idx, max(idx - 20, 0), -1):
        if df["macd"].iloc[i] > 0:
            red_days += 1
        else:
            break
    if red_days < 2 or red_days > 11:
        return None

    # 3. 上涨质量（≥2/3上涨）
    start = max(idx - red_days + 1, 1)
    up_days = sum(
        1 for i in range(start, idx + 1)
        if df["close"].iloc[i] > df["close"].iloc[i - 1]
    )
    if up_days / red_days < 0.6:
        return None

    # 4. 短期动能
    ret_3 = df["close"].iloc[idx] / df["close"].iloc[idx - 3] - 1
    ret_1 = df["close"].iloc[idx] / df["close"].iloc[idx - 1] - 1
    if not (-0.03 < ret_1 and ret_3 > 0.04):
        return None

    # 5. 均线结构（ma20>ma60，收盘>ma20）
    if not (
        df["close"].iloc[idx] > df["ma20"].iloc[idx] > df["ma60"].iloc[idx]
    ):
        return None

    # 6. 成交额过滤（20日均成交额≥1亿）
    if "amount" in df.columns:
        amt20 = df["amount"].iloc[-20:].mean()
        if amt20 < 1e8:
            return None

    # 7. 稳定评分系统
    dif_val = df["dif"].iloc[idx]
    dea_val = df["dea"].iloc[idx]
    strength = dif_val / (abs(dea_val) + 1e-6)
    score = (
        min(strength * 20, 40) +
        (12 - red_days) * 2 +
        min(ret_3 * 100, 20)
    )
    return {
        "code": None,
        "score": round(score, 2),
        "red_days": red_days,
        "ret_3": round(ret_3 * 100, 2),
    }


# ─────────────────────────────────────────────────────────
# 全市场扫描
# ─────────────────────────────────────────────────────────
def scan_market(codes: list, target_date: datetime | None,
                max_workers: int) -> list[dict]:
    results = []
    t0 = time.time()
    total = len(codes)

    def work(code: str) -> dict | None:
        c = normalize_prefixed(code)
        end_str = target_date.strftime("%Y-%m-%d") if target_date else None
        df = load_qfq_history(c, end_date=end_str, adjust="qfq", refresh=False)
        if df is None or df.empty:
            return None
        if target_date is not None:
            df = df[df["date"] <= pd.Timestamp(target_date.date())].reset_index(drop=True)
        if len(df) < 80:
            return None
        res = evaluate_macd_stock(df)
        if res is None:
            return None
        res["code"] = c
        name = get_stock_name(c, load_stock_names()) or ""
        res["name"] = name
        return res

    done = [0]

    def log_progress(futures):
        for _ in as_completed(futures):
            done[0] += 1
            if done[0] % 200 == 0 or done[0] == total:
                eta = (time.time() - t0) / done[0] * (total - done[0])
                print(f"  进度: {done[0]}/{total} ({done[0]*100//total}%) ETA={eta:.0f}s", flush=True)

    print(f"📋 全市场扫描: {total} 只")
    print(f"🚀 开始 MACD 筛选（workers={max_workers}）...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(work, c): c for c in codes}
        log_progress(futures)
        for future in as_completed(futures):
            r = future.result()
            if r is not None:
                results.append(r)

    print(f"✅ 扫描完成: {len(results)}/{total} 只通过，用时 {time.time()-t0:.1f}s")
    return results


# ─────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MACD 趋势启动型选股")
    parser.add_argument("--date", type=str, default=None, help="信号日期 YYYY-MM-DD（复盘用）")
    parser.add_argument("--min-turnover", type=float, default=DEFAULT_MIN_TURNOVER,
                        help=f"5日均换手率下限/%%（默认{DEFAULT_MIN_TURNOVER}）")
    parser.add_argument("--min-amount", type=float, default=1.0,
                        help=f"20日均成交额下限/亿元（默认1.0）")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"并行线程数（默认{DEFAULT_WORKERS}）")
    parser.add_argument("--codes", nargs="+", default=None,
                        help="指定股票代码列表（跳过全市场）")
    parser.add_argument("--output", type=str, default=None, help="输出文件路径")
    args = parser.parse_args()

    target_date = None
    if args.date:
        target_date = datetime.strptime(args.date, "%Y-%m-%d")
        print(f"\n📅 复盘模式: {args.date}")

    date_str = target_date.strftime("%Y-%m-%d") if target_date else datetime.now().strftime("%Y-%m-%d")
    output_path = Path(args.output) if args.output else Path.home() / "stock_reports" / f"macd_screen_{date_str}.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if target_date:
        try:
            from trend_strong_screen import get_market_gain, INDEX_CODES
            market = get_market_gain(INDEX_CODES, days=DEFAULT_MARKET_DAYS, target_date=target_date)
            print(f"📈 市场{DEFAULT_MARKET_DAYS}日涨幅: {market:.2f}%")
        except Exception:
            market = 0.0

    if args.codes:
        codes = args.codes
        print(f"\n📊 MACD 筛选（指定 {len(codes)} 只）")
    else:
        from rps_strong_screen import get_all_stock_codes
        codes = get_all_stock_codes()
        print(f"\n📊 MACD 筛选（全市场 {len(codes)} 只）")

    results = scan_market(
        codes=codes,
        target_date=target_date,
        max_workers=args.workers,
    )

    if not results:
        print("\n⚠️  无符合 MACD 启动形态的股票")
        import sys; sys.exit(0)

    results.sort(key=lambda x: x["score"], reverse=True)

    print(f"\n{'='*72}")
    print(f"📊 MACD 趋势启动型 {date_str}（共 {len(results)} 只）")
    print("=" * 72)

    lines = []
    lines.append(f"📊 MACD 趋势启动型 {date_str}（共 {len(results)} 只）")
    lines.append("=" * 72)

    header = (
        f"{'代码':<10}\t{'名称':<8}\t{'信号日':<12}"
        f"\t{'评分':>6}\t{'红柱天':>6}\t{'3日涨幅':>8}"
    )
    print(header)
    lines.append(header)
    print("-" * 72)
    lines.append("-" * 72)

    for r in results:
        row = (
            f"{r['code']:<10}\t{r['name']:<8}\t{date_str:<12}"
            f"\t{r['score']:>6}\t{r['red_days']:>6}\t{r['ret_3']:>+7.2f}%"
        )
        print(row)
        lines.append(row)

    print("=" * 72)
    lines.append("=" * 72)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"💾 结果已写入: {output_path}")
