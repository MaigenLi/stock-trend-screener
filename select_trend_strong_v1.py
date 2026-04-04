#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
趋势强势股筛选器
================
从全市场筛选趋势强势股，基于：
  1. 均线多头排列（价格站在多条均线上方）
  2. 动量加速（短期涨幅 + 相对强弱）
  3. 量价配合（放量上涨）

使用方法：
  python select_trend_strong.py                        # 默认 Top30
  python select_trend_strong.py --top-n 50            # 前50只
  python select_trend_strong.py --score-threshold 60  # 评分>60
  python select_trend_strong.py --min-volume 1e8      # 成交额>1亿
  python select_trend_strong.py --codes sh600036      # 指定股票
"""

import os
import sys
import time
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

WORKSPACE = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(WORKSPACE))

from stock_common import (
    get_complete_kline,
    get_complete_kline_df,
    get_stock_snapshot,
    normalize_stock_code,
)

# ── 股票代码列表 ──────────────────────────────────────────
STOCK_CODES_FILE = Path("/home/hfie/stock_code/results/stock_codes.txt")

# ── 默认参数 ──────────────────────────────────────────────
DEFAULT_TOP_N = 30
DEFAULT_SCORE_THRESHOLD = 50
DEFAULT_MIN_VOLUME = 5e7       # 5000万
DEFAULT_MIN_DAYS = 60           # 上市>60交易日

# ── 评分权重（趋势强势股版）───────────────────────────────
WEIGHT_TREND = 0.50     # 趋势因子 50%（核心）
WEIGHT_MOMENTUM = 0.30  # 动量因子 30%
WEIGHT_VOLUME = 0.20    # 量价因子 20%


# ============================================================
#  工具函数
# ============================================================

def get_all_stock_codes() -> List[str]:
    """从 stock_code/results/stock_codes.txt 读取股票列表"""
    if not STOCK_CODES_FILE.exists():
        raise FileNotFoundError(f"股票代码文件不存在: {STOCK_CODES_FILE}")
    codes = []
    with open(STOCK_CODES_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            code = line.lower()
            if code.startswith(('sh', 'sz', 'bj')):
                codes.append(code)
            elif code.isdigit() and len(code) == 6:
                if code.startswith(('60', '68', '90')):
                    codes.append(f"sh{code}")
                elif code.startswith(('00', '30', '20')):
                    codes.append(f"sz{code}")
                elif code.startswith(('43', '83', '87', '92')):
                    codes.append(f"bj{code}")
                else:
                    codes.append(f"sh{code}")
    return codes


def compute_ma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n).mean()


def compute_ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()


# ============================================================
#  核心评分函数
# ============================================================

def score_trend_strong(df: pd.DataFrame) -> Tuple[float, Dict]:
    """
    趋势强势评分（满分100）
    
    维度：
      - 价格在均线上方的数量（MA5/MA10/MA20/MA60/MA120）
      - 均线多头排列数（MA5>MA10, MA10>MA20, MA20>MA60, MA60>MA120）
      - 均线发散程度（短期均线/长期均线比值）
      - 5日均线斜率
    """
    if df is None or len(df) < 120:
        return 0.0, {}

    close = df["close"]
    ma5 = compute_ma(close, 5)
    ma10 = compute_ma(close, 10)
    ma20 = compute_ma(close, 20)
    ma60 = compute_ma(close, 60)
    ma120 = compute_ma(close, 120)

    c = float(close.iloc[-1])

    # 1. 价格在均线上方（每条均线 8 分，满分 40）
    above_scores = []
    for ma, name in [(ma5, "MA5"), (ma10, "MA10"), (ma20, "MA20"), (ma60, "MA60"), (ma120, "MA120")]:
        val = float(ma.iloc[-1])
        above_scores.append(1 if c > val else 0)
    
    above_score = sum(above_scores) / 5 * 40

    # 2. 均线多头排列（每组 8 分，满分 32）
    bull_pairs = 0
    if float(ma5.iloc[-1]) > float(ma10.iloc[-1]):
        bull_pairs += 1
    if float(ma10.iloc[-1]) > float(ma20.iloc[-1]):
        bull_pairs += 1
    if float(ma20.iloc[-1]) > float(ma60.iloc[-1]):
        bull_pairs += 1
    if float(ma60.iloc[-1]) > float(ma120.iloc[-1]):
        bull_pairs += 1
    bull_score = bull_pairs / 4 * 32

    # 3. 均线发散度（MA5 / MA60 ratio，满分 20）
    ma60_val = float(ma60.iloc[-1])
    if ma60_val > 0:
        divergence = c / ma60_val  # 价格在60日线多少倍
        div_score = min(max((divergence - 1) * 50, 0), 20)  # 每超1%得0.5分，上限20
    else:
        div_score = 0

    # 4. 5日均线斜率（满分 8）
    if len(ma5) >= 6:
        ma5_slope = float(ma5.iloc[-1]) / float(ma5.iloc[-6]) - 1
        slope_score = min(max(ma5_slope * 200, 0), 8)  # 0.5%/日斜率得满分
    else:
        slope_score = 0

    total = above_score + bull_score + div_score + slope_score
    
    factors = {
        "above_count": sum(above_scores),
        "above_ma5": above_scores[0],
        "above_ma10": above_scores[1],
        "above_ma20": above_scores[2],
        "above_ma60": above_scores[3],
        "above_ma120": above_scores[4],
        "above_score": round(above_score, 2),
        "bull_pairs": bull_pairs,
        "bull_score": round(bull_score, 2),
        "divergence_ratio": round(c / ma60_val, 4) if ma60_val > 0 else 0,
        "div_score": round(div_score, 2),
        "ma5_slope_pct": round(slope_score / 200 * 100, 3) if slope_score > 0 else 0,
        "slope_score": round(slope_score, 2),
    }
    return total, factors


def score_momentum(df: pd.DataFrame) -> Tuple[float, Dict]:
    """
    动量评分（满分100）
    
    维度：
      - 20日累计涨幅（满分 35）
      - 10日累计涨幅（满分 25）
      - 创20日新高（满分 40）
    """
    if df is None or len(df) < 25:
        return 0.0, {}

    close = df["close"]

    # 20日涨幅
    if len(close) >= 21:
        gain_20d = float(close.iloc[-1]) / float(close.iloc[-21]) - 1
        gain_20d = max(gain_20d, 0)  # 只看正向动量
        gain_20d_score = min(gain_20d * 100, 35)
    else:
        gain_20d = 0.0
        gain_20d_score = 0.0

    # 10日涨幅
    if len(close) >= 11:
        gain_10d = float(close.iloc[-1]) / float(close.iloc[-11]) - 1
        gain_10d = max(gain_10d, 0)
        gain_10d_score = min(gain_10d * 100, 25)
    else:
        gain_10d = 0.0
        gain_10d_score = 0.0

    # 创20日新高
    if len(close) >= 22:
        high_20d = float(close.iloc[-22:-1].max())
        near_high_ratio = float(close.iloc[-1]) / high_20d - 1 if high_20d > 0 else 0
        if near_high_ratio >= 0:  # 创新高
            new_high_score = 40
        elif near_high_ratio >= -0.02:  # 接近新高（2%以内）
            new_high_score = 25
        else:
            new_high_score = max(0, 15 + near_high_ratio * 200)
    else:
        new_high_score = 0.0

    total = gain_20d_score + gain_10d_score + new_high_score

    factors = {
        "gain_20d_pct": round(gain_20d * 100, 3),
        "gain_10d_pct": round(gain_10d * 100, 3),
        "new_high_score": round(new_high_score, 2),
        "gain_20d_score": round(gain_20d_score, 2),
        "gain_10d_score": round(gain_10d_score, 2),
    }
    return total, factors


def score_vol_price(df: pd.DataFrame) -> Tuple[float, Dict]:
    """
    量价评分（满分100）
    
    维度：
      - 量比（5日均量基准，满分 35）
      - 成交额放大（20日均额基准，满分 35）
      - 放量上涨配合（满分 30）
    """
    if df is None or len(df) < 10:
        return 0.0, {}

    close = df["close"]
    amount = df["amount"]
    volume = df["volume"]

    # 量比
    if len(df) >= 6:
        vol_5d_avg = float(volume.iloc[-6:-1].mean())
        vol_today = float(volume.iloc[-1])
        vol_ratio = vol_today / vol_5d_avg if vol_5d_avg > 0 else 1.0
    else:
        vol_ratio = 1.0

    # 成交额放大
    if len(amount) >= 21:
        amt_20d_avg = float(amount.iloc[-21:-1].mean())
        amt_today = float(amount.iloc[-1])
        amt_ratio = amt_today / amt_20d_avg if amt_20d_avg > 0 else 1.0
    else:
        amt_ratio = 1.0

    # 价格变化
    if len(close) >= 2:
        price_change = float(close.iloc[-1]) / float(close.iloc[-2]) - 1
    else:
        price_change = 0.0

    # 量比得分
    vr_score = min(max((vol_ratio - 1) * 25, 0), 35)

    # 成交额放大得分
    ar_score = min(max((amt_ratio - 1) * 15, 0), 35)

    # 量价配合：放量 + 上涨
    if price_change > 0.01 and vol_ratio > 1.2:
        match_score = 30
    elif price_change > 0 and vol_ratio > 1:
        match_score = 20
    elif price_change > 0:
        match_score = 10
    else:
        match_score = 0

    total = vr_score + ar_score + match_score

    factors = {
        "vol_ratio": round(vol_ratio, 3),
        "amt_ratio": round(amt_ratio, 3),
        "price_change_pct": round(price_change * 100, 3),
        "vr_score": round(vr_score, 2),
        "ar_score": round(ar_score, 2),
        "match_score": round(match_score, 2),
    }
    return total, factors


def evaluate_stock(code: str, min_volume: float = DEFAULT_MIN_VOLUME,
                   exclude_st: bool = True) -> Optional[Dict]:
    """
    评估单只股票，返回评分结果或 None（被过滤）
    """
    try:
        result = get_complete_kline(code, allow_realtime_patch=True)
        df = result.data

        # 基本面过滤
        if df is None or df.empty:
            return None

        # 新股过滤
        if len(df) < DEFAULT_MIN_DAYS:
            return None

        # 成交额过滤
        if len(df) >= 20:
            avg_amount = float(df["amount"].iloc[-20:].mean())
            if avg_amount < min_volume:
                return None

        # ST 股过滤
        name = ""
        try:
            snap = get_stock_snapshot(code)
            name = getattr(snap, 'name', '') or ''
        except Exception:
            pass

        if exclude_st and name and ('ST' in name or 'S' in name):
            return None

        # 三维度评分
        trend_score, trend_factors = score_trend_strong(df)
        momentum_score, momentum_factors = score_momentum(df)
        vol_score, vol_factors = score_vol_price(df)

        # 加权总分
        total = (
            trend_score * WEIGHT_TREND +
            momentum_score * WEIGHT_MOMENTUM +
            vol_score * WEIGHT_VOLUME
        )

        return {
            "code": code,
            "name": name,
            "score": round(total, 2),
            "trend_score": round(trend_score, 2),
            "momentum_score": round(momentum_score, 2),
            "vol_score": round(vol_score, 2),
            "factors": {
                "trend": trend_factors,
                "momentum": momentum_factors,
                "volume": vol_factors,
            },
            "passed": True,
            "data_complete": result.is_complete,
        }

    except Exception as e:
        return None


def scan_market(codes: List[str], top_n: int = DEFAULT_TOP_N,
                min_volume: float = DEFAULT_MIN_VOLUME,
                score_threshold: float = DEFAULT_SCORE_THRESHOLD,
                max_workers: int = 30) -> List[Tuple[str, str, float, Dict]]:
    """
    扫描全市场，返回趋势强势股列表
    
    Returns:
        [(code, name, score, factors), ...] 按 score 降序
    """
    total = len(codes)
    print(f"🚀 开始扫描 {total} 只股票...")
    print(f"   参数: top_n={top_n}, min_volume={min_volume/1e8:.1f}亿, score_threshold={score_threshold}")
    print(f"   权重: 趋势={WEIGHT_TREND*100:.0f}% 动量={WEIGHT_MOMENTUM*100:.0f}% 量价={WEIGHT_VOLUME*100:.0f}%")
    print()

    results = []
    done = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(evaluate_stock, code, min_volume): code for code in codes}

        for future in as_completed(futures):
            done += 1
            if done % 500 == 0:
                elapsed = time.time() - t0
                eta = elapsed / done * (total - done) if done > 0 else 0
                print(f"  进度: {done}/{total} ({done/total*100:.1f}%) ETA={eta:.0f}s", end="\r")

            result = future.result()
            if result is not None and result["score"] >= score_threshold:
                results.append((
                    result["code"],
                    result["name"],
                    result["score"],
                    result["factors"],
                ))

    print(f"\n  扫描完成！{time.time()-t0:.1f}秒，共 {len(results)} 只通过阈值筛选")

    # 排序
    results.sort(key=lambda x: x[2], reverse=True)
    return results[:top_n]


def print_result(results: List[Tuple[str, str, float, Dict]], title: str = "趋势强势股"):
    """格式化打印结果"""
    if not results:
        print("\n⚠️  未筛选到符合条件的股票")
        return

    print(f"\n{'='*90}")
    print(f"📈 {title}（共 {len(results)} 只）")
    print(f"{'='*90}")
    print(f"{'代码':<10} {'名称':<10} {'总分':>6} {'趋势':>6} {'动量':>6} {'量价':>6} {'MA5上方':^5} {'MA多头':^5} {'20日涨幅':>8} {'量比':>6}")
    print(f"{'-'*90}")

    for code, name, score, factors in results:
        f_trend = factors.get("trend", {})
        f_mom = factors.get("momentum", {})
        f_vol = factors.get("volume", {})

        above_count = f_trend.get("above_count", 0)
        bull_pairs = f_trend.get("bull_pairs", 0)
        gain_20d = f_mom.get("gain_20d_pct", 0)
        vol_ratio = f_vol.get("vol_ratio", 0)

        print(f"{code:<10} {name:<10} {score:>6.1f} "
              f"{factors.get('trend_score', 0):>6.1f} "
              f"{factors.get('momentum_score', 0):>6.1f} "
              f"{factors.get('vol_score', 0):>6.1f} "
              f"{above_count}/5   {bull_pairs}/4   "
              f"{gain_20d:>7.2f}% {vol_ratio:>6.2f}")

    print(f"{'-'*90}")
    print(f"评分说明：总分 = 趋势×50% + 动量×30% + 量价×20%")
    print(f"          趋势 = 价格在均线上方(40) + 均线多头排列(32) + 均线发散度(20) + 斜率(8)")
    print(f"          动量 = 20日涨幅(35) + 10日涨幅(25) + 创新高(40)")
    print(f"          量价 = 量比(35) + 成交额放大(35) + 量价配合(30)")


# ============================================================
#  主入口
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="趋势强势股筛选器")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N, help=f"返回前N只（默认{DEFAULT_TOP_N}）")
    parser.add_argument("--score-threshold", type=float, default=DEFAULT_SCORE_THRESHOLD,
                        help=f"评分阈值（默认{DEFAULT_SCORE_THRESHOLD}）")
    parser.add_argument("--min-volume", type=float, default=DEFAULT_MIN_VOLUME,
                        help=f"最低成交额阈值（默认{DEFAULT_MIN_VOLUME/1e8:.1f}亿）")
    parser.add_argument("--codes", nargs="+", default=None, help="指定股票代码")
    parser.add_argument("--workers", type=int, default=30, help="并行线程数（默认30）")
    args = parser.parse_args()

    # 获取股票列表
    if args.codes:
        codes = [normalize_stock_code(c) for c in args.codes]
        print(f"📋 指定股票: {codes}")
    else:
        codes = get_all_stock_codes()
        print(f"📋 全市场股票: {len(codes)} 只")

    # 扫描
    results = scan_market(
        codes,
        top_n=args.top_n,
        min_volume=args.min_volume,
        score_threshold=args.score_threshold,
        max_workers=args.workers,
    )

    # 输出
    print_result(results)
