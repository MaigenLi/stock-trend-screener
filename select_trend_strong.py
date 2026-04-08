#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
趋势强势股筛选器 v2
==================
改进点（相比 v1）：
  1. RSI-14 超买过滤：高位股预警/过滤，避免追高
  2. 相对强弱（个股 vs 上证/深证指数）：剔除随波逐流

改进说明：
  - RSI>75 → 扣20分；RSI>82 → 扣40分；RSI>88 → 直接过滤
  - 相对强弱 = 个股20日涨幅 - 指数20日涨幅（使用本地TDX指数日线）
  - 相对强弱 < -5% → 动量得分打5折；<-10% → 过滤

使用方法：
  python select_trend_strong.py                        # 默认 Top30
  python select_trend_strong.py --top-n 50            # 前50只
  python select_trend_strong.py --score-threshold 60  # 评分>60
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
STOCK_CODES_FILE = Path.home() / "stock_code" / "results" / "stock_codes.txt"

# ── 默认参数 ──────────────────────────────────────────────
DEFAULT_TOP_N = 30
DEFAULT_SCORE_THRESHOLD = 50
DEFAULT_MIN_VOLUME = 5e7       # 5000万
DEFAULT_MIN_DAYS = 60           # 上市>60交易日

# ── 评分权重（趋势强势股版）───────────────────────────────
WEIGHT_TREND = 0.50     # 趋势因子 50%（核心）
WEIGHT_MOMENTUM = 0.30  # 动量因子 30%（含相对强弱调整）
WEIGHT_VOLUME = 0.20    # 量价因子 20%

# ── 指数代码 ──────────────────────────────────────────────
INDEX_CODES = ["sh000001", "sz399001", "sh000300", "sz399006"]  # 上证、深证、沪深300、创业板指

# ── RSI 参数 ──────────────────────────────────────────────
RSI_PERIOD = 14
RSI_PENALTY_75 = 20   # RSI 75~82 扣20分
RSI_PENALTY_82 = 40   # RSI 82~88 扣40分
RSI_FILTER = 88        # RSI>88 直接过滤

# ── 相对强弱参数 ──────────────────────────────────────────
REL_STRENGTH_DISCOUNT = 0.5   # 相对强弱 < -5% 时动量得分打5折
REL_STRENGTH_FILTER = -10.0   # 相对强弱 < -10% 直接过滤（百分比）


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


def compute_rsi(series: pd.Series, n: int = 14) -> pd.Series:
    """计算 RSI-n"""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    # 使用指数移动平均
    avg_gain = gain.ewm(alpha=1/n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/n, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def get_index_kline(code: str, days: int = 25) -> Optional[pd.DataFrame]:
    """
    获取指数K线（使用本地TDX数据，无需网络）。
    返回最近 N 天的 DataFrame 或 None。
    """
    try:
        result = get_complete_kline(code, allow_realtime_patch=True)
        df = result.data
        if df is None or df.empty or len(df) < days:
            return None
        return df.tail(days).reset_index(drop=True)
    except Exception:
        return None


def get_market_gain(index_codes: List[str], days: int = 21) -> float:
    """
    获取市场基准涨幅（多指数平均）。
    返回近 N 日指数平均涨幅百分比。
    如果所有指数都失败，返回 0（表示不调整）。
    """
    gains = []
    for idx_code in index_codes:
        df = get_index_kline(idx_code, days + 5)
        if df is None or len(df) < days + 1:
            continue
        close = df["close"]
        # 用近 days 天的收盘价计算涨幅
        start_price = float(close.iloc[-(days)])
        end_price = float(close.iloc[-1])
        if start_price > 0:
            gain_pct = (end_price / start_price - 1) * 100
            gains.append(gain_pct)
    if gains:
        return sum(gains) / len(gains)  # 多指数平均
    return 0.0


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
        divergence = c / ma60_val
        div_score = min(max((divergence - 1) * 50, 0), 20)
    else:
        div_score = 0

    # 4. 5日均线斜率（满分 8）
    if len(ma5) >= 6:
        ma5_slope = float(ma5.iloc[-1]) / float(ma5.iloc[-6]) - 1
        slope_score = min(max(ma5_slope * 200, 0), 8)
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


def score_momentum(df: pd.DataFrame, market_gain: float = 0.0) -> Tuple[float, Dict]:
    """
    动量评分（满分100）— v2 新增相对强弱调整

    维度：
      - 20日累计涨幅（满分 35）
      - 10日累计涨幅（满分 25）
      - 创20日新高（满分 40）
      - 相对强弱调整（市场基准对比）
    """
    if df is None or len(df) < 25:
        return 0.0, {}

    close = df["close"]

    # 20日涨幅
    if len(close) >= 21:
        gain_20d = float(close.iloc[-1]) / float(close.iloc[-21]) - 1
        gain_20d = max(gain_20d, 0)  # 只看正向动量
        gain_20d_score = min(gain_20d * 100, 35)
        gain_20d_pct = gain_20d * 100
    else:
        gain_20d = 0.0
        gain_20d_score = 0.0
        gain_20d_pct = 0.0

    # 10日涨幅
    if len(close) >= 11:
        gain_10d = float(close.iloc[-1]) / float(close.iloc[-11]) - 1
        gain_10d = max(gain_10d, 0)
        gain_10d_score = min(gain_10d * 100, 25)
        gain_10d_pct = gain_10d * 100
    else:
        gain_10d = 0.0
        gain_10d_score = 0.0
        gain_10d_pct = 0.0

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

    # ── 相对强弱调整（v2 新增）───────────────────────────
    rel_strength = gain_20d_pct - market_gain  # 个股涨幅 - 市场平均涨幅

    if rel_strength < REL_STRENGTH_FILTER:
        # 跑输大盘超过10% → 动量得分打5折
        momentum_raw = gain_20d_score + gain_10d_score + new_high_score
        momentum_adjusted = momentum_raw * REL_STRENGTH_DISCOUNT
        rel_strength_applied = True
    else:
        momentum_adjusted = gain_20d_score + gain_10d_score + new_high_score
        rel_strength_applied = False

    total = momentum_adjusted

    factors = {
        "gain_20d_pct": round(gain_20d_pct, 3),
        "gain_10d_pct": round(gain_10d_pct, 3),
        "new_high_score": round(new_high_score, 2),
        "gain_20d_score": round(gain_20d_score, 2),
        "gain_10d_score": round(gain_10d_score, 2),
        # 相对强弱
        "market_gain_pct": round(market_gain, 3),
        "rel_strength_pct": round(rel_strength, 3),
        "rel_strength_applied": rel_strength_applied,
    }
    return total, factors


def score_vol_price(df: pd.DataFrame) -> Tuple[float, Dict]:
    """
    量价评分（满分100）
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
                   exclude_st: bool = True,
                   market_gain: float = 0.0) -> Optional[Dict]:
    """
    评估单只股票，返回评分结果或 None（被过滤）。
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

        # ── RSI-14 计算（v2 新增）─────────────────────────
        rsi_val = 50.0
        if len(df) >= RSI_PERIOD + 1:
            rsi_series = compute_rsi(df["close"], RSI_PERIOD)
            rsi_val = float(rsi_series.iloc[-1])
        else:
            rsi_val = 50.0

        # RSI 超买过滤
        if rsi_val > RSI_FILTER:
            return None  # RSI>88 直接过滤

        # ── 三维度评分 ────────────────────────────────────
        trend_score, trend_factors = score_trend_strong(df)
        momentum_score, momentum_factors = score_momentum(df, market_gain=market_gain)
        vol_score, vol_factors = score_vol_price(df)

        # ── RSI 惩罚（v2 新增，在加权前扣除）─────────────
        rsi_penalty = 0
        if rsi_val > RSI_PENALTY_82:  # 82~88
            rsi_penalty = RSI_PENALTY_82
        elif rsi_val > RSI_PENALTY_75:  # 75~82
            rsi_penalty = RSI_PENALTY_75

        # 加权总分
        total = (
            trend_score * WEIGHT_TREND +
            momentum_score * WEIGHT_MOMENTUM +
            vol_score * WEIGHT_VOLUME
        ) - rsi_penalty
        total = max(total, 0)  # 不出现负分

        return {
            "code": code,
            "name": name,
            "score": round(total, 2),
            "trend_score": round(trend_score, 2),
            "momentum_score": round(momentum_score, 2),
            "vol_score": round(vol_score, 2),
            "rsi": round(rsi_val, 2),
            "rsi_penalty": rsi_penalty,
            "factors": {
                "trend": trend_factors,
                "momentum": momentum_factors,
                "volume": vol_factors,
            },
            "passed": True,
            "data_complete": result.is_complete,
        }

    except Exception:
        return None


def scan_market(codes: List[str], top_n: int = DEFAULT_TOP_N,
                min_volume: float = DEFAULT_MIN_VOLUME,
                score_threshold: float = DEFAULT_SCORE_THRESHOLD,
                max_workers: int = 30) -> List[Tuple[str, str, float, Dict]]:
    """
    扫描全市场，返回趋势强势股列表
    """
    total = len(codes)
    print(f"🚀 开始扫描 {total} 只股票...")

    # ── 预先计算市场基准涨幅（v2 新增）───────────────────
    t0_market = time.time()
    market_gain = get_market_gain(INDEX_CODES, days=21)
    print(f"   市场基准（近21日）：上证+深证平均涨幅 {market_gain:+.2f}%（耗时 {time.time()-t0_market:.1f}s）")

    print(f"   参数: top_n={top_n}, min_volume={min_volume/1e8:.1f}亿, score_threshold={score_threshold}")
    print(f"   权重: 趋势={WEIGHT_TREND*100:.0f}% 动量={WEIGHT_MOMENTUM*100:.0f}% 量价={WEIGHT_VOLUME*100:.0f}%")
    print(f"   RSI过滤: >{RSI_FILTER} 过滤, >{RSI_PENALTY_82} 扣{RSI_PENALTY_82}分, >{RSI_PENALTY_75} 扣{RSI_PENALTY_75}分")
    print(f"   相对强弱: <{REL_STRENGTH_FILTER}% 过滤, <{-5}% 动量5折")
    print()

    results = []
    done = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(evaluate_stock, code, min_volume, True, market_gain): code
            for code in codes
        }

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
                    result["rsi"],
                    result["rsi_penalty"],
                ))

    print(f"\n  扫描完成！{time.time()-t0:.1f}秒，共 {len(results)} 只通过阈值筛选")

    # 排序
    results.sort(key=lambda x: x[2], reverse=True)
    return results[:top_n]


def print_result(results: List[Tuple], title: str = "趋势强势股 v2"):
    """格式化打印结果"""
    if not results:
        print("\n⚠️  未筛选到符合条件的股票")
        return

    print(f"\n{'='*100}")
    print(f"📈 {title}（共 {len(results)} 只）")
    print(f"{'='*100}")
    print(f"{'代码':<10} {'名称':<10} {'总分':>6} {'趋势':>6} {'动量':>6} {'量价':>6} "
          f"{'RSI':>5} {'20日涨幅':>8} {'相对强弱':>8} {'量比':>6}")
    print(f"{'-'*100}")

    for item in results:
        code, name, score, factors = item[0], item[1], item[2], item[3]
        rsi = item[4] if len(item) > 4 else 0
        rsi_penalty = item[5] if len(item) > 5 else 0

        f_mom = factors.get("momentum", {})
        f_vol = factors.get("volume", {})

        gain_20d = f_mom.get("gain_20d_pct", 0)
        rel_strength = f_mom.get("rel_strength_pct", 0)
        vol_ratio = f_vol.get("vol_ratio", 0)

        penalty_str = f"-{rsi_penalty}" if rsi_penalty > 0 else ""
        print(f"{code:<10} {name:<10} {score:>6.1f} "
              f"{factors.get('trend_score', 0):>6.1f} "
              f"{factors.get('momentum_score', 0):>6.1f} "
              f"{factors.get('vol_score', 0):>6.1f} "
              f"{rsi:>5.1f}{penalty_str:<4} "
              f"{gain_20d:>7.2f}% {rel_strength:>+7.2f}% {vol_ratio:>6.2f}")

    print(f"{'-'*100}")
    print(f"评分说明：总分 = 趋势×50% + 动量×30% + 量价×20% - RSI惩罚")
    print(f"          趋势 = 价格在均线上方(40) + 均线多头排列(32) + 均线发散度(20) + 斜率(8)")
    print(f"          动量 = 20日涨幅(35) + 10日涨幅(25) + 创新高(40)，再按相对强弱调整")
    print(f"          量价 = 量比(35) + 成交额放大(35) + 量价配合(30)")
    print(f"v2 改进：RSI>88 过滤，RSI>82 扣40分，RSI>75 扣20分；相对强弱 < -10% 过滤，< -5% 动量5折")


# ============================================================
#  主入口
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="趋势强势股筛选器 v2")
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
