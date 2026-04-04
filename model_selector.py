#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
股票综合评分模型与选股脚本
===========================
基于趋势、动量、量价三维度评分的选股系统

用法：
  python model_selector.py                          # 默认筛选（Top30，成交额>5000万）
  python model_selector.py --top-n 50               # 返回前50只
  python model_selector.py --min-volume 1e8         # 成交额阈值1亿
  python model_selector.py --score-threshold 60    # 评分阈值60
  python model_selector.py --include-st             # 包含ST股
  python model_selector.py --explore                 # 仅运行数据分析
  python model_selector.py --codes sh600000 sz000001 # 指定股票
"""

import os
import sys
import time
import random
import struct
import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

# ── 路径设置 ──────────────────────────────────────────────
WORKSPACE = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(WORKSPACE))

from stock_common import (
    get_complete_kline,
    get_complete_kline_df,
    get_complete_kline_batch,
    get_stock_snapshot,
    normalize_stock_code,
    get_default_tdx_dir,
)

# 数据目录
TDX_DATA_DIR = Path(os.path.expanduser("~/stock_data/vipdoc"))

# ── 默认参数 ──────────────────────────────────────────────
DEFAULT_TOP_N = 30
DEFAULT_MIN_VOLUME = 5e7      # 5000万成交额
DEFAULT_SCORE_THRESHOLD = 50  # 评分阈值
DEFAULT_EXCLUDE_ST = True
DEFAULT_WINDOW = 120           # 分析窗口

# 评分因子权重
WEIGHT_TREND = 0.30
WEIGHT_MOMENTUM = 0.25
WEIGHT_VOLUME = 0.25

# ── 全局市场指数代码 ──────────────────────────────────────
INDEX_CODES = ["sh000001", "sz399001", "sz399006"]  # 上证/深证/创业板


# ============================================================
#  第一阶段：数据探索与分析
# ============================================================

def get_all_stock_codes() -> List[str]:
    """获取全市场所有股票代码"""
    codes = []
    for market in ["sh", "sz", "bj"]:
        day_dir = TDX_DATA_DIR / market / "lday"
        if not day_dir.exists():
            continue
        prefix = market
        for fp in day_dir.glob(f"{prefix}*.day"):
            pure = fp.stem[len(prefix):]
            # 排除指数
            if pure.isdigit() and len(pure) == 6:
                codes.append(f"{market}{pure}")
    return codes


def sample_stocks(codes: List[str], n: int = 50, seed: int = 42) -> List[str]:
    """随机抽取n只股票"""
    random.seed(seed)
    return random.sample(codes, min(n, len(codes)))


def explore_stock_data(code: str) -> Dict[str, Any]:
    """分析单只股票的数据特征"""
    try:
        df = get_complete_kline_df(code, allow_realtime_patch=False)
        if df is None or df.empty:
            return None

        df = df.sort_values("date").reset_index(drop=True)
        result = {
            "code": code,
            "record_count": len(df),
            "date_range": (str(df["date"].iloc[0].date()) if len(df) > 0 else None,
                           str(df["date"].iloc[-1].date()) if len(df) > 0 else None),
        }

        # 价格统计
        closes = df["close"].values
        if len(closes) > 0:
            result["price_mean"] = float(np.mean(closes))
            result["price_std"] = float(np.std(closes))
            result["price_min"] = float(np.min(closes))
            result["price_max"] = float(np.max(closes))

        # 涨跌幅统计
        if len(df) > 1:
            changes = df["close"].pct_change().dropna().values * 100
            result["change_mean"] = float(np.mean(changes))
            result["change_std"] = float(np.std(changes))
            result["change_max"] = float(np.max(changes))
            result["change_min"] = float(np.min(changes))

        # 成交额统计
        amounts = df["amount"].values
        if len(amounts) > 0:
            result["amount_mean"] = float(np.mean(amounts))
            result["amount_std"] = float(np.std(amounts))
            result["amount_median"] = float(np.median(amounts))
            # 异常值检测（成交额 > 3倍标准差）
            amean, astd = np.mean(amounts), np.std(amounts)
            outliers = np.sum((amounts > amean + 3 * astd) | (amounts < amean - 3 * astd))
            result["amount_outliers"] = int(outliers)

        # 换手率估算（假设流通股本=成交额/价格，简化计算）
        if len(df) > 0 and "volume" in df.columns:
            volumes = df["volume"].values
            result["volume_mean"] = float(np.mean(volumes))
            result["volume_std"] = float(np.std(volumes))

        # 波动率（20日标准差）
        if len(closes) >= 20:
            ret20 = np.diff(closes[-21:]) / closes[-21:-1] * 100
            result["volatility_20d"] = float(np.std(ret20))

        # ATR（简化版：日内波幅均值）
        if len(df) >= 14:
            tr = (df["high"] - df["low"]).values
            result["atr_14"] = float(np.mean(tr[-14:]))

        return result
    except Exception as e:
        return {"code": code, "error": str(e)}


def explore_market_features(codes: List[str], sample_n: int = 50) -> Dict[str, Any]:
    """市场特征分析"""
    print("\n" + "=" * 60)
    print("📊 第一阶段：数据探索与分析")
    print("=" * 60)

    # 采样分析
    sample_codes = sample_stocks(codes, sample_n)
    print(f"\n📌 采样 {len(sample_codes)} 只股票进行数据特征分析...")

    stock_results = []
    with ThreadPoolExecutor(max_workers=min(20, len(sample_codes))) as executor:
        futures = {executor.submit(explore_stock_data, code): code for code in sample_codes}
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result:
                stock_results.append(result)
            print(f"  [{i}/{len(sample_codes)}] {sample_codes[i-1]} 完成", end="\r")

    print()

    if not stock_results:
        return {}

    # 汇总统计
    record_counts = [r.get("record_count", 0) for r in stock_results if "record_count" in r]
    price_means = [r.get("price_mean", 0) for r in stock_results if "price_mean" in r]
    amount_means = [r.get("amount_mean", 0) for r in stock_results if "amount_mean" in r]
    volatilities = [r.get("volatility_20d", 0) for r in stock_results if "volatility_20d" in r]

    print(f"\n📈 采样股票数据特征汇总：")
    print(f"  • 记录数：均值={np.mean(record_counts):.0f}, 中位数={np.median(record_counts):.0f}, "
          f"范围=[{min(record_counts)}, {max(record_counts)}]")
    print(f"  • 价格：均值={np.mean(price_means):.2f}, 中位数={np.median(price_means):.2f}")
    print(f"  • 成交额：均值={np.mean(amount_means)/1e8:.2f}亿, 中位数={np.median(amount_means)/1e8:.2f}亿")
    print(f"  • 波动率(20d)：均值={np.mean(volatilities):.2f}%, 中位数={np.median(volatilities):.2f}%")

    # 涨跌幅分布
    changes = []
    for r in stock_results:
        if "change_mean" in r:
            changes.append(r["change_mean"])
    if changes:
        print(f"  • 日均涨跌幅：均值={np.mean(changes):.4f}%, 标准差={np.std(changes):.2f}%")

    # 成交额异常值比例
    total_outliers = sum(r.get("amount_outliers", 0) for r in stock_results)
    total_records = sum(r.get("record_count", 0) for r in stock_results)
    if total_records > 0:
        print(f"  • 成交额异常值比例：{total_outliers/total_records*100:.2f}%")

    # 全市场日均成交额趋势（用采样股票代表）
    print(f"\n📊 全市场日均成交额估算（基于采样）：")
    market_amount_mean = np.mean(amount_means) * (len(codes) / len(sample_codes))
    print(f"  • 全市场日均成交额估算：{market_amount_mean/1e8:.2f}亿")

    # 均线排列分析
    print(f"\n📊 趋势排列分析（采样股票）：")
    ma_arrangements = {"bull": 0, "neutral": 0, "bear": 0}
    for code in sample_codes[:20]:  # 快速检查20只
        try:
            df = get_complete_kline_df(code, allow_realtime_patch=False)
            if df is not None and len(df) >= 60:
                ma5 = df["close"].rolling(5).mean().iloc[-1]
                ma20 = df["close"].rolling(20).mean().iloc[-1]
                ma60 = df["close"].rolling(60).mean().iloc[-1]
                if ma5 > ma20 > ma60:
                    ma_arrangements["bull"] += 1
                elif ma5 < ma20 < ma60:
                    ma_arrangements["bear"] += 1
                else:
                    ma_arrangements["neutral"] += 1
        except:
            continue
    total_ma = sum(ma_arrangements.values())
    if total_ma > 0:
        print(f"  • 均线多头排列：{ma_arrangements['bull']/total_ma*100:.1f}%")
        print(f"  • 均线空头排列：{ma_arrangements['bear']/total_ma*100:.1f}%")
        print(f"  • 均线混乱：{ma_arrangements['neutral']/total_ma*100:.1f}%")

    return {
        "sample_count": len(stock_results),
        "market_estimate": market_amount_mean,
        "ma_arrangements": ma_arrangements,
    }


# ============================================================
#  第二阶段：综合评分模型
# ============================================================

def compute_ma(series: pd.Series, n: int) -> pd.Series:
    """计算移动平均线"""
    return series.rolling(n).mean()


def compute_ema(series: pd.Series, n: int) -> pd.Series:
    """计算指数移动平均线"""
    return series.ewm(span=n, adjust=False).mean()


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """计算ATR"""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr


def score_trend_factor(df: pd.DataFrame) -> Tuple[float, Dict[str, Any]]:
    """
    趋势因子（权重30%）
    - 价格在关键均线上方（20/60/120日均线）
    - 均线多头排列程度
    - 短期趋势方向（5日均线 vs 20日均线）
    """
    if df is None or len(df) < 120:
        return 0.0, {}

    close = df["close"]
    ma5 = compute_ma(close, 5)
    ma20 = compute_ma(close, 20)
    ma60 = compute_ma(close, 60)
    ma120 = compute_ma(close, 120)

    current_close = float(close.iloc[-1])

    # 价格在均线上方
    above_ma20 = 1 if current_close > float(ma20.iloc[-1]) else 0
    above_ma60 = 1 if current_close > float(ma60.iloc[-1]) else 0
    above_ma120 = 1 if current_close > float(ma120.iloc[-1]) else 0

    # 均线多头排列
    ma_bull_count = 0
    if float(ma5.iloc[-1]) > float(ma20.iloc[-1]):
        ma_bull_count += 1
    if float(ma20.iloc[-1]) > float(ma60.iloc[-1]):
        ma_bull_count += 1
    if float(ma60.iloc[-1]) > float(ma120.iloc[-1]):
        ma_bull_count += 1

    # 短期趋势（5日均线方向）
    ma5_slope = float(ma5.iloc[-1]) / float(ma5.iloc[-5]) - 1 if len(ma5) >= 5 else 0

    # 计算得分
    above_score = (above_ma20 * 0.3 + above_ma60 * 0.3 + above_ma120 * 0.4) * 40  # 0-40
    bull_score = (ma_bull_count / 3) * 40  # 0-40
    slope_score = min(max(ma5_slope * 100, 0), 20)  # 0-20，限制在±20%

    total = above_score + bull_score + slope_score
    factors = {
        "above_ma20": above_ma20,
        "above_ma60": above_ma60,
        "above_ma120": above_ma120,
        "ma_bull_count": ma_bull_count,
        "ma5_slope_pct": round(ma5_slope * 100, 3),
        "above_score": round(above_score, 2),
        "bull_score": round(bull_score, 2),
        "slope_score": round(slope_score, 2),
    }
    return total, factors


def score_momentum_factor(df: pd.DataFrame, index_df: Optional[pd.DataFrame] = None) -> Tuple[float, Dict[str, Any]]:
    """
    动量因子（权重25%）
    - 20日累计涨幅
    - 相对强弱（个股vs大盘）
    - 创阶段性新高/新低
    """
    if df is None or len(df) < 30:
        return 0.0, {}

    close = df["close"]

    # 20日累计涨幅
    if len(close) >= 21:
        gain_20d = float(close.iloc[-1]) / float(close.iloc[-21]) - 1
    else:
        gain_20d = 0.0

    # 10日累计涨幅
    if len(close) >= 11:
        gain_10d = float(close.iloc[-1]) / float(close.iloc[-11]) - 1
    else:
        gain_10d = 0.0

    # 相对强弱（个股 vs 大盘）
    relative_strength = 0.0
    if index_df is not None and len(index_df) >= 21:
        index_close = index_df["close"]
        index_gain = float(index_close.iloc[-1]) / float(index_close.iloc[-21]) - 1
        relative_strength = gain_20d - index_gain

    # 创阶段性新高/新低（20日区间）
    if len(close) >= 20:
        high_20d = float(close.iloc[-21:-1].max())
        low_20d = float(close.iloc[-21:-1].min())
        near_high = (current_close := float(close.iloc[-1])) / high_20d - 1 if high_20d > 0 else 0
        near_low = low_20d / current_close if current_close > 0 else 0
        new_high = 1 if current_close >= high_20d * 0.98 else 0  # 接近20日高点
        new_low = 1 if current_close <= low_20d * 1.02 else 0
    else:
        near_high = near_low = new_high = new_low = 0.0

    # 计算得分
    gain_score = min(max(gain_20d * 100, 0), 35)  # 0-35
    rs_score = min(max(relative_strength * 100, 0), 35)  # 0-35
    high_low_score = (new_high * 15 + min(max((1 - near_low) * 10, 0), 15))  # 0-30

    total = gain_score + rs_score + high_low_score
    factors = {
        "gain_20d_pct": round(gain_20d * 100, 3),
        "gain_10d_pct": round(gain_10d * 100, 3),
        "relative_strength": round(relative_strength * 100, 3),
        "near_high_20d_pct": round(near_high * 100, 3),
        "new_high": new_high,
        "new_low": new_low,
        "gain_score": round(gain_score, 2),
        "rs_score": round(rs_score, 2),
        "high_low_score": round(high_low_score, 2),
    }
    return total, factors


def score_volume_factor(df: pd.DataFrame) -> Tuple[float, Dict[str, Any]]:
    """
    量价因子（权重25%）
    - 放量上涨（量比）
    - 成交额放大程度
    - 量价配合程度
    """
    if df is None or len(df) < 20:
        return 0.0, {}

    close = df["close"]
    amount = df["amount"]

    # 量比（今日成交量/5日均量）
    if len(df) >= 6:
        vol_5d_avg = float(df["volume"].iloc[-6:-1].mean())
        vol_today = float(df["volume"].iloc[-1])
        volume_ratio = vol_today / vol_5d_avg if vol_5d_avg > 0 else 1.0
    else:
        volume_ratio = 1.0

    # 成交额放大程度（今日成交额/20日均成交额）
    if len(amount) >= 21:
        amount_20d_avg = float(amount.iloc[-21:-1].mean())
        amount_today = float(amount.iloc[-1])
        amount_ratio = amount_today / amount_20d_avg if amount_20d_avg > 0 else 1.0
    else:
        amount_ratio = 1.0

    # 量价配合（涨幅与量比的关系）
    if len(close) >= 2:
        price_change = float(close.iloc[-1]) / float(close.iloc[-2]) - 1
    else:
        price_change = 0.0

    # 放量上涨：量比>1 且 价格上涨
    rising = 1 if price_change > 0 else 0
    volume_up = 1 if volume_ratio > 1.0 else 0
    vol_price_match = rising * volume_up

    # 计算得分
    vr_score = min(max(volume_ratio - 1, 0) * 10, 30)  # 量比>1时加分，0-30
    ar_score = min(max(amount_ratio - 1, 0) * 8, 25)  # 成交额放大，0-25
    match_score = vol_price_match * 20 + (rising * 5)  # 量价配合，0-25

    total = vr_score + ar_score + match_score
    factors = {
        "volume_ratio": round(volume_ratio, 3),
        "amount_ratio": round(amount_ratio, 3),
        "price_change_pct": round(price_change * 100, 3),
        "rising": rising,
        "vol_price_match": vol_price_match,
        "vr_score": round(vr_score, 2),
        "ar_score": round(ar_score, 2),
        "match_score": round(match_score, 2),
    }
    return total, factors


def filter_basic_conditions(df: Optional[pd.DataFrame], exclude_st: bool = True,
                             min_volume: float = 0, min_days: int = 60) -> Tuple[bool, str]:
    """
    基本面筛选（辅助）
    - 非ST股票（通过名称判断）
    - 非新股（上市>60交易日）
    - 成交额下限（避免僵尸股）
    """
    if df is None or df.empty:
        return False, "无数据"

    # 新股判断（数据记录太少）
    if len(df) < min_days:
        return False, f"新股（仅{len(df)}交易日<{min_days}）"

    # 僵尸股判断（成交额过低）
    if len(df) >= 20:
        avg_amount = float(df["amount"].iloc[-20:].mean())
        if avg_amount < min_volume:
            return False, f"成交额不足（{avg_amount/1e8:.2f}亿<{min_volume/1e8:.2f}亿）"

    return True, "通过"


def score_stock(code: str, exclude_st: bool = True, min_volume: float = 5e7) -> Optional[Dict[str, Any]]:
    """
    评估单只股票的评分

    返回：
        {
            "code": str,
            "name": str,
            "score": float,           # 总分 0-100
            "trend_score": float,
            "momentum_score": float,
            "volume_score": float,
            "factors": dict,           # 各因子详情
            "passed": bool,
            "filter_reason": str,
        }
    """
    try:
        result = get_complete_kline(code, allow_realtime_patch=True)
        df = result.data

        passed, filter_reason = filter_basic_conditions(df, exclude_st, min_volume)
        if not passed:
            return {
                "code": code,
                "name": "",
                "score": 0.0,
                "trend_score": 0.0,
                "momentum_score": 0.0,
                "volume_score": 0.0,
                "factors": {},
                "passed": False,
                "filter_reason": filter_reason,
            }

        # 获取股票名称
        name = ""
        try:
            if result.realtime_snapshot:
                name = result.realtime_snapshot.get("name", "") or ""
            if not name:
                snap = get_stock_snapshot(code)
                name = getattr(snap, "name", "") or ""
        except Exception:
            name = ""

        # 过滤ST股
        if exclude_st and ("ST" in name or "*ST" in name or "S" in name):
            return {
                "code": code,
                "name": name,
                "score": 0.0,
                "trend_score": 0.0,
                "momentum_score": 0.0,
                "volume_score": 0.0,
                "factors": {},
                "passed": False,
                "filter_reason": "ST股",
            }

        # 计算三维度评分
        trend_score, trend_factors = score_trend_factor(df)
        momentum_score, momentum_factors = score_momentum_factor(df)
        volume_score, volume_factors = score_volume_factor(df)

        # 加权总分
        total_score = (
            trend_score * WEIGHT_TREND +
            momentum_score * WEIGHT_MOMENTUM +
            volume_score * WEIGHT_VOLUME
        )

        return {
            "code": code,
            "name": name,
            "score": round(total_score, 2),
            "trend_score": round(trend_score, 2),
            "momentum_score": round(momentum_score, 2),
            "volume_score": round(volume_score, 2),
            "factors": {
                "trend": trend_factors,
                "momentum": momentum_factors,
                "volume": volume_factors,
            },
            "passed": True,
            "filter_reason": "通过",
            "data_complete": result.is_complete,
            "used_realtime_patch": result.used_realtime_patch,
        }

    except Exception as e:
        return {
            "code": code,
            "name": "",
            "score": 0.0,
            "trend_score": 0.0,
            "momentum_score": 0.0,
            "volume_score": 0.0,
            "factors": {},
            "passed": False,
            "filter_reason": f"异常: {str(e)[:50]}",
        }


def score_stocks_batch(codes: List[str], exclude_st: bool = True, min_volume: float = 5e7,
                       max_workers: int = 15) -> List[Dict[str, Any]]:
    """批量评分"""
    results = []
    total = len(codes)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(score_stock, code, exclude_st, min_volume): code
            for code in codes
        }

        done = 0
        for future in as_completed(futures):
            done += 1
            result = future.result()
            if result:
                results.append(result)
            if done % 50 == 0 or done == total:
                print(f"  ⏳ 进度: {done}/{total} ({done*100//total}%)", end="\r")

    print(f"  ✅ 完成: {done}/{total}")
    return results


# ============================================================
#  第三阶段：选股接口
# ============================================================

def select_stocks(
    codes: Optional[List[str]] = None,
    top_n: int = DEFAULT_TOP_N,
    min_volume: float = DEFAULT_MIN_VOLUME,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    exclude_st: bool = DEFAULT_EXCLUDE_ST,
    max_workers: int = 15,
    verbose: bool = True,
) -> List[Tuple[str, str, float, Dict[str, Any]]]:
    """
    选股主函数

    参数：
        codes: 股票代码列表，None则扫描全市场
        top_n: 返回前N只股票
        min_volume: 最低成交额阈值（元）
        score_threshold: 最低评分阈值
        exclude_st: 是否排除ST股
        max_workers: 并行线程数
        verbose: 是否显示详细日志

    返回：
        [(code, name, score, {factors}), ...]，按评分降序排列
    """
    start_time = time.time()

    if verbose:
        print("\n" + "=" * 60)
        print("🔍 股票综合评分模型选股系统")
        print("=" * 60)
        print(f"📋 参数：top_n={top_n}, min_volume={min_volume/1e8:.1f}亿, "
              f"score_threshold={score_threshold}, exclude_st={exclude_st}")

    # 获取股票列表
    if codes is None:
        if verbose:
            print("\n📂 加载全市场股票列表...")
        codes = get_all_stock_codes()
        if verbose:
            print(f"   全市场股票数量：{len(codes)} 只")

    # 批量评分
    if verbose:
        print(f"\n📊 正在评分...")
    results = score_stocks_batch(codes, exclude_st=exclude_st, min_volume=min_volume,
                                  max_workers=max_workers)

    # 过滤并排序
    passed_results = [r for r in results if r["passed"] and r["score"] >= score_threshold]
    passed_results.sort(key=lambda x: x["score"], reverse=True)

    if verbose:
        passed_count = len(passed_results)
        total_count = len(results)
        avg_score = np.mean([r["score"] for r in results]) if results else 0
        print(f"\n📈 筛选结果：")
        print(f"   • 扫描股票：{total_count} 只")
        print(f"   • 通过筛选：{passed_count} 只")
        print(f"   • 平均评分：{avg_score:.2f}")
        print(f"   • 最高评分：{passed_results[0]['score'] if passed_results else 0:.2f}")
        print(f"   • 最低评分（Top{top_n}）：{passed_results[min(top_n-1, len(passed_results)-1)]['score'] if passed_results else 0:.2f}")

    # 返回格式化结果
    output = []
    for r in passed_results[:top_n]:
        output.append((r["code"], r["name"], r["score"], r))  # 4th element is full result dict

    elapsed = time.time() - start_time
    if verbose:
        print(f"\n⏱️ 耗时：{elapsed:.1f}秒")
        print("=" * 60)

    return output


def print_results(results: List[Tuple[str, str, float, Dict[str, Any]]], top_n: int = 30):
    """打印选股结果"""
    if not results:
        print("\n⚠️ 没有符合条件的股票")
        return

    print(f"\n🏆 选股结果（共{len(results)}只）：")
    print("-" * 100)
    print(f"{'排名':<4} {'代码':<10} {'名称':<10} {'评分':<6} {'趋势':<6} {'动量':<6} {'量价':<6} "
          f"{'20日涨幅':<8} {'量比':<6} {'说明'}")
    print("-" * 100)

    for i, (code, name, score, result) in enumerate(results[:top_n], 1):
        # result is the full dict from score_stock
        if isinstance(result, dict):
            factors = result.get("factors", {})
            trend_score = result.get("trend_score", 0)
            momentum_score = result.get("momentum_score", 0)
            volume_score = result.get("volume_score", 0)
        else:
            factors = {}
            trend_score = momentum_score = volume_score = 0

        trend = factors.get("trend", {})
        momentum = factors.get("momentum", {})
        volume = factors.get("volume", {})

        gain_20d = momentum.get("gain_20d_pct", 0)
        vol_ratio = volume.get("volume_ratio", 0)
        desc = ""

        if trend.get("new_high"):
            desc = "创新高"
        elif volume.get("vol_price_match"):
            desc = "放量上涨"

        print(f"{i:<4} {code:<10} {name:<10} {score:<6.1f} "
              f"{trend_score:<6.1f} {momentum_score:<6.1f} {volume_score:<6.1f} "
              f"{gain_20d:<8.2f} {vol_ratio:<6.2f} {desc}")

    if len(results) > top_n:
        print(f"... 还有 {len(results) - top_n} 只")


# ============================================================
#  主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="股票综合评分模型选股系统")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N, help=f"返回前N只（默认{DEFAULT_TOP_N}）")
    parser.add_argument("--min-volume", type=float, default=DEFAULT_MIN_VOLUME,
                        help=f"最低成交额阈值元（默认{DEFAULT_MIN_VOLUME}）")
    parser.add_argument("--score-threshold", type=float, default=DEFAULT_SCORE_THRESHOLD,
                        help=f"最低评分阈值（默认{DEFAULT_SCORE_THRESHOLD}）")
    parser.add_argument("--include-st", action="store_true", help="包含ST股（默认排除）")
    parser.add_argument("--explore", action="store_true", help="仅运行数据分析")
    parser.add_argument("--codes", nargs="+", default=None, help="指定股票代码")
    parser.add_argument("--workers", type=int, default=15, help="并行线程数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（用于采样分析）")

    args = parser.parse_args()

    exclude_st = not args.include_st

    # 数据探索模式
    if args.explore:
        random.seed(args.seed)
        all_codes = get_all_stock_codes()
        print(f"\n📊 全市场股票总数：{len(all_codes)} 只")
        explore_market_features(all_codes, sample_n=50)
        return

    # 指定股票代码
    if args.codes:
        codes = [normalize_stock_code(c) for c in args.codes]
    else:
        codes = None

    # 运行选股
    results = select_stocks(
        codes=codes,
        top_n=args.top_n,
        min_volume=args.min_volume,
        score_threshold=args.score_threshold,
        exclude_st=exclude_st,
        max_workers=args.workers,
        verbose=True,
    )

    # 打印结果
    print_results(results, top_n=args.top_n)

    # 保存结果到文件
    if results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = WORKSPACE / "stock_trend" / "results" / f"model_select_{timestamp}.json"
        result_file.parent.mkdir(parents=True, exist_ok=True)

        save_data = [
            {
                "rank": i + 1,
                "code": code,
                "name": name,
                "score": score,
                "factors": factors,
            }
            for i, (code, name, score, factors) in enumerate(results)
        ]
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        print(f"\n💾 结果已保存至：{result_file}")


if __name__ == "__main__":
    main()
