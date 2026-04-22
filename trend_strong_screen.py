#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
趋势强势股筛选器 v2
==================
改进点（相比 v1）：
  1. RSI-14 超买过滤：高位股预警/过滤，避免追高
  2. 相对强弱（个股 vs 上证/深证/沪深300/创业板指数）：剔除随波逐流

数据来源：AkShare 前复权日线（与 gain_turnover 策略一致）

使用方法：
  python select_trend_strong.py                        # 默认 Top30
  python select_trend_strong.py --top-n 50            # 前50只
  python select_trend_strong.py --score-threshold 60  # 评分>60
  python select_trend_strong.py --codes sh600036       # 指定股票
  python select_trend_strong.py --date 2026-04-08     # 指定截止日期（复盘用）
"""

import re
import os
import sys
import time
import json
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, date as Date
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import akshare as ak

WORKSPACE = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(WORKSPACE))

from stock_trend.gain_turnover import (
    load_qfq_history,
    get_stock_name,
    load_stock_names,
    normalize_prefixed,
)

# ── 指数代码（AkShare格式）───────────────────────────────
INDEX_CODES = ["sh000001", "sz399001", "sh000300", "sz399006"]

# ── 指数缓存（进程内1小时有效）───────────────────────────
_INDEX_CACHE: Dict[str, Tuple[pd.DataFrame, float]] = {}
_INDEX_LOCK = threading.Lock()


def _get_index_history(code: str, days: int = 30) -> Optional[pd.DataFrame]:
    """获取指数日线（缓存1小时，AkShare来源）。"""
    now = time.time()
    key = code
    if key in _INDEX_CACHE:
        cached_df, cached_at = _INDEX_CACHE[key]
        if now - cached_at < 3600 and len(cached_df) >= days:
            return cached_df.tail(days).reset_index(drop=True)
    try:
        sym = code if code.startswith(("sh", "sz")) else f"sh{code}"
        df = ak.stock_zh_index_daily(symbol=sym)
        if df is not None and not df.empty:
            df = df.sort_values("date").reset_index(drop=True)
            _INDEX_CACHE[key] = (df, now)
            return df.tail(days).reset_index(drop=True)
    except Exception:
        pass
    if key in _INDEX_CACHE:
        return _INDEX_CACHE[key][0].tail(days).reset_index(drop=True)
    return None


# ── 股票代码列表 ────────────────────────────────────────
STOCK_CODES_FILE = Path.home() / "stock_code" / "results" / "stock_codes.txt"


def get_all_stock_codes() -> List[str]:
    """从 stock_code/results/stock_codes.txt 读取股票列表"""
    if not STOCK_CODES_FILE.exists():
        raise FileNotFoundError(f"股票代码文件不存在: {STOCK_CODES_FILE}")
    codes = []
    with open(STOCK_CODES_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            code = line.lower()
            if code.startswith(("sh", "sz", "bj")):
                codes.append(code)
            elif code.isdigit() and len(code) == 6:
                if code.startswith(("60", "68", "90")):
                    codes.append(f"sh{code}")
                elif code.startswith(("00", "30", "20")):
                    codes.append(f"sz{code}")
                elif code.startswith(("43", "83", "87", "92")):
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
    avg_gain = gain.ewm(alpha=1/n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/n, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def get_index_kline(code: str, days: int = 25,
                    target_date: Optional[datetime] = None) -> Optional[pd.DataFrame]:
    """
    获取指数K线（AkShare，缓存1小时）。
    target_date: 若指定则以该日期为截止日期。
    """
    df = _get_index_history(code, days + 10)
    if df is None or len(df) < days:
        return None
    if target_date is not None:
        from datetime import date as DateClass
        ts = target_date.date() if hasattr(target_date, "date") else target_date
        df = df[df["date"] <= ts]
        if df.empty:
            return None
    return df.tail(days).reset_index(drop=True)


def get_market_gain(index_codes: List[str], days: int = 21,
                    target_date: Optional[datetime] = None) -> float:
    """
    获取市场基准涨幅（多指数平均）。
    """
    gains = []
    for idx_code in index_codes:
        df = get_index_kline(idx_code, days + 5, target_date=target_date)
        if df is None or len(df) < days + 1:
            continue
        close = df["close"]
        start_price = float(close.iloc[-(days)])
        end_price = float(close.iloc[-1])
        if start_price > 0:
            gain_pct = (end_price / start_price - 1) * 100
            gains.append(gain_pct)
    if gains:
        return sum(gains) / len(gains)
    return 0.0


# ── 默认参数 ──────────────────────────────────────────────
DEFAULT_TOP_N = 30
DEFAULT_SCORE_THRESHOLD = 50
DEFAULT_WORKERS = 8   # 默认8线程，避免AkShare并发过高导致超时
DEFAULT_MIN_VOLUME = 5e7       # 5000万
DEFAULT_MIN_DAYS = 60           # 上市>60交易日

# ── 评分权重（趋势强势股版）───────────────────────────────
WEIGHT_TREND = 0.50     # 趋势因子 50%（核心）
WEIGHT_MOMENTUM = 0.30  # 动量因子 30%（含相对强弱调整）
WEIGHT_VOLUME = 0.20    # 量价因子 20%

# ── RSI 参数 ──────────────────────────────────────────────
RSI_PERIOD = 14
RSI_FILTER = 88        # RSI>88 直接过滤

# ── 相对强弱参数 ──────────────────────────────────────────
REL_STRENGTH_DISCOUNT = 0.5   # 相对强弱 < -5% 时动量得分打5折
REL_STRENGTH_FILTER = -10.0   # 相对强弱 < -10% 直接过滤（百分比）


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
    for ma in [ma5, ma10, ma20, ma60, ma120]:
        val = float(ma.iloc[-1])
        above_scores.append(1 if c > val else 0)

    above_score = sum(above_scores) / 5 * 40

    # 2. 均线多头排列（每组 8 分，满分 32）
    bull_pairs = 0
    if float(ma5.iloc[-1]) > float(ma10.iloc[-1]): bull_pairs += 1
    if float(ma10.iloc[-1]) > float(ma20.iloc[-1]): bull_pairs += 1
    if float(ma20.iloc[-1]) > float(ma60.iloc[-1]): bull_pairs += 1
    if float(ma60.iloc[-1]) > float(ma120.iloc[-1]): bull_pairs += 1
    bull_score = bull_pairs / 4 * 32

    # 3. 均线发散度（满分 20）
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
        "above_ma5": above_scores[0], "above_ma10": above_scores[1],
        "above_ma20": above_scores[2], "above_ma60": above_scores[3],
        "above_ma120": above_scores[4],
        "above_score": round(above_score, 2),
        "bull_pairs": bull_pairs, "bull_score": round(bull_score, 2),
        "divergence_ratio": round(c / ma60_val, 4) if ma60_val > 0 else 0,
        "div_score": round(div_score, 2),
        "ma5_slope_pct": round(slope_score / 200 * 100, 3) if slope_score > 0 else 0,
        "slope_score": round(slope_score, 2),
    }
    return total, factors


def score_momentum(df: pd.DataFrame, market_gain: float = 0.0) -> Tuple[float, Dict]:
    """
    动量评分（满分100）— 新增近强于中加分
    维度：
      - 20日累计涨幅（满分 35，≥30%满分）
      - 10日累计涨幅（满分 25，≥15%满分）
      - 创20日新高（满分 40）
      - 近强于中加分（+10，10日涨幅>20日涨幅×0.6时触发）
      - 相对强弱调整（市场基准对比）
    """
    if df is None or len(df) < 25:
        return 0.0, {}

    close = df["close"]

    # 20日涨幅（满分35，≥30%满分，线性递增）
    if len(close) >= 21:
        gain_20d = float(close.iloc[-1]) / float(close.iloc[-21]) - 1
        gain_20d = max(gain_20d, 0)
        gain_20d_score = min(gain_20d * 100 / 30.0 * 35.0, 35.0)  # ≥30%得35分
        gain_20d_pct = gain_20d * 100
    else:
        gain_20d = 0.0; gain_20d_score = 0.0; gain_20d_pct = 0.0

    # 10日涨幅（满分25，≥15%满分，线性递增）
    if len(close) >= 11:
        gain_10d = float(close.iloc[-1]) / float(close.iloc[-11]) - 1
        gain_10d = max(gain_10d, 0)
        gain_10d_score = min(gain_10d * 100 / 15.0 * 25.0, 25.0)  # ≥15%得25分
        gain_10d_pct = gain_10d * 100
    else:
        gain_10d = 0.0; gain_10d_score = 0.0; gain_10d_pct = 0.0

    # 创20日新高
    if len(close) >= 22:
        high_20d = float(close.iloc[-21:-1].max())
        near_high_ratio = float(close.iloc[-1]) / high_20d - 1 if high_20d > 0 else 0
        if near_high_ratio >= 0:
            new_high_score = 40
        elif near_high_ratio >= -0.02:
            new_high_score = 25
        else:
            new_high_score = max(0, 15 + near_high_ratio * 200)
    else:
        new_high_score = 0.0

    # 近强于中加分：10日涨幅 > 20日涨幅×0.6 时 +10分
    recent_strong_bonus = 0.0
    if gain_20d_pct > 0 and gain_10d_pct > gain_20d_pct * 0.6:
        recent_strong_bonus = 10.0

    # 相对强弱调整
    rel_strength = gain_20d_pct - market_gain

    momentum_raw = gain_20d_score + gain_10d_score + new_high_score + recent_strong_bonus
    if rel_strength < REL_STRENGTH_FILTER:
        momentum_adjusted = momentum_raw * REL_STRENGTH_DISCOUNT
        rel_strength_applied = True
    else:
        momentum_adjusted = momentum_raw
        rel_strength_applied = False

    total = momentum_adjusted

    factors = {
        "gain_20d_pct": round(gain_20d_pct, 3),
        "gain_10d_pct": round(gain_10d_pct, 3),
        "new_high_score": round(new_high_score, 2),
        "gain_20d_score": round(gain_20d_score, 2),
        "gain_10d_score": round(gain_10d_score, 2),
        "recent_strong_bonus": round(recent_strong_bonus, 2),
        "market_gain_pct": round(market_gain, 3),
        "rel_strength_pct": round(rel_strength, 3),
        "rel_strength_applied": rel_strength_applied,
    }
    return total, factors


def score_vol_price(df: pd.DataFrame) -> Tuple[float, Dict]:
    """量价评分（满分100）"""
    if df is None or len(df) < 10:
        return 0.0, {}

    close = df["close"]
    amount_col = df["amount"]
    volume = df["volume"]

    # 量比
    if len(df) >= 6:
        vol_5d_avg = float(volume.iloc[-6:-1].mean())
        vol_today = float(volume.iloc[-1])
        vol_ratio = vol_today / vol_5d_avg if vol_5d_avg > 0 else 1.0
    else:
        vol_ratio = 1.0

    # 成交额放大
    if len(amount_col) >= 21:
        amt_20d_avg = float(amount_col.iloc[-21:-1].mean())
        amt_today = float(amount_col.iloc[-1])
        amt_ratio = amt_today / amt_20d_avg if amt_20d_avg > 0 else 1.0
    else:
        amt_ratio = 1.0

    # 价格变化
    if len(close) >= 2:
        price_change = float(close.iloc[-1]) / float(close.iloc[-2]) - 1
    else:
        price_change = 0.0

    vr_score = min(max((vol_ratio - 1) * 25, 0), 35)
    ar_score = min(max((amt_ratio - 1) * 15, 0), 35)

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


def evaluate_stock(code: str,
                  min_volume: float = DEFAULT_MIN_VOLUME,
                  exclude_st: bool = True,
                  market_gain: float = 0.0,
                  names_cache: Optional[Dict[str, str]] = None,
                  target_date: Optional[datetime] = None) -> Optional[Dict]:
    """
    评估单只股票，返回评分结果或 None（被过滤）。
    target_date: 若指定则以该日期为截止日期。
    """
    try:
        end_date = target_date.strftime("%Y-%m-%d") if target_date else None
        df = load_qfq_history(code, end_date=end_date, adjust="qfq")
        if df is None or df.empty:
            return None
        if len(df) < DEFAULT_MIN_DAYS:
            return None

        close = df["close"].astype(float)
        amount_vals = df["amount"].astype(float)

        # 成交额过滤
        if len(df) >= 21:
            avg_amount = float(amount_vals.iloc[-20:].mean())
            if avg_amount < min_volume:
                return None

        # 名称
        name = ""
        if names_cache is not None:
            name = get_stock_name(code, names_cache)

        if exclude_st and name and re.search(r'S[T*]|^\*ST', name):
            return None

        # RSI-14
        rsi_val = 50.0
        if len(df) >= RSI_PERIOD + 1:
            rsi_series = compute_rsi(close, RSI_PERIOD)
            rsi_val = float(rsi_series.iloc[-1])

        if rsi_val > RSI_FILTER:
            return None

        # 10日涨幅 > 40% 直接过滤（涨幅过高风险大）
        if len(close) >= 11:
            gain_10d = float(close.iloc[-1]) / float(close.iloc[-11]) - 1
            if gain_10d > 0.40:
                return None

        # 三维度评分
        trend_score, trend_factors = score_trend_strong(df)
        momentum_score, momentum_factors = score_momentum(df, market_gain=market_gain)
        vol_score, vol_factors = score_vol_price(df)

        # RSI 惩罚
        # RSI 惩罚：真正按 RSI 区间扣分
        rsi_penalty = 0
        if rsi_val > 82:
            rsi_penalty = 5
        elif rsi_val > 75:
            rsi_penalty = 2

        raw = trend_score * WEIGHT_TREND + momentum_score * WEIGHT_MOMENTUM
        # 归一化：按实际满分归一（趋势满分100，动量满分110），再×100 → 0~100
        total = raw / (WEIGHT_TREND * 100.0 + WEIGHT_MOMENTUM * 110.0) * 100.0 - rsi_penalty
        total = max(total, 0)

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
            "data_date": str(df["date"].iloc[-1]),
        }

    except Exception:
        return None


def scan_market(codes: List[str],
               top_n: int = DEFAULT_TOP_N,
               min_volume: float = DEFAULT_MIN_VOLUME,
               score_threshold: float = DEFAULT_SCORE_THRESHOLD,
               max_workers: int = DEFAULT_WORKERS,
               target_date: Optional[datetime] = None) -> List[Tuple]:
    """
    扫描全市场，返回趋势强势股列表。
    target_date: 若指定则以该日期为截止日期。
    """
    total = len(codes)

    # 预加载名称缓存
    names_cache = load_stock_names()

    if target_date:
        date_str = target_date.strftime("%Y-%m-%d")
        print(f"🚀 扫描 {total} 只股票（截止日期: {date_str}）...")
    else:
        print(f"🚀 开始扫描 {total} 只股票...")

    # 预先计算市场基准涨幅
    t0_market = time.time()
    market_gain = get_market_gain(INDEX_CODES, days=21, target_date=target_date)
    if target_date:
        print(f"   市场基准（近21日，{target_date.strftime('%Y-%m-%d')}）: {market_gain:+.2f}%（耗时 {time.time()-t0_market:.1f}s）")
    else:
        print(f"   市场基准（近21日）: {market_gain:+.2f}%（耗时 {time.time()-t0_market:.1f}s）")

    print(f"   参数: top_n={top_n}, min_volume={min_volume/1e8:.1f}亿, score_threshold={score_threshold}")
    print(f"   权重: 趋势={WEIGHT_TREND*100:.0f}% 动量={WEIGHT_MOMENTUM*100:.0f}% 量价={WEIGHT_VOLUME*100:.0f}%")
    print(f"   RSI过滤: >{RSI_FILTER} 过滤, >82 扣5分, >75 扣2分")
    print(f"   相对强弱: <{REL_STRENGTH_FILTER}% 过滤, <{-5}% 动量5折")
    print()

    results = []
    done = [0]
    t0 = time.time()
    lock = threading.Lock()

    def process(code: str):
        result = evaluate_stock(code, min_volume, True, market_gain, names_cache, target_date)
        with lock:
            done[0] += 1
            if done[0] % 500 == 0:
                elapsed = time.time() - t0
                eta = elapsed / done[0] * (total - done[0]) if done[0] > 0 else 0
                print(f"  进度: {done[0]}/{total} ({done[0]/total*100:.1f}%) ETA={eta:.0f}s", end="\r")
        if result is not None and result["score"] >= score_threshold:
            with lock:
                results.append((
                    result["code"], result["name"], result["score"],
                    result["factors"], result["rsi"], result["rsi_penalty"],
                    result.get("data_date", ""),
                ))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(process, codes))

    print(f"\n  扫描完成！{time.time()-t0:.1f}秒，共 {len(results)} 只通过阈值筛选")

    results.sort(key=lambda x: x[2], reverse=True)
    return results if top_n <= 0 else results[:top_n]


def print_result(results: List[Tuple], title: str = "趋势强势股 v2"):
    """格式化打印结果"""
    if not results:
        print("\n⚠️  未筛选到符合条件的股票")
        return

    def _pad(s: str, width: int) -> str:
        """按显示宽度填充：中文2字符，ASCII 1字符"""
        import wcwidth
        dwidth = sum(wcwidth.wcwidth(c) for c in s)
        return s + " " * (width - dwidth)

    def _f(ls: list[str], widths: list[int]) -> str:
        """用显示宽度对齐分隔"""
        parts = [_pad(str(ls[i]), widths[i]) for i in range(len(ls))]
        return "  ".join(parts)

    col_widths = [10, 8, 6, 6, 6, 6, 9, 9, 9, 5, 9, 6, 12]

    print(f"\n{'='*110}")
    print(f"📈 {title}（共 {len(results)} 只）")
    print(f"{'='*110}")
    print(_f(["代码", "名称", "总分", "趋势", "动量", "量价", "RSI", "20日涨幅", "10日涨幅", "近强", "相对强弱", "量比", "数据日"], col_widths))
    print("-"*110)

    for item in results:
        code, name, score, factors = item[0], item[1], item[2], item[3]
        rsi = item[4] if len(item) > 4 else 0
        rsi_penalty = item[5] if len(item) > 5 else 0
        data_date = item[6] if len(item) > 6 else ""

        f_trend = factors.get("trend", {})
        f_mom = factors.get("momentum", {})
        f_vol = factors.get("volume", {})

        gain_20d = f_mom.get("gain_20d_pct", 0)
        gain_10d = f_mom.get("gain_10d_pct", 0)
        recent_strong_bonus = f_mom.get("recent_strong_bonus", 0)
        rel_strength = f_mom.get("rel_strength_pct", 0)
        vol_ratio = f_vol.get("vol_ratio", 0)

        trend_s = f_trend.get("above_score", 0) + f_trend.get("bull_score", 0) + f_trend.get("div_score", 0) + f_trend.get("slope_score", 0)
        momentum_s = f_mom.get("gain_20d_score", 0) + f_mom.get("gain_10d_score", 0) + f_mom.get("new_high_score", 0) + recent_strong_bonus
        vol_s = f_vol.get("vr_score", 0) + f_vol.get("ar_score", 0) + f_vol.get("match_score", 0)

        penalty_str = f"-{rsi_penalty}" if rsi_penalty > 0 else ""
        recent_strong_str = f"+{int(recent_strong_bonus)}" if recent_strong_bonus > 0 else "-"
        print(_f([
            code,
            name,
            f"{score:.1f}",
            f"{trend_s:.1f}",
            f"{momentum_s:.1f}",
            f"{vol_s:.1f}",
            f"{rsi:.1f}{penalty_str}",
            f"{gain_20d:+.2f}%",
            f"{gain_10d:+.2f}%",
            recent_strong_str,
            f"{rel_strength:+.2f}%",
            f"{vol_ratio:.2f}",
            str(data_date),
        ], col_widths))

    print("-"*110)
    print("评分说明：总分 = 趋势×50% + 动量×30% + 量价×20% - RSI惩罚")
    print("趋势 = 价格在均线上方(40) + 均线多头排列(32) + 均线发散度(20) + 斜率(8)")
    print("动量 = 20日涨幅(35) + 10日涨幅(25) + 创新高(40) + 近强于中(10)，按相对强弱调整")
    print("量价 = 量比(35) + 成交额放大(35) + 量价配合(30)")
    print("说明：近强=近强于中加分（+10），10日涨幅>20日涨幅×0.6时触发")


# ============================================================
#  主入口
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="趋势强势股筛选器 v2（AkShare前复权）")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N,
                        help=f"返回前N只（默认{DEFAULT_TOP_N}）")
    parser.add_argument("--score-threshold", type=float, default=DEFAULT_SCORE_THRESHOLD,
                        help=f"评分阈值（默认{DEFAULT_SCORE_THRESHOLD}）")
    parser.add_argument("--min-volume", type=float, default=DEFAULT_MIN_VOLUME,
                        help=f"最低成交额阈值（默认{DEFAULT_MIN_VOLUME/1e8:.1f}亿）")
    parser.add_argument("--codes", nargs="+", default=None, help="指定股票代码")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help=f"并行线程数（默认{DEFAULT_WORKERS}）")
    parser.add_argument("--date", type=str, default=None,
                        help="指定截止日期（YYYY-MM-DD），默认当前/最近交易日")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="输出文件路径（默认 ~/stock_reports/trend_strong_YYYY-MM-DD.txt）")
    args = parser.parse_args()

    # 解析目标日期
    target_date = None
    if args.date:
        if args.date.lower() == "today":
            target_date = datetime.now()
        else:
            try:
                target_date = datetime.strptime(args.date, "%Y-%m-%d")
            except ValueError:
                print(f"❌ 日期格式错误: {args.date}，应为 YYYY-MM-DD")
                sys.exit(1)

    # 获取股票列表
    if args.codes:
        codes = [normalize_prefixed(c) for c in args.codes]
        print(f"📋 指定股票: {codes}")
    else:
        codes = get_all_stock_codes()
        print(f"📋 全市场股票: {len(codes)} 只")

    if target_date:
        print(f"📅 复盘模式: 截止日期 {target_date.strftime('%Y-%m-%d')}")

    # 扫描
    results = scan_market(
        codes,
        top_n=args.top_n,
        min_volume=args.min_volume,
        score_threshold=args.score_threshold,
        max_workers=args.workers,
        target_date=target_date,
    )

    # 输出
    date_str = target_date.strftime('%Y-%m-%d') if target_date else datetime.now().strftime('%Y-%m-%d')
    title = f"趋势强势股 v2（{date_str}）"

    # 捕获输出写入文件
    import io, contextlib
    output_buffer = io.StringIO()
    with contextlib.redirect_stdout(output_buffer):
        print_result(results, title=title)
    output_text = output_buffer.getvalue()

    # 打印到终端
    print(output_text)

    # 保存到文件
    output_path = args.output
    if output_path is None:
        out_dir = Path.home() / "stock_reports"
        out_dir.mkdir(exist_ok=True)
        output_path = out_dir / f"trend_strong_{date_str}.txt"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output_text)
    print(f"💾 结果已保存: {output_path}")
