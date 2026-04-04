#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
趋势向上 + 放量突破选股器
==========================
专门捕捉"均线多头排列 + 放量突破关键阻力位"的强势股

核心理念：
  放量突破 = 量能先于价格异动
  趋势向上 = 均线系统呈多头扩散
  两者共振 = 主升浪信号

选股逻辑：
  1. 趋势向上：价格站稳20日均线，20日线向上
  2. 放量突破：今日成交量 > 5日均量×1.5，且突破20日/60日均线或前期高点
  3. 评分排序：三维度综合评分

用法：
  python select_breakout.py                        # 默认 Top30
  python select_breakout.py --top-n 50            # 前50只
  python select_breakout.py --score-threshold 55  # 评分>55
  python select_breakout.py --codes sh600036       # 指定股票
"""

import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

WORKSPACE = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(WORKSPACE))

from stock_common import (
    get_complete_kline,
    get_stock_snapshot,
    normalize_stock_code,
)

STOCK_CODES_FILE = Path("/home/hfie/stock_code/results/stock_codes.txt")

# ── 默认参数 ──────────────────────────────────────────────
DEFAULT_TOP_N = 30
DEFAULT_SCORE_THRESHOLD = 50
DEFAULT_MIN_VOLUME = 5e7       # 5000万
DEFAULT_MIN_DAYS = 60           # 上市>60交易日

# ── 评分权重 ──────────────────────────────────────────────
WEIGHT_BREAKOUT = 0.40   # 突破因子 40%（核心）
WEIGHT_VOLUME = 0.35     # 放量因子 35%
WEIGHT_TREND = 0.25      # 趋势因子 25%


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


# ============================================================
#  突破因子（满分100）
# ============================================================

def score_breakout(df: pd.DataFrame) -> Tuple[float, Dict]:
    """
    放量突破因子
    
    核心信号：
      1. 突破20日均线（收盘价 > MA20，且昨日收盘 < MA20）
      2. 突破60日均线（收盘价 > MA60，且昨日收盘 < MA60）
      3. 突破前期高点（20日内高点）
      4. 突破多条均线（同时突破 MA5+MA20，或 MA20+MA60）
    
    评分逻辑：
      - 突破类型（什么位置突破）：0-50分
      - 突破幅度（超越多少）：0-30分  
      - 突破力度（阳线力度）：0-20分
    """
    if df is None or len(df) < 25:
        return 0.0, {}

    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_ = df["open"]

    ma5 = compute_ma(close, 5)
    ma20 = compute_ma(close, 20)
    ma60 = compute_ma(close, 60)

    c = float(close.iloc[-1])
    c_prev = float(close.iloc[-2])
    h = float(high.iloc[-1])
    o = float(open_.iloc[-1])

    ma5_prev = float(ma5.iloc[-2]) if len(ma5) >= 2 else 0
    ma20_prev = float(ma20.iloc[-2]) if len(ma20) >= 2 else 0
    ma20_curr = float(ma20.iloc[-1])
    ma60_curr = float(ma60.iloc[-1])

    # ── 1. 识别突破类型 ────────────────────────────────────
    break_signals = []

    # 突破MA20（当日阳线突破）
    if c > ma20_curr and c_prev <= ma20_prev:
        break_signals.append(("break_ma20", True))

    # 突破MA60
    if c > ma60_curr and len(close) >= 21:
        ma60_prev = float(ma60.iloc[-2])
        if c_prev <= ma60_prev:
            break_signals.append(("break_ma60", True))

    # 突破20日高点
    if len(close) >= 22:
        high_20d = float(close.iloc[-22:-1].max())
        if c >= high_20d * 0.98:  # 接近/突破20日高点
            break_signals.append(("break_high20", True))

    # 突破5+20均线（双重突破）
    ma5_curr_v = float(ma5.iloc[-1]) if len(ma5) >= 1 else 0
    if c > ma5_curr_v and c > ma20_curr:
        ma5_prev_v = float(ma5.iloc[-2]) if len(ma5) >= 2 else 0
        ma20_prev_v = float(ma20.iloc[-2]) if len(ma20) >= 2 else 0
        if c_prev <= ma5_prev_v or c_prev <= ma20_prev_v:
            break_signals.append(("break_ma5_ma20", True))

    # ── 2. 计算突破幅度 ────────────────────────────────────
    max_break_pct = 0.0
    if len(close) >= 22:
        high_20d = float(close.iloc[-22:-1].max())
        max_break_pct = (c / high_20d - 1) * 100 if high_20d > 0 else 0
    break_ma20_pct = (c / ma20_curr - 1) * 100 if ma20_curr > 0 else 0
    break_ma60_pct = (c / ma60_curr - 1) * 100 if ma60_curr > 0 else 0
    break_pct = max(max_break_pct, break_ma20_pct, break_ma60_pct)

    # ── 3. 突破力度（阳线实体 vs 突破空间）────────────────
    body = c - o
    body_ratio = body / (c - o + 0.001) if (c - o) > 0 else 0  # 阳线实体占比
    upper_shadow = max(0, h - c)
    full_range = h - float(low.iloc[-1]) if len(low) >= 1 else c - o
    body_ratio = body / (full_range + 0.001)  # 实体/总振幅

    # ── 4. 评分计算 ────────────────────────────────────────
    break_type_score = 0
    if "break_ma5_ma20" in [s[0] for s in break_signals]:
        break_type_score = 50  # 双重突破，最高
    elif "break_ma60" in [s[0] for s in break_signals]:
        break_type_score = 40  # 突破60日线
    elif "break_ma20" in [s[0] for s in break_signals]:
        break_type_score = 30  # 突破20日线
    elif "break_high20" in [s[0] for s in break_signals]:
        break_type_score = 25  # 突破20日高点

    # 有突破信号但无明确类型识别
    if break_type_score == 0 and any([
        c > ma5_curr if len(ma5) >= 1 else False,
        c > ma20_curr,
    ]):
        # 价格在均线上方，但非今日突破，弱信号
        break_type_score = 10

    # 突破幅度得分（0-30）
    break_magnitude_score = min(max(break_pct * 10, 0), 30)

    # 力度得分（0-20）
    body_score = body_ratio * 20

    total = break_type_score + break_magnitude_score + body_score

    # ── 5. 缠论/结构补充 ────────────────────────────────────
    # 检查是否在山脚/山腰（非突破但趋势向上）
    above_ma20 = 1 if c > ma20_curr else 0
    ma20_up = 1 if ma20_curr > ma20_prev else 0  # 20日均线向上

    factors = {
        "break_signals": [s[0] for s in break_signals],
        "break_count": len(break_signals),
        "break_type_score": round(break_type_score, 2),
        "break_pct": round(break_pct, 3),
        "break_magnitude_score": round(break_magnitude_score, 2),
        "body_ratio": round(body_ratio, 3),
        "body_score": round(body_score, 2),
        "above_ma20": above_ma20,
        "ma20_up": ma20_up,
        "price_vs_ma20": round(c / ma20_curr - 1, 4) * 100 if ma20_curr > 0 else 0,
        "price_vs_ma60": round(c / ma60_curr - 1, 4) * 100 if ma60_curr > 0 else 0,
    }
    return total, factors


# ============================================================
#  放量因子（满分100）
# ============================================================

def score_volume(df: pd.DataFrame) -> Tuple[float, Dict]:
    """
    放量因子
    
    核心理念：
      - 量比（当日量 / 5日均量）反映是否明显放量
      - 连续放量（3日均量 > 5日均量）反映资金持续流入
      - 量价配合（放量上涨 > 缩量回调）
    
    评分：
      - 量比得分：0-40（量比>2.0满分）
      - 连续放量：0-30（持续放量满分）
      - 量价配合：0-30（放量且上涨满分）
    """
    if df is None or len(df) < 10:
        return 0.0, {}

    close = df["close"]
    volume = df["volume"]
    amount = df["amount"]

    # 量比（当日量 / 5日均量）
    vol_5d_avg = float(volume.iloc[-6:-1].mean()) if len(volume) >= 6 else float(volume.mean())
    vol_today = float(volume.iloc[-1])
    vol_ratio = vol_today / vol_5d_avg if vol_5d_avg > 0 else 1.0

    # 连续放量（3日均量 > 5日均量）
    if len(volume) >= 6:
        vol_3d_avg = float(volume.iloc[-3:].mean())
        vol_5d_avg_base = float(volume.iloc[-6:-3].mean())
        continuous_breakout = vol_3d_avg / (vol_5d_avg_base + 0.001) if vol_5d_avg_base > 0 else 1.0
    else:
        continuous_breakout = 1.0

    # 量价配合
    if len(close) >= 2:
        price_change = float(close.iloc[-1]) / float(close.iloc[-2]) - 1
    else:
        price_change = 0.0

    # 放量上涨 = 量比>1 且 涨幅>0
    # 缩量回调 = 量比<1 且 跌幅<0
    if price_change > 0 and vol_ratio > 1.0:
        vol_price_match = 1
    elif price_change < 0 and vol_ratio < 1.0:
        vol_price_match = 0.5  # 缩量回调，良性
    elif price_change > 0 and vol_ratio <= 1.0:
        vol_price_match = 0.3  # 缩量上涨
    else:
        vol_price_match = 0

    # 评分
    # 量比得分（0-40）：量比1.0=0分，量比2.0=40分
    vr_score = min(max((vol_ratio - 1.0) * 40, 0), 40)

    # 连续放量（0-30）
    cb_score = min(max((continuous_breakout - 1.0) * 30, 0), 30)

    # 量价配合（0-30）
    vp_score = vol_price_match * 30

    total = vr_score + cb_score + vp_score

    factors = {
        "vol_ratio": round(vol_ratio, 3),
        "continuous_breakout": round(continuous_breakout, 3),
        "price_change_pct": round(price_change * 100, 3),
        "vol_price_match": vol_price_match,
        "vr_score": round(vr_score, 2),
        "cb_score": round(cb_score, 2),
        "vp_score": round(vp_score, 2),
    }
    return total, factors


# ============================================================
#  趋势因子（满分100）
# ============================================================

def score_trend_up(df: pd.DataFrame) -> Tuple[float, Dict]:
    """
    趋势向上因子
    
    条件：
      - 价格在20日均线上方（山腰以上）
      - 20日均线向上倾斜
      - 20日线在60日线上方（趋势多头）
      - 均线呈发散状（短期 > 长期）
    
    评分：
      - 位置得分：0-35（价格相对均线位置）
      - 方向得分：0-35（均线方向）
      - 多头得分：0-30（均线排列）
    """
    if df is None or len(df) < 65:
        return 0.0, {}

    close = df["close"]
    ma5 = compute_ma(close, 5)
    ma20 = compute_ma(close, 20)
    ma60 = compute_ma(close, 60)

    c = float(close.iloc[-1])
    ma20_curr = float(ma20.iloc[-1])
    ma20_5d_ago = float(ma20.iloc[-6]) if len(ma20) >= 6 else ma20_curr
    ma60_curr = float(ma60.iloc[-1])
    ma5_curr = float(ma5.iloc[-1])

    # 1. 位置得分：价格在20日线上方（0-35）
    if c > ma20_curr:
        position_ratio = c / ma20_curr - 1
        position_score = min(position_ratio * 100 * 2, 35)  # 每超1%得2分，上限35
    else:
        position_ratio = 1 - c / ma20_curr
        position_score = max(0, 15 - position_ratio * 100 * 5)  # 低于均线扣分

    # 2. 方向得分：20日均线向上（0-35）
    ma20_slope = (ma20_curr / ma20_5d_ago - 1) * 100 if ma20_5d_ago > 0 else 0
    if ma20_slope > 0:
        direction_score = min(ma20_slope * 50, 35)  # 0.2%/日斜率满分
    else:
        direction_score = max(ma20_slope * 50 + 15, 0)  # 向下斜率扣分

    # 3. 多头排列得分（0-30）
    bull_count = 0
    if ma5_curr > ma20_curr:
        bull_count += 1
    if ma20_curr > ma60_curr:
        bull_count += 1
    bull_score = bull_count / 2 * 30

    total = position_score + direction_score + bull_score

    factors = {
        "c_above_ma20": 1 if c > ma20_curr else 0,
        "position_score": round(position_score, 2),
        "ma20_slope_pct": round(ma20_slope, 3),
        "direction_score": round(direction_score, 2),
        "bull_count": bull_count,
        "bull_score": round(bull_score, 2),
        "c_ma20_pct": round((c / ma20_curr - 1) * 100, 3) if ma20_curr > 0 else 0,
        "ma5_ma20_pct": round((ma5_curr / ma20_curr - 1) * 100, 3) if ma20_curr > 0 else 0,
    }
    return total, factors


# ============================================================
#  综合评估
# ============================================================

def evaluate_stock(code: str, min_volume: float = DEFAULT_MIN_VOLUME,
                   exclude_st: bool = True) -> Optional[Dict]:
    """评估单只股票"""
    try:
        result = get_complete_kline(code, allow_realtime_patch=True)
        df = result.data

        if df is None or df.empty:
            return None

        # 基础过滤
        if len(df) < DEFAULT_MIN_DAYS:
            return None

        if len(df) >= 20:
            avg_amount = float(df["amount"].iloc[-20:].mean())
            if avg_amount < min_volume:
                return None

        # 名称 & ST过滤
        name = ""
        try:
            snap = get_stock_snapshot(code)
            name = getattr(snap, 'name', '') or ''
        except Exception:
            pass

        if exclude_st and name and ('ST' in name or 'S' in name or '*' in name):
            return None

        # 三维度评分
        break_score, break_factors = score_breakout(df)
        vol_score, vol_factors = score_volume(df)
        trend_score, trend_factors = score_trend_up(df)

        # 加权总分
        total = (
            break_score * WEIGHT_BREAKOUT +
            vol_score * WEIGHT_VOLUME +
            trend_score * WEIGHT_TREND
        )

        return {
            "code": code,
            "name": name,
            "score": round(total, 2),
            "break_score": round(break_score, 2),
            "vol_score": round(vol_score, 2),
            "trend_score": round(trend_score, 2),
            "factors": {
                "breakout": break_factors,
                "volume": vol_factors,
                "trend": trend_factors,
            },
            "passed": True,
            "data_complete": result.is_complete,
        }

    except Exception:
        return None


def scan_market(codes: List[str], top_n: int = DEFAULT_TOP_N,
                min_volume: float = DEFAULT_MIN_VOLUME,
                score_threshold: float = DEFAULT_SCORE_THRESHOLD,
                max_workers: int = 30) -> List[Tuple]:
    """扫描全市场"""
    total = len(codes)
    print(f"🚀 开始扫描 {total} 只股票（趋势向上 + 放量突破）...")
    print(f"   参数: top_n={top_n}, min_volume={min_volume/1e8:.1f}亿, score_threshold={score_threshold}")
    print(f"   权重: 突破={WEIGHT_BREAKOUT*100:.0f}% 放量={WEIGHT_VOLUME*100:.0f}% 趋势={WEIGHT_TREND*100:.0f}%")
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

    results.sort(key=lambda x: x[2], reverse=True)
    return results[:top_n]


def print_result(results: List[Tuple], title: str = "趋势向上 + 放量突破"):
    """格式化输出"""
    if not results:
        print("\n⚠️  未筛选到符合条件的股票")
        return

    print(f"\n{'='*100}")
    print(f"📈 {title}（共 {len(results)} 只）")
    print(f"{'='*100}")
    print(f"{'代码':<10} {'名称':<10} {'总分':>6} {'突破':>6} {'放量':>6} {'趋势':>6} "
          f"{'突破信号':^18} {'量比':>5} {'20日均线上':^8} {'20日涨幅':>8}")
    print(f"{'-'*100}")

    for code, name, score, factors in results:
        fb = factors.get("breakout", {})
        fv = factors.get("volume", {})
        ft = factors.get("trend", {})

        signals = "/".join(fb.get("break_signals", [])[:2]) if fb.get("break_signals") else "-"
        vol_ratio = fv.get("vol_ratio", 0)
        ma20_up = "↑" if ft.get("ma20_up", 0) == 1 else "↓"
        c_above = ft.get("c_above_ma20", 0)
        gain_pct = fb.get("break_pct", 0)

        print(f"{code:<10} {name:<10} {score:>6.1f} "
              f"{factors.get('break_score', 0):>6.1f} "
              f"{factors.get('vol_score', 0):>6.1f} "
              f"{factors.get('trend_score', 0):>6.1f} "
              f"{signals:<18} {vol_ratio:>5.2f} {ma20_up}{c_above:<7} {gain_pct:>7.2f}%")

    print(f"{'-'*100}")
    print(f"评分说明：总分 = 突破×40% + 放量×35% + 趋势×25%")
    print(f"          突破 = 突破类型(50) + 突破幅度(30) + 阳线力度(20)")
    print(f"          放量 = 量比(40) + 连续放量(30) + 量价配合(30)")
    print(f"          趋势 = 价格位置(35) + 均线方向(35) + 多头排列(30)")


# ============================================================
#  主入口
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="趋势向上 + 放量突破选股器")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N, help=f"返回前N只（默认{DEFAULT_TOP_N}）")
    parser.add_argument("--score-threshold", type=float, default=DEFAULT_SCORE_THRESHOLD,
                        help=f"评分阈值（默认{DEFAULT_SCORE_THRESHOLD}）")
    parser.add_argument("--min-volume", type=float, default=DEFAULT_MIN_VOLUME,
                        help=f"最低成交额（默认{DEFAULT_MIN_VOLUME/1e8:.1f}亿）")
    parser.add_argument("--codes", nargs="+", default=None, help="指定股票")
    parser.add_argument("--workers", type=int, default=30, help="并行线程数")
    args = parser.parse_args()

    if args.codes:
        codes = [normalize_stock_code(c) for c in args.codes]
        print(f"📋 指定股票: {codes}")
    else:
        codes = get_all_stock_codes()
        print(f"📋 全市场股票: {len(codes)} 只")

    results = scan_market(
        codes,
        top_n=args.top_n,
        min_volume=args.min_volume,
        score_threshold=args.score_threshold,
        max_workers=args.workers,
    )

    print_result(results)
