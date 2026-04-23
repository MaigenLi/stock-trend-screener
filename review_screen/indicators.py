#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
技术指标计算（趋势 + 量能）
"""

import numpy as np
import pandas as pd


def compute_macd(closes: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD指标"""
    ema_fast = pd.Series(closes).ewm(span=fast, adjust=False).mean().values
    ema_slow = pd.Series(closes).ewm(span=slow, adjust=False).mean().values
    dif = ema_fast - ema_slow
    dea = pd.Series(dif).ewm(span=signal, adjust=False).mean().values
    macd = 2.0 * (dif - dea)
    return dif, dea, macd


def compute_ma(closes: np.ndarray, periods: tuple = (5, 10, 20, 60)):
    """均线"""
    return {p: pd.Series(closes).rolling(p).mean().values for p in periods}


def detect_volume_price_wave(
    close: np.ndarray,
    volume: np.ndarray,
    lookback: int = 20,
    high: np.ndarray | None = None,
    low: np.ndarray | None = None,
) -> dict:
    """
    周期模式识别（结构波段版）

    涨跌定义（3日结构）：
      涨：c2 > c0 AND c1 < c2
          - 连续2日上涨：c0 < c1 < c2
          - 上下引线：c0 > c1 < c2 且 c0 < c2
          - 平再涨：c0 <= c1 < c2 且 c0 < c2
      跌：c2 < c0 AND c1 > c2（同理反向）
      平盘：不满足上述条件

    波段构造原则：
      1. 波段从第一个非中立日启动
      2. 中立日吸收到当前波段，不切断
      3. 反向非中立日出现时，先前波段确认结束，新波段开始

    Args:
        close: 收盘价数组（date升序）
        volume: 成交量数组
        lookback: 回溯天数

    Returns:
        recent_up_gt_down:  最近涨段均量 > 最近跌段均量
        up_vs_down_ratio:   涨段均量 / 跌段均量
        all_up_gt_down:     全部涨段均量 > 全部跌段均量
        wave_count:         识别出的波段数量
        last_wave_dir:      最近波段方向
        pattern_score:      模式评分（0-1）
        waves:              波段详情列表
    """
    n = min(lookback, len(close) - 1)
    scan_start = len(close) - n  # 实际扫描起始索引

    if n < 5:
        return {
            "recent_up_gt_down": False,
            "up_vs_down_ratio": 0.0,
            "all_up_gt_down": False,
            "wave_count": 0,
            "last_wave_dir": None,
            "pattern_score": 0.0,
            "waves": [],
        }

    # ── Step 1: 每日状态分类 ─────────────────────────────────
    # i=0,1 → 无足够历史，无法判断（中立）
    raw_states = ["neutral"] * 2  # 前2天中立
    for i in range(2, len(close)):
        c0, c1, c2 = close[i - 2], close[i - 1], close[i]
        if c2 > c0 and c1 < c2:
            raw_states.append("up")
        elif c2 < c0 and c1 > c2:
            raw_states.append("down")
        else:
            raw_states.append("neutral")

    # ── Step 2: 主波段构造（中立日吸收，反向需确认）────────────
    # 规则：
    #   1. 中立日不切段，吸收到当前主波段
    #   2. 反向信号出现时，不立即切段，先进入“候选反向”
    #   3. 候选反向累计到 2 个有效反向信号，才确认切换主波段
    waves = []
    wave_start = None
    wave_dir = None

    candidate_dir = None
    candidate_start = None
    candidate_count = 0

    for i in range(scan_start, len(close)):
        state = raw_states[i]

        if wave_dir is None:
            if state != "neutral":
                wave_start = i
                wave_dir = state
            continue

        # 当前没有候选反向
        if candidate_dir is None:
            if state == "neutral" or state == wave_dir:
                continue
            # 首次反向，进入候选观察
            candidate_dir = state
            candidate_start = i
            candidate_count = 1
            continue

        # 已经处于候选反向观察中
        if state == "neutral":
            continue

        if state == candidate_dir:
            candidate_count += 1
            if candidate_count >= 2:
                # 反向确认，前波段截止到候选起点前一天
                waves.append({
                    "start_idx": wave_start,
                    "end_idx": candidate_start - 1,
                    "direction": wave_dir,
                })
                wave_start = candidate_start
                wave_dir = candidate_dir
                candidate_dir = None
                candidate_start = None
                candidate_count = 0
            continue

        if state == wave_dir:
            # 候选反向失败，仍属于原主波段
            candidate_dir = None
            candidate_start = None
            candidate_count = 0
            continue

    # 若存在未确认的候选反向，视为噪音，继续并入当前主波段
    if wave_dir is not None and wave_start is not None:
        waves.append({
            "start_idx": wave_start,
            "end_idx": len(close) - 1,
            "direction": wave_dir,
        })

    if len(waves) < 2:
        return {
            "recent_up_gt_down": False,
            "up_vs_down_ratio": 0.0,
            "all_up_gt_down": False,
            "wave_count": len(waves),
            "last_wave_dir": waves[-1]["direction"] if waves else None,
            "pattern_score": 0.0,
            "waves": [],
        }

    # ── Step 3: 计算每个波段的均量、涨跌幅、高低点 ──────────────────
    high_arr = high if high is not None else close
    low_arr = low if low is not None else close
    for w in waves:
        s, e = w["start_idx"], w["end_idx"]
        w["len"] = e - s + 1
        w["avg_volume"] = float(np.mean(volume[s:e + 1]))
        w["price_change"] = (close[e] / close[s] - 1) * 100 if close[s] > 0 else 0.0
        w["wave_high"] = float(np.max(high_arr[s:e + 1]))
        w["wave_low"] = float(np.min(low_arr[s:e + 1]))

    # ── Step 4: 分离涨段和跌段 ────────────────────────────────
    up_waves = [w for w in waves if w["direction"] == "up"]
    down_waves = [w for w in waves if w["direction"] == "down"]

    if not up_waves or not down_waves:
        return {
            "recent_up_gt_down": False,
            "up_vs_down_ratio": 0.0,
            "all_up_gt_down": False,
            "wave_count": len(waves),
            "last_wave_dir": waves[-1]["direction"] if waves else None,
            "pattern_score": 0.0,
            "waves": waves,
        }

    # ── Step 5: 最近一个完整涨段 vs 最近一个完整跌段 ────────────
    last_up = up_waves[-1]
    last_down = down_waves[-1]

    recent_up_gt_down = last_up["avg_volume"] > last_down["avg_volume"]
    up_vs_down_ratio = last_up["avg_volume"] / max(last_down["avg_volume"], 1.0)

    # 全部涨段 vs 全部跌段
    all_up_avg = float(np.mean([w["avg_volume"] for w in up_waves]))
    all_down_avg = float(np.mean([w["avg_volume"] for w in down_waves]))
    all_up_gt_down = all_up_avg > all_down_avg

    # ── Step 6: 模式评分 ──────────────────────────────────────
    score = 0.0
    if recent_up_gt_down:
        score += 0.4
    if all_up_gt_down:
        score += 0.3
    if waves and waves[-1]["direction"] == "up":
        score += 0.3

    return {
        "recent_up_gt_down": recent_up_gt_down,
        "up_vs_down_ratio": round(up_vs_down_ratio, 2),
        "all_up_gt_down": all_up_gt_down,
        "wave_count": len(waves),
        "last_wave_dir": waves[-1]["direction"],
        "pattern_score": round(score, 2),
        "waves": waves,
    }
def compute_volume_metrics(
    volume: np.ndarray,
    amount: np.ndarray,
    close: np.ndarray,
    turnover_true: np.ndarray | None = None,
) -> dict:
    """
    量能指标

    Returns:
        vol_ratio: 当日量比（当日成交量 / 5日均成交量）
        vol_trend: 5日均量 / 20日均量（量能中期趋势）
        turnover: 换手率估算（成交量/成交额 × 100）
        vol_up_days_ratio: 近5日上涨日占比
        vol_up_vs_down: 近5日涨时均量 / 跌时均量
        vol_consec_strong: 连续放量天数（近5日中量比>1.0的天数）
        vol_recent_3: 近3日均量 / 前5日均量（启动爆发力）
    """
    vol_5 = np.nanmean(volume[-5:]) if len(volume) >= 5 else np.nanmean(volume)
    vol_20 = np.nanmean(volume[-20:]) if len(volume) >= 20 else np.nanmean(volume)
    vol_3_recent = np.nanmean(volume[-3:]) if len(volume) >= 3 else np.nanmean(volume)
    vol_5_prior = np.nanmean(volume[-8:-3]) if len(volume) >= 8 else vol_5
    vol_ratio = float(volume[-1] / vol_5) if vol_5 > 0 else 0.0
    vol_trend = float(vol_5 / vol_20) if vol_20 > 0 else 0.0
    # 换手率：AkShare的turnover列直接是百分比，无需再×100
    if turnover_true is not None and len(turnover_true) == len(volume):
        turnover_est = float(turnover_true[-1])  # 直接是百分比
    else:
        amt_5 = np.nanmean(amount[-5:]) if len(amount) >= 5 else np.nanmean(amount)
        turnover_est = float(volume[-1] / amt_5 * 100) if amt_5 > 0 else 0.0

    # 近5日均换手率
    if turnover_true is not None and len(turnover_true) >= 5:
        avg_turnover_5 = float(np.nanmean(turnover_true[-5:]))  # 直接是百分比
    else:
        avg_turnover_5 = turnover_est

    # ── 量能结构：涨时放量 vs 跌时缩量 ──────────────────────────────
    # 近5日（排除今天）中，上涨日均量 vs 下跌日均量
    n = min(5, len(close) - 1)
    up_vols = []
    down_vols = []
    for i in range(len(close) - n, len(close)):
        if close[i] > close[i - 1]:
            up_vols.append(volume[i])
        elif close[i] < close[i - 1]:
            down_vols.append(volume[i])

    vol_up_avg = np.mean(up_vols) if up_vols else 0.0
    vol_down_avg = np.mean(down_vols) if down_vols else 0.0
    vol_up_vs_down = float(vol_up_avg / vol_down_avg) if vol_down_avg > 0 else 0.0

    # 近5日上涨日占比（用于判断多空节奏）
    up_days_count = len(up_vols)
    down_days_count = len(down_vols)
    vol_up_days_ratio = up_days_count / (up_days_count + down_days_count) if (up_days_count + down_days_count) > 0 else 0.5

    # 连续放量天数（近5日量比>1.0的天数）
    vol_consec_strong = 0
    for i in range(len(volume) - 1, max(len(volume) - 6, -1), -1):
        vr = volume[i] / np.nanmean(volume[max(0, i-4):i+1]) if i >= 4 else volume[i] / np.nanmean(volume[:i+1])
        if vr > 1.0:
            vol_consec_strong += 1
        else:
            break

    # 近3日均量 / 前5日均量（启动爆发力）
    vol_recent_3 = float(vol_3_recent / vol_5_prior) if vol_5_prior > 0 else 0.0

    return {
        "vol_ratio": vol_ratio,
        "vol_trend": vol_trend,
        "turnover_est": avg_turnover_5,  # 5日均换手率
        "turnover_today": turnover_est,   # 当日换手率
        "vol_up_vs_down": vol_up_vs_down,
        "vol_up_days_ratio": vol_up_days_ratio,
        "vol_consec_strong": vol_consec_strong,
        "vol_recent_3": vol_recent_3,
    }


def compute_rsi(closes: np.ndarray, period: int = 14) -> float:
    """RSI（Wilder平滑）"""
    if len(closes) < period + 1:
        return 50.0
    delta = np.diff(closes)
    gain = np.clip(delta, 0, None)
    loss = np.clip(-delta, 0, None)
    avg_g = float(gain[:period].mean())
    avg_l = float(loss[:period].mean())
    rs = avg_g / avg_l if avg_l > 1e-12 else 100.0
    rsi_val = 100.0 - 100.0 / (1.0 + rs)
    for i in range(period, len(gain)):
        avg_g = (avg_g * 13.0 + gain[i]) / 14.0
        avg_l = (avg_l * 13.0 + loss[i]) / 14.0
        rs = avg_g / avg_l if avg_l > 1e-12 else 100.0
        rsi_val = 100.0 - 100.0 / (1.0 + rs)
    return float(rsi_val)


def count_red_days(macd: np.ndarray, idx: int) -> int:
    """从idx往回数，连续红柱天数"""
    count = 0
    for i in range(idx, -1, -1):
        if macd[i] > 0:
            count += 1
        else:
            break
    return count


def compute_all(df: pd.DataFrame) -> dict:
    """
    计算单只股票全部指标（基于最新一根K线）

    Args:
        df: 前复权日线，date升序

    Returns:
        指标字典
    """
    if len(df) < 35:
        return {}

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    volume = df["volume"].values
    amount = df["amount"].values

    idx = len(close) - 1  # 最新一根

    # 流通市值（亿元）
    outs = df["outstanding_share"].astype(float).values if "outstanding_share" in df.columns else None
    market_cap = float(close[idx] * outs[idx] / 1e8) if outs is not None else 0.0

    # 优先使用AkShare原始换手率（true_turnover列），否则估算
    if "turnover" in df.columns:
        turnover_true = df["turnover"].astype(float).values
    else:
        # 估算：成交量/流通股本（outstanding_share）
        if outs is not None:
            turnover_true = volume / outs * 100.0
        else:
            turnover_true = np.full_like(volume, np.nan, dtype=float)

    dif, dea, macd = compute_macd(close)
    ma = compute_ma(close)
    ma5_arr = ma[5]
    ma10_arr = ma[10]
    ma20_arr = ma[20]
    ma60_arr = ma[60]
    vol_metrics = compute_volume_metrics(volume, amount, close, turnover_true)
    rsi = compute_rsi(close)
    red_days = count_red_days(macd, idx)

    # 涨幅
    gain1 = (close[idx] / close[idx - 1] - 1) * 100 if close[idx - 1] > 0 else 0.0
    gain3 = (close[idx] / close[idx - 3] - 1) * 100 if close[idx - 3] > 0 else 0.0
    gain5 = (close[idx] / close[idx - 5] - 1) * 100 if close[idx - 5] > 0 else 0.0
    gain20 = (close[idx] / close[idx - 20] - 1) * 100 if close[idx - 20] > 0 else 0.0

    # ── 回调支撑检查 ───────────────────────────────────────────────
    # 收盘价距5日线的距离（%，负数表示在线下）
    ma5 = ma5_arr[idx]
    ma10 = ma10_arr[idx]
    ma20_val = ma20_arr[idx]
    ma5_distance_pct = (close[idx] - ma5) / ma5 * 100.0 if ma5 > 0 else 0.0
    # 最低价距5日线的距离（回调深度）
    low_near = df["low"].values
    low_distance_pct = (low_near[idx] - ma5) / ma5 * 100.0 if ma5 > 0 else 0.0
    # 最近3天是否曾跌破5日线
    broke_ma5_recently = False
    if idx >= 3:
        for i in range(idx - 2, idx + 1):
            if low_near[i] < ma[5][i]:
                broke_ma5_recently = True
                break
    # 最近是否曾跌破10日线（超过1天）
    broke_ma10_count = 0
    if idx >= 3:
        for i in range(idx - 2, idx + 1):
            if low_near[i] < ma[10][i]:
                broke_ma10_count += 1

    # ── 缩量整理检查 ───────────────────────────────────────────────
    # 近5日中，涨时量明显大于跌时量（健康）
    has_consolidation_pattern = (
        vol_metrics["vol_up_vs_down"] > 1.3 and
        vol_metrics["vol_up_days_ratio"] >= 0.5
    )
    # 整理时间是否过长（超过10天横盘=危险）
    consolidation_days = 0
    if idx >= 20:
        # 检查最近10天是否在高位横盘（涨幅小+波动小）
        recent_gain = (close[idx] / close[idx - 10] - 1) * 100
        recent_vol_avg = np.nanmean(volume[idx - 9:idx + 1])
        vol_avg_20 = np.nanmean(volume[idx - 20:idx + 1])
        vol_ratio_now = recent_vol_avg / vol_avg_20 if vol_avg_20 > 0 else 0.0
        if abs(recent_gain) < 5.0 and vol_ratio_now < 1.1:
            consolidation_days = 10  # 疑似横盘整理

    # ── 周期量价模式识别（核心）───────────────────────────────
    # 识别近20日内的涨跌波段，对比涨段均量 vs 跌段均量
    wave_pattern = detect_volume_price_wave(close, volume, lookback=60, high=high, low=low)

    # ── 止损位参考 ──────────────────────────────────────────────────
    # 红柱区间起始日前的低点（作为参考止损位）
    stop_loss_ref = None
    if idx >= 2:
        red_start_idx = idx - red_days + 1
        # 取红柱区间前1天到当前的所有低点中的最小值
        ref_start = max(0, red_start_idx - 1)
        ref_end = idx + 1
        if ref_end > ref_start:
            stop_loss_ref = float(np.min(low_near[ref_start:ref_end]))

    return {
        # 价格/均线
        "close": close[idx],
        "ma5": ma5,
        "ma10": ma10,
        "ma20": ma20_val,
        "ma60": ma[60][idx] if len(df) >= 60 else np.nan,
        # MACD
        "dif": dif[idx],
        "dea": dea[idx],
        "macd": macd[idx],
        "red_days": red_days,
        # 量能（使用真实换手率）
        **vol_metrics,
        "turnover": float(turnover_true[idx]),  # 直接是百分比
        "market_cap": market_cap,  # 流通市值（亿元）
        # RSI
        "rsi": rsi,
        # 涨幅
        "gain1": gain1,
        "gain3": gain3,
        "gain5": gain5,
        "gain20": gain20,
        # 回调支撑
        "ma5_distance_pct": ma5_distance_pct,
        "low_distance_pct": low_distance_pct,
        "broke_ma5_recently": broke_ma5_recently,
        "broke_ma10_count": broke_ma10_count,
        # 整理模式
        "has_consolidation_pattern": has_consolidation_pattern,
        "consolidation_days": consolidation_days,
        # 周期量价模式（波段识别）
        "wave_pattern_score": wave_pattern["pattern_score"],
        "wave_recent_up_gt_down": wave_pattern["recent_up_gt_down"],
        "wave_up_vs_down_ratio": wave_pattern["up_vs_down_ratio"],
        "wave_all_up_gt_down": wave_pattern["all_up_gt_down"],
        "wave_count": wave_pattern["wave_count"],
        "wave_last_dir": wave_pattern["last_wave_dir"],
        # 止损参考
        "stop_loss_ref": stop_loss_ref,
        # 原始序列
        "_dif": dif,
        "_dea": dea,
        "_macd": macd,
        "_close": close,
        "_low": low_near,
        "_idx": idx,
        "waves": wave_pattern["waves"],
        "_ma20": ma20_arr,
        "_ma60": ma60_arr,
        "_turnover": turnover_true * 100.0,  # 换手率数组（%）
        "_dates": df["date"].values,
    }
