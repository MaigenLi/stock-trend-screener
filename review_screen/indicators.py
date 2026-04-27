#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
技术指标计算(趋势 + 量能)
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
    周期模式识别(结构波段版)

    涨跌定义(3日结构):
      涨:c2 > c0 AND c1 < c2
          - 连续2日上涨:c0 < c1 < c2
          - 上下引线:c0 > c1 < c2 且 c0 < c2
          - 平再涨:c0 <= c1 < c2 且 c0 < c2
      跌:c2 < c0 AND c1 > c2(同理反向)
      平盘:不满足上述条件

    波段构造原则:
      1. 波段从第一个非中立日启动
      2. 中立日吸收到当前波段,不切断
      3. 反向非中立日出现时,先前波段确认结束,新波段开始

    Args:
        close: 收盘价数组(date升序)
        volume: 成交量数组
        lookback: 回溯天数

    Returns:
        recent_up_gt_down:  最近涨段均量 > 最近跌段均量
        up_vs_down_ratio:   涨段均量 / 跌段均量
        all_up_gt_down:     全部涨段均量 > 全部跌段均量
        wave_count:         识别出的波段数量
        last_wave_dir:      最近波段方向
        pattern_score:      模式评分(0-1)
        waves:              波段详情列表
    """
    n = min(lookback, len(close) - 1)

    # 固定锚点：lookback 窗口起点，不再使用随机局部最低点
    scan_start = len(close) - n

    if n < 5 or len(close) - scan_start < 5:
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
    # i=0,1 → 无足够历史,无法判断(中立)
    raw_states = ["neutral"] * 2  # 前2天中立
    for i in range(2, len(close)):
        c0, c1, c2 = close[i - 2], close[i - 1], close[i]
        if c2 > c0 and c1 < c2:
            raw_states.append("up")
        elif c2 < c0 and c1 > c2:
            raw_states.append("down")
        else:
            raw_states.append("neutral")

    def _find_next_wave_from(start: int, direction: str) -> tuple | None:
        """
        从 start 索引向前扫描,找到第一个满足连续三连的方向波段。

        三连阳(direction='up'):close[i+1] > close[i] > close[i-1](连续3根阳线)
        三连阴(direction='down'):close[i+1] < close[i] < close[i-1](连续3根阴线)

        找到时返回 (wave_start, wave_end),否则返回 None。
        wave_end 是第三根K线的索引(最后一根确认K线)。
        """
        cmp_up = direction == "up"
        for i in range(start, len(close) - 1):
            c_im1 = close[i - 1] if i > 0 else None
            c_i = close[i]
            c_ip1 = close[i + 1]
            if c_im1 is None:
                continue
            ok = (c_ip1 > c_i > c_im1) if cmp_up else (c_ip1 < c_i < c_im1)
            if ok:
                return (i, i + 1)   # wave_start=i(第一根阳线日), wave_end=i+1
        return None

    # ── Step 2: 从固定起点 scan_start 扫描找第一组三连 ───────────────
    # 不再使用随机锚点：固定从窗口起点开始扫描
    waves = []
    up_res = _find_next_wave_from(scan_start, "up")
    if up_res is None:
        # 找不到三连阳 → 无法构建波段
        return {
            "recent_up_gt_down": False,
            "up_vs_down_ratio": 0.0,
            "all_up_gt_down": False,
            "wave_count": 0,
            "last_wave_dir": None,
            "pattern_score": 0.0,
            "waves": [],
        }

    u1_start, u1_end = up_res
    current_dir = "up"
    wave_start = u1_start
    wave_end = u1_end

    # 从 u1_end 之后交替寻找下一个波段
    while True:
        next_dir = "down" if current_dir == "up" else "up"
        res = _find_next_wave_from(wave_end + 1, next_dir)
        if res is None:
            break
        next_start, next_end = res
        # 前一波段截止到当前波段起点前一天
        waves.append({
            "start_idx": wave_start,
            "end_idx": next_start - 1,
            "direction": current_dir,
        })
        wave_start = next_start
        wave_end = next_end
        current_dir = next_dir

    # 最后一段延伸至窗口末尾
    waves.append({
        "start_idx": wave_start,
        "end_idx": len(close) - 1,
        "direction": current_dir,
    })

    # ── Step 2.5: 噪声过滤 ──────────────────────────────────
    # 2天波段且|涨跌幅|<2% → 视为噪声，合并到前一波段
    # 注意：合并前的涨跌幅按波段内计算来确定是否为噪声
    if len(waves) >= 3:
        merged = []
        for w in waves:
            s, e = w["start_idx"], w["end_idx"]
            days = e - s + 1
            # 计算波段内涨跌幅（用于噪声判定）
            if days == 2:
                inner_pct = abs((close[e] / close[s] - 1) * 100)
            else:
                inner_pct = 999.0  # 非2天不可能是噪声
            if days == 2 and inner_pct < 2.0:
                # 噪声！合并到前一个波段
                if merged:
                    merged[-1]["end_idx"] = e
                else:
                    w["len"] = days  # 第一条保留
                    merged.append(w)
            else:
                merged.append(w)
        waves = merged

    # 合并连续同向波段（噪声过滤可能导致中间方向波段被吃）
    if len(waves) >= 2:
        merged2 = [waves[0]]
        for w in waves[1:]:
            if w["direction"] == merged2[-1]["direction"]:
                # 同向合并
                merged2[-1]["end_idx"] = w["end_idx"]
            else:
                merged2.append(w)
        waves = merged2

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
    for i, w in enumerate(waves):
        s, e = w["start_idx"], w["end_idx"]
        w["len"] = e - s + 1
        w["avg_volume"] = float(np.mean(volume[s:e + 1]))
        # 涨跌幅定义（统一口径）：
        # - 从上一波段终点收盘价到本波段终点收盘价
        # - 第一个波段无前波段可比较，用波段内涨跌幅
        if i > 0:
            prev_end = waves[i - 1]["end_idx"]
            prev_close = close[prev_end]
            w["price_change"] = (close[e] / prev_close - 1) * 100 if prev_close > 0 else 0.0
        else:
            w["price_change"] = (close[e] / close[s] - 1) * 100 if close[s] > 0 else 0.0
        w["wave_high"] = float(np.max(high_arr[s:e + 1]))
        w["wave_low"] = float(np.min(low_arr[s:e + 1]))
        # 量能爆发力:波段内最大量 / 波段前5日均量
        max_vol = float(np.max(volume[s:e + 1]))
        prev_avg = float(np.mean(volume[max(s - 5, 0):s])) if s > 0 else 1.0
        w["volume_power"] = max_vol / max(prev_avg, 1.0)

    # ── 转换为 Wave 数据结构 ────────────────────────────────
    from structure_analyzer import Wave as WaveObj
    wave_objects = [
        WaveObj(
            start=w["start_idx"], end=w["end_idx"],
            direction=w["direction"],
            pct=w["price_change"], days=w["len"],
            avg_volume=w["avg_volume"],
            volume_power=w.get("volume_power", 0.0),
            wave_high=w["wave_high"], wave_low=w["wave_low"],
        )
        for w in waves
    ]
    up_waves = [w for w in wave_objects if w.direction == "up"]
    down_waves = [w for w in wave_objects if w.direction == "down"]

    if not up_waves or not down_waves:
        return {
            "recent_up_gt_down": False,
            "up_vs_down_ratio": 0.0,
            "all_up_gt_down": False,
            "wave_count": len(wave_objects),
            "last_wave_dir": wave_objects[-1].direction if wave_objects else None,
            "pattern_score": 0.0,
            "waves": wave_objects,
            "structure_result": None,
        }

    # ── Step 4: 量能对比 ────────────────────────────────────
    last_up = up_waves[-1]
    last_down = down_waves[-1]
    recent_up_gt_down = last_up.avg_volume > last_down.avg_volume
    up_vs_down_ratio = last_up.avg_volume / max(last_down.avg_volume, 1.0)

    all_up_avg = float(np.mean([w.avg_volume for w in up_waves]))
    all_down_avg = float(np.mean([w.avg_volume for w in down_waves]))
    all_up_gt_down = all_up_avg > all_down_avg

    # ── Step 5: 结构分析(复用 structure_analyzer)──────────────
    from structure_analyzer import analyze_structure
    result = analyze_structure(up_waves, down_waves)

    # ── Step 6: 模式评分 ──────────────────────────────────────
    score = 0.0
    if recent_up_gt_down:
        score += 0.25     # 涨段放量
    if all_up_gt_down:
        score += 0.20     # 长期量价健康
    if wave_objects and wave_objects[-1].direction == "up":
        score += 0.05     # 趋势延续中
    if result and result.is_strong:
        score += 0.30     # 涨得多跌得少
    if result and result.up_speed > result.down_speed:
        score += 0.20     # 涨效率 > 跌效率

    return {
        "recent_up_gt_down": recent_up_gt_down,
        "up_vs_down_ratio": round(up_vs_down_ratio, 2),
        "all_up_gt_down": all_up_gt_down,
        "wave_count": len(wave_objects),
        "last_wave_dir": wave_objects[-1].direction if wave_objects else None,
        # 涨跌幅强弱
        "up_stronger_than_down": result.is_strong if result else False,
        "up_down_ratio": result.strength_ratio if result else 0.0,
        # 主升浪 / 二次启动
        "is_main_trend": result.is_main_trend if result else False,
        "is_second_break": result.is_second_break if result else False,
        # 结构评分与说明
        "structure_score": result.score if result else 0.0,
        "structure_reason": result.reason if result else "数据不足",
        # 效率指标
        "up_efficiency": last_up.speed if last_up else 0.0,
        "down_efficiency": last_down.speed if last_down else 0.0,
        "efficiency_ratio": round(last_up.speed / max(last_down.speed, 0.01), 2) if (last_up and last_down) else 0.0,
        # 综合评分
        "pattern_score": round(score, 2),
        "structure_result": result,
        "waves": wave_objects,
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
        vol_ratio: 当日量比(当日成交量 / 5日均成交量)
        vol_trend: 5日均量 / 20日均量(量能中期趋势)
        turnover_est: 当日换手率(AkShare的turnover列本身已是%,无需×100)
        vol_up_days_ratio: 近5日上涨日占比
        vol_up_vs_down: 近5日涨时均量 / 跌时均量
        vol_consec_strong: 连续放量天数(近5日中量比>1.0的天数)
        vol_recent_3: 近3日均量 / 前5日均量(启动爆发力)
    """
    vol_5 = np.nanmean(volume[-5:]) if len(volume) >= 5 else np.nanmean(volume)
    vol_20 = np.nanmean(volume[-20:]) if len(volume) >= 20 else np.nanmean(volume)
    vol_3_recent = np.nanmean(volume[-3:]) if len(volume) >= 3 else np.nanmean(volume)
    vol_5_prior = np.nanmean(volume[-8:-3]) if len(volume) >= 8 else vol_5
    vol_ratio = float(volume[-1] / vol_5) if vol_5 > 0 else 0.0
    vol_trend = float(vol_5 / vol_20) if vol_20 > 0 else 0.0
    # 当日换手率:AkShare的turnover列本身已是%(如2.5=2.5%),无需×100
    if turnover_true is not None and len(turnover_true) == len(volume):
        turnover_est = float(turnover_true[-1])
    else:
        # 备用估算(量纲:成交量/流通股本×100)
        amt_5 = np.nanmean(amount[-5:]) if len(amount) >= 5 else np.nanmean(amount)
        turnover_est = float(volume[-1] / amt_5 * 100) if amt_5 > 0 else 0.0

    # 近5日均换手率
    if turnover_true is not None and len(turnover_true) >= 5:
        avg_turnover_5 = float(np.nanmean(turnover_true[-5:]))
    else:
        avg_turnover_5 = turnover_est

    # ── 量能结构:涨时放量 vs 跌时缩量 ──────────────────────────────
    # 近5日(排除今天)中,上涨日均量 vs 下跌日均量
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

    # 近5日上涨日占比(用于判断多空节奏)
    up_days_count = len(up_vols)
    down_days_count = len(down_vols)
    vol_up_days_ratio = up_days_count / (up_days_count + down_days_count) if (up_days_count + down_days_count) > 0 else 0.5

    # 连续放量天数(近5日量比>1.0的天数)
    vol_consec_strong = 0
    for i in range(len(volume) - 1, max(len(volume) - 6, -1), -1):
        vr = volume[i] / np.nanmean(volume[max(0, i-4):i+1]) if i >= 4 else volume[i] / np.nanmean(volume[:i+1])
        if vr > 1.0:
            vol_consec_strong += 1
        else:
            break

    # 近3日均量 / 前5日均量(启动爆发力)
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
        "vol_ma5": float(vol_5),
        "vol_ma60": float(np.nanmean(volume[-60:])) if len(volume) >= 60 else float(vol_5),
    }


def compute_rsi(closes: np.ndarray, period: int = 14) -> float:
    """RSI(Wilder平滑, 统一复用gain_turnover)"""
    from stock_trend.gain_turnover import compute_rsi_scalar
    return compute_rsi_scalar(np.asarray(closes), period)


def count_red_days(macd: np.ndarray, idx: int) -> int:
    """从idx往回数,连续红柱天数"""
    count = 0
    for i in range(idx, -1, -1):
        if macd[i] > 0:
            count += 1
        else:
            break
    return count


def compute_all(df: pd.DataFrame, ma10_break_window: int = 3) -> dict:
    """
    计算单只股票全部指标(基于最新一根K线)

    Args:
        df: 前复权日线,date升序

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

    # 流通市值(亿元)
    outs = df["outstanding_share"].astype(float).values if "outstanding_share" in df.columns else None
    market_cap = float(close[idx] * outs[idx] / 1e8) if outs is not None else 0.0

    # 优先使用AkShare原始换手率(true_turnover列),否则估算
    if "turnover" in df.columns:
        turnover_true = df["turnover"].astype(float).values
    else:
        # 估算:成交量/流通股本(outstanding_share)
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
    low_60d = float(np.nanmin(close[max(0, idx-60):idx+1])) if idx >= 0 else float(close[idx])

    # ── 回调支撑检查 ───────────────────────────────────────────────
    # 收盘价距5日线的距离(%,负数表示在线下)
    ma5 = ma5_arr[idx]
    ma10 = ma10_arr[idx]
    ma20_val = ma20_arr[idx]
    ma5_distance_pct = (close[idx] - ma5) / ma5 * 100.0 if ma5 > 0 else 0.0
    # 最低价距5日线的距离(回调深度)
    low_near = df["low"].values
    low_distance_pct = (low_near[idx] - ma5) / ma5 * 100.0 if ma5 > 0 else 0.0
    # 最近3天是否曾跌破5日线
    broke_ma5_recently = False
    if idx >= 3:
        for i in range(idx - 2, idx + 1):
            if low_near[i] < ma[5][i]:
                broke_ma5_recently = True
                break
    # 最近 N 天是否曾跌破10日线(超过1天)
    broke_ma10_count = 0
    if idx >= ma10_break_window:
        for i in range(idx - ma10_break_window + 1, idx + 1):
            if low_near[i] < ma[10][i]:
                broke_ma10_count += 1

    # ── 缩量整理检查 ───────────────────────────────────────────────
    # 近5日中,涨时量明显大于跌时量(健康)
    has_consolidation_pattern = (
        vol_metrics["vol_up_vs_down"] > 1.3 and
        vol_metrics["vol_up_days_ratio"] >= 0.5
    )
    # 整理时间是否过长(超过10天横盘=危险)
    consolidation_days = 0
    if idx >= 20:
        # 检查最近10天是否在高位横盘(涨幅小+波动小)
        recent_gain = (close[idx] / close[idx - 10] - 1) * 100
        recent_vol_avg = np.nanmean(volume[idx - 9:idx + 1])
        vol_avg_20 = np.nanmean(volume[idx - 20:idx + 1])
        vol_ratio_now = recent_vol_avg / vol_avg_20 if vol_avg_20 > 0 else 0.0
        if abs(recent_gain) < 5.0 and vol_ratio_now < 1.1:
            consolidation_days = 10  # 疑似横盘整理

    # ── 周期量价模式识别(核心)───────────────────────────────
    # 识别近20日内的涨跌波段,对比涨段均量 vs 跌段均量
    wave_pattern = detect_volume_price_wave(close, volume, lookback=60, high=high, low=low)

    # ── 止损位参考 ──────────────────────────────────────────────────
    # 红柱区间起始日前的低点(作为参考止损位)
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
        # 量能(使用真实换手率)
        **vol_metrics,
        "turnover": float(turnover_true[idx]),  # 直接是百分比
        "market_cap": market_cap,  # 流通市值(亿元)
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
        # 周期量价模式(波段识别)
        "wave_pattern_score": wave_pattern["pattern_score"],
        "wave_recent_up_gt_down": wave_pattern["recent_up_gt_down"],
        "wave_up_vs_down_ratio": wave_pattern["up_vs_down_ratio"],
        "wave_all_up_gt_down": wave_pattern["all_up_gt_down"],
        # 涨跌幅强弱(从 structure_analyzer 来)
        "up_stronger_than_down": wave_pattern.get("up_stronger_than_down", False),
        "up_down_ratio": wave_pattern.get("up_down_ratio", 0.0),
        # 主升浪 / 二次启动
        "is_main_trend": wave_pattern.get("is_main_trend", False),
        "is_second_break": wave_pattern.get("is_second_break", False),
        "structure_score": wave_pattern.get("structure_score", 0.0),
        "structure_reason": wave_pattern.get("structure_reason", ""),
        "up_efficiency": wave_pattern.get("up_efficiency", 0.0),
        "down_efficiency": wave_pattern.get("down_efficiency", 0.0),
        "efficiency_ratio": wave_pattern.get("efficiency_ratio", 0.0),
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
        "low_60d": low_60d,
        "vol_ma5": vol_metrics["vol_ma5"],
        "vol_ma60": vol_metrics["vol_ma60"],
        "_ma20": ma20_arr,
        "_ma60": ma60_arr,
        "_turnover": turnover_true,  # 换手率数组(AkShare的turnover列本身就是百分比,如2.5=2.5%)
        "_dates": df["date"].values,
    }
