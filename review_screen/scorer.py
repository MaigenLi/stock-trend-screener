#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评分系统（基于实战经验）
"""

import numpy as np


def _find_ascending_start(ups: list) -> int:
    """
    扫描找到第一个连续递增三联 ups[i] < ups[i+1] < ups[i+2]
    返回 u1 的索引 i，之前的波段全部丢弃不参与评分
    """
    for i in range(len(ups) - 2):
        if ups[i]["wave_high"] < ups[i+1]["wave_high"] < ups[i+2]["wave_high"]:
            return i
    return 0  # 找不到则从初始评分（退化为旧行为）


def score_wave_quality(waves: list) -> float:
    """
    波段结构质量评分

    先扫描找到第一个连续递增三联 u1<u3<u5，
    从此之前的上涨/下跌波段全部丢弃不参与评分。

    上涨波段（从找到的 u1 开始）：
      u3 > u1 → +2
      u5 > u3 且 > max(全部前高) → +8
      u5 > u3 但未破历史高 → +1
      u5 < u3 → -3

    下跌波段（与找到的 u1 对应的下跌波段开始）：
      低点 > prev_low → +0（不加分）
      低点 < prev_low → -1（新低）
    """
    if not waves:
        return 0.0

    score = 0.0

    # 分离上坡和下坡（从老到新）
    ups = [w for w in waves if w["direction"] == "up"]
    downs = [w for w in waves if w["direction"] == "down"]

    # 找到第一个连续递增三联，丢弃之前的一切
    start_u_idx = _find_ascending_start(ups)

    # ── 上涨波段：从 start_u_idx 开始评分 ─────────────────────────
    for i in range(start_u_idx + 1, len(ups)):
        curr_h = ups[i]["wave_high"]
        prev_h = ups[i - 1]["wave_high"]

        if curr_h > prev_h:
            # 如果是 u1→u3 第一次比较（即 u1 为 start_u_idx）
            if i == start_u_idx + 1:
                score += 2.0
            else:
                max_prior = max(ups[k]["wave_high"] for k in range(start_u_idx, i))
                if curr_h > max_prior:
                    score += 8.0  # 创历史新高
                else:
                    score += 1.0  # 高于prev但不破历史
        elif curr_h < prev_h:
            score -= 3.0

    # ── 下跌波段：从 start_u_idx 对应的下跌波段开始评分 ────────────
    # 如果 start_u_idx > 0，跳过前 start_u_idx 个下跌波段的比较
    #（这些下跌波段属于被丢弃的上涨波段区间）
    d_start = max(1, start_u_idx)
    for i in range(d_start, len(downs)):
        curr_lo = downs[i]["wave_low"]
        prev_lo = downs[i - 1]["wave_low"]
        if curr_lo < prev_lo:
            score -= 1.0
        # curr_lo > prev_lo → +0，不加分不减分

    return round(score, 1)


def score_stock(ind: dict) -> float:
    """
    综合评分（0-100）

    评分维度：
    1. DIF强度（0-25分）：开口大小，多空力量对比
    2. 红柱新鲜度（0-20分）：天数越少越新鲜
    3. 量能质量（0-25分）：
       - 换手率（0-8分）
       - 量比（0-5分）
       - 涨时放量/跌时缩量结构（0-8分）
       - 近3日爆发力（0-4分）
    4. 趋势质量（0-20分）：
       - 均线多头（0-15分）
       - 回调支撑（0-5分）
    5. 整理模式（0-10分）：缩量整理加分
    """
    score = 0.0

    # ── 0. 波段结构质量 ───────────────────────────────
    waves = ind.get("waves", [])
    wave_quality = score_wave_quality(waves)
    score += wave_quality

    # ── 1. DIF强度（0-25分）──────────────────────────
    dea = ind["dea"]
    dif = ind["dif"]
    if dea != 0:
        dif_strength = min(abs(dif / dea), 10.0)
    else:
        dif_strength = 0.0
    score += min(dif_strength * 2.5, 25.0)

    # ── 2. 红柱新鲜度（0-20分）──────────────────────
    rd = ind["red_days"]
    red_bonus = max(0, 12 - rd) * 2.0
    score += min(red_bonus, 20.0)

    # ── 3. 量能质量（0-25分）────────────────────────
    # 3.1 换手率（0-8分）
    to = ind.get("turnover_est", 0.0)
    if to >= 8:
        to_score = 8.0
    elif to >= 5:
        to_score = 4.0 + (to - 5) * (4.0 / 3.0)
    elif to >= 3:
        to_score = 2.0 + (to - 3) * (2.0 / 2.0)
    else:
        to_score = max(0, to * 0.7)
    score += to_score

    # 3.2 量比（0-5分）
    vr = ind.get("vol_ratio", 1.0)
    if vr >= 3.0:
        vr_score = 5.0
    elif vr >= 2.0:
        vr_score = 3.0 + (vr - 2.0) * (2.0 / 1.0)
    elif vr >= 1.3:
        vr_score = 1.0 + (vr - 1.3) * (2.0 / 0.7)
    else:
        vr_score = max(0, vr * 0.8)
    score += vr_score

    # 3.3 涨时放量/跌时缩量结构（0-8分）—— 改为用波段识别
    # 核心：最近涨段均量 > 最近跌段均量
    wave_score = ind.get("wave_pattern_score", 0.0)
    wave_ratio = ind.get("wave_up_vs_down_ratio", 0.0)
    # 波段模式评分（0-1）映射到0-8分
    vu_score = wave_score * 8.0
    score += vu_score

    # 3.4 近3日爆发力（0-4分）
    vr3 = ind.get("vol_recent_3", 1.0)
    if vr3 >= 2.0:
        vr3_score = 4.0
    elif vr3 >= 1.5:
        vr3_score = 2.0 + (vr3 - 1.5) * (2.0 / 0.5)
    elif vr3 >= 1.2:
        vr3_score = 1.0 + (vr3 - 1.2) * (1.0 / 0.3)
    else:
        vr3_score = max(0, vr3 * 0.8)
    score += vr3_score


    # ── 4. 趋势质量（0-20分）────────────────────────
    # 4.1 均线多头（0-15分）
    close = ind["close"]
    ma5 = ind["ma5"]
    ma10 = ind["ma10"]
    ma20 = ind["ma20"]

    ma_score = 0.0
    if close > ma5: ma_score += 3.0
    if ma5 > ma10: ma_score += 3.0
    if ma10 > ma20: ma_score += 3.0
    if close > ma20: ma_score += 3.0
    if ma5 > ma20: ma_score += 3.0  # 短期与中期拉开距离
    score += ma_score

    # 4.2 回调支撑（0-5分）
    # 收盘在MA5上方，且近日未跌破MA5
    ma5_dist = ind.get("ma5_distance_pct", 0.0)
    broke_ma5 = ind.get("broke_ma5_recently", False)
    broke_ma10 = ind.get("broke_ma10_count", 0)

    support_score = 0.0
    if ma5_dist > 0 and not broke_ma5:
        support_score = 5.0  # 完美：收盘在线上且未破
    elif ma5_dist > 0 and broke_ma5:
        support_score = 3.0  # 收盘线上但曾破
    elif ma5_dist > -2 and not broke_ma5:
        support_score = 2.0  # 略低于MA5但快速收回
    elif ma5_dist > -3 and broke_ma10 == 0:
        support_score = 1.0  # 略低于MA5但MA10未破
    score += support_score

    # ── 5. 整理模式（0-10分）────────────────────────
    # 有缩量整理模式 + 近5日上涨日占比高 = 健康
    has_con = ind.get("has_consolidation_pattern", False)
    up_ratio = ind.get("vol_up_days_ratio", 0.5)

    if has_con and up_ratio >= 0.7:
        consolidation_score = 10.0
    elif has_con and up_ratio >= 0.5:
        consolidation_score = 7.0
    elif up_ratio >= 0.7:
        consolidation_score = 5.0
    elif up_ratio >= 0.5:
        consolidation_score = 3.0
    else:
        consolidation_score = 0.0
    score += consolidation_score

    return round(min(score, 100.0), 1)


def score_detail(ind: dict) -> dict:
    """详细评分拆解"""
    dea = ind["dea"]
    dif = ind["dif"]
    dif_strength = min(abs(dif / dea), 10.0) if dea != 0 else 0.0
    dif_score = min(dif_strength * 2.5, 25.0)

    rd = ind["red_days"]
    red_score = min(max(0, 12 - rd) * 2.0, 20.0)

    to = ind.get("turnover_est", 0.0)
    to_score = min(8.0, max(0, 2.0 if to < 3 else (4.0 if to < 5 else (8.0 if to >= 8 else 4.0 + (to - 5) * (4.0 / 3.0)))))

    vr = ind.get("vol_ratio", 1.0)
    vr_score = min(5.0, max(0, 1.0 if vr < 1.3 else (3.0 if vr < 2.0 else (5.0 if vr >= 3.0 else 3.0 + (vr - 2.0) * 2.0))))

    vu = ind.get("vol_up_vs_down", 0.0)
    vu_score = min(8.0, max(0, 2.0 if vu < 1.2 else (5.0 if vu < 1.5 else (8.0 if vu >= 2.0 else 5.0 + (vu - 1.5) * 3.0))))

    vr3 = ind.get("vol_recent_3", 1.0)
    vr3_score = min(4.0, max(0, 1.0 if vr3 < 1.2 else (2.0 if vr3 < 1.5 else (4.0 if vr3 >= 2.0 else 2.0 + (vr3 - 1.5) * 2.0))))

    close = ind["close"]
    ma_score = 0.0
    if close > ind["ma5"]: ma_score += 3.0
    if ind["ma5"] > ind["ma10"]: ma_score += 3.0
    if ind["ma10"] > ind["ma20"]: ma_score += 3.0
    if close > ind["ma20"]: ma_score += 3.0
    if ind["ma5"] > ind["ma20"]: ma_score += 3.0

    ma5_dist = ind.get("ma5_distance_pct", 0.0)
    broke_ma5 = ind.get("broke_ma5_recently", False)
    broke_ma10 = ind.get("broke_ma10_count", 0)
    if ma5_dist > 0 and not broke_ma5: support_score = 5.0
    elif ma5_dist > 0 and broke_ma5: support_score = 3.0
    elif ma5_dist > -2 and not broke_ma5: support_score = 2.0
    elif ma5_dist > -3 and broke_ma10 == 0: support_score = 1.0
    else: support_score = 0.0

    has_con = ind.get("has_consolidation_pattern", False)
    up_ratio = ind.get("vol_up_days_ratio", 0.5)
    if has_con and up_ratio >= 0.7: con_score = 10.0
    elif has_con and up_ratio >= 0.5: con_score = 7.0
    elif up_ratio >= 0.7: con_score = 5.0
    elif up_ratio >= 0.5: con_score = 3.0
    else: con_score = 0.0

    waves = ind.get("waves", [])
    wave_quality = score_wave_quality(waves)

    return {
        "dif_score": round(dif_score, 1),
        "red_score": round(red_score, 1),
        "turnover_score": round(to_score, 1),
        "volume_score": round(vr_score, 1),
        "vol_structure_score": round(vu_score, 1),
        "vol_burst_score": round(vr3_score, 1),
        "ma_score": round(ma_score, 1),
        "support_score": round(support_score, 1),
        "wave_quality_score": round(wave_quality, 1),
        "consolidation_score": round(con_score, 1),
        "total": score_stock(ind),
    }
