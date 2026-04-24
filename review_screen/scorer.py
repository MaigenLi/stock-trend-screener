#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评分系统（基于实战经验）
"""

import numpy as np

from utils import w_get, find_ascending_start

# 阶段标签常量
PHASE_MARKUP = "主升浪"
PHASE_SECOND_BREAK = "二次启动"
_REASON_SECOND_BREAK_VOL = "回调缩量+放量反弹"
PHASE_WASH = "洗盘"
PHASE_DISTRIBUTION = "出货"
PHASE_ACCUMULATION = "吸筹"
PHASE_UNCLEAR = "不明"


def score_wave_quality(waves: list) -> float:
    """
    波段结构质量评分

    先扫描找到第一个连续递增三联 u1<u3<u5，
    从此之前的上涨/下跌波段全部丢弃不参与评分。

    上涨波段价格结构（从找到的 u1 开始）：
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
    ups = [w for w in waves if w_get(w, "direction") == "up"]
    downs = [w for w in waves if w_get(w, "direction") == "down"]

    # 找到第一个连续递增三联，丢弃之前的一切
    start_u_idx = find_ascending_start(ups)

    # ── 上涨波段：从 start_u_idx 开始评分 ─────────────────────────
    for i in range(start_u_idx + 1, len(ups)):
        curr_h = w_get(ups[i], "wave_high")
        prev_h = w_get(ups[i - 1], "wave_high")

        if curr_h > prev_h:
            if i == start_u_idx + 1:
                score += 2.0
            else:
                max_prior = max(w_get(ups[k], "wave_high") for k in range(start_u_idx, i))
                if curr_h > max_prior:
                    score += 8.0  # 创历史新高
                else:
                    score += 1.0  # 高于prev但不破历史
        elif curr_h < prev_h:
            score -= 3.0

    # ── 下跌波段 ─────────────────────────────────────────────
    d_start = start_u_idx
    for i in range(d_start, len(downs)):
        curr_lo = w_get(downs[i], "wave_low")
        prev_lo = w_get(downs[i - 1], "wave_low") if i > 0 else curr_lo
        if curr_lo < prev_lo:
            score -= 1.0

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

    # ★ 强势结构与主升浪评分（额外加分，不改变 wave_quality 纯分）
    try:
        from structure_analyzer import analyze_structure
        up_wobj = [w for w in waves if w_get(w, "direction") == "up"]
        down_wobj = [w for w in waves if w_get(w, "direction") == "down"]
        struct_res = analyze_structure(up_wobj, down_wobj)
        if struct_res:
            if struct_res.is_strong:
                score += 5.0
            if struct_res.up_speed > struct_res.down_speed:
                score += 3.0
    except ImportError:
        pass

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
    ma5_dist = ind.get("ma5_distance_pct", 0.0)
    broke_ma5 = ind.get("broke_ma5_recently", False)
    broke_ma10 = ind.get("broke_ma10_count", 0)

    support_score = 0.0
    if ma5_dist > 0 and not broke_ma5:
        support_score = 5.0
    elif ma5_dist > 0 and broke_ma5:
        support_score = 3.0
    elif ma5_dist > -2 and not broke_ma5:
        support_score = 2.0
    elif ma5_dist > -3 and broke_ma10 == 0:
        support_score = 1.0
    score += support_score

    # ── 5. 整理模式（0-10分）────────────────────────
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
    if to >= 8:
        to_score = 8.0
    elif to >= 5:
        to_score = 4.0 + (to - 5) * (4.0 / 3.0)
    elif to >= 3:
        to_score = 2.0 + (to - 3) * (2.0 / 2.0)
    else:
        to_score = max(0, to * 0.7)

    vr = ind.get("vol_ratio", 1.0)
    if vr >= 3.0:
        vr_score = 5.0
    elif vr >= 2.0:
        vr_score = 3.0 + (vr - 2.0) * (2.0 / 1.0)
    elif vr >= 1.3:
        vr_score = 1.0 + (vr - 1.3) * (2.0 / 0.7)
    else:
        vr_score = max(0, vr * 0.8)
    vr_score = min(vr_score, 5.0)

    vu = ind.get("vol_up_vs_down", 0.0)
    if vu >= 2.0:
        vu_score = 8.0
    elif vu >= 1.5:
        vu_score = 5.0 + (vu - 1.5) * (3.0 / 0.5)
    elif vu >= 1.2:
        vu_score = 2.0 + (vu - 1.2) * (3.0 / 0.3)
    else:
        vu_score = max(0, vu * (2.0 / 1.2))
    vu_score = min(vu_score, 8.0)

    vr3 = ind.get("vol_recent_3", 1.0)
    if vr3 >= 2.0:
        vr3_score = 4.0
    elif vr3 >= 1.5:
        vr3_score = 2.0 + (vr3 - 1.5) * (2.0 / 0.5)
    elif vr3 >= 1.2:
        vr3_score = 1.0 + (vr3 - 1.2) * (1.0 / 0.3)
    else:
        vr3_score = max(0, vr3 * 0.8)
    vr3_score = min(vr3_score, 4.0)

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
    if ma5_dist > 0 and not broke_ma5:
        support_score = 5.0
    elif ma5_dist > 0 and broke_ma5:
        support_score = 3.0
    elif ma5_dist > -2 and not broke_ma5:
        support_score = 2.0
    elif ma5_dist > -3 and broke_ma10 == 0:
        support_score = 1.0
    else:
        support_score = 0.0

    has_con = ind.get("has_consolidation_pattern", False)
    up_ratio = ind.get("vol_up_days_ratio", 0.5)
    if has_con and up_ratio >= 0.7:
        con_score = 10.0
    elif has_con and up_ratio >= 0.5:
        con_score = 7.0
    elif up_ratio >= 0.7:
        con_score = 5.0
    elif up_ratio >= 0.5:
        con_score = 3.0
    else:
        con_score = 0.0

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


def classify_phase(ind: dict) -> tuple:
    """
    判定股票当前所处阶段。
    返回 (phase_label, reason)
    phase_label: "主升浪" / "二次启动" / "洗盘" / "出货" / "吸筹" / "不明"
    """
    if not ind:
        return (PHASE_UNCLEAR, "数据不足")

    waves = ind.get("waves", [])
    close = ind.get("close", 0)
    ma5 = ind.get("ma5", 0)
    ma10 = ind.get("ma10", 0)
    ma20 = ind.get("ma20", 0)
    ma60 = ind.get("ma60", 0)
    wq = score_wave_quality(waves)
    red_days = ind.get("red_days", 0)

    # 找最近涨跌段
    up_waves = [w for w in waves if w_get(w, "direction") == "up"]
    down_waves = [w for w in waves if w_get(w, "direction") == "down"]
    last_up = up_waves[-1] if up_waves else None
    last_down = down_waves[-1] if down_waves else None
    last_wave = waves[-1] if waves else None

    # ═══════════════════════════════════════════
    # 1️⃣ 出货：放量大跌 + 趋势破位
    # ═══════════════════════════════════════════
    if (last_wave and w_get(last_wave, "direction") == "down" and
        last_down and w_get(last_down, "pct") < -12 and
        close < ma20):
        return (PHASE_DISTRIBUTION, "放量大跌失守MA20")

    # ═══════════════════════════════════════════
    # 2️⃣ 主升浪：多头排列 + 波段结构强 + 红柱
    # ═══════════════════════════════════════════
    if (close > ma5 and ma5 > ma10 and ma10 > ma20 and
        wq > 5 and red_days >= 3):
        try:
            from structure_analyzer import analyze_structure
            struct_res = analyze_structure(up_waves, down_waves)
            if struct_res and struct_res.is_strong:
                return (PHASE_MARKUP, f"均线多头排+波评{wq:+.0f}+{struct_res.reason}")
        except Exception:
            pass
        return (PHASE_MARKUP, f"均线多头排列+波评{wq:+.0f}")

    # ═══════════════════════════════════════════
    # 3️⃣ 二次启动：回调后放量反弹，接近前高
    # ═══════════════════════════════════════════
    if (last_wave and w_get(last_wave, "direction") == "up" and
        last_up and last_down and
        abs(w_get(last_down, "pct")) >= 5 and
        abs(w_get(last_down, "pct")) <= 15 and
        close > ma20 and
        w_get(last_up, "avg_volume") >= w_get(last_down, "avg_volume") * 0.8):
        prev_up = up_waves[-2] if len(up_waves) >= 2 else None
        if prev_up and w_get(last_up, "wave_high") >= w_get(prev_up, "wave_high") * 0.85:
            return (PHASE_SECOND_BREAK, f"回调{-w_get(last_down,'pct'):.0f}%后放量反弹")
        return (PHASE_SECOND_BREAK, _REASON_SECOND_BREAK_VOL)

    # ═══════════════════════════════════════════
    # 4️⃣ 洗盘：回调缩量 + 中期趋势未破
    # ═══════════════════════════════════════════
    if (last_wave and w_get(last_wave, "direction") == "down" and
        last_down and last_up and
        abs(w_get(last_down, "pct")) <= 12 and
        w_get(last_down, "avg_volume") < w_get(last_up, "avg_volume") * 1.1 and
        ma20 > ma60):
        return (PHASE_WASH, f"缩量回调{w_get(last_down,'pct'):.0f}%")

    # ═══════════════════════════════════════════
    # 5️⃣ 吸筹：窄幅震荡 + 中期趋势向上
    # ═══════════════════════════════════════════
    if (len(waves) >= 4 and
        all(abs(w_get(w, "pct")) < 8 for w in waves[-4:]) and
        ma20 > ma60 and
        close > ma20):
        return (PHASE_ACCUMULATION, "窄幅震荡整理")

    return (PHASE_UNCLEAR, "结构不清晰")
