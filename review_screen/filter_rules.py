#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
筛选规则（基于实战经验设计）
"""

from dataclasses import dataclass


@dataclass
class FilterConfig:
    """筛选参数配置"""

    # ── 数据长度 ──────────────────────────────────────────
    min_bars: int = 60          # 最少K线根数

    # ── 趋势条件 ──────────────────────────────────────────
    require_close_above_ma20: bool = True  # 收盘>MA20（核心）
    require_ma20_above_ma60: bool = True  # MA20>MA60
    require_close_above_ma60: bool = True  # 收盘>MA60
    require_wave_ma20_above_ma60: bool = True  # 波段内每日MA20>MA60

    # ── MACD条件 ──────────────────────────────────────────
    require_macd_positive: bool = True      # MACD > 0
    require_macd_2days: bool = True         # 连续2天红柱
    require_dif_rising_2days: bool = True  # DIF连续2日上涨
    require_dea_rising: bool = True        # DEA上涨
    max_red_days: int = 30               # 红柱天数上限（>20=主升浪，<15=新鲜启动）

    # ── 涨幅条件 ──────────────────────────────────────────
    min_gain3: float = 3.0     # 3日涨幅下限
    max_gain3: float = 22.0    # 3日涨幅上限（>18%过热，>22%风险大）
    max_gain20: float = 60.0   # 20日涨幅上限

    # ── 量能条件 ──────────────────────────────────────────
    min_turnover: float = 1.0          # 换手率下限（%）
    min_vol_ratio: float = 1.0           # 当日量比下限
    min_vol_up_vs_down: float = 1.2     # 波段涨/跌均量比下限
    min_vol_consec_strong: int = 2      # 连续放量天数下限

    # ── 波段方向 ──────────────────────────────────────────
    require_latest_wave_down: bool = False  # 要求最新波段为下跌（蓄势找买点）
    min_latest_down_length: int = 2         # 下跌波段最小天数
    require_first_up_ascending: bool = False  # 前三上涨波段必须u1<u3<u5（开启更严格的结构质量要求）
    strict_trend_only: bool = False         # 严格趋势（latest-wave-down专用：只要求MA条件）

    # ── 回调支撑 ──────────────────────────────────────────
    allow_broke_ma5_recently: bool = True   # 允许近期跌破MA5（强势回调）
    max_broke_ma10_days: int = 2          # 近3天跌破MA10上限

    # ── 周期量价模式（核心）────────────────────────────────
    require_wave_up_gt_down: bool = True  # 必须满足涨段均量>跌段均量
    min_wave_up_ratio: float = 1.2       # 波段涨/跌均量比下限
    require_wave_pattern_score: float = 0.4  # 波段模式评分下限

    # ── 风险过滤 ──────────────────────────────────────────
    max_rsi: float = 85.0          # RSI上限（提高，允许主升浪）
    min_up_ratio: float = 0.5      # 红柱区间上涨日占比下限


def check_filters(ind: dict, cfg: FilterConfig = None) -> tuple[bool, str]:
    """
    检查单只股票是否通过全部筛选条件
    Returns: (是否通过, 拒绝原因)
    """
    if cfg is None:
        cfg = FilterConfig()

    # ── 数据长度 ──────────────────────────────────────────
    if len(ind) < 5:
        return False, "数据不足"

    # ── 波段结构：前三上涨波段必须 u1 < u3 < u5（所有模式适用）─────
    waves = ind.get("waves", [])
    up_waves = [w for w in waves if w["direction"] == "up"]
    if len(up_waves) < 3:
        return False, f"上涨波段不足3个（仅{len(up_waves)}个），无法验证结构"
    u1_h = up_waves[0]["wave_high"]
    u3_h = up_waves[1]["wave_high"]
    u5_h = up_waves[2]["wave_high"]
    if not (u1_h < u3_h < u5_h):
        return False, f"前三上涨波段未满足u1<u3<u5（{u1_h:.2f}/{u3_h:.2f}/{u5_h:.2f}），结构不健康"
    # ── latest-wave-down 模式：下跌段宽松，前一个上涨波段严格验证 ──
    if cfg.require_latest_wave_down:
        waves = ind.get("waves", [])
        if not waves:
            return False, "无波段数据"
        last_wave = waves[-1]
        if last_wave["direction"] != "down":
            return False, f"最新波段为{last_wave['direction']}非下跌"
        if last_wave["len"] < cfg.min_latest_down_length:
            return False, f"下跌波段仅{last_wave['len']}天<{cfg.min_latest_down_length}天"

        # ① 当前下跌波段：只验证MA趋势
        if not (ind["close"] > ind["ma20"]):
            return False, f"收盘{ind['close']:.2f}≤MA20{ind['ma20']:.2f}"
        if not (ind["ma20"] > ind["ma60"]):
            return False, f"MA20{ind['ma20']:.2f}≤MA60{ind['ma60']:.2f}"
        if not (ind["close"] > ind["ma60"]):
            return False, f"收盘{ind['close']:.2f}≤MA60{ind['ma60']:.2f}"

        # ② 前一个上涨波段：必须满足波段内每日MA20>MA60 & close>MA20 & close>MA60 & 换手率达标
        if len(waves) < 2:
            return False, "无前序上涨波段"
        prev_wave = waves[-2]
        if prev_wave["direction"] != "up":
            return False, f"前序波段为{prev_wave['direction']}非上涨"

        s = prev_wave["start_idx"]
        e = prev_wave["end_idx"]
        ma20_arr = ind["_ma20"]
        ma60_arr = ind["_ma60"]
        close_arr = ind["_close"]
        turnover_arr = ind.get("_turnover")

        market_cap = ind.get("market_cap", 0)
        if market_cap >= 500:
            min_turnover = 0.5
        elif market_cap >= 100:
            min_turnover = 1.0
        elif market_cap >= 30:
            min_turnover = 1.5
        else:
            min_turnover = 2.0

        bad_days = []
        for i in range(s, e + 1):
            reasons = []
            if ma20_arr[i] <= ma60_arr[i]:
                reasons.append("MA20≤MA60")
            if close_arr[i] <= ma20_arr[i]:
                reasons.append("close≤MA20")
            if close_arr[i] <= ma60_arr[i]:
                reasons.append("close≤MA60")
            if turnover_arr is not None and len(turnover_arr) > i:
                if turnover_arr[i] < min_turnover:
                    reasons.append(f"换手{turnover_arr[i]:.2f}%<{min_turnover}%")
            if reasons:
                bad_days.append((i, ", ".join(reasons)))

        if bad_days:
            bad_info = [f"{str(ind['_dates'][i])[:10]}({r})" for i, r in bad_days[:3]]
            return False, f"前涨段{bad_info[0]}等日期不满足条件"

        return True, "通过（下跌波段蓄势，前涨段量价健康）"



    # ── 趋势条件（核心：收盘>MA20）────────────────────────────
    if cfg.require_close_above_ma20:
        if not (ind["close"] > ind["ma20"]):
            return False, f"收盘{ind['close']:.2f}≤MA20{ind['ma20']:.2f}"

    # ── 均线多头排列：MA20>MA60，收盘>MA60 ──────────────────────────
    if cfg.require_ma20_above_ma60:
        if not (ind["ma20"] > ind["ma60"]):
            return False, f"MA20{ind['ma20']:.2f}≤MA60{ind['ma60']:.2f}"
    if cfg.require_close_above_ma60:
        if not (ind["close"] > ind["ma60"]):
            return False, f"收盘{ind['close']:.2f}≤MA60{ind['ma60']:.2f}"

    # ── 硬性条件：波段内每个交易日必须 MA20>MA60, close>MA20, close>MA60, 换手率达标 ──────────────────────
    if cfg.require_wave_ma20_above_ma60:
        waves = ind.get("waves", [])
        if waves:
            last_wave = waves[-1]
            s = last_wave["start_idx"]
            e = last_wave["end_idx"]
            ma20_arr = ind["_ma20"]
            ma60_arr = ind["_ma60"]
            close_arr = ind["_close"]
            turnover_arr = ind.get("_turnover")  # 数组，每日的换手率（%）

            # 市值分档换手率下限（与量能条件一致）
            market_cap = ind.get("market_cap", 0)
            if market_cap >= 500:
                min_turnover = 0.5
            elif market_cap >= 100:
                min_turnover = 1.0
            elif market_cap >= 30:
                min_turnover = 1.5
            else:
                min_turnover = 2.0

            bad_days = []
            for i in range(s, e + 1):
                reason = []
                if ma20_arr[i] <= ma60_arr[i]:
                    reason.append(f"MA20≤MA60")
                if close_arr[i] <= ma20_arr[i]:
                    reason.append(f"close≤MA20")
                if close_arr[i] <= ma60_arr[i]:
                    reason.append(f"close≤MA60")
                if turnover_arr is not None and len(turnover_arr) > i:
                    if turnover_arr[i] < min_turnover:
                        reason.append(f"换手率{turnover_arr[i]:.2f}%<{min_turnover}%")
                if reason:
                    bad_days.append((i, ", ".join(reason)))

            if bad_days:
                bad_info = [f"{str(ind['_dates'][i])[:10]}({r})" for i, r in bad_days[:3]]
                return False, f"波段内{bad_info[0]}等日期不满足条件（{market_cap:.0f}亿档）"

    # ── MACD条件 ──────────────────────────────────────────
    if cfg.require_macd_positive:
        if ind["macd"] <= 0:
            return False, f"MACD={ind['macd']:.4f}≤0"

    if cfg.require_macd_2days:
        macd = ind["_macd"]
        idx = ind["_idx"]
        if idx < 1 or not (macd[idx] > 0 and macd[idx - 1] > 0):
            return False, "MACD未连续2天红柱"

    if cfg.require_dif_rising_2days:
        dif = ind["_dif"]
        idx = ind["_idx"]
        if idx < 2 or not (dif[idx] > dif[idx - 1] > dif[idx - 2]):
            return False, "DIF未连续2日上涨"

    if cfg.require_dea_rising:
        dea = ind["_dea"]
        idx = ind["_idx"]
        if idx < 1 or not (dea[idx] > dea[idx - 1]):
            return False, "DEA未上涨"

    red_days = ind["red_days"]
    if red_days > cfg.max_red_days:
        return False, f"红柱天数{red_days}>30（已过度运行）"

    # ── 涨幅条件 ──────────────────────────────────────────
    gain1 = ind.get("gain1", 0)
    gain3 = ind.get("gain3", 0)
    gain20 = ind.get("gain20", 0)

    if gain3 <= cfg.min_gain3:
        return False, f"3日涨幅{gain3:.2f}%<3%（动能不足）"

    if gain3 >= cfg.max_gain3:
        return False, f"3日涨幅{gain3:.2f}%≥{cfg.max_gain3}%（已过热）"

    if gain20 >= cfg.max_gain20:
        return False, f"20日涨幅{gain20:.2f}%≥60%（位置太高）"

    # ── 量能条件（按市值分档）──────────────────────────────────────
    turnover = ind.get("turnover_est", 0)
    market_cap = ind.get("market_cap", 0)
    # 市值分档换手率门槛：大盘股低换手正常，小盘股需要更高换手
    if market_cap >= 500:      # 大盘 >500亿
        min_turnover = 0.5
    elif market_cap >= 100:    # 中盘 100-500亿
        min_turnover = 1.0
    elif market_cap >= 30:     # 小盘 30-100亿
        min_turnover = 1.5
    else:                       # 微盘 <30亿
        min_turnover = 2.0
    if turnover < min_turnover:
        return False, f"换手率{turnover:.2f}%<{min_turnover}%（{market_cap:.0f}亿市值档）"

    if ind.get("vol_ratio", 0) < cfg.min_vol_ratio:
        return False, f"量比{ind.get('vol_ratio', 0):.2f}<{cfg.min_vol_ratio}（量能偏弱）"

    # ── 周期量价模式（核心检查）──────────────────────────────
    if cfg.require_wave_up_gt_down:
        if not ind.get("wave_recent_up_gt_down", False):
            ratio = ind.get("wave_up_vs_down_ratio", 0.0)
            return False, f"涨段均量<跌段均量（量价结构不健康）"

    if ind.get("wave_up_vs_down_ratio", 0.0) < cfg.min_wave_up_ratio:
        ratio = ind.get("wave_up_vs_down_ratio", 0.0)
        return False, f"波段涨/跌量比={ratio:.2f}<{cfg.min_wave_up_ratio}"

    if ind.get("wave_pattern_score", 0.0) < cfg.require_wave_pattern_score:
        score = ind.get("wave_pattern_score", 0.0)
        return False, f"波段模式评分{score:.1f}<{cfg.require_wave_pattern_score}（量价配合不佳）"

    # ── 风险过滤 ──────────────────────────────────────────
    if not cfg.allow_broke_ma5_recently:
        if ind.get("broke_ma5_recently", False):
            return False, "近期曾跌破MA5（支撑不稳）"

    if ind.get("broke_ma10_count", 0) > cfg.max_broke_ma10_days:
        return False, f"近3天{ind.get('broke_ma10_count', 0)}天跌破MA10（趋势偏弱）"

    # ── 风险过滤 ──────────────────────────────────────────
    if ind.get("rsi", 50) > cfg.max_rsi:
        return False, f"RSI={ind.get('rsi', 0):.1f}>{cfg.max_rsi}（已过热）"

    # ── 上涨质量 ─────────────────────────────────────────
    macd = ind["_macd"]
    close = ind["_close"]
    idx = ind["_idx"]
    rd = ind["red_days"]
    if rd >= 2:
        start = max(1, idx - rd + 1)
        up_count = sum(
            1 for i in range(start, idx + 1)
            if close[i] > close[i - 1]
        )
        up_ratio = up_count / rd
        if up_ratio < cfg.min_up_ratio:
            return False, f"红柱区间上涨日{up_ratio:.0%}<50%（质量不足）"

    return True, "通过"
