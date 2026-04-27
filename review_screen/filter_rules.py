#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
筛选规则（基于实战经验设计）
"""

from dataclasses import dataclass

from utils import w_get, find_ascending_start


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

    # ── MACD条件（宽松化）────────────────────────────────
    require_macd_positive: bool = True      # MACD > 0（方向判断）
    require_macd_2days: bool = False        # 取消：连续2天红柱（太严）
    require_dif_rising_2days: bool = False   # 取消：连续2日DIF上涨 → 改为单日
    require_dif_rising_today: bool = True   # 新增：当日 DIF 上涨即可
    require_dea_rising: bool = True        # DEA上涨
    max_red_days: int = 30               # 红柱天数上限

    # ── 涨幅条件 ──────────────────────────────────────────
    min_gain5: float = 3.0     # 5日涨幅下限
    max_gain5: float = 22.0    # 5日涨幅上限（>18%过热，>22%风险大）
    max_gain20: float = 60.0   # 20日涨幅上限

    # ── 量能条件 ──────────────────────────────────────────
    min_turnover: float = 1.0          # 换手率下限（%）
    min_vol_ratio: float = 1.0           # 当日量比下限
    min_vol_up_vs_down: float = 1.2     # 波段涨/跌均量比下限
    min_vol_consec_strong: int = 2      # 连续放量天数下限

    # ── 波段方向 ──────────────────────────────────────────
    require_latest_wave_down: bool = False  # 要求最新波段为下跌（蓄势找买点）
    min_latest_down_length: int = 2         # 下跌波段最小天数
    strict_trend_only: bool = False         # 严格趋势（latest-wave-down专用：只要求MA条件）

    # ── 回调支撑 ──────────────────────────────────────────
    allow_broke_ma5_recently: bool = True   # 允许近期跌破MA5（强势回调）
    max_broke_ma10_days: int = 3          # 跌破MA10的天数窗口上限（compute_all 用此值计算 broke_ma10_count）

    # ── 周期量价模式（核心）────────────────────────────────
    require_wave_up_gt_down: bool = True  # 必须满足涨段均量>跌段均量
    min_wave_up_ratio: float = 1.2       # 波段涨/跌均量比下限
    require_wave_pattern_score: float = 0.4  # 波段模式评分下限

    # ── 上涨质量 ──────────────────────────────────────────
    min_up_ratio: float = 0.5      # 红柱区间上涨日占比下限

    # ── 位置评分（低位加分）───────────────────────────────
    enable_position_score: bool = True  # 开启位置评分（低位加分）
    position_bonus_max: float = 10.0     # 位置评分上限（分）
    position_low_threshold: float = 15.0 # 近60日低点涨幅 < 此值 → 加分

    # ── 软过滤（非核心条件不过不拒绝，降分）─────────────────
    soft_filter_turnover: bool = True    # 换手率软过滤
    soft_filter_vol_ratio: bool = True   # 量比软过滤
    soft_penalty_turnover: float = 8.0  # 换手率不达标扣分
    soft_penalty_vol_ratio: float = 5.0  # 量比不达标扣分

    # ── 恢复模式（趋势破坏后重建）────────────────────────────
    enable_recovery_mode: bool = True   # 开启恢复模式
    recovery_min_low: float = -25.0     # 允许的最大回调幅度（%）


def check_filters(ind: dict, cfg: FilterConfig = None) -> tuple[bool, str, float]:
    """
    检查单只股票是否通过筛选条件
    Returns: (是否通过, 拒绝原因, 软扣分)
    软扣分：非核心条件不达标时不拒绝，但扣分累加到总分
    """
    if cfg is None:
        cfg = FilterConfig()

    soft_penalty = 0.0

    # ── 数据长度 ──────────────────────────────────────────
    if len(ind) < 5:
        return False, "数据不足", 0.0

    # ── 波段结构：扫描找到第一个连续递增对 u1<u3 ──────────────
    waves = ind.get("waves", [])
    up_waves = [w for w in waves if w_get(w, "direction") == "up"]

    # ── 容错路径：只有1个上涨波段但有真实上涨动力 ─────────────────────
    # 目标：捕捉还在启动初期的股票（还没走出完整的两波上涨）
    if len(up_waves) == 1:
        gain20 = ind.get("gain20", 0)
        close = ind["close"]
        ma20 = ind["ma20"]
        ma60 = ind["ma60"]
        # 近20日涨幅 > 0：方向向上
        # close > ma60：价格站稳
        # ma20 > ma60：中期趋势向上
        if gain20 > 0 and close > ma60 and ma20 > ma60:
            # 近20日上涨天数（粗估：看趋势是否健康）
            if ind.get("wave_up_vs_down_ratio", 0) >= 1.0:
                # 单波但量价健康：允许通过，标记为单波蓄势
                pass  # 继续后续检查
            else:
                return False, f"单波+量能不足（波量比={ind.get('wave_up_vs_down_ratio', 0):.2f}<1.0）", 0.0
        else:
            return False, f"上涨波段仅1个且近20日涨幅{gain20:.1f}%<0或均线不满足", 0.0

    if len(up_waves) < 2:
        # ── 容错路径：只有1个上涨波段但有真实上涨动力 ─────────────────
        # 目标：捕捉还在启动初期的股票（还没走出完整的两波上涨）
        if len(up_waves) == 1:
            gain20 = ind.get("gain20", 0)
            close = ind["close"]
            ma20 = ind["ma20"]
            ma60 = ind["ma60"]
            # 近20日涨幅>0 + 价格站稳MA60 + 均线多头 → 容许通过
            if gain20 > 0 and close > ma60 and ma20 > ma60 and ind.get("wave_up_vs_down_ratio", 0) >= 1.0:
                pass  # 单波蓄势通过，继续后续检查
            else:
                reason = (
                    f"单波+近20日涨幅{gain20:.1f}%<0或"
                    if gain20 <= 0 else
                    f"单波+波量比={ind.get('wave_up_vs_down_ratio', 0):.2f}<1.0或"
                )
                return False, f"{reason}条件不满足", 0.0
        else:
            return False, f"上涨波段不足2个（仅{len(up_waves)}个）", 0.0

    found_idx = find_ascending_start(up_waves, default=None)
    if found_idx is None:
        return False, "未找到有效递增对 u1<u3", 0.0
        return False, "未找到有效递增对 u1<u3", 0.0

    # ── latest-wave-down 模式 ──────────────────────────────
    if cfg.require_latest_wave_down:
        wave_list = ind.get("waves", [])
        if not wave_list:
            return False, "无波段数据", 0.0
        last_wave = wave_list[-1]
        if w_get(last_wave, "direction") != "down":
            return False, f"最新波段为{w_get(last_wave, 'direction')}非下跌", 0.0
        if w_get(last_wave, "len") < cfg.min_latest_down_length:
            return False, f"下跌波段仅{w_get(last_wave, 'len')}天<{cfg.min_latest_down_length}天", 0.0

        # 当前下跌波段：只验证MA趋势
        if not (ind["close"] > ind["ma20"]):
            return False, f"收盘{ind['close']:.2f}≤MA20{ind['ma20']:.2f}", 0.0
        if not (ind["ma20"] > ind["ma60"]):
            return False, f"MA20{ind['ma20']:.2f}≤MA60{ind['ma60']:.2f}", 0.0
        if not (ind["close"] > ind["ma60"]):
            return False, f"收盘{ind['close']:.2f}≤MA60{ind['ma60']:.2f}", 0.0

        # 前一个上涨波段严格验证
        if len(waves) < 2:
            return False, "无前序上涨波段", 0.0
        prev_wave = waves[-2]
        if w_get(prev_wave, "direction") != "up":
            return False, f"前序波段为{w_get(prev_wave, 'direction')}非上涨", 0.0

        s = w_get(prev_wave, "start_idx")
        e = w_get(prev_wave, "end_idx")
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
            return False, f"前涨段{bad_info[0]}等日期不满足条件", 0.0

        return True, "通过（下跌波段蓄势，前涨段量价健康）", 0.0

    # ── 检测是否触发恢复模式 ────────────────────────────────
    recovery_mode = False
    if cfg.enable_recovery_mode:
        close = ind["close"]
        ma20 = ind["ma20"]
        ma60 = ind["ma60"]
        # 恢复模式：价格跌破MA20 但 NOT 全面破位（MA60 还向上或者价格刚触底反弹）
        # 核心逻辑：
        #   场景A: close<ma20 但 ma20>ma60 → 正常回调（old逻辑，保留）
        #   场景B: close<ma20 且 ma20<ma60（全面破位）→ 检查是否从底部反弹
        #          → 条件：近5日从低点反弹 > 3% AND 最低点未创新低
        # 两种场景任一种成立则进入恢复模式
        if close < ma20:
            low_60d = ind.get("low_60d", close)
            gain_from_low = (close / low_60d - 1) * 100 if low_60d > 0 else 0
            
            if gain_from_low < cfg.recovery_min_low:
                recovery_mode = False  # 还在创新低，不恢复
            elif ma20 > ma60:
                # 场景A：正常回调，MA趋势还在
                recovery_mode = True
            else:
                # 场景B：全面破位，但要最近从低点反弹
                gain5 = ind.get("gain5", 0)
                if gain5 > 3.0 and gain_from_low > cfg.recovery_min_low:
                    recovery_mode = True  # 从底部反弹

    # ── 核心硬条件（趋势MA约束）─────────────────────────────
    if recovery_mode:
        # 恢复模式：只需要 MA20 > MA60，放宽收盘位置
        if cfg.require_ma20_above_ma60:
            if not (ind["ma20"] > ind["ma60"]):
                return False, f"MA20{ind['ma20']:.2f}≤MA60{ind['ma60']:.2f}", 0.0
        if cfg.require_close_above_ma60:
            if not (ind["close"] > ind["ma60"]):
                return False, f"收盘{ind['close']:.2f}≤MA60{ind['ma60']:.2f}", 0.0
    else:
        # 正常模式：完整趋势要求
        if cfg.require_close_above_ma20:
            if not (ind["close"] > ind["ma20"]):
                return False, f"收盘{ind['close']:.2f}≤MA20{ind['ma20']:.2f}", 0.0
        if cfg.require_ma20_above_ma60:
            if not (ind["ma20"] > ind["ma60"]):
                return False, f"MA20{ind['ma20']:.2f}≤MA60{ind['ma60']:.2f}", 0.0
        if cfg.require_close_above_ma60:
            if not (ind["close"] > ind["ma60"]):
                return False, f"收盘{ind['close']:.2f}≤MA60{ind['ma60']:.2f}", 0.0

    # ── MACD条件（宽松化）──────────────────────────────────
    if cfg.require_macd_positive:
        if ind["macd"] <= 0:
            return False, f"MACD={ind['macd']:.4f}≤0", 0.0

    # DIF 改为单日上涨
    if cfg.require_dif_rising_today:
        dif = ind["_dif"]
        idx = ind["_idx"]
        if idx < 1 or not (dif[idx] > dif[idx - 1]):
            return False, "DIF未上涨", 0.0

    if cfg.require_dea_rising:
        dea = ind["_dea"]
        idx = ind["_idx"]
        if idx < 1 or not (dea[idx] > dea[idx - 1]):
            return False, "DEA未上涨", 0.0

    red_days = ind["red_days"]
    if red_days > cfg.max_red_days:
        return False, f"红柱天数{red_days}>30（已过度运行）", 0.0

    # ── 涨幅条件 ──────────────────────────────────────────
    gain5 = ind.get("gain5", 0)
    gain20 = ind.get("gain20", 0)

    if gain5 <= cfg.min_gain5:
        return False, f"5日涨幅{gain5:.1f}%<{cfg.min_gain5}%（动能不足）", 0.0

    if gain5 >= cfg.max_gain5:
        return False, f"5日涨幅{gain5:.1f}%≥{cfg.max_gain5}%（已过热）", 0.0

    if gain20 >= cfg.max_gain20:
        return False, f"20日涨幅{gain20:.1f}%≥60%（位置太高）", 0.0

    # ── 相对强度门槛（跑输市场太远的直接拒绝）────────────────────────
    # 使用外部计算的 relative_strength（个股20日涨幅 - 指数20日涨幅）
    # 未提供时用 gain20 作为代理（假设指数≈0基准）
    rel = ind.get("relative_strength", None)
    if rel is None:
        rel = gain20  # 代理：个股20日涨幅 vs 市场（假设市场横盘）
    if rel < -10.0:
        return False, f"相对强度{rel:.1f}%<{'-10.0'}%（跑输市场太远）", 0.0

    # ── 量能条件（按市值分档 + 软拒绝）────────────────────────
    turnover = ind.get("turnover_est", 0)
    market_cap = ind.get("market_cap", 0)
    if market_cap >= 500:
        min_turnover = 0.5
    elif market_cap >= 100:
        min_turnover = 1.0
    elif market_cap >= 30:
        min_turnover = 1.5
    else:
        min_turnover = 2.0

    if turnover < min_turnover:
        if cfg.soft_filter_turnover:
            soft_penalty += cfg.soft_penalty_turnover
        else:
            return False, f"换手率{turnover:.2f}%<{min_turnover}%（{market_cap:.0f}亿市值档）", 0.0

    vol_ratio = ind.get("vol_ratio", 1.0)
    if vol_ratio < cfg.min_vol_ratio:
        if cfg.soft_filter_vol_ratio:
            soft_penalty += cfg.soft_penalty_vol_ratio
        else:
            return False, f"量比{vol_ratio:.2f}<{cfg.min_vol_ratio}（量能偏弱）", 0.0

    # 绝对量门槛（新增）：近5日均量 > 近60日均量 * 0.8
    vol_ma5 = ind.get("vol_ma5", 0)
    vol_ma60 = ind.get("vol_ma60", 0)
    if vol_ma60 > 0 and vol_ma5 < vol_ma60 * 0.8:
        soft_penalty += 5.0

    # ── 周期量价模式（改为软扣分）────────────────────────────
    wave_up_ratio = ind.get("wave_up_vs_down_ratio", 0.0)
    if cfg.require_wave_up_gt_down:
        if not ind.get("wave_recent_up_gt_down", False):
            soft_penalty += 8.0

    if wave_up_ratio < cfg.min_wave_up_ratio and wave_up_ratio > 0:
        soft_penalty += 6.0

    wave_pattern_score = ind.get("wave_pattern_score", 0.0)
    if wave_pattern_score < cfg.require_wave_pattern_score:
        soft_penalty += 5.0

    # ── 上涨质量 ─────────────────────────────────────────
    close = ind["_close"]
    idx = ind["_idx"]
    rd = ind["red_days"]
    if rd >= 2:
        start = max(1, idx - rd + 1)
        up_count = sum(
            1 for i in range(start, idx + 1)
            if close[i] > close[i - 1]
        )
        up_ratio_calc = up_count / rd
        if up_ratio_calc < cfg.min_up_ratio:
            soft_penalty += 6.0

    return True, "通过", soft_penalty
