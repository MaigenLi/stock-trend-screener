#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
明日买入准备度过滤器
====================

在复盘通过的基础上，增加"明日是否能买"的判断。
这是从"复盘分析"到"实盘决策"的关键桥梁。

P0 核心：过滤掉不能买、不能追的股票
P1 增强：加入大盘环境判断
P2 积累：建立模拟验证数据

用法（建议通过 screen_trend_filter.py 的 --buy-ready 调用）:
    from buy_ready import BuyReadyConfig, apply_buy_ready_filter, get_market_mode, compute_risk_reward
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json

import numpy as np

WORKSPACE = Path(__file__).parent.parent.parent.resolve()
CACHE_DIR = WORKSPACE / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ── 大盘环境检测 ─────────────────────────────────────────────
# HS300数据缓存文件
_HS300_CACHE = CACHE_DIR / "hs300_ma20.json"


def get_market_mode(target_date: datetime | None = None) -> dict:
    """
    检测大盘环境（HS300 vs MA20）

    Returns:
        dict{
            "mode": "牛市" | "震荡" | "熊市",
            "hs300": float,      # 当前HS300点位
            "ma20": float,       # HS300 MA20值
            "ratio": float,      # (hs300/ma20 - 1) * 100
            "trend": "up" | "down",
            "signal": str,       # 人类可读描述
        }

    大盘规则:
    - 牛市: HS300 > MA20 AND ratio > 1%
    - 震荡: HS300 > MA20 but ratio <= 1%
    - 熊市: HS300 < MA20
    """
    import akshare as ak

    try:
        df = ak.stock_zh_index_daily(symbol="sh000300")
        df = df.tail(30)
        closes = df["close"].values
        ma20 = float(np.mean(closes[-20:]))
        hs300 = float(closes[-1])
        ratio = (hs300 / ma20 - 1) * 100
        trend = "up" if hs300 > ma20 else "down"

        if ratio > 1.0:
            mode = "牛市"
        elif ratio > 0:
            mode = "震荡"
        else:
            mode = "熊市"

        return {
            "mode": mode,
            "hs300": round(hs300, 2),
            "ma20": round(ma20, 2),
            "ratio": round(ratio, 3),
            "trend": trend,
            "signal": f"🐂 {mode}" if trend == "up" else f"🐻 {mode}",
        }
    except Exception as e:
        # HS300获取失败，返回中性假设
        return {
            "mode": "震荡",
            "hs300": 0,
            "ma20": 0,
            "ratio": 0,
            "trend": "up",
            "signal": "⚠️ 大盘未知",
        }


@dataclass
class BuyReadyConfig:
    """
    明日买入准备度配置
    ========================

    P0 配置（必须满足）:
    - 今日涨幅: 0.3% ~ 8%（不能跌、不能暴涨）
    - 非涨停/跌停（涨跌幅 < 9.5%）
    - RSI(14) < 80（不能过热追高）
    - 距MA5 < 8%（不能偏离太远）

    P1 配置（大盘相关）:
    - 熊市时提高门槛（减少买入）

    P2 配置（模拟验证）:
    - 记录每次信号，用于事后验证
    """

    # ── P0: 硬性过滤 ────────────────────────────────────────
    min_gain1: float = 0.3       # 今日涨幅下限（%），避免选跌的
    max_gain1: float = 8.0      # 今日涨幅上限（%），避免追暴涨
    max_rsi: float = 80.0        # RSI上限（>80过热）
    max_ma5_dist: float = 8.0    # 距MA5上限（%）

    # ── P1: 大盘环境调整 ────────────────────────────────────
    enable_market_adjust: bool = True  # 是否根据大盘调整阈值
    bear_min_gain1: float = 1.0       # 熊市时提高涨幅下限
    bear_max_gain1: float = 5.0        # 熊市时降低涨幅上限
    bear_max_ma5_dist: float = 5.0     # 熊市时缩小偏离容忍度

    # ── 涨停排除 ────────────────────────────────────────────
    limit_pct: float = 9.5       # 涨停 threshold（%），>9.5% 视为涨停

    # ── 止损配置 ────────────────────────────────────────────
    stop_loss_warning: float = 5.0    # 亏损至此百分比（%）报警
    stop_loss_forced: float = 8.0     # 亏损至此百分比（%）建议卖出

    # ── 风险收益评分 ────────────────────────────────────────
    enable_risk_reward: bool = True  # 是否计算风险收益评分

    # ── P2: 模拟验证 ────────────────────────────────────────
    simulation_file: Path = field(
        default_factory=lambda: Path.home() / "stock_reports" / "simulation_log.json"
    )

    def get_effective_thresholds(self, market_mode: str) -> dict:
        """根据大盘模式返回实际使用的阈值"""
        if not self.enable_market_adjust:
            return {
                "min_gain1": self.min_gain1,
                "max_gain1": self.max_gain1,
                "max_ma5_dist": self.max_ma5_dist,
            }

        if market_mode == "熊市":
            return {
                "min_gain1": self.bear_min_gain1,
                "max_gain1": self.bear_max_gain1,
                "max_ma5_dist": self.bear_max_ma5_dist,
            }
        elif market_mode == "震荡":
            return {
                "min_gain1": max(self.min_gain1, 0.5),
                "max_gain1": min(self.max_gain1, 6.0),
                "max_ma5_dist": min(self.max_ma5_dist, 6.0),
            }
        else:  # 牛市
            return {
                "min_gain1": self.min_gain1,
                "max_gain1": self.max_gain1,
                "max_ma5_dist": self.max_ma5_dist,
            }


def is_limit_up(ind: dict, limit_pct: float = 9.5) -> bool:
    """判断是否涨停或跌停（不能买卖）"""
    gain1 = ind.get("gain1", 0)
    return gain1 >= limit_pct or gain1 <= -limit_pct


def compute_risk_reward(ind: dict, cfg: BuyReadyConfig) -> dict:
    """
    计算风险收益评分（用于优先级排序）

    综合维度：
    1. 止损空间（越小越好）→ 买入安全边际
    2. 潜在涨幅（相对历史位置）→ 越高位置的股票上涨空间越小
    3. RSI健康度（50-75最佳）→ 太低是弱势，太高是过热
    4. 量能配合度（上涨日放量）→ 量价配合才健康

    Returns:
        dict{
            "rr_score": float,        # 综合风险收益评分（0-100，越高越好）
            "stop_loss_pct": float,  # 止损幅度（%）
            "upside_pct": float,     # 潜在涨幅（%）
            "rsi_score": float,      # RSI健康分（0-25）
            "vol_score": float,      # 量能分（0-25）
        }
    """
    close = ind.get("close", 0)
    sl_ref = ind.get("stop_loss_ref")
    ma20 = ind.get("ma20", 0)
    ma60 = ind.get("ma60", 0)
    rsi = ind.get("rsi", 50)
    gain20 = ind.get("gain20", 0)
    vol_ratio = ind.get("vol_ratio", 1.0)
    vol_up_vs_down = ind.get("vol_up_vs_down", 1.0)
    ma5_dist = ind.get("ma5_distance_pct", 0)
    phase = ind.get("phase", "不明")

    # ── 止损幅度 ──────────────────────────────────────────
    if sl_ref and close > 0:
        stop_loss_pct = (close - sl_ref) / close * 100
    else:
        # 无止损参考，用近20日低点估算
        low = ind.get("low_distance_pct", 0)
        stop_loss_pct = abs(low) if low < 0 else 5.0

    # ── 潜在涨幅估算（MA20以上空间 / 历史波动）───────────
    # 越高位的股票继续上涨空间越小（但也不能太低，那是弱势）
    if ma20 > 0:
        dist_from_ma20 = (close / ma20 - 1) * 100
        # 距MA20 5-20% 是最佳区间（不是追高，但也确认了趋势）
        if 5 <= dist_from_ma20 <= 20:
            upside_pct = 20 - dist_from_ma20  # 离MA20越近，上涨空间越大
        elif dist_from_ma20 < 5:
            upside_pct = 15 + (5 - dist_from_ma20)  # 贴近MA20，有均线支撑
        else:
            upside_pct = max(0, 10 - (dist_from_ma20 - 20) * 0.5)  # 过高，追高风险大
    else:
        upside_pct = 10.0

    # ── RSI健康分（0-25）───────────────────────────────────
    if 50 <= rsi <= 70:
        rsi_score = 25.0  # 最佳区间
    elif 45 <= rsi < 50 or 70 < rsi <= 75:
        rsi_score = 20.0
    elif 40 <= rsi < 45 or 75 < rsi <= 80:
        rsi_score = 12.0
    elif rsi < 40:
        rsi_score = 5.0   # 弱势，可能还在跌
    else:
        rsi_score = 0.0   # 过热

    # ── 量能分（0-25）──────────────────────────────────────
    vol_score = 0.0
    if vol_ratio >= 1.3:
        vol_score += 10.0
    elif vol_ratio >= 1.0:
        vol_score += 7.0
    elif vol_ratio >= 0.8:
        vol_score += 3.0

    if vol_up_vs_down >= 1.5:
        vol_score += 10.0
    elif vol_up_vs_down >= 1.2:
        vol_score += 7.0
    elif vol_up_vs_down >= 1.0:
        vol_score += 4.0
    else:
        vol_score += 0.0

    vol_score = min(vol_score, 25.0)

    # ── 止损分（0-25）─────────────────────────────────────
    if stop_loss_pct <= 2.0:
        sl_score = 25.0
    elif stop_loss_pct <= 3.0:
        sl_score = 22.0
    elif stop_loss_pct <= 5.0:
        sl_score = 18.0
    elif stop_loss_pct <= 8.0:
        sl_score = 12.0
    else:
        sl_score = max(0, 10 - (stop_loss_pct - 8.0))

    # ── 位置分（0-25）─────────────────────────────────────
    pos_score = 0.0
    if phase == "主升浪":
        pos_score += 15.0
    elif phase == "二次启动":
        pos_score += 12.0
    elif phase == "洗盘":
        pos_score += 8.0

    # MA5偏离：贴近MA5最好（+5），太远扣分
    if abs(ma5_dist) <= 3.0:
        pos_score += 5.0
    elif abs(ma5_dist) <= 6.0:
        pos_score += 2.0

    pos_score = min(pos_score, 25.0)

    # ── 综合评分 ──────────────────────────────────────────
    rr_score = min(sl_score + rsi_score + vol_score + pos_score, 100.0)

    return {
        "rr_score": round(rr_score, 1),
        "stop_loss_pct": round(stop_loss_pct, 2),
        "upside_pct": round(upside_pct, 1),
        "rsi_score": round(rsi_score, 1),
        "vol_score": round(vol_score, 1),
        "sl_score": round(sl_score, 1),
        "pos_score": round(pos_score, 1),
    }


def apply_buy_ready_filter(
    ind: dict,
    cfg: BuyReadyConfig,
    market_mode: str = "震荡",
) -> tuple[bool, str, dict]:
    """
    明日买入准备度过滤

    Returns:
        (是否准备就绪, 拒绝原因, 风险收益分析)
    """
    rr = compute_risk_reward(ind, cfg) if cfg.enable_risk_reward else {}
    thresholds = cfg.get_effective_thresholds(market_mode)

    # ── P0-1: 涨停排除 ─────────────────────────────────────
    if is_limit_up(ind, cfg.limit_pct):
        return False, f"涨停/跌停（今日涨幅{ind.get('gain1', 0):.1f}%）", rr

    # ── P0-2: 今日涨幅范围 ─────────────────────────────────
    gain1 = ind.get("gain1", 0)
    if gain1 < thresholds["min_gain1"]:
        return False, f"涨幅{gain1:.1f}%<{thresholds['min_gain1']}%（偏弱）", rr
    if gain1 > thresholds["max_gain1"]:
        return False, f"涨幅{gain1:.1f}%>{thresholds['max_gain1']}%（追高风险）", rr

    # ── P0-3: RSI过滤 ──────────────────────────────────────
    rsi = ind.get("rsi", 50)
    if rsi >= cfg.max_rsi:
        return False, f"RSI={rsi:.1f}>={cfg.max_rsi}（过热）", rr

    # ── P0-4: 距MA5偏离度 ────────────────────────────────
    ma5_dist = ind.get("ma5_distance_pct", 0)
    if ma5_dist > thresholds["max_ma5_dist"]:
        return False, f"距MA5+{ma5_dist:.1f}%>{thresholds['max_ma5_dist']}%（偏离过远）", rr
    if ma5_dist < -10:  # 跌破MA5超过10%，可能是下跌趋势
        return False, f"距MA5{ma5_dist:.1f}%<-10%（下跌趋势）", rr

    # ── P0-5: 连续两天下跌排除（弱势）──────────────────────
    # 检查近2日是否有连续下跌
    gain_arr = ind.get("_gain_list", [])
    if len(gain_arr) >= 2:
        if gain_arr[-1] < 0 and gain_arr[-2] < 0:
            return False, "连续两天下跌（弱势）", rr

    return True, "买入准备就绪", rr


def get_stop_loss_alert(
    ind: dict,
    entry_price: float,
    cfg: BuyReadyConfig,
) -> dict:
    """
    止损风险评估（买入后持续监控）

    Args:
        ind: 当前股票指标
        entry_price: 买入价格
        cfg: 止损配置

    Returns:
        dict{
            "current_loss_pct": float,  # 当前亏损百分比
            "status": "safe" | "warning" | "forced_sell",
            "message": str,
            "action": str,
        }
    """
    current_price = ind.get("close", 0)
    if entry_price <= 0 or current_price <= 0:
        return {
            "current_loss_pct": 0,
            "status": "safe",
            "message": "价格数据不足",
            "action": "持有观察",
        }

    loss_pct = (current_price - entry_price) / entry_price * 100

    if loss_pct >= cfg.stop_loss_forced:
        return {
            "current_loss_pct": round(loss_pct, 2),
            "status": "forced_sell",
            "message": f"亏损{loss_pct:.1f}% ≥ {cfg.stop_loss_forced}%",
            "action": f"🚨 建议卖出（亏损{loss_pct:.1f}%）",
        }
    elif loss_pct >= cfg.stop_loss_warning:
        return {
            "current_loss_pct": round(loss_pct, 2),
            "status": "warning",
            "message": f"亏损{loss_pct:.1f}% ≥ {cfg.stop_loss_warning}%",
            "action": f"⚠️ 注意止损（亏损{loss_pct:.1f}%）",
        }
    else:
        return {
            "current_loss_pct": round(loss_pct, 2),
            "status": "safe",
            "message": f"当前盈亏{loss_pct:+.1f}%",
            "action": "持有" if loss_pct >= 0 else "持有观察",
        }


# ── P2: 模拟验证日志 ─────────────────────────────────────────
def log_signal(
    cfg: BuyReadyConfig,
    stock_code: str,
    stock_name: str,
    signal_date: str,
    entry_price: float,
    stop_loss_ref: float | None,
    rr_score: float,
    stop_loss_pct: float,
    market_mode: str,
) -> None:
    """
    记录模拟信号（用于事后验证）

    记录格式:
    {
        "signal_date": "YYYY-MM-DD",
        "stocks": [
            {
                "code": "000001",
                "name": "平安银行",
                "entry_price": 12.50,
                "stop_loss_ref": 11.80,
                "rr_score": 78.5,
                "stop_loss_pct": 5.6,
                "market_mode": "牛市",
                "result": null,  # 待后续填入
                "holding_days": null,  # 待后续填入
                "exit_price": null,
                "pnl_pct": null,
            }
        ]
    }
    """
    cfg.simulation_file.parent.mkdir(parents=True, exist_ok=True)

    # 读取现有日志
    if cfg.simulation_file.exists():
        try:
            data = json.loads(cfg.simulation_file.read_text(encoding="utf-8"))
        except Exception:
            data = {"signals": []}
    else:
        data = {"signals": []}

    # 追加新信号
    data["signals"].append({
        "code": stock_code,
        "name": stock_name,
        "signal_date": signal_date,
        "entry_price": entry_price,
        "stop_loss_ref": stop_loss_ref,
        "rr_score": rr_score,
        "stop_loss_pct": stop_loss_pct,
        "market_mode": market_mode,
        "result": None,
        "holding_days": None,
        "exit_price": None,
        "pnl_pct": None,
    })

    cfg.simulation_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def update_signal_result(
    cfg: BuyReadyConfig,
    stock_code: str,
    signal_date: str,
    exit_price: float,
    holding_days: int,
) -> dict | None:
    """
    更新信号结果（卖出后调用）

    填入 result/pnl_pct/holding_days/exit_price
    返回更新后的记录
    """
    if not cfg.simulation_file.exists():
        return None

    try:
        data = json.loads(cfg.simulation_file.read_text(encoding="utf-8"))
    except Exception:
        return None

    # 找到对应信号
    for sig in data["signals"]:
        if sig["code"] == stock_code and sig["signal_date"] == signal_date and sig["result"] is None:
            entry = sig["entry_price"]
            sig["exit_price"] = exit_price
            sig["holding_days"] = holding_days
            sig["pnl_pct"] = round((exit_price - entry) / entry * 100, 2)
            sig["result"] = "win" if sig["pnl_pct"] > 0 else "loss"
            cfg.simulation_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            return sig

    return None


def get_simulation_stats(cfg: BuyReadyConfig) -> dict:
    """获取模拟验证统计"""
    if not cfg.simulation_file.exists():
        return {"total": 0, "message": "尚无模拟记录"}

    try:
        data = json.loads(cfg.simulation_file.read_text(encoding="utf-8"))
    except Exception:
        return {"total": 0, "message": "读取失败"}

    signals = data.get("signals", [])
    completed = [s for s in signals if s["result"] is not None]

    if not completed:
        return {"total": len(signals), "completed": 0, "message": "尚无完成交易"}

    wins = [s for s in completed if s["result"] == "win"]
    pnls = [s["pnl_pct"] for s in completed]

    return {
        "total_signals": len(signals),
        "completed": len(completed),
        "win_count": len(wins),
        "win_rate": round(len(wins) / len(completed) * 100, 1),
        "avg_pnl": round(sum(pnls) / len(pnls), 2),
        "max_pnl": round(max(pnls), 2),
        "min_pnl": round(min(pnls), 2),
        "avg_holding_days": round(sum(s["holding_days"] for s in completed) / len(completed), 1),
    }
