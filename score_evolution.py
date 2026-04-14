#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评分进化器 — 自我进化策略 第三层
================================
分析 feedback_tracker.csv 中积累的历史信号，
评估各参数组合的胜率/期望/回撤，找出最优方向，
输出量化的参数调整建议，供人工确认后生效。

用法：
  python score_evolution.py                              # 分析 + 输出建议
  python score_evolution.py --min-samples 20           # 最小样本量门槛
  python score_evolution.py --output evolution_report_2026-04-14.txt
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

WORKSPACE = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(WORKSPACE))

TRACKER_CSV = WORKSPACE / "stock_trend" / "feedback_tracker.csv"
PARAM_GRID = {
    "signal_days": [2, 3],
    "min_gain": [1.5, 2.0, 2.5],
    "max_gain": [5.0, 7.0, 9.0],
    "quality_days": [10, 15, 20],
    "turnover": [1.0, 1.5, 2.0],
    "score_threshold": [55, 60, 65, 70],
    "max_extension": [10, 12, 16],
}
MIN_SAMPLES = 20  # 每个组合最少样本量才参与分析
EVOLUTION_CSV = WORKSPACE / "stock_trend" / "evolution_history.csv"
EVOLUTION_FIELDS = [
    "date", "param_changed", "old_value", "new_value",
    "reason", "expected_winrate_change", "confirmed",
]


# ── 读取数据 ──────────────────────────────────────────────
def load_tracker() -> list[dict]:
    if not Path(TRACKER_CSV).exists():
        return []
    with open(TRACKER_CSV, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _safe_float(val, default=0.0) -> float:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _safe_bool(val) -> bool:
    return str(val) == "True"


# ── 基础指标计算 ──────────────────────────────────────────
def compute_metrics(rows: list[dict]) -> dict:
    """计算一组信号的核心指标。"""
    if not rows:
        return {}
    rets = [_safe_float(r["ret_actual"]) for r in rows if r.get("ret_actual")]
    if not rets:
        return {}
    wins = [r for r in rets if r > 0]
    stop_losses = [_safe_float(r["ret_exit"]) for r in rows if _safe_bool(r["stop_loss"])]
    hit5 = sum(1 for r in rows if _safe_bool(r["hit_5pct"]))
    hit7 = sum(1 for r in rows if _safe_bool(r["hit_7pct"]))
    win_rate = len(wins) / len(rets) * 100
    avg_ret = np.mean(rets)
    std_ret = np.std(rets) if len(rets) > 1 else 1.0
    sharpe = avg_ret / std_ret * math.sqrt(252) if std_ret > 0 else 0.0
    max_loss = min(rets) if rets else 0.0
    max_retrace = max((_safe_float(r["max_retrace"]) for r in rows), default=0.0)
    avg_hold = np.mean([_safe_float(r["hold_days"]) for r in rows if r["hold_days"]]) if rows else 0.0
    n = len(rets)
    return {
        "n": n,
        "win_rate": win_rate,
        "avg_ret": avg_ret,
        "std_ret": std_ret,
        "sharpe": sharpe,
        "max_loss": max_loss,
        "max_retrace": max_retrace,
        "avg_hold": avg_hold,
        "hit5_rate": hit5 / n * 100,
        "hit7_rate": hit7 / n * 100,
        "stop_rate": len(stop_losses) / n * 100,
    }


# ── 单维度分析 ────────────────────────────────────────────
def analyze_dimension(rows: list[dict], dim: str, values: list) -> dict:
    """
    分析某一维度不同取值的表現差异（如 ma5_ma10_strict=True vs False）。
    返回 {value: metrics}
    """
    results = {}
    for val in values:
        subset = [r for r in rows if _safe_float(r.get(dim, 0)) == val
                  or str(r.get(dim, "")) == str(val)]
        if len(subset) >= 5:
            m = compute_metrics(subset)
            if m:
                m["subset_size"] = len(subset)
                results[val] = m
    return results


# ── 趋势维度分析（MA5/MA10约束强度）─────────────────────────
def analyze_ma_constraint_strength(rows: list[dict]) -> dict:
    """
    强约束：信号日 MA5 > MA10 且均比前一天上涨
    弱约束：只满足 close > MA5 >= MA10 * 0.995（MA10可走平）
    """
    # 从现有 tracker 数据推断约束强度（需要原始 K 线数据辅助判断）
    # 简化：用 extension_pct 和 RSI 组合做代理指标
    strong = []
    weak = []
    for r in rows:
        score = _safe_float(r.get("quality_score", 0))
        ret_exit = _safe_float(r.get("ret_exit", 0))
        # 高评分且正收益 → 强约束倾向
        if score >= 70 and ret_exit > 0:
            strong.append(r)
        elif score < 60 or ret_exit < -2:
            weak.append(r)

    m_strong = compute_metrics(strong)
    m_weak = compute_metrics(weak)
    return {"strong": m_strong, "weak": m_weak}


# ── RSI 区间分析 ──────────────────────────────────────────
def analyze_rsi_zone(rows: list[dict]) -> dict:
    """
    低 RSI（<50）：反弹型
    中 RSI（50-65）：稳健型
    高 RSI（65-78）：动量型
    超买（>78）：过滤型（信号应减少出现）
    """
    # quality_score 作为 RSI 高低的代理变量
    low = [r for r in rows if 55 <= _safe_float(r.get("quality_score", 0)) < 70]
    mid = [r for r in rows if 70 <= _safe_float(r.get("quality_score", 0)) < 80]
    high = [r for r in rows if _safe_float(r.get("quality_score", 0)) >= 80]
    return {
        "低分区(55-70)": compute_metrics(low),
        "中分区(70-80)": compute_metrics(mid),
        "高分区(≥80)": compute_metrics(high),
    }


# ── 核心进化建议生成 ─────────────────────────────────────
def generate_evolution_advice(rows: list[dict]) -> list[dict]:
    """
    分析 feedback_tracker 历史，生成参数调整建议。
    返回 [{param, old, new, confidence, reason}, ...]
    """
    advice = []
    if len(rows) < MIN_SAMPLES:
        return [{"type": "insufficient_data", "n": len(rows), "min": MIN_SAMPLES}]

    # 1. 止盈档位分析：+5% vs +7% 触发率 vs 实际收益
    hit5_rets = [_safe_float(r["ret_actual"]) for r in rows if _safe_bool(r["hit_5pct"])]
    hit7_rets = [_safe_float(r["ret_actual"]) for r in rows if _safe_bool(r["hit_7pct"])]
    all_rets = [_safe_float(r["ret_actual"]) for r in rows if r.get("ret_actual")]

    avg_all = np.mean(all_rets) if all_rets else 0

    # 2. 止损触发率分析
    stop_count = sum(1 for r in rows if _safe_bool(r["stop_loss"]))
    stop_rate = stop_count / len(rows) * 100 if rows else 0

    # 3. score_threshold 敏感性
    low_thresh = [r for r in rows if 55 <= _safe_float(r.get("quality_score", 0)) < 65]
    high_thresh = [r for r in rows if _safe_float(r.get("quality_score", 0)) >= 70]
    m_low = compute_metrics(low_thresh)
    m_high = compute_metrics(high_thresh)

    # 生成建议
    # 建议1: 提高 score_threshold
    if m_high and m_low:
        if m_high.get("win_rate", 0) - m_low.get("win_rate", 0) > 10:
            advice.append({
                "type": "param_adjust",
                "param": "score_threshold",
                "old_value": 60.0,
                "new_value": 65.0,
                "confidence": min(90, (m_high["win_rate"] - m_low["win_rate"]) * 3),
                "reason": (
                    f"高评分(≥70)胜率{m_high['win_rate']:.1f}% vs 低评分(55-65)胜率{m_low['win_rate']:.1f}%，"
                    f"差距{m_high['win_rate']-m_low['win_rate']:.1f}%，建议提高门槛"
                ),
            })

    # 建议2: 如果止损率过高，考虑放宽 max_extension
    if stop_rate > 15:
        advice.append({
            "type": "param_adjust",
            "param": "max_extension",
            "old_value": 16.0,
            "new_value": 18.0,
            "confidence": min(85, stop_rate * 2),
            "reason": f"止损触发率{stop_rate:.1f}%（>15%），适当放宽偏离MA20容忍度可能减少假信号",
        })
    elif stop_rate < 5:
        advice.append({
            "type": "param_adjust",
            "param": "max_extension",
            "old_value": 16.0,
            "new_value": 14.0,
            "confidence": 70,
            "reason": f"止损率仅{stop_rate:.1f}%（<5%），可收紧偏离MA20约束以提高信号质量",
        })

    # 建议3: 分析质量窗口长度
    short_q = [r for r in rows if _safe_float(r.get("hold_days", 0)) <= 3]
    long_q = [r for r in rows if _safe_float(r.get("hold_days", 0)) > 3]
    m_short = compute_metrics(short_q)
    m_long = compute_metrics(long_q)
    if m_short and m_long:
        if m_short["avg_ret"] > m_long["avg_ret"] + 0.5:
            advice.append({
                "type": "param_adjust",
                "param": "hold_days",
                "old_value": 3,
                "new_value": 2,
                "confidence": min(80, (m_short["avg_ret"] - m_long["avg_ret"]) * 20),
                "reason": (
                    f"持有≤3天平均收益{m_short['avg_ret']:+.2f}% > 持有>3天{m_long['avg_ret']:+.2f}%，"
                    "建议缩短持有期或 quality_days"
                ),
            })

    # 建议4: 如果 +7% 触发率高但实际收益不高，说明"虚假繁荣"
    if hit7_rets and hit5_rets:
        avg_h7 = np.mean(hit7_rets)
        avg_h5 = np.mean(hit5_rets)
        if avg_h7 < avg_h5 + 0.3:
            advice.append({
                "type": "param_adjust",
                "param": "max_gain",
                "old_value": 7.0,
                "new_value": 5.0,
                "confidence": 65,
                "reason": f"+7%触发后平均收益{avg_h7:.2f}% 与 +5%触发{avg_h5:.2f}% 差异小，说明高期望难实现，建议降低上限",
            })

    # 建议5: 汇总统计
    m_all = compute_metrics(rows)
    advice.append({
        "type": "summary",
        "total_signals": len(rows),
        "win_rate": m_all.get("win_rate", 0),
        "avg_ret": m_all.get("avg_ret", 0),
        "sharpe": m_all.get("sharpe", 0),
        "max_loss": m_all.get("max_loss", 0),
        "stop_rate": stop_rate,
    })

    return advice


# ── 格式化输出 ────────────────────────────────────────────
def format_evolution_report(
    advice: list[dict],
    rows: list[dict],
    date_str: str,
) -> str:
    lines = []
    lines.append("=" * 80)
    lines.append(f"🧬 评分进化报告 | {date_str} | 数据量: {len(rows)} 条信号")
    lines.append("=" * 80)

    summary = next((a for a in advice if a["type"] == "summary"), None)
    if summary:
        lines.append(f"""
📊 历史信号统计（{summary['total_signals']} 条）
   胜率: {summary['win_rate']:.1f}%   平均收益: {summary['avg_ret']:+.3f}%/笔
   夏普: {summary['sharpe']:.2f}    最大亏损: {summary['max_loss']:+.2f}%
   止损率: {summary['stop_rate']:.1f}%""")

    param_advices = [a for a in advice if a["type"] == "param_adjust"]
    if param_advices:
        lines.append("\n🛠️  参数调整建议（按置信度排序）：")
        lines.append(f"{'参数':<20} {'当前值':>8} {'建议值':>8} {'置信度':>8}  依据")
        lines.append("-" * 80)
        for a in sorted(param_advices, key=lambda x: -x["confidence"]):
            conf = f"{a['confidence']:.0f}%"
            lines.append(
                f"{a['param']:<20} {a['old_value']:>8} {a['new_value']:>8} {conf:>8}  {a['reason']}"
            )
        lines.append("")
        lines.append("⚙️  手动生效方法：")
        for a in sorted(param_advices, key=lambda x: -x["confidence"]):
            if a["confidence"] >= 70:
                lines.append(
                    f"  sed -i 's/{a['old_value']}.*# {a['param']}/{a['new_value']}  # {a['param']}/' gain_turnover.py"
                )
    else:
        lines.append("\n✅ 当前参数表现良好，无需调整")

    if summary:
        m = compute_metrics(rows)
        if m:
            lines.append(f"\n📈 盈亏分布：")
            rets = sorted([_safe_float(r["ret_actual"]) for r in rows if r.get("ret_actual")])
            if rets:
                p25 = rets[int(len(rets)*0.25)]
                p50 = rets[int(len(rets)*0.50)]
                p75 = rets[int(len(rets)*0.75)]
                lines.append(f"   25%分位: {p25:+.2f}%   中位数: {p50:+.2f}%   75%分位: {p75:+.2f}%")
                loss_count = sum(1 for r in rets if r < -2)
                big_win = sum(1 for r in rets if r > 5)
                lines.append(f"   大赢(>+5%): {big_win}笔   大亏(<-2%): {loss_count}笔")

    return "\n".join(lines)


# ── 写入进化历史 ──────────────────────────────────────────
def record_evolution(advice: list[dict], date_str: str):
    """将本次进化建议记录到历史文件。"""
    p = Path(EVOLUTION_CSV)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=EVOLUTION_FIELDS)
            writer.writeheader()

    param_advices = [a for a in advice if a["type"] == "param_adjust"]
    if not param_advices:
        return

    with open(p, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EVOLUTION_FIELDS)
        for a in param_advices:
            writer.writerow({
                "date": date_str,
                "param_changed": a["param"],
                "old_value": a["old_value"],
                "new_value": a["new_value"],
                "reason": a["reason"][:100],
                "expected_winrate_change": f"{a['confidence']:.0f}%",
                "confirmed": "pending",
            })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评分进化器 — 分析反馈数据，生成参数调整建议")
    parser.add_argument("--min-samples", type=int, default=MIN_SAMPLES, help=f"最小样本量（默认{MIN_SAMPLES}）")
    parser.add_argument("--output", "-o", type=str, default=None, help="报告输出路径")
    args = parser.parse_args()

    rows = load_tracker()
    date_str = datetime.now().strftime("%Y-%m-%d")

    if len(rows) < args.min_samples:
        print(f"⚠️  数据量不足: {len(rows)} 条 < {args.min_samples} 条（最小样本量）")
        print(f"   继续分析（数据仅供参考）...")

    advice = generate_evolution_advice(rows)
    report_text = format_evolution_report(advice, rows, date_str)
    print("\n" + report_text)

    # 记录进化历史
    record_evolution(advice, date_str)

    # 写入报告
    if args.output:
        out = Path(args.output)
    else:
        out = WORKSPACE / "stock_trend" / f"evolution_report_{date_str}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report_text, encoding="utf-8")
    print(f"\n💾 报告已写入: {out.resolve()}")
    evolution_hist = Path(EVOLUTION_CSV)
    if evolution_hist.exists():
        print(f"📝 进化历史已记录: {evolution_hist.resolve()}")
