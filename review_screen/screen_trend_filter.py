#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
复盘选股 · 实战强化版（v3）
==========================

在 screen.py 基础筛选之上，整合了全部P0/P1/P2改进：

  第一步：基础筛选（screen.py 原版 — 均线/量能/波段结构）
  第二步：趋势强化（三条件，过滤趋势不够强的）
  第三步：量能精选（宽松化，主要用于排序）
  第四步：买入准备度（新增P0 — 过滤不能买/不能追的）
  第五步：大盘环境 + 分层排序（新增P1）

P0 改进：
  ✅ 去除量比/涨跌量比硬过滤（改为宽松软条件）
  ✅ 买入准备度过滤（涨停/过热/偏离/弱势排除）
  ✅ 止损风险评级（stop_loss_ref → 实际风控）
  ✅ RSI/MA5偏离/涨幅范围过滤

P1 改进：
  ✅ 大盘HS300环境检测（牛市/震荡/熊市）
  ✅ 熊市时自动提高买入门槛
  ✅ 风险收益评分（用于优先级排序）

P2 改进：
  ✅ 模拟验证日志（记录信号→结果）
  ✅ 分层排名（强信号/中信号/弱信号）
  ✅ 每次买入自动记录，用于事后验证

用法：
    python screen_trend_filter.py --date 2026-04-23
    python screen_trend_filter.py --date 2026-04-23 --show-rejected
    python screen_trend_filter.py --date 2026-04-23 --top-n 20
    python screen_trend_filter.py --date 2026-04-23 --simulate
    python screen_trend_filter.py --date 2026-04-23 --skip-step 3  # 跳过第三步
"""

import sys
import json
import time
import argparse
from datetime import datetime
from pathlib import Path

WORKSPACE = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(WORKSPACE))

from stock_trend.review_screen.screen import (
    evaluate_stock, normalize_code, load_stock_names, get_stock_name,
    scan_market, _make_row, _header_row, _COLS,
)
from stock_trend.review_screen.data_cache import preload_all_codes
from stock_trend.review_screen.filter_rules import FilterConfig
from stock_trend.review_screen.buy_ready import (
    BuyReadyConfig, apply_buy_ready_filter, compute_risk_reward,
    get_market_mode, get_stop_loss_alert, log_signal, get_simulation_stats,
)

DEFAULT_WORKERS = 8


# ── 趋势强化过滤条件（第二步）────────────────────────────────
TREND_FILTER = {
    "gain20_min": 13.8,            # 20日涨幅 ≥ 13.8%（浮点余量）
    "wave_quality_min": 4,         # 波段质量 ≥ 4
    "ma20_60_separation_min": 3.5, # MA20 vs MA60 ≥ 3.5%
}

# ── 量能宽松过滤条件（第三步 — 已放宽为软条件）───────────────
VOLUME_FILTER = {
    "vol_ratio_min": 0.70,         # 量比下限（宽松化，P0核心改进）
    "vol_up_down_min": 0.90,       # 涨跌量比下限（宽松化）
    "soft_penalty_only": True,      # True=只扣分不拒绝，False=硬过滤
}


def apply_trend_filter(ind: dict) -> bool:
    """第二步：趋势强化过滤"""
    gain20 = ind.get("gain20", 0)
    wave_quality = ind.get("wave_quality_score", 0)
    ma20 = ind.get("ma20", 0)
    ma60 = ind.get("ma60", 0)
    ma_sep = (ma20 / ma60 - 1) * 100 if ma60 > 0 else 0
    return (
        gain20 >= TREND_FILTER["gain20_min"]
        and wave_quality >= TREND_FILTER["wave_quality_min"]
        and ma_sep >= TREND_FILTER["ma20_60_separation_min"]
    )


def apply_volume_soft(ind: dict) -> tuple[bool, float]:
    """
    第三步：量能宽松过滤（软条件）
    Returns: (是否通过, 扣分)
    """
    vol_ratio = ind.get("vol_ratio", 0)
    vol_up_down = ind.get("vol_up_vs_down", 0)
    penalty = 0.0
    passed = True

    if vol_ratio < VOLUME_FILTER["vol_ratio_min"]:
        penalty += 5.0 * (VOLUME_FILTER["vol_ratio_min"] - vol_ratio)
        if not VOLUME_FILTER["soft_penalty_only"]:
            passed = False

    if vol_up_down < VOLUME_FILTER["vol_up_down_min"]:
        penalty += 4.0 * (VOLUME_FILTER["vol_up_down_min"] - vol_up_down)
        if not VOLUME_FILTER["soft_penalty_only"]:
            passed = False

    return passed, penalty


def get_trend_reject_reason(ind: dict) -> str:
    gain20 = ind.get("gain20", 0)
    wave_quality = ind.get("wave_quality_score", 0)
    ma20 = ind.get("ma20", 0)
    ma60 = ind.get("ma60", 0)
    ma_sep = (ma20 / ma60 - 1) * 100 if ma60 > 0 else 0
    reasons = []
    if gain20 < TREND_FILTER["gain20_min"]:
        reasons.append(f"20日涨幅{gain20:.1f}%<{TREND_FILTER['gain20_min']}%")
    if wave_quality < TREND_FILTER["wave_quality_min"]:
        reasons.append(f"波段质量{wave_quality:.1f}<{TREND_FILTER['wave_quality_min']}")
    if ma_sep < TREND_FILTER["ma20_60_separation_min"]:
        reasons.append(f"MA20/60分离{ma_sep:.1f}%<{TREND_FILTER['ma20_60_separation_min']}%")
    return "趋势过滤: " + ", ".join(reasons)


def get_reject_reason_summary(ind: dict) -> str:
    """简洁拒绝原因（用于表格显示）"""
    reasons = []
    gain1 = ind.get("gain1", 0)
    rsi = ind.get("rsi", 50)
    ma5_dist = ind.get("ma5_distance_pct", 0)

    if gain1 < 0.3: reasons.append(f"涨幅{gain1:.1f}%")
    if rsi >= 80: reasons.append(f"RSI{rsi:.0f}过热")
    if abs(ma5_dist) > 8: reasons.append(f"距MA5±{ma5_dist:.0f}%")
    if gain1 >= 9.5: reasons.append("涨停")
    return "; ".join(reasons) if reasons else ""


def rank_signal(rr_score: float, trend_score: float, vol_penalty: float,
                  stop_loss_pct: float = 999.0) -> tuple[str, int]:
    """
    分层排名（强信号/中信号/弱信号）
    结合风险收益评分 + 趋势评分 + 量能扣分 + 止损距离

    止损距离 > 15% → 直接降级（风险收益比太差）
    """
    # 止损过宽直接惩罚，严重者降级
    sl_penalty = max(0, (stop_loss_pct - 15.0) * 1.5) if stop_loss_pct > 15 else 0
    effective = rr_score * 0.55 + trend_score * 0.45 - vol_penalty - sl_penalty

    if effective >= 75 and stop_loss_pct <= 12:
        return "🟢强信号", 1
    elif effective >= 60 and stop_loss_pct <= 15:
        return "🔵中信号", 2
    elif effective >= 45 and stop_loss_pct <= 20:
        return "🟡弱信号", 3
    else:
        return "⚪观察", 4


def main():
    parser = argparse.ArgumentParser(description="复盘选股 · 实战强化版（v3）")
    parser.add_argument("--date", type=str, required=True, help="信号日期 YYYY-MM-DD")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help=f"并行线程数")
    parser.add_argument("--codes", nargs="+", default=None, help="指定股票代码列表")
    parser.add_argument("--output", type=str, default=None, help="输出文件路径")
    parser.add_argument("--waves", action="store_true", help="显示Top10波段详情")
    parser.add_argument("--show-rejected", action="store_true", help="显示被过滤拒绝的股票及原因")
    parser.add_argument("--skip-step", type=int, default=0, choices=[0, 3],
                        help="跳过某一步（3=跳过第三步量能过滤）")
    parser.add_argument("--top-n", type=int, default=0, help="只输出TopN（0=全部）")
    parser.add_argument("--gain20", type=float, default=TREND_FILTER["gain20_min"],
                        help=f"20日涨幅下限（默认{TREND_FILTER['gain20_min']}）")
    parser.add_argument("--wave-quality", type=float, default=TREND_FILTER["wave_quality_min"],
                        help=f"波段质量下限（默认{TREND_FILTER['wave_quality_min']}）")
    parser.add_argument("--ma-sep", type=float, default=TREND_FILTER["ma20_60_separation_min"],
                        help=f"MA20/60分离下限（默认{TREND_FILTER['ma20_60_separation_min']}）")
    parser.add_argument("--simulate", action="store_true", help="显示模拟验证统计")
    parser.add_argument("--strict", action="store_true", help="严格模式（硬过滤量能）")
    parser.add_argument("--market-adjust", action="store_true", default=True,
                        help="根据大盘环境调整阈值（默认开启）")
    args = parser.parse_args()

    # 更新阈值
    TREND_FILTER["gain20_min"] = args.gain20
    TREND_FILTER["wave_quality_min"] = args.wave_quality
    TREND_FILTER["ma20_60_separation_min"] = args.ma_sep
    if args.strict:
        VOLUME_FILTER["soft_penalty_only"] = False
    if args.skip_step == 3:
        VOLUME_FILTER["vol_ratio_min"] = 0.0
        VOLUME_FILTER["vol_up_down_min"] = 0.0
        VOLUME_FILTER["soft_penalty_only"] = True

    try:
        target_date = datetime.strptime(args.date, "%Y-%m-%d")
    except ValueError:
        print(f"❌ 日期格式错误: {args.date}，应为 YYYY-MM-DD")
        sys.exit(1)

    date_str = args.date

    # ── P1: 大盘环境检测（第一步）────────────────────────────
    print()
    print("━" * 60)
    print("📊 第一步：大盘环境检测（HS300 vs MA20）")
    print("━" * 60)
    market = get_market_mode(target_date)
    cfg_market = "牛市" if market["mode"] == "牛市" else ("熊市" if market["mode"] == "熊市" else "震荡")
    print(f"   {market['signal']}  HS300={market['hs300']} MA20={market['ma20']}  偏离{market['ratio']:+.2f}%")
    print(f"   阈值调整: {'开启' if args.market_adjust else '关闭'}")
    print()

    # ── 加载股票 ─────────────────────────────────────────
    if args.codes:
        codes = args.codes
        print(f"📊 指定 {len(codes)} 只")
    else:
        codes = preload_all_codes()
        print(f"📊 全市场 {len(codes)} 只")

    # ── 第一步：基础筛选 ──────────────────────────────────
    cfg = FilterConfig()

    print()
    print("━" * 60)
    print("📋 第二步：基础筛选（均线/量能/波段结构）")
    print("━" * 60)

    base_results = scan_market(
        codes=codes,
        target_date=target_date,
        max_workers=args.workers,
        cfg=cfg,
        return_failed=False,
    )
    print(f"   基础通过: {len(base_results)} 只")

    # ── 第二步：趋势强化 ──────────────────────────────────
    print()
    print("━" * 60)
    print("🔍 第三步：趋势强化过滤")
    print(f"   gain20 ≥ {TREND_FILTER['gain20_min']}%")
    print(f"   波段质量 ≥ {TREND_FILTER['wave_quality_min']}")
    print(f"   MA20/60分离 ≥ {TREND_FILTER['ma20_60_separation_min']}%")
    print("━" * 60)

    trend_passed = []
    trend_rejected = []
    for r in base_results:
        if apply_trend_filter(r):
            trend_passed.append(r)
        else:
            trend_rejected.append(r)

    print(f"   趋势通过: {len(trend_passed)} 只")

    if not trend_passed:
        print("⚠️  无股票通过趋势过滤")
        sys.exit(0)

    # ── 第三步：量能软过滤（宽松化）───────────────────────
    print()
    print("━" * 60)
    print(f"🔥 第四步：量能精选（宽松化）")
    print(f"   量比 ≥ {VOLUME_FILTER['vol_ratio_min']}（软扣分）")
    print(f"   涨跌量比 ≥ {VOLUME_FILTER['vol_up_down_min']}（软扣分）")
    print("━" * 60)

    # 对所有趋势通过的股票计算量能扣分
    for r in trend_passed:
        ok, penalty = apply_volume_soft(r)
        r["_vol_penalty"] = penalty

    trend_passed.sort(key=lambda x: x["score"] - x.get("_vol_penalty", 0), reverse=True)

    # ── 第四步：买入准备度过滤 ───────────────────────────
    print()
    print("━" * 60)
    print("✅ 第五步：买入准备度 + 风险收益评分")
    print(f"   过滤: 涨停/涨幅<0.3%/{'>'}8%/RSI≥80/距MA5>8%/连续下跌")
    print(f"   评分: 止损空间 + RSI健康 + 量能配合 + 位置质量")
    print("━" * 60)

    cfg_buy = BuyReadyConfig(
        enable_market_adjust=args.market_adjust,
        stop_loss_warning=5.0,
        stop_loss_forced=8.0,
    )
    market_mode = cfg_market

    buy_ready_passed = []
    buy_ready_failed = []
    buy_reasons = {}

    for r in trend_passed:
        ready, reason, rr = apply_buy_ready_filter(r, cfg_buy, market_mode)
        r["_buy_ready"] = ready
        r["_buy_reason"] = reason
        r["_rr"] = rr

        if ready:
            # 计算综合排名分
            rr_score = rr.get("rr_score", 50)
            trend_score = r.get("score", 50)
            vol_penalty = r.get("_vol_penalty", 0)
            stop_loss_pct = rr.get("stop_loss_pct", 999.0)
            signal_type, _ = rank_signal(rr_score, trend_score, vol_penalty, stop_loss_pct)
            r["_signal_type"] = signal_type
            r["_combined_score"] = round(
                rr_score * 0.55 + trend_score * 0.45 - vol_penalty
                - max(0, (stop_loss_pct - 15.0) * 1.5 if stop_loss_pct > 15 else 0), 1
            )
            r["_stop_loss_pct"] = stop_loss_pct
            buy_ready_passed.append(r)
        else:
            buy_reasons[r["code"]] = reason
            buy_ready_failed.append(r)

    print(f"   买入就绪: {len(buy_ready_passed)} 只")
    print(f"   暂不推荐: {len(buy_ready_failed)} 只")

    if not buy_ready_passed:
        print("⚠️  无股票满足买入准备条件")
        sys.exit(0)

    # ── 最终排序（综合风险收益 + 趋势）───────────────────
    buy_ready_passed.sort(key=lambda x: x["_combined_score"], reverse=True)

    # ── 输出 ──────────────────────────────────────────────
    output_path = Path(args.output) if args.output else (
        Path.home() / "stock_reports" / f"review_screen_trend_{date_str}.txt"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    display = buy_ready_passed[:args.top_n] if args.top_n > 0 else buy_ready_passed
    total = len(buy_ready_passed)

    print(f"\n{'='*130}")
    print(f"📊 复盘选股 · 实战强化版 v3 {date_str}（{market['signal']} | 买入就绪 {len(display)} 只）")
    print(f"{'='*130}")

    # 风险收益表头
    print(f"\n🏆 优先买入名单（按风险收益+趋势综合评分）")
    print(f"{'代码':<12} {'名称':<8} {'信号':<8} {'综合分':>7} {'RSI':>5} {'量比':>6} {'20日%':>7} {'止损%':>6} {'涨停':<5} {'MA5距':>6} {'止损价':>7} {'风险':<12} {'建议'}")
    print("-" * 130)

    lines = [
        f"📊 复盘选股 · 实战强化版 v3 {date_str}（{market['signal']} | 买入就绪 {total} 只）",
        f"过滤链: 全市场{len(codes)} → 基础{len(base_results)} → 趋势{len(trend_passed)} → 量能软过滤 → 买入就绪{total}",
        f"大盘: {market['signal']} HS300={market['hs300']} MA20={market['ma20']} ({market['ratio']:+.2f}%)",
        "━" * 130,
        "🏆 优先买入名单（按风险收益+趋势综合评分）",
    ]

    # 统计
    strong = sum(1 for r in buy_ready_passed if r["_signal_type"] == "🟢强信号")
    medium = sum(1 for r in buy_ready_passed if r["_signal_type"] == "🔵中信号")
    weak = sum(1 for r in buy_ready_passed if r["_signal_type"] == "🟡弱信号")
    print(f"\n📈 信号分布: 🟢强{strong}  🔵中{medium}  🟡弱{weak}  →  建议重点关注前{strong + medium}只")

    for r in display:
        rr = r["_rr"]
        gain1 = r.get("gain1", 0)
        rsi = r.get("rsi", 50)
        vol_ratio = r.get("vol_ratio", 0)
        ma5_dist = r.get("ma5_distance_pct", 0)
        stop_loss_pct = rr.get("stop_loss_pct", 0)
        sl_price = r.get("stop_loss_ref", 0)
        is_limit = "⚠️涨停" if gain1 >= 9.5 else ("⚠️跌停" if gain1 <= -9.5 else "")
        risk = f"止损{stop_loss_pct:.1f}%" if stop_loss_pct <= 5 else f"⚠️宽止损{stop_loss_pct:.1f}%"
        action = "买入" if r["_signal_type"] == "🟢强信号" else ("观察" if r["_signal_type"] == "🔵中信号" else "谨慎")

        print(
            f"{r['code']:<12} {r['name']:<8} {r['_signal_type']:<8} {r['_combined_score']:>7.1f} "
            f"{rsi:>5.1f} {vol_ratio:>6.2f} {r.get('gain20', 0):>+7.1f}% {stop_loss_pct:>6.1f}% "
            f"{is_limit:<5} {ma5_dist:>+6.1f}% {sl_price:>7.2f}  {risk:<12} {action}"
        )

    print("━" * 130)

    # ── 止损风险表 ───────────────────────────────────────
    print(f"\n🛡️  止损风险评级（Top{len(display)}）")
    print(f"{'代码':<12} {'名称':<8} {'信号':<8} {'综合':>6} {'止损%':>7} {'止损价':>8} {'RSI':>6} {'上涨%':>6} {'位置':<8}")
    print("-" * 80)
    for r in display[:20]:
        rr = r["_rr"]
        sl_price = r.get("stop_loss_ref", 0) or 0
        sl_pct = rr.get("stop_loss_pct", 0)
        phase = r.get("phase", "不明")

        if sl_pct <= 3:
            risk_icon = "🟢理想"
        elif sl_pct <= 5:
            risk_icon = "🔵正常"
        elif sl_pct <= 8:
            risk_icon = "🟡偏大"
        else:
            risk_icon = "🔴过大"

        print(
            f"{r['code']:<12} {r['name']:<8} {r['_signal_type']:<8} {r['_combined_score']:>6.1f} "
            f"{sl_pct:>6.1f}% {sl_price:>8.2f} {r.get('rsi', 0):>6.1f} "
            f"{r.get('gain20', 0):>+6.1f}% {phase:<8}"
        )

    # ── 不推荐名单 ───────────────────────────────────────
    if args.show_rejected and buy_ready_failed:
        print(f"\n❌ 暂不推荐（{len(buy_ready_failed)} 只）")
        print(f"{'代码':<12} {'名称':<8} {'趋势分':>7} {'拒绝原因'}")
        print("-" * 60)
        buy_ready_failed.sort(key=lambda x: x["score"], reverse=True)
        for r in buy_ready_failed[:20]:
            print(f"{r['code']:<12} {r['name']:<8} {r['score']:>7.1f}  {buy_reasons.get(r['code'], '')}")

    # ── P2: 模拟验证 ─────────────────────────────────────
    if args.simulate:
        print(f"\n📊 模拟验证统计")
        print("━" * 40)
        stats = get_simulation_stats(BuyReadyConfig())
        if stats.get("total", 0) > 0:
            print(f"   总信号: {stats['total_signals']}")
            print(f"   已完成: {stats['completed']}")
            print(f"   胜率: {stats['win_rate']}%")
            print(f"   平均收益: {stats['avg_pnl']:+.2f}%")
            print(f"   最大盈利: {stats['max_pnl']:+.2f}%")
            print(f"   最大亏损: {stats['min_pnl']:+.2f}%")
            print(f"   平均持有: {stats['avg_holding_days']} 天")
        else:
            print(f"   {stats['message']}（尚无模拟记录）")

    # ── 写入文件 ─────────────────────────────────────────
    file_lines = [
        f"📊 复盘选股 · 实战强化版 v3 {date_str}",
        f"大盘环境: {market['signal']} HS300={market['hs300']} vs MA20={market['ma20']} ({market['ratio']:+.2f}%)",
        f"过滤链: 全市场{len(codes)} → 基础{len(base_results)} → 趋势{len(trend_passed)} → 买入就绪{total}",
        f"信号分布: 🟢强{strong}  🔵中{medium}  🟡弱{weak}",
        "",
        f"🏆 优先买入名单（共 {total} 只）",
    ]

    for r in display:
        rr = r["_rr"]
        file_lines.append(
            f"{r['code']} {r['name']} | {r['_signal_type']} 综合{r['_combined_score']:.1f} | "
            f"止损{rr.get('stop_loss_pct',0):.1f}%({r.get('stop_loss_ref',0):.2f}) | "
            f"RSI{r.get('rsi',0):.1f} | 20日{r.get('gain20',0):+.1f}% | "
            f"量比{r.get('vol_ratio',0):.2f} | {r['_buy_reason']}"
        )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(file_lines) + "\n")

    print(f"\n💾 结果已写入: {output_path}")

    # ── 汇总 ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"📋 实战强化版 v3 汇总")
    print(f"{'='*60}")
    print(f"  大盘: {market['signal']}")
    print(f"  全市场 {len(codes)} → 基础 {len(base_results)} → 趋势 {len(trend_passed)}")
    print(f"  → 买入就绪 {len(buy_ready_passed)} 只")
    print(f"  信号分布: 🟢强{strong}  🔵中{medium}  🟡弱{weak}")
    print(f"  建议: 重点关注前 {strong + medium} 只（🟢+🔵）")
    print()
    print(f"  💡 使用说明:")
    print(f"  - 🟢强信号: 可直接买入，风险收益比优秀")
    print(f"  - 🔵中信号: 观察，等待更好买点或轻仓试探")
    print(f"  - 🟡弱信号: 不建议新买入，持有观察")
    print(f"  - 止损: 亏损≥5%报警，≥8%强制建议卖出")
    print(f"  - 持仓: 同时最多持有 5-8 只，不要分散过度")


if __name__ == "__main__":
    main()
