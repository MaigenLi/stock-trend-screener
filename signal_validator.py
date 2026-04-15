#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
信号验证器 — 自我进化策略 第一层
================================
读取昨日选股结果，用今日数据验证信号质量。
真实回报率定义：T+1 开盘价买入 → T+1 收盘价卖出（持有1天）。

用法：
  python signal_validator.py                                          # 自动找昨日输出文件
  python signal_validator.py --input daily_screen_2026-04-11.txt
  python signal_validator.py --codes sz002990 sz000001             # 指定个股
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

WORKSPACE = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(WORKSPACE))

from stock_trend.gain_turnover import (
    get_stock_name,
    load_qfq_history,
    normalize_prefixed,
)


# ── 数据类 ────────────────────────────────────────────────
@dataclass
class SignalValidation:
    code: str
    name: str
    signal_date: str        # 信号日（昨日筛选日期）
    signal_close: float     # 信号日收盘价（参考基准）

    # T+1 日行情
    open_today: float       # T+1 开盘价（真实买入价）
    close_today: float      # T+1 收盘价（真实卖出价）
    high_today: float       # T+1 最高价
    low_today: float        # T+1 最低价

    # 真实收益（持有1天，开盘买→收盘卖）
    ret_actual: float       # (收盘-开盘)/开盘 × 100%

    # 参考收益（信号日收盘→T+1收盘，作为代理指标）
    ret_signal: float       # (T+1收盘-信号收盘)/信号收盘 × 100%

    # 当日最高收益（信号收盘→日内高点）
    ret_high: float         # (日内高点-信号收盘)/信号收盘 × 100%

    # 止盈止损（均以 signal_close 为基准）
    hit_3pct: bool
    hit_5pct: bool
    hit_7pct: bool
    hit_10pct: bool
    stop_loss: bool         # T+1 最低价 ≤ signal_close × 0.98

    # 评分
    quality_score: float
    evolution_tag: str


# ── 解析昨日选股输出文件 ──────────────────────────────────
def parse_screen_output(path: Path) -> list[tuple[str, str, str, float]]:
    """
    解析选股结果文件，返回 (code, name, signal_date, signal_close) 列表。
    文件列顺序（固定）：
      0=代码 1=名称 2=日期 3=总分 4=窗口涨幅 5=20日额 6=5日换手 7=RSI 8=偏离MA20 9=收盘 10+=EPS/ROE...
    示例行：
      sz002866 传艺科技 2026-04-13 92.2 +4.92% 4.97 16.94% 62.5 +7.75% 22.75 - - ✗ -
    """
    results = []
    date_pat = re.compile(r"^(\d{4}-\d{2}-\d{2})$")
    code_pat = re.compile(r"^(sh|sz|bj)(\d{6})$")

    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 10:
            continue
        m_code = code_pat.match(parts[0].lower())
        if not m_code:
            continue
        code = f"{m_code.group(1)}{m_code.group(2)}"
        name = parts[1]
        # 信号日：parts[2] 必为 YYYY-MM-DD
        if not date_pat.match(parts[2]):
            continue
        sig_date = parts[2]
        # 收盘价固定在 parts[9]（列位置固定）
        try:
            sig_close = float(parts[9])
            if sig_close <= 0:
                continue
        except ValueError:
            continue
        results.append((code, name, sig_date, sig_close))
    return results


# ── 核心验证 ──────────────────────────────────────────────
def validate_signal(
    code: str,
    name: str,
    signal_date: str,
    signal_close: float,
) -> Optional[SignalValidation]:
    """
    用 T+1 日数据验证昨日信号。
    真实收益 = (T+1收盘 - T+1开盘) / T+1开盘 × 100%
    """
    df = load_qfq_history(code, max_age_hours=0)
    if df is None or df.empty:
        return None

    sig_ts = pd.Timestamp(signal_date)
    post = df[df["date"] > sig_ts]       # T+1 及之后
    if post.empty:
        # T+1 数据尚未就绪（常见于收盘后 1-2 小时内）
        latest_date = df["date"].max().strftime("%Y-%m-%d")
        if latest_date <= signal_date:
            # 最新数据不晚于信号日，说明今天数据确实未更新
            print(f"  ⚠️ {code} {name}: T+1 数据未就绪（最新 {latest_date} ≤ 信号 {signal_date}）")
        return None
    today_row = post.iloc[0]             # T+1 当天

    open_today = float(today_row["open"])
    close_today = float(today_row["close"])
    high_today = float(today_row["high"])
    low_today = float(today_row["low"])
    today_date = str(today_row["date"])[:10]

    if open_today <= 0 or signal_close <= 0:
        return None

    # 真实收益：T+1开盘买入 → T+1收盘卖出
    ret_actual = (close_today - open_today) / open_today * 100.0
    # 参考收益：信号收盘 → T+1收盘
    ret_signal = (close_today - signal_close) / signal_close * 100.0
    # 日内高点收益（信号收盘为基准）
    ret_high = (high_today - signal_close) / signal_close * 100.0
    # 跳空幅度
    open_gap = (open_today - signal_close) / signal_close * 100.0

    # 止盈止损（均以 signal_close 为基准）
    # 止盈：日内高点突破即算触发
    hit_3pct = high_today >= signal_close * 1.03
    hit_5pct = high_today >= signal_close * 1.05
    hit_7pct = high_today >= signal_close * 1.07
    hit_10pct = high_today >= signal_close * 1.10
    # 止损：需收盘跌破止损线才算（盘中刺破但收回复活不算）
    stop_loss = close_today <= signal_close * 0.98

    # 质量评分（基于 ret_actual）
    score = 50.0
    if ret_actual >= 6.0:
        score += 35.0
    elif ret_actual >= 5.0:
        score += 30.0
    elif ret_actual >= 4.0:
        score += 25.0
    elif ret_actual >= 3.0:
        score += 20.0
    elif ret_actual >= 2.0:
        score += 15.0
    elif ret_actual >= 0:
        score += 8.0
    else:
        score -= 20.0
    if hit_5pct:
        score += 10.0
    if stop_loss:
        score -= 15.0
    if open_gap >= 9.0:
        score -= 10.0
    elif open_gap >= 5.0:
        score -= 5.0

    score = max(0.0, min(100.0, score))

    if score >= 85:
        tag = "🟢优秀"
    elif score >= 70:
        tag = "🔵良好"
    elif score >= 55:
        tag = "🟡及格"
    else:
        tag = "🔴失效"

    return SignalValidation(
        code=normalize_prefixed(code),
        name=name,
        signal_date=today_date,
        signal_close=round(signal_close, 2),
        open_today=round(open_today, 2),
        close_today=round(close_today, 2),
        high_today=round(high_today, 2),
        low_today=round(low_today, 2),
        ret_actual=round(ret_actual, 2),
        ret_signal=round(ret_signal, 2),
        ret_high=round(ret_high, 2),
        hit_3pct=hit_3pct,
        hit_5pct=hit_5pct,
        hit_7pct=hit_7pct,
        hit_10pct=hit_10pct,
        stop_loss=stop_loss,
        quality_score=round(score, 1),
        evolution_tag=tag,
    )


def find_latest_screen_output() -> Optional[Path]:
    """
    找昨日的选股结果文件（以文件内日期为准，而非文件修改时间）。
    16:40 执行时，今日 screening 大概率还在运行中，取前一天的输出最准确。
    支持格式：
      - daily_screen_YYYY-MM-DD.txt（gain_turnover_screen.py 默认输出）
      - daily_screen_YYYY-MM-DD.txt
    """
    reports_dir = Path.home() / "stock_reports"
    today = datetime.now()
    from datetime import timedelta
    yesterday = (today - timedelta(days=1)).strftime("%Y-%m-%d")

    # 支持两种文件名格式
    patterns = ["daily_screen_*.txt"]
    candidates: list[Path] = []
    for pat in patterns:
        candidates.extend(reports_dir.glob(pat))

    if not candidates:
        return None

    # 优先：找到昨天日期的文件
    for p in sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True):
        fname = p.name
        m = re.search(r"(\d{4}-\d{2}-\d{2})", fname)
        if m and m.group(1) == yesterday:
            return p

    # 兜底：没有昨天的文件，则取最新修改的文件
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def run_validation(
    codes: list[tuple[str, str, str, float]],
) -> list[SignalValidation]:
    """
    codes: (code, name, signal_date, signal_close)
    signal_date: 信号发出的日期（筛选数据截止日）
    signal_close: 信号日收盘价
    """
    t0 = time.time()
    validations = []
    skipped = 0

    for code, name, sig_date, sig_close in codes:
        c = normalize_prefixed(code)
        result = validate_signal(c, name, sig_date, sig_close)
        if result:
            validations.append(result)
        else:
            skipped += 1

    validations.sort(key=lambda x: x.quality_score, reverse=True)

    if validations:
        scores = [v.quality_score for v in validations]
        rets = [v.ret_actual for v in validations]
        print(
            f"\n📊 信号验证（{len(validations)} 只）| {time.time()-t0:.1f}s\n"
            f"   均分={np.mean(scores):.1f}  "
            f"上涨={sum(1 for r in rets if r>0)}/{len(rets)}  "
            f"均收益={np.mean(rets):+.2f}%  "
            f"+3%={sum(1 for v in validations if v.hit_3pct)}  "
            f"+5%={sum(1 for v in validations if v.hit_5pct)}  "
            f"止损={sum(1 for v in validations if v.stop_loss)}"
        )
    else:
        print(f"\n📊 信号验证（0 只）| {time.time()-t0:.1f}s")
    
    if skipped > 0:
        print(f"  ⚠️  跳过 {skipped} 只（T+1 数据未就绪）")
        print(f"  💡 建议：等待 cache_qfq_daily.py 更新今日数据（16:10 后）或稍后重试")
    
    return validations


def format_report(validations: list[SignalValidation]) -> str:
    lines = []
    today = datetime.now().strftime("%Y-%m-%d")
    lines.append("=" * 130)
    lines.append(f"📋 信号验证报告 | 验证日: {today} | {len(validations)} 只")
    lines.append("=" * 130)

    if validations:
        rets = [v.ret_actual for v in validations]
        scores = [v.quality_score for v in validations]
        lines.append(
            f"  均分={np.mean(scores):.1f}  上涨={sum(1 for r in rets if r>0)}/{len(rets)}  "
            f"均收益={np.mean(rets):+.2f}%  "
            f"+3%={sum(1 for v in validations if v.hit_3pct)}  "
            f"+5%={sum(1 for v in validations if v.hit_5pct)}  "
            f"止损={sum(1 for v in validations if v.stop_loss)}"
        )
    lines.append("-" * 130)

    # 列宽定义（用于填充宽度计算）
    # 标题行（webchat 会压缩连续空格，列间用 \t 保证结构可辨认）
    hdr = (
        f"{'代码':<12}"
        f"\t{'名称':<10}"
        f"\t{'信号日':<12}"
        f"\t{'信号价':>8}"
        f"\t{'今开':>8}"
        f"\t{'今收':>8}"
        f"\t{'真实收益':>8}"
        f"\t{'参收益':>8}"
        f"\t{'高涨':>8}"
        f"\t{'+3':>3}"
        f"\t{'+5':>3}"
        f"\t{'止损':>4}"
        f"\t{'评分':>6}"
        f"\t{'评价':>8}"
    )
    lines.append(hdr)
    lines.append("-" * 130)

    # 数据行
    for v in validations:
        row = (
            f"{v.code:<12}"
            f"\t{v.name:<10}"
            f"\t{v.signal_date:<12}"
            f"\t{v.signal_close:>8.2f}"
            f"\t{v.open_today:>8.2f}"
            f"\t{v.close_today:>8.2f}"
            f"\t{v.ret_actual:>+8.2f}"
            f"\t{v.ret_signal:>+8.2f}"
            f"\t{v.ret_high:>+8.2f}"
            f"\t{'✓' if v.hit_3pct else '-':>3}"
            f"\t{'✓' if v.hit_5pct else '-':>3}"
            f"\t{'✓' if v.stop_loss else '-':>4}"
            f"\t{v.quality_score:>6.1f}"
            f"\t{v.evolution_tag:>8}"
        )
        lines.append(row)
    lines.append("-" * 130)
    lines.append("注: 真实收益=T+1开盘买入→T+1收盘卖出(持有1天)  参收益=信号收盘→T+1收盘(参考)")
    lines.append("评价: 🟢优秀≥85  🔵良好≥70  🟡及格≥55  🔴失效<55")

    lines.append("\n─── 失效股 ───")
    bad = [v for v in validations if v.quality_score < 55]
    if bad:
        for v in bad:
            reasons = []
            if v.stop_loss:
                reasons.append("触发-2%止损")
            if v.ret_actual < 0:
                reasons.append(f"收盘{v.ret_actual:+.1f}%")
            gap = (v.open_today - v.signal_close) / v.signal_close * 100
            if gap >= 5:
                reasons.append(f"跳空高开{gap:.1f}%")
            if not reasons:
                reasons.append(f"评分过低({v.quality_score:.0f})")
            lines.append(f"  {v.code} {v.name}: {'; '.join(reasons)}")
    else:
        lines.append("  无失效股")

    lines.append("\n─── 评分分布 ───")
    for lo, hi, lbl in [(85, 100, "🟢优秀"), (70, 85, "🔵良好"), (55, 70, "🟡及格"), (0, 55, "🔴失效")]:
        count = sum(1 for s in validations if lo <= s.quality_score < hi)
        lines.append(f"  {lbl} ({lo}-{hi}): {count:>3}")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="信号验证器")
    parser.add_argument("--input", "-i", type=str, default=None, help="昨日选股输出文件")
    parser.add_argument("--codes", nargs="+", default=None, help="指定股票代码")
    parser.add_argument("--names", nargs="+", default=None, help="对应名称")
    parser.add_argument("--output", "-o", type=str, default=None, help="报告输出路径")
    args = parser.parse_args()

    if args.codes:
        codes = list(zip(args.codes, args.names or [get_stock_name(c, {}) for c in args.codes]))
    else:
        input_path = Path(args.input) if args.input else find_latest_screen_output()
        if not input_path or not input_path.exists():
            print("❌ 未找到昨日选股文件，请用 --input 指定")
            sys.exit(1)
        codes = parse_screen_output(input_path)
        print(f"📋 解析 {input_path.name} → {len(codes)} 只")

    if not codes:
        print("❌ 无股票可验证")
        sys.exit(1)

    validations = run_validation(codes)
    report_text = format_report(validations)
    print("\n" + report_text)

    out = Path(args.output) if args.output else Path.home() / "stock_reports" / f"signal_validation_{datetime.now().strftime('%Y-%m-%d')}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report_text, encoding="utf-8")
    print(f"\n💾 已写入: {out.resolve()}")
