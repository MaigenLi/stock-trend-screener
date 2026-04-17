#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
信号验证器 — 自我进化策略 第一层
================================
读取前一日选股结果，用当日数据验证信号质量。

用法：
  python signal_validator.py                            # 默认今天，验证最近一次筛选的信号
  python signal_validator.py --date 2026-04-16         # 用04-16收盘数据，验证04-15的信号
  python signal_validator.py --date 2026-04-16 --input xxx.txt  # 指定信号文件
  python signal_validator.py --codes sz002990 sz000001           # 指定个股

--date 语义：
  验证日（target_date）= --date 参数，表示用这天的数据来验证
  信号文件 = 验证日的前一个交易日（自动查找最近的 screening 文件）

例子（假设今天 2026-04-17收盘后）：
  python signal_validator.py
    → 用04-17数据验证 triple_screen_2026-04-16.txt（昨天跑出的信号）
  python signal_validator.py --date 2026-04-16
    → 用04-16数据验证 daily_screen_2026-04-15.txt（更早的信号，复盘用）
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
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
    load_stock_names,
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


# ── 交易日工具 ────────────────────────────────────────────
def get_trading_day_file(target_date: datetime, reports_dir: Path) -> Optional[tuple[Path, str]]:
    """
    找 target_date 前最近一个有选股输出文件的交易日。
    返回 (文件路径, 信号日期字符串)。
    信号日期 = target_date 的前一个交易日。
    """
    # 尝试依次往前找（最多5个日历日）
    candidates: list[Path] = []
    # triple_screen 优先于 daily_screen（同一日期时取 triple_screen）
    for pat in ["daily_screen_*.txt", "signal_validation_*.txt"]:
        candidates.extend(reports_dir.glob(pat))
    triple_files = list(reports_dir.glob("triple_screen_*.txt"))
    # triple_screen 加入（重复日期会用 triple 覆盖 daily）
    candidates.extend(triple_files)

    if not candidates:
        return None

    date_pat = re.compile(r"(\d{4}-\d{2}-\d{2})")

    # 找 target_date 之前最近的文件（triple_screen 优先）
    best_file = None
    best_file_date: Optional[datetime] = None
    best_sig_date = None
    target_date_only = target_date.date()
    for p in candidates:
        m = date_pat.search(p.name)
        if not m:
            continue
        file_date_str = m.group(1)
        try:
            file_date = datetime.strptime(file_date_str, "%Y-%m-%d")
        except ValueError:
            continue
        # 信号日 = target_date 的前一个交易日（严格 < target_date，仅比较日期）
        if file_date.date() >= target_date_only:
            continue
        # triple_screen 优先（同一日期时，保留 triple_screen）
        is_triple = "triple" in p.name
        is_best_triple = best_file and "triple" in best_file.name if best_file else False
        update = False
        if best_file is None:
            update = True
        elif file_date > best_file_date:
            update = True
        elif file_date == best_file_date and is_triple and not is_best_triple:
            update = True
        if update:
            best_file = p
            best_file_date = file_date
            best_sig_date = file_date_str

    if best_file and best_sig_date:
        return best_file, best_sig_date
    return None


def is_trading_day(date_str: str, reports_dir: Path) -> bool:
    """根据选股文件判断某日期是否是交易日（有文件=交易日）。"""
    date_pat = re.compile(r"(\d{4}-\d{2}-\d{2})")
    for pat in ["daily_screen_*.txt", "triple_screen_*.txt"]:
        for p in reports_dir.glob(pat):
            m = date_pat.search(p.name)
            if m and m.group(1) == date_str:
                return True
    return False


# ── 解析选股输出文件 ──────────────────────────────────────
def parse_screen_output(path: Path) -> list[tuple[str, str, str, float]]:
    """
    解析选股结果文件，返回 (code, name, signal_date, signal_close) 列表。
    文件列顺序（固定，tab 分隔）：
      0=代码 1=名称 2=日期 3=总分 4=窗口涨幅 5=RPS综合 6=趋势 7=5日换手 8=RSI 9=风险 10=RSI动量 11=量加速 12=偏离MA20 13=收盘 14+=扣分/连号/连档...
    """
    results = []
    date_pat = re.compile(r"^(\d{4}-\d{2}-\d{2})$")
    code_pat = re.compile(r"^(sh|sz|bj)(\d{6})$")

    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 14:
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
        # 收盘价固定在 parts[13]（列位置固定）
        try:
            sig_close = float(parts[13])
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
    target_date: Optional[datetime] = None,
) -> Optional[SignalValidation]:
    """
    用 target_date 的数据验证 signal_date 发出的信号。
    target_date=None → 用最新数据（T+1）
    target_date=某日 → 用该日数据
    """
    # load_qfq_history 支持 end_date 参数
    end_date = target_date.strftime("%Y-%m-%d") if target_date else None
    df = load_qfq_history(code, end_date=end_date, adjust="qfq")
    if df is None or df.empty:
        return None

    sig_ts = pd.Timestamp(signal_date)
    post = df[df["date"] > sig_ts]       # T+1 及之后
    if post.empty:
        latest_date = df["date"].max().strftime("%Y-%m-%d")
        if latest_date <= signal_date:
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
    hit_3pct = high_today >= signal_close * 1.03
    hit_5pct = high_today >= signal_close * 1.05
    hit_7pct = high_today >= signal_close * 1.07
    hit_10pct = high_today >= signal_close * 1.10
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


def run_validation(
    codes: list[tuple[str, str, str, float]],
    target_date: Optional[datetime] = None,
) -> list[SignalValidation]:
    """
    codes: (code, name, signal_date, signal_close)
    target_date: 用该日数据验证（None=用最新数据）
    """
    t0 = time.time()
    validations = []
    skipped = 0

    for code, name, sig_date, sig_close in codes:
        c = normalize_prefixed(code)
        result = validate_signal(c, name, sig_date, sig_close, target_date)
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
        date_note = f"（数据截止 {target_date.strftime('%Y-%m-%d')}）" if target_date else "（最新数据）"
        print(f"  ⚠️  跳过 {skipped} 只（T+1 数据未就绪 {date_note}）")

    return validations


def format_report(validations: list[SignalValidation], signal_date_str: str, validate_date_str: str) -> str:
    lines = []
    lines.append("=" * 130)
    lines.append(f"📋 信号验证报告 | 验证日: {validate_date_str} | 信号日: {signal_date_str} | {len(validations)} 只")
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
    parser.add_argument("--date", type=str, default=None,
                        help="验证日（用该日数据验证前一个交易日信号），默认今天。例: --date 2026-04-16")
    parser.add_argument("--input", "-i", type=str, default=None, help="手动指定信号文件路径")
    parser.add_argument("--codes", nargs="+", default=None, help="指定股票代码")
    parser.add_argument("--names", nargs="+", default=None, help="对应名称")
    parser.add_argument("--output", "-o", type=str, default=None, help="报告输出路径")
    args = parser.parse_args()

    # ── 确定验证日 target_date ─────────────────────────────
    reports_dir = Path.home() / "stock_reports"
    if args.date:
        try:
            target_date = datetime.strptime(args.date, "%Y-%m-%d")
        except ValueError:
            print(f"❌ 日期格式错误: {args.date}，应为 YYYY-MM-DD")
            sys.exit(1)
    else:
        target_date = datetime.now()   # 默认今天

    # ── 确定信号文件 ───────────────────────────────────────
    input_path = None
    signal_date_str = None

    if args.input:
        input_path = Path(args.input)
        # 从文件名提取信号日期
        m = re.search(r"(\d{4}-\d{2}-\d{2})", input_path.name)
        if m:
            signal_date_str = m.group(1)
    else:
        # 自动找 target_date 前最近一个交易日的选股文件
        result = get_trading_day_file(target_date, reports_dir)
        if result:
            input_path, signal_date_str = result
            print(f"📋 自动找到信号文件: {input_path.name}")
            print(f"   信号日: {signal_date_str} → 验证日: {target_date.strftime('%Y-%m-%d')}")
        else:
            print(f"❌ 在 {reports_dir} 中未找到 {target_date.strftime('%Y-%m-%d')} 之前的选股文件")
            print(f"   请先用 triple_screen.py 或 gain_turnover_screen.py 生成选股报告")
            sys.exit(1)

    if not input_path or not input_path.exists():
        print(f"❌ 信号文件不存在: {input_path}")
        sys.exit(1)

    # ── 解析股票列表 ───────────────────────────────────────
    if args.codes:
        codes = list(zip(args.codes, args.names or [get_stock_name(c, {}) for c in args.codes]))
        # 手动指定时，signal_date 用 --date 的前一天（或 --date 本身）
        sig_date_fallback = (target_date - timedelta(days=1)).strftime("%Y-%m-%d")
        codes = [(c, n, sig_date_fallback, 0.0) for c, n in codes]
    else:
        parsed = parse_screen_output(input_path)
        if not parsed:
            print(f"❌ {input_path.name} 中未解析到股票")
            sys.exit(1)
        codes = parsed
        # 从解析结果确认信号日期（以文件内日期为准）
        signal_date_str = parsed[0][2] if parsed else signal_date_str

    print(f"📋 解析 {input_path.name} → {len(codes)} 只 | 验证日: {target_date.strftime('%Y-%m-%d')} | 信号日: {signal_date_str}")

    # ── 运行验证 ───────────────────────────────────────────
    validate_date_str = target_date.strftime("%Y-%m-%d")
    validations = run_validation(codes, target_date if args.date else None)
    report_text = format_report(validations, signal_date_str or "", validate_date_str)
    print("\n" + report_text)

    # 输出文件名：默认用 validate_date
    out_name = f"signal_validation_{validate_date_str}.txt"
    if args.output:
        out = Path(args.output)
    else:
        out = reports_dir / out_name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report_text, encoding="utf-8")
    print(f"\n💾 已写入: {out.resolve()}")
