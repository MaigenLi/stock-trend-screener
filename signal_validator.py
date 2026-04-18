#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
信号验证器 — 自我进化策略 第一层
================================
读取前一日选股结果，用当日数据验证信号质量。

用法：
  python signal_validator.py                            # 默认今天，验证最近一次筛选的信号
  python signal_validator.py --date 2026-04-16         # 用04-16收盘数据，验证04-15的信号
  python signal_validator.py --date 2026-04-16 --multi  # 用04-15+04-16收盘数据，验证04-14的信号（T+1+T+2）
  python signal_validator.py --date 2026-04-16 --input xxx.txt  # 指定信号文件
  python signal_validator.py --codes sz002990 sz000001           # 指定个股

--date 语义：
  验证日（target_date）= --date 参数，表示用这天的数据来验证
  信号文件 = 验证日的前一个交易日（自动查找最近的 screening 文件）

--multi 语义（需配合 --date）：
  用 target_date 和 target_date-1 两天数据，验证信号日（target_date-2）的表现。
  即 --date 2026-04-16 --multi → 信号日=04-14，验证 T+1(04-15) 和 T+2(04-16)。

例子（假设今天 2026-04-17收盘后）：
  python signal_validator.py
    → 用04-17数据验证 triple_screen_2026-04-16.txt（昨天跑出的信号）
  python signal_validator.py --date 2026-04-16
    → 用04-16数据验证 triple_screen_2026-04-15.txt
  python signal_validator.py --date 2026-04-16 --multi
    → 用04-15+04-16数据验证 triple_screen_2026-04-14.txt（T+1+T+2双日验证）
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass, field
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
    quality_score: float    # 当日收益评分（ret_actual）
    holding_score: float   # 持有期收益评分（ret_signal，multi日验证用）
    evolution_tag: str


@dataclass
class MultiDaySignal:
    """同一只股票的多日验证结果"""
    code: str
    name: str
    signal_date: str
    signal_close: float
    t1: Optional[SignalValidation] = None   # T+1 结果
    t2: Optional[SignalValidation] = None   # T+2 结果
    combined_score: float = 50.0
    combined_tag: str = "🔴失效"


# ── 交易日工具 ────────────────────────────────────────────
def get_trading_day_file(target_date: datetime, reports_dir: Path) -> Optional[tuple[Path, str]]:
    """
    找 target_date 前最近一个有选股输出文件的交易日。
    返回 (文件路径, 信号日期字符串)。
    信号日期 = target_date 的前一个交易日。
    """
    candidates: list[Path] = []
    for pat in ["daily_screen_*.txt", "signal_validation_*.txt"]:
        candidates.extend(reports_dir.glob(pat))
    triple_files = list(reports_dir.glob("triple_screen_*.txt"))
    candidates.extend(triple_files)

    if not candidates:
        return None

    date_pat = re.compile(r"(\d{4}-\d{2}-\d{2})")

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
        if file_date.date() >= target_date_only:
            continue
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




def find_latest_screen_output() -> Optional[Path]:
    """
    找昨日的选股结果文件（以文件内日期为准，而非文件修改时间）。
    16:40 执行时，今日 screening 大概率还在运行中，取前一天的输出最准确。
    """
    reports_dir = Path.home() / 'stock_reports'
    today = datetime.now()
    candidates = list(reports_dir.glob('daily_screen_*.txt'))
    # 也搜索 triple_screen
    candidates.extend(reports_dir.glob('triple_screen_*.txt'))
    if not candidates:
        return None

    # 优先：找到前一交易日（跳过周末）日期的文件
    prev_trading = get_n_trading_days_before(today, 1)
    prev_str = prev_trading.strftime('%Y-%m-%d')
    for p in sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True):
        fname = p.name
        m = re.search(r'(\d{4}-\d{2}-\d{2})', fname)
        if m:
            file_date = m.group(1)
            if file_date == prev_str:
                return p

    # 兜底：没有昨天的文件，则取最新修改的文件
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def get_n_trading_days_before(target_date: datetime, n: int) -> datetime:
    """
    返回 target_date 之前第 n 个交易日（跳过周末）。
    用于 --date 2026-04-14 → 找前2个交易日 = 04-10（04-14周一，前2个交易日是04-12周六(跳过)→04-11周五→04-10周四）。
    """
    current = target_date
    days_back = 0
    while days_back < n:
        current -= timedelta(days=1)
        # 0=Mon, 5=Sat, 6=Sun
        if current.weekday() < 5:
            days_back += 1
    return current


def get_signal_file_2days_before(target_date: datetime, reports_dir: Path) -> Optional[tuple[Path, str]]:
    """
    找 target_date 前第二个交易日的选股文件。
    signal_date = target_date 前2个交易日（跳过周末）。
    """
    signal_date = get_n_trading_days_before(target_date, 2)
    signal_str = signal_date.strftime("%Y-%m-%d")

    # 精确找该日期的文件（triple_screen 优先）
    for pat in [f"triple_screen_{signal_str}.txt", f"daily_screen_{signal_str}.txt"]:
        candidates = list(reports_dir.glob(pat))
        if candidates:
            # triple_screen 优先
            best = sorted(candidates, key=lambda p: ("triple" not in p.name, 0))[0]
            return best, signal_str

    # 没找到精确日期，往前找最近的文件
    candidates: list[Path] = []
    for pat in ["daily_screen_*.txt", "signal_validation_*.txt", "triple_screen_*.txt"]:
        candidates.extend(reports_dir.glob(pat))

    date_pat = re.compile(r"(\d{4}-\d{2}-\d{2})")
    best_file = None
    best_file_date: Optional[datetime] = None

    for p in candidates:
        m = date_pat.search(p.name)
        if not m:
            continue
        try:
            file_date = datetime.strptime(m.group(1), "%Y-%m-%d")
        except ValueError:
            continue
        if file_date.date() >= signal_date.date():
            continue
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

    if best_file and best_file_date:
        return best_file, best_file_date.strftime("%Y-%m-%d")
    return None


# ── 解析选股输出文件 ──────────────────────────────────────
def parse_screen_output(path: Path) -> list[tuple[str, str, str, float]]:
    """
    解析选股结果文件，返回 (code, name, signal_date, signal_close) 列表。
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
        if not date_pat.match(parts[2]):
            continue
        sig_date = parts[2]
        try:
            sig_close = float(parts[13])
            if sig_close <= 0:
                continue
        except ValueError:
            continue
        results.append((code, name, sig_date, sig_close))
    return results


# ── 核心验证（单日）────────────────────────────────────────
def _calc_score(ret_actual: float, hit_5pct: bool, stop_loss: bool, open_gap: float) -> float:
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
    return max(0.0, min(100.0, score))


def _tag(score: float) -> str:
    if score >= 85:
        return "🟢优秀"
    elif score >= 70:
        return "🔵良好"
    elif score >= 55:
        return "🟡及格"
    return "🔴失效"


def _build_signal_validation(
    code: str, name: str, today_date: str,
    signal_date: str, signal_close: float,
    today_row: pd.Series,
    ref_price: float | None = None,
) -> SignalValidation:
    """
    ref_price: 收益计算的参考价。
    - T+1 时 = None（以信号日收盘为基准）
    - T+2 时 = T+1_open（以T+1开盘价为基准，衡量持有两天的真实收益）
    """
    open_today = float(today_row["open"])
    close_today = float(today_row["close"])
    high_today = float(today_row["high"])
    low_today = float(today_row["low"])
    ref = ref_price if ref_price is not None else signal_close

    ret_actual = (close_today - open_today) / open_today * 100.0
    ret_signal = (close_today - ref) / ref * 100.0
    ret_high = (high_today - ref) / ref * 100.0
    open_gap = (open_today - ref) / ref * 100.0

    hit_3pct = high_today >= ref * 1.03
    hit_5pct = high_today >= ref * 1.05
    hit_7pct = high_today >= ref * 1.07
    hit_10pct = high_today >= ref * 1.10
    stop_loss = close_today <= ref * 0.98

    # quality_score: 基于单日实际波动（ret_actual）
    quality_score = _calc_score(ret_actual, hit_5pct, stop_loss, open_gap)
    # holding_score: 基于持有期收益（ret_signal）
    holding_score = _calc_score(ret_signal, hit_5pct, stop_loss, open_gap)

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
        quality_score=round(quality_score, 1),
        holding_score=round(holding_score, 1),
        evolution_tag=_tag(quality_score),
    )


def validate_signal(
    code: str,
    name: str,
    signal_date: str,
    signal_close: float,
    target_date: Optional[datetime] = None,
) -> Optional[SignalValidation]:
    """
    用 target_date 的数据验证 signal_date 发出的信号。
    """
    end_date = target_date.strftime("%Y-%m-%d") if target_date else None
    df = load_qfq_history(code, end_date=end_date, adjust="qfq")
    if df is None or df.empty:
        return None

    sig_ts = pd.Timestamp(signal_date)
    post = df[df["date"] > sig_ts]
    if post.empty:
        latest_date = df["date"].max().strftime("%Y-%m-%d")
        if latest_date <= signal_date:
            print(f"  ⚠️ {code} {name}: T+1 数据未就绪（最新 {latest_date} ≤ 信号 {signal_date}）")
        return None
    today_row = post.iloc[0]
    today_date = str(today_row["date"])[:10]

    open_today = float(today_row["open"])
    if open_today <= 0 or signal_close <= 0:
        return None

    return _build_signal_validation(code, name, today_date, signal_date, signal_close, today_row)


def run_validation(
    codes: list[tuple[str, str, str, float]],
    target_date: Optional[datetime] = None,
) -> list[SignalValidation]:
    """验证单日（T+1）。"""
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


# ── 格式化辅助 ───────────────────────────────────────────
def _cw(s: str) -> int:
    """显示宽度：中文/全角=2，其他=1"""
    return sum(2 if ord(c) > 0x3000 or ord(c) > 0x1100 else 1 for c in str(s))


def _fixw(s: str, w: int) -> str:
    """定宽截断/填充，保证每个值正好占 w 个显示单元格（用空格填充）"""
    s = str(s)
    d = _cw(s)
    if d > w:
        # 截断：从右往左删，直到宽度≤w（保留整数部分）
        result = []
        count = 0
        for c in reversed(s):
            dw = 2 if ord(c) > 0x3000 or ord(c) > 0x1100 else 1
            if count + dw > w:
                break
            result.append(c)
            count += dw
        s = "".join(reversed(result))
        d = _cw(s)
    return s + " " * (w - d)


# ── 多日验证（T+1 + T+2）─────────────────────────────────
def run_multi_validation(
    codes: list[tuple[str, str, str, float]],
    target_date: datetime,
) -> list[MultiDaySignal]:
    """
    用 target_date-1 和 target_date 两天的数据，验证 signal_date 的信号。
    signal_date = target_date - 2 个工作日（由调用方保证）。
    返回每只股票 T+1+T+2 的联合验证结果。
    """
    t0 = time.time()
    results: list[MultiDaySignal] = []
    skipped = 0

    # 一次性加载数据到 target_date
    end_str = target_date.strftime("%Y-%m-%d")

    for code, name, sig_date, sig_close in codes:
        c = normalize_prefixed(code)
        df = load_qfq_history(c, end_date=end_str, adjust="qfq")
        if df is None or df.empty:
            skipped += 1
            continue

        sig_ts = pd.Timestamp(sig_date)
        post = df[df["date"] > sig_ts]
        if post.empty:
            skipped += 1
            continue

        t1_row = post.iloc[0]
        t1_val = _build_signal_validation(
            c, name, str(t1_row["date"])[:10],
            sig_date, sig_close, t1_row
        )

        t2_val = None
        if len(post) >= 2:
            t2_row = post.iloc[1]
            t2_val = _build_signal_validation(
                c, name, str(t2_row["date"])[:10],
                sig_date, sig_close, t2_row,
                ref_price=t1_val.open_today,  # T+2 收益以 T+1 开盘价为基准
            )

        # 综合评分 = T+1评分 + T+2评分（各占50%权重，但T+2权重略低）
        if t2_val:
            combined = t1_val.quality_score * 0.55 + t2_val.quality_score * 0.45
        else:
            combined = t1_val.quality_score

        combined = max(0.0, min(100.0, combined))
        tag = _tag(combined) if t2_val else t1_val.evolution_tag

        results.append(MultiDaySignal(
            code=c,
            name=name,
            signal_date=sig_date,
            signal_close=round(sig_close, 2),
            t1=t1_val,
            t2=t2_val,
            combined_score=round(combined, 1),
            combined_tag=tag,
        ))

    results.sort(key=lambda x: x.combined_score, reverse=True)

    # 打印汇总
    if results:
        t1_scores = [r.t1.quality_score for r in results]
        t2_scores = [r.t2.holding_score for r in results if r.t2]  # T+2评分基于持有期
        t1_rets = [r.t1.ret_actual for r in results]
        t2_rets = [r.t2.ret_signal for r in results if r.t2]  # T+2持有期收益
        print(
            f"\n📊 双日验证（{len(results)} 只）| {time.time()-t0:.1f}s\n"
            f"   T+1 均分={np.mean(t1_scores):.1f}  上涨={sum(1 for r in t1_rets if r>0)}/{len(t1_rets)}  "
            f"均收益={np.mean(t1_rets):+.2f}%\n"
            f"   T+2 均分={np.mean(t2_scores):.1f}  上涨={sum(1 for r in t2_rets if r>0)}/{len(t2_rets)}  "
            f"均收益={np.mean(t2_rets):+.2f}%\n"
            f"   综合均分={np.mean([r.combined_score for r in results]):.1f}  "
            f"+3%={sum(1 for r in results if r.t1.hit_3pct)}/{len(results)}  "
            f"+5%={sum(1 for r in results if r.t1.hit_5pct)}/{len(results)}"
        )
    else:
        print(f"\n📊 双日验证（0 只）| {time.time()-t0:.1f}s")

    if skipped > 0:
        print(f"  ⚠️  跳过 {skipped} 只（T+1 数据未就绪）")

    return results


# ── 报告格式化 ────────────────────────────────────────────
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

    hdr = " ".join([
        _fixw('代码', 10), _fixw('名称', 8), _fixw('信号日', 12), _fixw('信号价', 8),
        _fixw('今开', 8), _fixw('今收', 8), _fixw('真实收益', 9), _fixw('参收益', 9),
        _fixw('高涨', 9), _fixw('+3', 4), _fixw('+5', 4), _fixw('止损', 5),
        _fixw('评分', 7), _fixw('评价', 8),
    ])
    lines.append(hdr)
    lines.append("-" * 130)

    for v in validations:
        parts = [
            _fixw(v.code, 10), _fixw(v.name, 8), _fixw(v.signal_date, 12),
            _fixw(f"{v.signal_close:.2f}", 8), _fixw(f"{v.open_today:.2f}", 8),
            _fixw(f"{v.close_today:.2f}", 8), _fixw(f"{v.ret_actual:+.2f}", 9),
            _fixw(f"{v.ret_signal:+.2f}", 9), _fixw(f"{v.ret_high:+.2f}", 9),
            _fixw('y' if v.hit_3pct else 'n', 4), _fixw('y' if v.hit_5pct else 'n', 4),
            _fixw('y' if v.stop_loss else 'n', 5), _fixw(f"{v.quality_score:.1f}", 7),
            _fixw(v.evolution_tag, 8),
        ]
        lines.append(" ".join(parts))
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


def format_multi_report(
    results: list[MultiDaySignal],
    signal_date_str: str,
    t1_date_str: str,
    t2_date_str: str,
) -> str:
    """双日验证报告"""
    lines = []
    lines.append("=" * 140)
    lines.append(f"📋 双日验证报告 | 信号日: {signal_date_str} | T+1: {t1_date_str} | T+2: {t2_date_str} | {len(results)} 只")
    lines.append("=" * 140)

    if results:
        t1_scores = [r.t1.quality_score for r in results]
        t2_scores = [r.t2.holding_score for r in results if r.t2]  # T+2评分基于持有期
        t1_rets = [r.t1.ret_actual for r in results]
        t2_rets = [r.t2.ret_signal for r in results if r.t2]  # T+2持有期收益
        combined = [r.combined_score for r in results]
        lines.append(
            f"  T+1 均分={np.mean(t1_scores):.1f}  上涨={sum(1 for r in t1_rets if r>0)}/{len(t1_rets)}  "
            f"均收益={np.mean(t1_rets):+.2f}%"
        )
        lines.append(
            f"  T+2 均分={np.mean(t2_scores):.1f}  上涨={sum(1 for r in t2_rets if r>0)}/{len(t2_rets)}  "
            f"均收益={np.mean(t2_rets):+.2f}%"
        )
        lines.append(
            f"  综合均分={np.mean(combined):.1f}  "
            f"T+1+3%={sum(1 for r in results if r.t1.hit_3pct)}/{len(results)}  "
            f"T+1+5%={sum(1 for r in results if r.t1.hit_5pct)}/{len(results)}  "
            f"T+1止损={sum(1 for r in results if r.t1.stop_loss)}/{len(results)}"
        )
    lines.append("-" * 140)

    col_widths = [
        ('代码', 10), ('名称', 8), ('信号日', 12), ('信号价', 9),
        ('T+1收', 8), ('T+1真实', 9), ('T+1高涨', 9), ('+3', 3), ('+5', 3), ('T+1评', 7),
        ('T+2收', 8), ('T+2真实', 9), ('T+2高涨', 9), ('+3', 3), ('+5', 3), ('T+2评', 7),
        ('综合', 7), ('评价', 8),
    ]
    hdr = " ".join(_fixw(lbl, w) for lbl, w in col_widths)
    lines.append(hdr)
    lines.append("-" * 140)

    for r in results:
        t1 = r.t1
        t2 = r.t2
        t1_3 = 'y' if t1.hit_3pct else 'n'
        t1_5 = 'y' if t1.hit_5pct else 'n'
        t2_3 = 'y' if t2 and t2.hit_3pct else 'n' if t2 else '#'
        t2_5 = 'y' if t2 and t2.hit_5pct else 'n' if t2 else '#'

        t2_c = t2.close_today if t2 else 0
        t2_r = t2.ret_signal if t2 else 0  # (T+2收-T+1开)/T+1开
        t2_h = t2.ret_high if t2 else 0     # (T+2高-T+1开)/T+1开
        t2_s = t2.quality_score if t2 else 0

        parts = [
            _fixw(r.code, 10),
            _fixw(r.name, 8),
            _fixw(r.signal_date, 12),
            _fixw(f"{r.signal_close:.2f}", 9),
            _fixw(f"{t1.close_today:.2f}", 8),
            _fixw(f"{t1.ret_actual:+.2f}", 9),
            _fixw(f"{t1.ret_high:+.2f}", 9),
            _fixw(t1_3, 3),
            _fixw(t1_5, 3),
            _fixw(f"{t1.quality_score:.1f}", 7),
            _fixw(f"{t2_c:.2f}", 8),
            _fixw(f"{t2_r:+.2f}", 9),
            _fixw(f"{t2_h:+.2f}", 9),
            _fixw(t2_3, 3),
            _fixw(t2_5, 3),
            _fixw(f"{t2.holding_score:.1f}", 7),  # T+2评基于持有期收益
            _fixw(f"{r.combined_score:.1f}", 7),
            _fixw(r.combined_tag, 8),
        ]
        row = " ".join(parts)
        lines.append(row)

    lines.append("-" * 140)
    lines.append("注: T+1真实=(T+1收-T+1开)/T+1开  T+1高涨=(T+1高-信号收)/信号收 | T+2真实=(T+2收-T+1开)/T+1开  T+2高涨=(T+2高-T+1开)/T+1开")
    lines.append("综合评分: T+1×55% + T+2×45%  评价: 🟢优秀≥85  🔵良好≥70  🟡及格≥55  🔴失效<55")

    lines.append("\n─── 综合评分分布 ───")
    for lo, hi, lbl in [(85, 100, "🟢优秀"), (70, 85, "🔵良好"), (55, 70, "🟡及格"), (0, 55, "🔴失效")]:
        count = sum(1 for s in results if lo <= s.combined_score < hi)
        lines.append(f"  {lbl} ({lo}-{hi}): {count:>3}")

    return "\n".join(lines)


# ── 主程序 ────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="信号验证器")
    parser.add_argument("--date", type=str, default=None,
                        help="验证日，默认今天。自动取前2个交易日作为信号日，验证T+1+T+2双日表现。例: --date 2026-04-14 → 信号日=04-10")
    parser.add_argument("--input", "-i", type=str, default=None, help="手动指定信号文件路径")
    parser.add_argument("--codes", nargs="+", default=None, help="指定股票代码")
    parser.add_argument("--names", nargs="+", default=None, help="对应名称")
    parser.add_argument("--output", "-o", type=str, default=None, help="报告输出路径")
    args = parser.parse_args()

    reports_dir = Path.home() / "stock_reports"

    # ── 确定目标日期 ─────────────────────────────────────
    if args.date:
        try:
            target_date = datetime.strptime(args.date, "%Y-%m-%d")
        except ValueError:
            print(f"❌ 日期格式错误: {args.date}，应为 YYYY-MM-DD")
            sys.exit(1)
    else:
        target_date = datetime.now()

    validate_date_str = target_date.strftime("%Y-%m-%d")

    # ── 确定信号文件 ─────────────────────────────────────
    input_path = None
    signal_date_str = None

    if args.input:
        input_path = Path(args.input)
        m = re.search(r"(\d{4}-\d{2}-\d{2})", input_path.name)
        if m:
            signal_date_str = m.group(1)
    else:
        # 自动找 target_date 前2个交易日的选股文件
        result = get_signal_file_2days_before(target_date, reports_dir)
        if result:
            input_path, signal_date_str = result
            sig_dt = datetime.strptime(signal_date_str, "%Y-%m-%d")
            t1_dt = get_n_trading_days_before(target_date, 1)
            print(f"📋 自动找到信号文件: {input_path.name}")
            print(f"   信号日: {signal_date_str} → T+1: {t1_dt.strftime('%Y-%m-%d')} → 验证日: {validate_date_str}")
        else:
            print(f"❌ 在 {reports_dir} 中未找到 {target_date.strftime('%Y-%m-%d')} 前2个交易日的选股文件")
            sys.exit(1)

    if not input_path or not input_path.exists():
        print(f"❌ 信号文件不存在: {input_path}")
        sys.exit(1)

    # ── 解析股票列表 ─────────────────────────────────────
    if args.codes:
        codes = list(zip(args.codes, args.names or [get_stock_name(c, {}) for c in args.codes]))
        sig_d = signal_date_str or (target_date - timedelta(days=1)).strftime("%Y-%m-%d")
        codes = [(c, n, sig_d, 0.0) for c, n in codes]
    else:
        parsed = parse_screen_output(input_path)
        if not parsed:
            print(f"❌ {input_path.name} 中未解析到股票")
            sys.exit(1)
        codes = parsed
        signal_date_str = parsed[0][2] if parsed else signal_date_str

    print(f"📋 解析 {input_path.name} → {len(codes)} 只 | 信号日: {signal_date_str}")

    # ── 运行验证（双日模式，自动） ─────────────────────────────────────────
    t1_date = get_n_trading_days_before(target_date, 1).strftime("%Y-%m-%d")
    t2_date = validate_date_str
    print(f"📊 双日验证: T+1={t1_date} T+2={t2_date}")
    multi_results = run_multi_validation(codes, target_date)
    report_text = format_multi_report(multi_results, signal_date_str or "", t1_date, t2_date)
    out_name = f"signal_validation_{t2_date}.txt"

    print("\n" + report_text)

    # 输出文件
    if args.output:
        out = Path(args.output)
    else:
        out = reports_dir / out_name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report_text, encoding="utf-8")
    print(f"\n💾 已写入: {out.resolve()}")
