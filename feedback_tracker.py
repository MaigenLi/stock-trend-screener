#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
反馈追踪器 — 自我进化策略 第二层
================================
将 signal_validator.py 的验证结果追加到 CSV 数据库，
记录每个信号的完整生命周期（买入→持有→卖出），供 L3 进化分析。

CSV 字段：
  signal_date, code, name, signal_close,
  verified_date, open_verified, close_verified, high_verified,
  ret_actual, ret_signal, ret_high,
  hit_3pct, hit_5pct, hit_7pct, hit_10pct, stop_loss,
  quality_score, evolution_tag,
  exit_date, exit_price, hold_days, ret_exit, max_retrace,
  best_price, best_return, closed

用法：
  python feedback_tracker.py                              # 追加今日验证结果
  python feedback_tracker.py --stats                      # 显示数据库统计
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

WORKSPACE = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(WORKSPACE))

from stock_trend.signal_validator import (
    SignalValidation,
    find_latest_screen_output,
    parse_screen_output,
    run_validation,
)
from stock_trend.gain_turnover import (
    load_qfq_history,
    normalize_prefixed,
)

TRACKER_CSV = WORKSPACE / "stock_trend" / "feedback_tracker.csv"
FIELDS = [
    "signal_date", "code", "name", "signal_close",
    "verified_date", "open_verified", "close_verified", "high_verified",
    "ret_actual", "ret_signal", "ret_high",
    "hit_3pct", "hit_5pct", "hit_7pct", "hit_10pct", "stop_loss",
    "quality_score", "evolution_tag",
    "exit_date", "exit_price", "hold_days", "ret_exit", "max_retrace",
    "best_price", "best_return", "closed",
]


def _ensure_csv():
    if not TRACKER_CSV.exists():
        TRACKER_CSV.parent.mkdir(parents=True, exist_ok=True)
        with open(TRACKER_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()


def _read_csv() -> list[dict]:
    _ensure_csv()
    with open(TRACKER_CSV, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _append_rows(rows: list[dict]):
    _ensure_csv()
    with open(TRACKER_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writerows(rows)


# ── 多日追踪 ──────────────────────────────────────────────
def track_signal_full(code: str, name: str, signal_date: str, signal_close: float) -> dict:
    """
    从信号日起，追踪最多 10 个交易日（或触发止盈/止损）。
    真实收益以 T+1 开盘买入 → 收盘卖出（持有1天）为标准。
    """
    df = load_qfq_history(code, max_age_hours=0)
    if df is None or df.empty:
        return {}

    sig_ts = pd.Timestamp(signal_date)
    post = df[df["date"] > sig_ts].head(10).reset_index(drop=True)
    if post.empty:
        return {}

    entry_open = float(post.iloc[0]["open"])
    entry_date = str(post.iloc[0]["date"])[:10]
    best_price = entry_open
    best_return = 0.0
    max_retrace = 0.0
    exit_date = ""
    exit_price = entry_open
    hold_days = 0
    closed = False
    hit_3pct = hit_5pct = hit_7pct = hit_10pct = stop_loss = False

    for i in range(len(post)):
        day_close = float(post.iloc[i]["close"])
        day_high = float(post.iloc[i]["high"])
        day_low = float(post.iloc[i]["low"])

        if day_high / entry_open - 1 >= 0.03:
            hit_3pct = True
        if day_high / entry_open - 1 >= 0.05:
            hit_5pct = True
        if day_high / entry_open - 1 >= 0.07:
            hit_7pct = True
        if day_high / entry_open - 1 >= 0.10:
            hit_10pct = True

        if day_low / entry_open - 1 <= -0.02:
            stop_loss = True
            exit_date = str(post.iloc[i]["date"])[:10]
            exit_price = day_low
            hold_days = i + 1
            ret_exit = (exit_price - entry_open) / entry_open * 100.0
            closed = True
            break

        if day_close > best_price:
            best_price = day_close
            best_return = (best_price - entry_open) / entry_open * 100.0
        day_retrace = (best_price - day_close) / best_price * 100.0 if best_price > 0 else 0.0
        max_retrace = max(max_retrace, day_retrace)

    if not closed:
        last = post.iloc[-1]
        exit_date = str(last["date"])[:10]
        exit_price = float(last["close"])
        hold_days = len(post)
        ret_exit = (exit_price - entry_open) / entry_open * 100.0

    # T+1 真实收益（开盘买→收盘卖）
    ret_actual = (exit_price - entry_open) / entry_open * 100.0 if hold_days > 0 else 0.0
    # 信号日收盘→T+1 收盘（代理指标）
    ret_signal = (exit_price - signal_close) / signal_close * 100.0 if signal_close > 0 else 0.0

    return {
        "signal_date": signal_date,
        "code": normalize_prefixed(code),
        "name": name,
        "signal_close": round(signal_close, 2),
        "verified_date": entry_date,
        "open_verified": round(entry_open, 2),
        "close_verified": round(exit_price, 2),
        "high_verified": round(float(post["high"].max()), 2),
        "ret_actual": round(ret_actual, 2),
        "ret_signal": round(ret_signal, 2),
        "ret_high": round((float(post["high"].max()) - signal_close) / signal_close * 100.0, 2),
        "hit_3pct": hit_3pct,
        "hit_5pct": hit_5pct,
        "hit_7pct": hit_7pct,
        "hit_10pct": hit_10pct,
        "stop_loss": stop_loss,
        "quality_score": 0.0,
        "evolution_tag": "",
        "exit_date": exit_date,
        "exit_price": round(exit_price, 2),
        "hold_days": hold_days,
        "ret_exit": round(ret_exit, 2),
        "max_retrace": round(max_retrace, 2),
        "best_price": round(best_price, 2),
        "best_return": round(best_return, 2),
        "closed": closed,
    }


def append_validation_results(validations: list[SignalValidation]):
    """追加验证结果到 tracker CSV（仅追加不在 CSV 中的信号）。"""
    existing = {(r["code"], r["signal_date"]) for r in _read_csv()}
    new_rows = []
    for v in validations:
        key = (v.code, v.signal_date)
        if key in existing:
            continue
        new_rows.append({
            "signal_date": v.signal_date,
            "code": v.code,
            "name": v.name,
            "signal_close": v.signal_close,
            "verified_date": v.signal_date,
            "open_verified": v.open_today,
            "close_verified": v.close_today,
            "high_verified": v.high_today,
            "ret_actual": v.ret_actual,
            "ret_signal": v.ret_signal,
            "ret_high": v.ret_high,
            "hit_3pct": v.hit_3pct,
            "hit_5pct": v.hit_5pct,
            "hit_7pct": v.hit_7pct,
            "hit_10pct": v.hit_10pct,
            "stop_loss": v.stop_loss,
            "quality_score": v.quality_score,
            "evolution_tag": v.evolution_tag,
            "exit_date": "",
            "exit_price": 0.0,
            "hold_days": 0,
            "ret_exit": 0.0,
            "max_retrace": 0.0,
            "best_price": 0.0,
            "best_return": 0.0,
            "closed": False,
        })
    if new_rows:
        _append_rows(new_rows)
        print(f"📝 追加 {len(new_rows)} 条新记录到反馈数据库")
    else:
        print("📝 无新记录，所有信号已在数据库中")


def show_stats():
    rows = _read_csv()
    if not rows:
        print("📊 反馈数据库为空")
        return

    total = len(rows)
    closed = [r for r in rows if r["closed"] == "True"]
    scores = [float(r["quality_score"]) for r in rows if r["quality_score"]]
    rets_actual = [float(r["ret_actual"]) for r in rows if r.get("ret_actual")]
    rets_exit = [float(r["ret_exit"]) for r in closed if r["ret_exit"]]
    wins_actual = [r for r in rets_actual if r > 0]

    print(f"\n📊 反馈数据库统计（总计 {total} 条信号）")
    if rets_actual:
        print(f"   1日收益: 均={np.mean(rets_actual):+.2f}%  胜率={len(wins_actual)}/{len(rets_actual)}={len(wins_actual)/len(rets_actual)*100:.1f}%  最大={max(rets_actual):+.2f}%  最小={min(rets_actual):+.2f}%")
    if rets_exit:
        wins_exit = [r for r in rets_exit if r > 0]
        print(f"   持仓收益: 均={np.mean(rets_exit):+.2f}%  胜率={len(wins_exit)}/{len(rets_exit)}={len(wins_exit)/len(rets_exit)*100:.1f}%  最大={max(rets_exit):+.2f}%  最小={min(rets_exit):+.2f}%")
    if scores:
        print(f"   质量分: 均={np.mean(scores):.1f}")
    print(f"   触发+3%={sum(1 for r in rows if r['hit_3pct']=='True')}  +5%={sum(1 for r in rows if r['hit_5pct']=='True')}  止损={sum(1 for r in rows if r['stop_loss']=='True')}")


if __name__ == "__main__":
    import pandas as pd

    parser = argparse.ArgumentParser(description="反馈追踪器")
    parser.add_argument("--input", "-i", type=str, default=None, help="指定昨日选股文件")
    parser.add_argument("--codes", nargs="+", default=None, help="指定股票")
    parser.add_argument("--names", nargs="+", default=None)
    parser.add_argument("--stats", action="store_true", help="显示统计")
    args = parser.parse_args()

    if args.stats:
        show_stats()
        sys.exit(0)

    if args.codes:
        codes = list(zip(args.codes, args.names or [r["name"] for r in _read_csv() if r["code"] in args.codes]))
    else:
        input_path = Path(args.input) if args.input else find_latest_screen_output()
        if not input_path or not input_path.exists():
            print("❌ 未找到选股文件，请用 --input 指定")
            sys.exit(1)
        codes = parse_screen_output(input_path)

    if not codes:
        print("❌ 无股票")
        sys.exit(1)

    validations = run_validation(codes)
    append_validation_results(validations)
    show_stats()
