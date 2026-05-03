#!/usr/bin/env python3
"""
date_utils.py — 日期/交易日公共工具
"""
from pathlib import Path
import pandas as pd

WORKSPACE = Path(__file__).parent.parent.parent.resolve()
_QFQ_DIR = WORKSPACE / ".cache" / "qfq_daily"

_TRADING_DAYS = None


def get_trading_days():
    """从 QFQ CSV 文件构建交易日历（升序）"""
    global _TRADING_DAYS
    if _TRADING_DAYS is not None:
        return _TRADING_DAYS
    dates = set()
    if _QFQ_DIR.exists():
        for f in _QFQ_DIR.glob("*_qfq.csv"):
            try:
                dates.update(pd.read_csv(f, usecols=["date"])["date"].tolist())
            except Exception:
                pass
    _TRADING_DAYS = sorted(set(dates))
    return _TRADING_DAYS


def validate_signal_date(date_str: str) -> str:
    """
    验证输入日期是否为有效交易日。
    返回该日期（如果有效）。
    如果无效，打印最近的前后交易日并退出。
    """
    import sys
    days = get_trading_days()
    if date_str in days:
        return date_str

    prev_day = None
    next_day = None
    for d in reversed(days):
        if d < date_str:
            prev_day = d
            break
    for d in days:
        if d > date_str:
            next_day = d
            break

    print(f"❌ {date_str} 不是有效交易日", file=sys.stderr)
    if prev_day and next_day:
        print(f"   最近的前一交易日: {prev_day}，后一交易日: {next_day}", file=sys.stderr)
    elif prev_day:
        print(f"   最近的前一交易日: {prev_day}", file=sys.stderr)
    elif next_day:
        print(f"   最近的后一交易日: {next_day}", file=sys.stderr)
    print(f"   请使用有效日期重新运行。", file=sys.stderr)
    sys.exit(1)


def _is_trading_day(date_str: str) -> bool:
    return date_str in get_trading_days()


def _prev_trading_day(date_str: str) -> str | None:
    days = get_trading_days()
    for d in reversed(days):
        if d < date_str:
            return d
    return None


def _next_trading_day(date_str: str) -> str | None:
    days = get_trading_days()
    for d in days:
        if d > date_str:
            return d
    return None
