#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
前复权日线数据缓存（复用 gain_turnover.py 的缓存体系）
========================================================
缓存路径：~/.openclaw/workspace/.cache/qfq_daily/{code}_qfq.csv
数据格式：与 cache_qfq_daily.py 完全一致（列：date, open, high, low, close, volume, amount, outstanding_share, turnover）
"""

import sys
from pathlib import Path

import pandas as pd

WORKSPACE = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(WORKSPACE))

# 直接复用 gain_turnover.py 的缓存逻辑
from stock_trend.gain_turnover import load_qfq_history as _load_qfq_history


def load_qfq_history(
    code: str,
    end_date: str | None = None,
    refresh: bool = False,
) -> pd.DataFrame | None:
    """
    加载前复权日线数据（复用 gain_turnover.py 的 CSV 缓存）。

    Args:
        code: 带 sh/sz 前缀的股票代码（如 'sh600186', 'sz000001'）
        end_date: YYYY-MM-DD，截止日期（不含）
        refresh: True=强制联网刷新

    Returns:
        DataFrame（按 date 升序）或 None（数据不足/加载失败）
    """
    df = _load_qfq_history(
        code=code,
        start_date=None,
        end_date=end_date,
        adjust="qfq",
        refresh=refresh,
    )

    if df is None or df.empty:
        return None

    # 按日期升序排列
    df = df.sort_values("date").reset_index(drop=True)

    return df


def preload_all_codes() -> list[str]:
    """加载全市场股票代码列表"""
    from stock_trend.gain_turnover import get_all_stock_codes

    codes = get_all_stock_codes()
    return codes
