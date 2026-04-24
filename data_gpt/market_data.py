#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
行情数据（前复权）
"""

import time
import pandas as pd
import akshare as ak

from .cache import load_cache, save_cache


def _fetch_from_akshare(code: str):
    for _ in range(3):  # 重试机制
        try:
            df = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                adjust="qfq"
            )

            df = df.rename(columns={
                "日期": "date",
                "开盘": "open",
                "收盘": "close",
                "最高": "high",
                "最低": "low",
                "成交量": "volume",
                "成交额": "amount"
            })

            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date")

            return df

        except Exception:
            time.sleep(1)

    return None


def load_qfq_history(code: str, refresh=False):
    """
    加载前复权K线
    """

    cache_key = f"{code}"

    if not refresh:
        df = load_cache("qfq_daily", cache_key, max_age=86400)
        if df is not None:
            return df

    df = _fetch_from_akshare(code)

    if df is not None and len(df) > 0:
        save_cache("qfq_daily", cache_key, df)

    return df
