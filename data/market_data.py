#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
行情数据模块（完全等价 gain_turnover.load_qfq_history）
"""

import time
import pandas as pd
import akshare as ak

from .cache import load_cache, save_cache


# =========================
# 原始获取逻辑（保持）
# =========================

def _fetch_qfq_from_akshare(code: str):
    for _ in range(3):  # 保留重试
        try:
            df = ak.stock_zh_a_hist(
                symbol=code,
                period="daily",
                adjust="qfq"
            )

            if df is None or len(df) == 0:
                return None

            # ⚠️ 字段完全保持原来风格
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


# =========================
# 对外接口（名字不变！）
# =========================

def load_qfq_history(code: str, refresh: bool = False):
    """
    完全兼容原 gain_turnover.load_qfq_history
    """

    cache_key = code

    # ⚠️ 原路径保持
    if not refresh:
        df = load_cache("qfq_daily", cache_key)
        if df is not None:
            return df

    df = _fetch_qfq_from_akshare(code)

    if df is not None:
        save_cache("qfq_daily", cache_key, df)

    return df
