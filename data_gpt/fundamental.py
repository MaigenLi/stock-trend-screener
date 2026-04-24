#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基本面数据（流通股本等）
"""

import time
import akshare as ak

from .cache import load_cache, save_cache


def _fetch_float_shares(code: str):
    """
    获取流通股本（股）
    """
    for _ in range(3):
        try:
            df = ak.stock_individual_info_em(symbol=code)

            for _, row in df.iterrows():
                if row["item"] == "流通股":
                    val = row["value"]

                    # 处理单位（万股 → 股）
                    if "万" in val:
                        return float(val.replace("万", "")) * 10000
                    else:
                        return float(val)

        except Exception:
            time.sleep(1)

    return None


def get_float_shares(code: str, refresh=False):
    """
    获取流通股本（带缓存）
    """

    cache_key = code

    if not refresh:
        val = load_cache("float_shares", cache_key, max_age=7 * 86400)
        if val:
            return val

    val = _fetch_float_shares(code)

    if val:
        save_cache("float_shares", cache_key, val)

    return val


# =========================
# 换手率计算（正确版）
# =========================

def compute_turnover(df, float_shares):
    """
    turnover = volume / float_shares
    """

    if float_shares is None or float_shares == 0:
        return None

    df = df.copy()
    df["turnover"] = df["volume"] / float_shares

    return df
