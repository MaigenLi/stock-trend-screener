#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基本面模块（从 gain_turnover 拆出）
"""

import time
import akshare as ak

from .cache import load_cache, save_cache


# =========================
# 流通股本（保持原实现风格）
# =========================

def _fetch_float_shares(code: str):
    for _ in range(3):
        try:
            df = ak.stock_individual_info_em(symbol=code)

            if df is None:
                return None

            for _, row in df.iterrows():
                if row["item"] == "流通股":
                    val = row["value"]

                    if "万" in val:
                        return float(val.replace("万", "")) * 10000
                    else:
                        return float(val)

        except Exception:
            time.sleep(1)

    return None


def get_float_shares(code: str, refresh=False):
    """
    完全兼容旧逻辑
    """

    cache_key = code

    if not refresh:
        val = load_cache("float_shares", cache_key)
        if val:
            return val

    val = _fetch_float_shares(code)

    if val:
        save_cache("float_shares", cache_key, val)

    return val


# =========================
# （保持你原来的错误逻辑也可以保留）
# =========================

def compute_turnover_legacy(df):
    """
    ⚠️ 原 gain_turnover 的旧算法（保留兼容）
    """
    df = df.copy()
    df["turnover"] = df["volume"] / df["amount"]
    return df


def compute_turnover(df, float_shares):
    """
    正确换手率（新接口，不影响旧代码）
    """
    if float_shares is None or float_shares == 0:
        return None

    df = df.copy()
    df["turnover"] = df["volume"] / float_shares
    return df
