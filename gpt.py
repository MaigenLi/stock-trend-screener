#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MACD 强势启动选股（实盘增强版）
============================

修复点：
1. 修复换手率（需要流通股本）
2. 修复红柱统计越界 bug
3. 增加均线趋势过滤
4. 稳定评分系统（避免除0爆炸）
5. 去除未来函数风险
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from gain_turnover import load_qfq_history


# =========================
# 技术指标
# =========================

def compute_macd(df: pd.DataFrame):
    close = df["close"]

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    macd = (dif - dea) * 2

    df["dif"] = dif
    df["dea"] = dea
    df["macd"] = macd

    return df


def compute_ma(df: pd.DataFrame):
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma60"] = df["close"].rolling(60).mean()
    return df


# =========================
# 核心选股逻辑
# =========================

def evaluate_macd_stock(df: pd.DataFrame):
    if len(df) < 80:
        return None

    df = compute_macd(df)
    df = compute_ma(df)

    idx = len(df) - 1

    # -------------------------
    # 1. 趋势过滤（核心）
    # -------------------------
    if not (
        df["dif"].iloc[idx] > 0
        and df["dif"].iloc[idx] > df["dif"].iloc[idx - 1] > df["dif"].iloc[idx - 2]
        and df["dea"].iloc[idx] > df["dea"].iloc[idx - 1]
        and df["macd"].iloc[idx] > df["macd"].iloc[idx - 1]
    ):
        return None

    # -------------------------
    # 2. 红柱启动阶段
    # -------------------------
    red_days = 0
    for i in range(idx, max(idx - 20, 0), -1):
        if df["macd"].iloc[i] > 0:
            red_days += 1
        else:
            break

    if red_days < 2 or red_days > 12:
        return None

    # -------------------------
    # 3. 上涨质量
    # -------------------------
    up_days = 0
    start = max(idx - red_days + 1, 1)  # 防止 i-1 越界

    for i in range(start, idx + 1):
        if df["close"].iloc[i] > df["close"].iloc[i - 1]:
            up_days += 1

    if up_days / red_days < 0.6:
        return None

    # -------------------------
    # 4. 短期动能
    # -------------------------
    ret_3 = df["close"].iloc[idx] / df["close"].iloc[idx - 3] - 1
    ret_1 = df["close"].iloc[idx] / df["close"].iloc[idx - 1] - 1

    if not (-0.03 < ret_1 and ret_3 > 0.04):
        return None

    # -------------------------
    # 5. 均线结构（关键新增）
    # -------------------------
    if not (
        df["close"].iloc[idx] > df["ma20"].iloc[idx] > df["ma60"].iloc[idx]
    ):
        return None

    # -------------------------
    # 6. 成交额过滤（替代错误换手率）
    # -------------------------
    if "amount" in df.columns:
        amt20 = df["amount"].iloc[-20:].mean()
        if amt20 < 1e8:  # 1亿过滤
            return None

    # -------------------------
    # 7. 稳定评分系统
    # -------------------------
    dif = df["dif"].iloc[idx]
    dea = df["dea"].iloc[idx]

    # 避免除0问题
    strength = dif / (abs(dea) + 1e-6)

    score = (
        min(strength * 20, 40) +     # 趋势强度
        (12 - red_days) * 2 +        # 越早启动越好
        min(ret_3 * 100, 20)         # 动能
    )

    return {
        "score": round(score, 2),
        "red_days": red_days,
        "ret_3": round(ret_3 * 100, 2),
    }


# =========================
# 主流程
# =========================

def run(date_str: str):
    target_date = datetime.strptime(date_str, "%Y-%m-%d")

    # 你可以替换成自己的股票池
    stock_list = load_stock_list()

    results = []

    for code in stock_list:
        try:
            df = load_qfq_history(code)

            if df is None or len(df) < 80:
                continue

            df = df[df["date"] <= target_date]

            if len(df) < 80:
                continue

            res = evaluate_macd_stock(df)

            if res:
                results.append({
                    "code": code,
                    **res
                })

        except Exception as e:
            continue

    # 排序输出
    results = sorted(results, key=lambda x: x["score"], reverse=True)

    print(f"\n=== MACD 强势股（{date_str}） ===")
    for r in results[:50]:
        print(r)


# =========================
# 股票列表（你可以换成自己的）
# =========================

def load_stock_list():
    # TODO: 替换成你的全市场股票池
    return [
        "000001",
        "000002",
        "600000",
        "600519",
    ]


# =========================
# CLI
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True, help="YYYY-MM-DD")

    args = parser.parse_args()

    run(args.date)
