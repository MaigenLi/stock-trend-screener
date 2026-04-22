#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
参数优化（训练集 / 验证集分离版）
"""

import argparse
import itertools
import pandas as pd
from datetime import datetime, timedelta

from real_screen import run
import rps_strong_screen as rps


# =========================
# 参数空间
# =========================

PARAM_GRID = {
    "MIN_GAIN": [3, 5, 7],
    "SIGNAL_DAYS": [3, 5],
    "MAX_GAIN": [25, 35],
    "W_RPS": [0.3, 0.4],
}


# =========================
# CLI
# =========================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--start", type=str, help="开始日期")
    parser.add_argument("--end", type=str, help="结束日期")
    parser.add_argument("--days", type=int, default=60)

    parser.add_argument("--split", type=float, default=0.7,
                        help="训练集比例（默认0.7）")

    return parser.parse_args()


# =========================
# 时间
# =========================

def resolve_dates(args):
    if args.start and args.end:
        return args.start, args.end

    end = datetime.now()
    start = end - timedelta(days=args.days)

    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def get_dates(start, end):
    df = rps.get_index_data()
    df = df[(df["date"] >= start) & (df["date"] <= end)]
    return df["date"].tolist()


# =========================
# 收益
# =========================

def get_ret(code, date, days):
    df = rps.get_stock_kline(code)
    df = df[df["date"] >= date]

    if len(df) < days + 1:
        return None

    if df.iloc[0]["date"] != date:
        return None

    buy = df.iloc[0]["close"]
    sell = df.iloc[days]["close"]

    return (sell / buy - 1) * 100


# =========================
# 评分
# =========================

def score_strategy(df):
    df = df.dropna()

    if len(df) < 20:
        return -999

    win_rate = (df["ret_3d"] > 0).mean()
    avg_ret = df["ret_3d"].mean()

    return avg_ret * 0.7 + win_rate * 100 * 0.3


# =========================
# 回测函数
# =========================

def run_backtest(params, dates):
    records = []

    for d in dates:
        dt = datetime.strptime(d, "%Y-%m-%d")

        try:
            picks = run(dt, custom_config=params)
        except:
            continue

        if not picks:
            continue

        for p in picks:
            r3 = get_ret(p.code, d, 3)

            records.append({
                "date": d,
                "code": p.code,
                "ret_3d": r3
            })

    df = pd.DataFrame(records)
    score = score_strategy(df)

    return score, len(df), df


# =========================
# 主流程
# =========================

def optimize(start_date, end_date, split_ratio):
    print("\n==============================")
    print("📊 参数优化（训练/验证）")
    print(f"总区间: {start_date} → {end_date}")
    print("==============================\n")

    dates = get_dates(start_date, end_date)

    # 切分
    split_idx = int(len(dates) * split_ratio)
    train_dates = dates[:split_idx]
    test_dates  = dates[split_idx:]

    print(f"训练集: {train_dates[0]} → {train_dates[-1]} ({len(train_dates)}天)")
    print(f"验证集: {test_dates[0]} → {test_dates[-1]} ({len(test_dates)}天)\n")

    keys = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())

    results = []

    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))

        print(f"\n🧪 参数: {params}")

        train_score, train_n, _ = run_backtest(params, train_dates)

        print(f"  Train → 样本:{train_n} 得分:{train_score:.2f}")

        if train_score < 0:
            continue

        test_score, test_n, _ = run_backtest(params, test_dates)

        print(f"  Test  → 样本:{test_n} 得分:{test_score:.2f}")

        results.append({
            "params": str(params),
            "train_score": train_score,
            "test_score": test_score,
            "gap": test_score - train_score,
            "train_samples": train_n,
            "test_samples": test_n
        })

    df = pd.DataFrame(results)

    # 排序：优先看验证集
    df = df.sort_values("test_score", ascending=False)

    print("\n🏆 最优参数（验证集TOP5）:")
    print(df.head(5))

    df.to_csv("param_opt_cv_result.csv", index=False, encoding="utf-8-sig")

    print("\n💾 已保存 param_opt_cv_result.csv")


# =========================

if __name__ == "__main__":
    args = parse_args()
    start_date, end_date = resolve_dates(args)

    optimize(start_date, end_date, args.split)
