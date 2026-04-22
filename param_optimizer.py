#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
# CLI 参数
# =========================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--start", type=str, help="开始日期 YYYY-MM-DD")
    parser.add_argument("--end", type=str, help="结束日期 YYYY-MM-DD")
    parser.add_argument("--days", type=int, default=30, help="回测天数（默认30）")

    return parser.parse_args()


# =========================
# 时间解析
# =========================

def resolve_dates(args):
    if args.start and args.end:
        return args.start, args.end

    end = datetime.now()
    start = end - timedelta(days=args.days)

    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


# =========================
# 交易日
# =========================

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

    # 防止未来函数（关键！）
    if df.iloc[0]["date"] != date:
        return None

    buy = df.iloc[0]["close"]
    sell = df.iloc[days]["close"]

    return (sell / buy - 1) * 100


# =========================
# 评分函数（重点）
# =========================

def score_strategy(df):
    df = df.dropna()

    if len(df) < 20:   # 样本太少直接淘汰
        return -999

    win_rate = (df["ret_3d"] > 0).mean()
    avg_ret = df["ret_3d"].mean()

    return avg_ret * 0.7 + win_rate * 100 * 0.3


# =========================
# 单组回测
# =========================

def run_one(params, dates):
    records = []

    print(f"\n🧪 参数: {params}")

    for d in dates:
        dt = datetime.strptime(d, "%Y-%m-%d")

        try:
            picks = run(dt, custom_config=params)
        except Exception as e:
            print(f"  ❌ {d} 失败: {e}")
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

    print(f"   样本数: {len(df)}  得分: {score:.2f}")

    return score, len(df), df


# =========================
# 主优化
# =========================

def optimize(start_date, end_date):
    print("\n==============================")
    print("📊 参数优化区间")
    print(f"开始: {start_date}")
    print(f"结束: {end_date}")
    print("==============================\n")

    dates = get_dates(start_date, end_date)

    keys = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())

    total = 1
    for v in values:
        total *= len(v)

    print(f"🔢 参数组合总数: {total}\n")

    results = []
    count = 0

    for combo in itertools.product(*values):
        count += 1
        params = dict(zip(keys, combo))

        print(f"\n进度: {count}/{total}")

        score, sample_size, _ = run_one(params, dates)

        results.append({
            "params": str(params),
            "score": score,
            "samples": sample_size
        })

    df = pd.DataFrame(results)
    df = df.sort_values("score", ascending=False)

    print("\n🏆 最优参数 TOP5:")
    print(df.head(5))

    output = "param_opt_result.csv"
    df.to_csv(output, index=False, encoding="utf-8-sig")

    print(f"\n💾 已保存: {output}")


# =========================
# 入口
# =========================

if __name__ == "__main__":
    args = parse_args()
    start_date, end_date = resolve_dates(args)

    optimize(start_date, end_date)
