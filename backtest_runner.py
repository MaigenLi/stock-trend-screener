#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
自动回测脚本

功能：
- 批量跑日期
- 调用 real_screen
- 计算 T+1 / T+3 / T+5 收益
- 输出统计结果
"""

import argparse
import pandas as pd
from datetime import datetime, timedelta

import rps_strong_screen as rps
from real_screen import run


# =========================
# 参数
# =========================

HOLD_DAYS = [1, 3, 5]

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--start", type=str, help="开始日期 YYYY-MM-DD")
    parser.add_argument("--end", type=str, help="结束日期 YYYY-MM-DD")
    parser.add_argument("--days", type=int, default=30, help="回测天数（默认30天）")

    return parser.parse_args()

# =========================
# 获取交易日列表
# =========================

def get_trade_dates(start, end):
    df = rps.get_index_data()  # 用指数做交易日
    df = df[(df["date"] >= start) & (df["date"] <= end)]
    return df["date"].tolist()


# =========================
# 获取未来收益
# =========================

def get_forward_return(code, buy_date, days):
    df = rps.get_stock_kline(code)

    df = df[df["date"] >= buy_date]
    if len(df) < days + 1:
        return None

    buy_price = df.iloc[0]["close"]
    sell_price = df.iloc[days]["close"]

    return (sell_price / buy_price - 1) * 100


def resolve_dates(args):
    if args.start and args.end:
        return args.start, args.end

    # 默认：最近 N 天
    end = datetime.now()
    start = end - timedelta(days=args.days)

    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


# =========================
# 主回测
# =========================

def backtest(start_date, end_date):
    print("\n==============================")
    print("📊 回测区间")
    print(f"开始: {start_date}")
    print(f"结束: {end_date}")
    print("==============================\n")

    dates = get_trade_dates(start_date, end_date)

    all_records = []

    for d in dates:
        print(f"\n=== 回测 {d} ===")

        target_date = datetime.strptime(d, "%Y-%m-%d")

        try:
            picks = run(target_date)
        except Exception as e:
            print(f"❌ 失败: {e}")
            continue

        if not picks:
            continue

        for p in picks:
            code = p.code

            record = {
                "date": d,
                "code": code,
            }

            for h in HOLD_DAYS:
                ret = get_forward_return(code, d, h)
                record[f"ret_{h}d"] = ret

            all_records.append(record)

    df = pd.DataFrame(all_records)

    if df.empty:
        print("无数据")
        return

    # =========================
    # 统计
    # =========================

    print("\n=== 回测结果 ===")

    for h in HOLD_DAYS:
        col = f"ret_{h}d"

        valid = df[col].dropna()

        if len(valid) == 0:
            continue

        win_rate = (valid > 0).mean()
        avg_ret = valid.mean()

        print(f"\nT+{h}")
        print(f"样本数: {len(valid)}")
        print(f"胜率: {win_rate:.2%}")
        print(f"平均收益: {avg_ret:.2f}%")

    # =========================
    # 保存
    # =========================

    output = "backtest_result.csv"
    df.to_csv(output, index=False, encoding="utf-8-sig")

    print(f"\n💾 已保存: {output}")


# =========================
# 入口
# =========================

if __name__ == "__main__":
    args = parse_args()

    start_date, end_date = resolve_dates(args)

    backtest(start_date, end_date)
