#!/usr/bin/env python3
"""查询缓存数据日期分布统计"""
import argparse
import pandas as pd
from pathlib import Path
from collections import Counter

CACHE_DIR = Path.home() / ".openclaw/workspace/.cache/qfq_daily"

def check_date(target_date: str, verbose: bool = False):
    """统计缓存中指定日期的数据情况"""
    csv_files = list(CACHE_DIR.glob("*.csv"))
    print(f"缓存目录: {CACHE_DIR}")
    print(f"总文件数: {len(csv_files)} 只")
    print()

    # 统计最新日期分布
    date_counter: dict[str, int] = {}
    files_checked = 0

    for f in csv_files:
        try:
            df = pd.read_csv(f)
            if "date" not in df.columns:
                continue
            latest = str(df["date"].iloc[-1])[:10]
            date_counter[latest] = date_counter.get(latest, 0) + 1
            files_checked += 1
        except Exception:
            continue

    # 总体分布
    print(f"=== 缓存数据日期分布（共 {files_checked} 只）===")
    for date, count in sorted(date_counter.items(), reverse=True)[:20]:
        bar = "█" * (count // 50)
        print(f"  {date}  {count:5d} 只  {bar}")

    # 指定日期统计
    if target_date:
        total = date_counter.get(target_date, 0)
        print(f"\n=== 指定日期: {target_date} ===")
        print(f"  有数据股票: {total} 只 / {files_checked} 只")
        if total > 0 and verbose:
            # 进一步检查这个日期的数据详情
            sample_files = [f for f in csv_files if True]  # 重新遍历找样本
            target_stocks = []
            for f in csv_files:
                try:
                    df = pd.read_csv(f)
                    if "date" not in df.columns:
                        continue
                    if str(df["date"].iloc[-1])[:10] == target_date:
                        pure = f.stem.replace("_qfq", "")
                        target_stocks.append(pure)
                except:
                    continue
            print(f"  股票代码样本（前20只）: {', '.join(target_stocks[:20])}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="查询缓存数据日期统计")
    parser.add_argument("--date", "-d", type=str, default=None,
                        help="指定查询日期，如 2026-04-17")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="显示详细信息")
    args = parser.parse_args()

    check_date(args.date, args.verbose)
