#!/usr/bin/env python3
"""
全市场扫描：指定区间涨幅大于10%的股票排名
用法: python top_gain_in_range.py <开始日期> <结束日期>
  例: python top_gain_in_range.py 2024-05-20 2024-09-19
"""

import sys
import argparse
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

import os
STOCK_TREND_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, STOCK_TREND_DIR)
from gain_turnover import load_qfq_history, get_all_stock_codes, load_stock_names, get_stock_name

MIN_GAIN = 10.0
_names = None

def _init_worker():
    global _names
    _names = load_stock_names()

def analyze_stock(code):
    global _names
    try:
        df = load_qfq_history(code, start_date=START, end_date=END)
        if df is None or len(df) < 20:
            return None

        start_price = float(df.iloc[0]['close'])
        end_price   = float(df.iloc[-1]['close'])
        gain_pct    = (end_price - start_price) / start_price * 100

        if gain_pct > MIN_GAIN:
            high = float(df['high'].max())
            low  = float(df['low'].min())
            turnover_avg = float(df['true_turnover'].mean()) if 'true_turnover' in df.columns else 0.0
            amount_avg   = float(df['amount'].mean())
            return {
                'code': code,
                'name': get_stock_name(code, _names),
                'start_price':  round(start_price, 3),
                'end_price':    round(end_price,   3),
                'gain_pct':     round(gain_pct,    2),
                'max_gain_pct': round((high - start_price) / start_price * 100, 2),
                'high':         round(high,  3),
                'low':          round(low,   3),
                'days':         len(df),
                'turnover_avg': round(turnover_avg, 2),
                'amount_avg_wan': round(amount_avg / 10000, 2),
            }
    except Exception:
        pass
    return None

def main():
    parser = argparse.ArgumentParser(description='区间涨幅排行榜')
    parser.add_argument('start_date', help='开始日期，格式 YYYY-MM-DD')
    parser.add_argument('end_date',   help='结束日期，格式 YYYY-MM-DD')
    args = parser.parse_args()

    global START, END
    START = args.start_date
    END   = args.end_date

    codes = get_all_stock_codes()
    print(f"全市场 {len(codes)} 只 | 区间 {START} → {END} | 涨幅>{MIN_GAIN}%")

    results = []
    with Pool(min(32, cpu_count()), initializer=_init_worker) as pool:
        for r in tqdm(pool.imap(analyze_stock, codes, chunksize=20),
                     total=len(codes), desc="扫描中"):
            if r:
                results.append(r)

    results.sort(key=lambda x: -x['gain_pct'])
    top200 = results[:200]

    print(f"\n{'='*90}")
    print(f"📈 {START} → {END} 涨幅>{MIN_GAIN}% 共{len(results)}只，列出前{len(top200)}只")
    print(f"{'='*90}")

    import pandas as pd
    df_out = pd.DataFrame(top200)
    df_out.index = range(1, len(df_out) + 1)
    df_out.index.name = 'rank'

    cols = ['code','name','start_price','end_price','gain_pct',
            'max_gain_pct','high','low','days','turnover_avg','amount_avg_wan']
    print(df_out[cols].to_string())

    out = f"top_gain_{START}_{END}.csv"
    df_out[cols].to_csv(out, encoding='utf-8-sig')
    print(f"\n✅ 已保存: {out}")

if __name__ == '__main__':
    main()