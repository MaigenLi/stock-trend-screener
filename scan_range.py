#!/usr/bin/env python3
"""
scan_range.py — 区间扫描器

用法：
  python scan_range.py --start 2024-05-20 --end 2024-09-19
  python scan_range.py --start 2024-05-20 --end 2024-09-19 --qfq
  python scan_range.py --start 2024-05-20 --end 2024-09-19 --top 50

过滤逻辑全部引用 screen_trend.py 的 check_limitup_channel / check_ma，
保持与单票分析完全一致的筛选标准。
"""

import sys, argparse
from pathlib import Path
from datetime import date, timedelta

WORKSPACE = Path.home() / ".openclaw/workspace"
sys.path.insert(0, str(WORKSPACE / "stock_trend"))

import numpy as np
import pandas as pd
from gain_turnover import (
    load_stock_names_akshare,
    get_all_stock_codes_akshare,
    load_qfq_history,
    load_raw_history,
    rolling_mean,
)

# ── 从 screen_trend.py 引入真实的筛选函数 ─────────────────
from screen_trend import (
    check_limitup_channel,
    check_ma,
    normalize_symbol,
)

CACHE_DIR   = WORKSPACE / ".cache" / "qfq_daily"
RAW_CACHE_DIR = WORKSPACE / ".cache" / "raw_daily"

# ── 预加载缓存（batch模式加速用）──
_price = {}   # code -> DataFrame (raw日线)，与 screen_trend._price 完全独立

def preload(signal_date=None, data_mode="raw"):
    """预加载所有缓存CSV到内存，batch扫描时直接查dict省去每次读文件"""
    global _price
    cache_dir = RAW_CACHE_DIR if data_mode == "raw" else CACHE_DIR
    if data_mode == "raw":
        files = [f for f in cache_dir.glob("*.csv")
                if not f.name.endswith("_qfq.csv")]
    else:
        files = list(cache_dir.glob("*_qfq.csv"))
    loaded = 0
    for f in files:
        code_raw = f.stem.replace("_qfq", "")
        key = normalize_symbol(code_raw)
        try:
            df = pd.read_csv(f, dtype={"date": str})
            df["date"] = pd.to_datetime(df["date"])
            if signal_date:
                df = df[df["date"] <= pd.to_datetime(signal_date)]
            _price[key] = df
            loaded += 1
        except Exception:
            pass
    print(f"预加载 {loaded} 只（{data_mode}），范围≤{signal_date or '最新'}", flush=True)

def _load_df(code, end_date=None, data_mode="raw"):
    """优先从内存缓存取，未命中则兜底用load_history"""
    key = normalize_symbol(code)
    df = _price.get(key)
    if df is None:
        return load_history(code, end_date=end_date, data_mode=data_mode)
    return df

def load_history(code: str, end_date: str = None, data_mode: str = "raw", refresh: bool = False):
    """统一加载接口。data_mode: 'raw'=原始复权, 'qfq'=前复权"""
    if data_mode == "qfq":
        return load_qfq_history(code, end_date=end_date, refresh=refresh)
    return load_raw_history(code, start_date=None, end_date=end_date, refresh=refresh)

def get_trading_dates(start_date: str, end_date: str) -> list[str]:
    """获取区间内所有交易日（排除周末）"""
    start = pd.Timestamp(start_date)
    end   = pd.Timestamp(end_date)
    dates = []
    cur = start
    while cur <= end:
        if cur.weekday() < 5:   # 排除周六(5)和周日(6)
            dates.append(str(cur)[:10])
        cur += timedelta(days=1)
    return dates

def scan_date(signal_date: str, data_mode: str = "raw") -> list[dict]:
    """扫描指定日期，返回全部满足条件的股票"""
    codes = get_all_stock_codes_akshare()
    names = load_stock_names_akshare()
    results = []

    for idx, code in enumerate(codes):
        try:
            df = _load_df(code, end_date=signal_date, data_mode=data_mode)
            #df = load_history(code, end_date=signal_date, data_mode=data_mode)
            if df is None or len(df) < 65:
                continue

            r_lim = check_limitup_channel(df, signal_date=signal_date, code=code)
            if r_lim:
                r_lim["code"] = code
                r_lim["name"] = names.get(code, code)
                results.append(r_lim)
                continue

            r = check_ma(df, signal_date=signal_date)
            if r:
                r["code"] = code
                r["name"] = names.get(code, code)
                results.append(r)
        except Exception:
            pass

        if (idx + 1) % 1000 == 0:
            print(f"  已扫描 {idx+1} 只 ... 当前 {len(results)} 只", flush=True)

    return results

def main():
    parser = argparse.ArgumentParser(description="区间扫描器")
    parser.add_argument("--start", type=str, required=True,  help="开始日期 YYYY-MM-DD")
    parser.add_argument("--end",   type=str, required=True,  help="结束日期 YYYY-MM-DD")
    parser.add_argument("--qfq",   action="store_true",        help="使用前复权数据（默认原始数据）")
    parser.add_argument("--top",   type=int, default=0,        help="每日显示前N只（0=全部）")
    args = parser.parse_args()

    data_mode = "qfq" if args.qfq else "raw"
    trading_dates = get_trading_dates(args.start, args.end)
    print(f"区间：{args.start} → {args.end}，共 {len(trading_dates)} 个交易日，数据模式：{data_mode}")

    preload(args.end, data_mode=data_mode)

    all_results = []
    for td in trading_dates:
        print(f"  扫描 {td} ...", end="", flush=True)
        results = scan_date(td, data_mode=data_mode)
        # 特殊通道排前，按 gain20 降序
        results.sort(key=lambda x: -x.get("gain20", 0))
        for r in results:
            r["signal_date"] = td
        all_results.extend(results)
        print(f" {len(results)} 只")

    mode_label = "qfq" if args.qfq else "raw"
    out_dir = WORKSPACE / "output"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{args.start}_{args.end}_{mode_label}.txt"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"区间扫描结果：{args.start} → {args.end}（{data_mode}）\n")
        f.write(f"交易日数：{len(trading_dates)}，总结果：{len(all_results)} 只\n")
        f.write(f"{'='*100}\n")
        f.write(f"{'信号日':<12} {'代码':<8} {'名称':<10} {'日期':<12} {'收盘':>8} {'20日涨':>8} {'特殊':>4}\n")
        f.write(f"{'-'*100}\n")

        for td in trading_dates:
            day_results = [r for r in all_results if r["signal_date"] == td]
            if not day_results:
                continue

            # 每日 limitup_results + normal_results 分开，各取 top
            lim = [r for r in day_results if r.get("_limitup") or r.get("limitup")]
            nor = [r for r in day_results if not r.get("_limitup") and not r.get("limitup")]
            if args.top > 0:
                lim = lim[:args.top]
                nor = nor[:args.top]
            day_results = lim + nor

            f.write(f"\n  === {td} 找到 {len(day_results)} 只 ===\n")
            for r in day_results:
                marker = "🔴特殊" if r.get("_limitup") or r.get("limitup") else ""
                gain20 = r.get("gain20", 0)
                close = r.get("close", r.get("close", 0))
                f.write(f"  {td}  {r['code']:<8} {r['name']:<10} "
                        f"{r['date']:<12} {float(close):>8.2f} {float(gain20):>+7.1f}%  {marker}\n")

    print(f"\n✅ 结果已写入：{out_path}")
    print(f"   交易日数：{len(trading_dates)}，总结果：{len(all_results)} 只")

if __name__ == "__main__":
    main()