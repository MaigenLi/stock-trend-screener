#!/usr/bin/env python3
"""
eval_double.py — 评估 screen_double.py 信号的实际收益
=====================================================
用法:
  python eval_double.py --date 2026-04-29
  python eval_double.py --date 2026-04-29 --top-n 100

原理:
  screen_double.py --date D 产生的信号 = 在 D-1 日收盘后选出的股票
  实际买入 = D-1+1 日（下一个交易日）开盘
  实际卖出 = D-1+3 日（持有3天后）收盘

  例如: --date 2026-04-29
    信号日 = 2026-04-24（上一个大交易日）
    买入日 = 2026-04-27（T+1开盘）
    卖出日 = 2026-04-29（T+3收盘）

信号文件: ~/stock_reports/screen_double_{signal_date}.txt
"""

import sys, json, argparse, time, re
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

WORKSPACE = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(WORKSPACE))

from stock_trend.gain_turnover import (
    load_qfq_history,
    get_stock_name,
    load_stock_names,
    normalize_symbol,
)

REPORTS_DIR = Path.home() / "stock_reports"

# ── 交易日历 ─────────────────────────────────────────────
_TRADING_DAYS = None

def get_trading_days():
    """从已有 QFQ CSV 文件构建交易日历（降序）"""
    global _TRADING_DAYS
    if _TRADING_DAYS is not None:
        return _TRADING_DAYS
    cache = WORKSPACE / ".cache" / "qfq_daily"
    dates = set()
    for f in cache.glob("*_qfq.csv"):
        df = pd.read_csv(f, usecols=["date"], nrows=1)
        dates.update(pd.read_csv(f, usecols=["date"])["date"].tolist())
    _TRADING_DAYS = sorted(set(dates))
    return _TRADING_DAYS


def prev_trading_day(date_str: str, n=1) -> str:
    """返回 date_str 之前第 n 个交易日（date_str 本身不算）"""
    days = get_trading_days()
    try:
        idx = days.index(date_str)
    except ValueError:
        # date_str 不在列表，查找最近的前一个交易日
        idx = -1
        for i, d in enumerate(days):
            if d >= date_str:
                idx = i - 1
                break
        if idx < 0:
            idx = 0
    # n=1 → 前一个交易日
    result_idx = max(0, idx - n)
    return days[result_idx]


def next_trading_day(date_str: str, n=1) -> str:
    """返回 date_str 之后第 n 个交易日（不含 date_str 本身）"""
    days = get_trading_days()
    try:
        idx = days.index(date_str)
    except ValueError:
        idx = -1
        for i, d in enumerate(days):
            if d >= date_str:
                idx = i
                break
        if idx < 0:
            idx = len(days) - 1
    result_idx = min(len(days) - 1, idx + n)
    return days[result_idx]


# ── 解析 screen_double 输出 ─────────────────────────────
def parse_screen_output(path: Path):
    """解析 screen_double_{date}.txt，返回 (code, name, signal_date, signal_close) 列表"""
    results = []
    date_pat = re.compile(r"^(\d{4}-\d{2}-\d{2})$")
    code_pat = re.compile(r"^((sh|sz|bj)?)(\d{6})$")

    txt = path.read_text(encoding="utf-8")
    for line in txt.splitlines():
        parts = line.split()  # 空格分隔（中文对齐用空格填充）
        if len(parts) < 12:
            continue
        p0 = parts[0].strip()
        m = code_pat.match(p0.lower())
        if not m:
            continue
        p2 = parts[2].strip()
        if not date_pat.match(p2):
            continue
        try:
            sig_close = float(parts[5].strip())
            if sig_close <= 0:
                continue
        except ValueError:
            continue
        prefix = m.group(1) or ""
        code = f"{prefix}{m.group(3)}"
        name = parts[1].strip()
        sig_date = p2
        results.append((code, name, sig_date, sig_close))
    return results


# ── 收益评估 ─────────────────────────────────────────────
def calc_return(code, buy_date, sell_date):
    """计算持有三天收益：buy_date开盘买入，sell_date收盘卖出"""
    df = load_qfq_history(normalize_symbol(code), end_date=sell_date)
    if df is None or len(df) < 2:
        return None, "数据不足"

    df = df.sort_values("date").reset_index(drop=True)
    dates = df["date"].tolist()
    closes = df["close"].values

    # buy_date/sell_date 是字符串，dates 是 pandas.Timestamp 列表，需要转换
    import pandas as pd
    buy_ts = pd.Timestamp(buy_date)
    sell_ts = pd.Timestamp(sell_date)

    try:
        buy_idx = dates.index(buy_ts)
    except ValueError:
        return None, f"{buy_date}不在数据中"

    # 前复权数据：用当天收盘代替开盘买入（近似）
    buy_price = float(closes[buy_idx])

    try:
        sell_idx = dates.index(sell_ts)
    except ValueError:
        return None, f"{sell_date}不在数据中"
    sell_price = float(closes[sell_idx])

    ret = (sell_price / buy_price - 1) * 100
    return round(ret, 2), None


# ── 报告 ────────────────────────────────────────────────
def eval_signal(sell_date: str, top_n: int = 250):
    """
    评估 screen_double_{signal_date}.txt 的收益。
    sell_date: 用户指定的卖出日期（持有3日后收盘卖出）
    信号日: 卖出日的前3个交易日（T-3，收盘前筛选产生信号）
    买入日: 信号日的下一个交易日（T+1开盘）
    """
    # 信号日 = 卖出日的前3个交易日
    signal_date = prev_trading_day(sell_date, n=3)
    # 买入日 = 信号日的下一个交易日（T+1开盘）
    buy_date = next_trading_day(signal_date, 1)

    signal_file = REPORTS_DIR / f"screen_double_{signal_date}.txt"

    if not signal_file.exists():
        print(f"❌ 信号文件不存在: {signal_file}")
        print(f"   请先运行: screen_double.py --date {signal_date}")
        return

    signals = parse_screen_output(signal_file)
    if not signals:
        print(f"❌ {signal_file.name} 中未解析到股票")
        return

    print(f"📋 信号文件: {signal_file.name}")
    print(f"   信号日（T-3）: {signal_date}  卖出日: {sell_date}")
    print(f"   买入日（T+1）: {buy_date}（开盘）  持有3日后卖出: {sell_date}（收盘）")
    print(f"   候选股票: {len(signals)} 只（取前 {min(top_n, len(signals))} 只评估）\n")

    names = load_stock_names()
    results = []
    errors = []

    for i, (code, name, sig_date, sig_close) in enumerate(signals[:top_n]):
        ret, err = calc_return(code, buy_date, sell_date)
        if err:
            errors.append((code, name, err))
            continue

        # 信号日收盘价（参考）
        sig_ret = round((ret / 100 + 1) * sig_close - sig_close, 2)

        results.append({
            "code": code,
            "name": name,
            "signal_date": sig_date,
            "sig_close": sig_close,
            "buy_date": buy_date,
            "sell_date": sell_date,
            "ret": ret,
            "sig_ret": sig_ret,
        })

    if not results:
        print("❌ 没有任何有效结果")
        return

    results.sort(key=lambda x: -x["ret"])

    # 统计
    rets = [r["ret"] for r in results]
    win = [r for r in results if r["ret"] > 0]
    loss = [r for r in results if r["ret"] <= 0]
    avg = sum(rets) / len(rets)

    print(f"{'='*110}")
    print(f"{'代码':<10}{'名称':<8}{'信号日':<12}{'信号价':>8} "
          f"{'买入日':<12}{'卖出日':<12}{'持有3日收益':>12}  {'信号参考收益':>12}")
    print(f"{'-'*110}")
    for r in results:
        tag = "🟢" if r["ret"] > 0 else "🔴"
        print(f"{r['code']:<10}{r['name']:<8}{r['signal_date']:<12}"
              f"{r['sig_close']:>8.2f} {r['buy_date']:<12}{r['sell_date']:<12}"
              f"{tag}{r['ret']:>+7.2f}%   {r['sig_ret']:>+9.2f}%")
    print(f"{'='*110}")

    print(f"\n📊 评估汇总（{len(results)} 只 / {len(signals)} 只）")
    print(f"   胜率: {len(win)}/{len(results)} = {len(win)/len(results)*100:.1f}%")
    print(f"   平均收益: {avg:+.2f}%")
    print(f"   最大盈利: {max(rets):+.2f}%  ({[r['code'] for r in results if r['ret']==max(rets)][0]})")
    print(f"   最大亏损: {min(rets):+.2f}%  ({[r['code'] for r in results if r['ret']==min(rets)][0]})")
    if errors:
        print(f"\n   数据异常（{len(errors)} 只）: {[r[0] for r in errors[:5]]}")

    # 保存结果
    out = {
        "signal_date": signal_date,
        "buy_date": buy_date,
        "sell_date": sell_date,
        "total_signals": len(signals),
        "evaluated": len(results),
        "win_rate": round(len(win)/len(results)*100, 1),
        "avg_return": round(avg, 2),
        "results": results,
    }
    json_path = REPORTS_DIR / f"eval_double_{signal_date}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n💾 JSON已保存: {json_path}")


# ── CLI ────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="评估 screen_double.py 信号的实际持有收益")
    parser.add_argument("--date", required=True, help="卖出日期（持有3日后的收盘日），如 2026-04-29")
    parser.add_argument("--top-n", type=int, default=250, help="评估前N只（默认250）")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  eval_double.py  —  收益评估")
    print(f"  评估日期: {args.date}")
    print(f"{'='*60}\n")

    t0 = time.time()
    eval_signal(args.date, args.top_n)
    print(f"\n⏱ 用时 {time.time()-t0:.1f}秒")


if __name__ == "__main__":
    main()
