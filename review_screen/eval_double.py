#!/usr/bin/env python3
"""
eval_double.py — screen_double 信号评估
===============================================================
用法:
  python eval_double.py --date 2026-04-08              # 默认持有10天
  python eval_double.py --date 2026-04-08 --hold-days 5
  python eval_double.py --date 2026-04-08 --top-n 300

原理:
  screen_double.py --date T 产生的信号 = T日收盘后选出的股票
  实际买入 = T+1 日开盘
  实际卖出 = T+hold_days 日收盘
  持有交易日数 = hold_days - 1

信号文件: ./output/screen_double_{signal_date}.txt
"""

import sys, json, argparse, time, re, unicodedata, subprocess
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

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

# ── 交易日历 ─────────────────────────────────────────────
from stock_trend.review_screen.date_utils import validate_signal_date, get_trading_days

def next_n_trading_day(date_str: str, n: int) -> str:
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
        parts = line.split()
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


# ── 收益评估 ───────────────────────────────
def calc_return(code, buy_date, sell_date):
    """计算持有收益：buy_date开盘买入，sell_date收盘卖出"""
    df = load_qfq_history(normalize_symbol(code), end_date=sell_date)
    if df is None or len(df) < 2:
        return None, None, None, "数据不足"

    df = df.sort_values("date").reset_index(drop=True)
    dates = df["date"].tolist()
    closes = df["close"].values
    opens = df["open"].values if "open" in df.columns else None

    buy_ts = pd.Timestamp(buy_date)
    sell_ts = pd.Timestamp(sell_date)

    try:
        buy_idx = dates.index(buy_ts)
    except ValueError:
        return None, None, None, f"{buy_date}不在数据中"

    if opens is None:
        return None, None, None, "缺少open字段"
    buy_price = float(opens[buy_idx])
    if buy_price <= 0:
        return None, None, None, f"{buy_date}开盘价无效"

    try:
        sell_idx = dates.index(sell_ts)
    except ValueError:
        return None, None, None, f"{sell_date}不在数据中"
    sell_price = float(closes[sell_idx])
    if sell_price <= 0:
        return None, None, None, f"{sell_date}收盘价无效"

    ret = (sell_price / buy_price - 1) * 100
    return round(ret, 2), round(buy_price, 2), round(sell_price, 2), None


# ── 报告 ────────────────────────────────────────────────
def eval_signal(signal_date: str, top_n: int = 300, hold_days: int = 10):
    """
    评估 screen_double_{signal_date}.txt 的收益。
    信号日 = T（screen_double --date 的日期）
    买入日 = T+1（开盘）
    卖出日 = T+hold_days（收盘，持有 hold_days-1 个交易日）
    """
    signal_file = OUTPUT_DIR / f"screen_double_{signal_date}.txt"

    if not signal_file.exists():
        print(f"⚠️  信号文件不存在: {signal_file}")
        print(f"   正在自动运行 screen_double.py --date {signal_date} ...")
        screen_py = WORKSPACE / "stock_trend" / "review_screen" / "screen_double.py"
        result = subprocess.run(
            [sys.executable, str(screen_py), "--date", signal_date],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"❌ screen_double.py 运行失败:\n{result.stderr}")
            return
        print(f"✅ screen_double.py 完成")
        if not signal_file.exists():
            print(f"❌ 仍无信号文件: {signal_file}")
            return

    buy_date = next_n_trading_day(signal_date, 1)
    sell_date = next_n_trading_day(signal_date, hold_days)

    signals = parse_screen_output(signal_file)
    if not signals:
        print(f"❌ {signal_file.name} 中未解析到股票")
        return

    hold_days_label = f"持有{hold_days - 1}交易日"
    sell_col_label = f"T+{hold_days}收"

    print(f"📋 信号文件: {signal_file.name}")
    print(f"   信号日（T）: {signal_date}  买入日（T+1开盘）: {buy_date}  卖出日（T+{hold_days}收盘）: {sell_date}")
    print(f"   候选股票: {len(signals)} 只（取前 {min(top_n, len(signals))} 只评估）\n")

    results = []
    errors = []

    for i, (code, name, sig_date, sig_close) in enumerate(signals[:top_n]):
        ret, buy_price, sell_price, err = calc_return(code, buy_date, sell_date)
        if err:
            errors.append((code, name, err))
            continue

        results.append({
            "code": code,
            "name": name,
            "signal_date": sig_date,
            "sig_close": sig_close,
            "buy_date": buy_date,
            "sell_date": sell_date,
            "buy_open": buy_price,
            "sell_close": sell_price,
            "ret": ret,
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
    gt20 = [r for r in results if r["ret"] > 20]
    gt10 = [r for r in results if r["ret"] > 10]
    gt0  = [r for r in results if r["ret"] > 0]

    # 每列宽度（手工指定，确保标题和数据完全对齐）
    # 中文标签按双字符宽度换算：label_cells = len(label) * 2
    W = [9, 10, 10, 9, 10, 9, 10, 9, 10]   # 代码 名称 信号日 信号价 买入日 买开 卖出日 卖收 T+10收 收益
    def pad(v, w, right=False):
        ev = sum(2 if unicodedata.east_asian_width(c) in ("W","F") else 1 for c in str(v))
        return " " * max(0, w - ev) + str(v) if not right else str(v) + " " * max(0, w - ev)

    def rpad(v, w):
        return pad(v, w, right=True)

    HDR = " ".join([
        rpad("   代码", W[0]), rpad("  名称", W[1]), rpad("信号日", W[2]),
        rpad("信号价", W[3]), rpad("买入日", W[4]), rpad("T+1开盘", W[5]),
        rpad("卖出日", W[6]), rpad(sell_col_label, W[7]), rpad("收益", W[8]),
    ])
    print("═" * (sum(W) + len(W) - 1))
    print(HDR)
    print("─" * (sum(W) + len(W) - 1))
    for r in results:
        tag = "\U0001F7E2" if r["ret"] > 0 else "\U0001f534"
        row = " ".join([
            pad(r["code"], W[0]), pad(r["name"], W[1]), pad(r["signal_date"], W[2]),
            rpad(r["sig_close"], W[3]), pad(r["buy_date"], W[4]), rpad(r["buy_open"], W[5]),
            pad(r["sell_date"], W[6]), rpad(r["sell_close"], W[7]),
            rpad(f"{tag}{r['ret']:>+7.2f}%", W[8]),
        ])
        print(row)
    print("─" * (sum(W) + len(W) - 1))
    print(f"{'='*94}")

    print(f"\n📊 评估汇总（{len(results)} 只 / {len(signals)} 只）")
    print(f"   胜率: {len(gt0)}/{len(results)} = {len(gt0)/len(results)*100:.1f}%")
    print(f"   收益率>10%: {len(gt10)}/{len(results)} = {len(gt10)/len(results)*100:.1f}%")
    print(f"   收益率>20%: {len(gt20)}/{len(results)} = {len(gt20)/len(results)*100:.1f}%")
    print(f"   平均收益: {avg:+.2f}%")
    print(f"   最大盈利: {max(rets):+.2f}%  ({[r['code'] for r in results if r['ret']==max(rets)][0]})")
    print(f"   最大亏损: {min(rets):+.2f}%  ({[r['code'] for r in results if r['ret']==min(rets)][0]})")
    if errors:
        print(f"\n   数据异常（{len(errors)} 只）: {[r[0] for r in errors[:5]]}")

    # 保存结果（供 analyze_winners.py 使用）
    out = {
        "signal_date": signal_date,
        "buy_date": buy_date,
        "sell_date": sell_date,
        "hold_days": hold_days,
        "total_signals": len(signals),
        "evaluated": len(results),
        "win_rate": round(len(gt0)/len(results)*100, 1),
        "avg_return": round(avg, 2),
        "results": results,
    }
    json_path = OUTPUT_DIR / f"eval_double_{signal_date}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n💾 JSON已保存: {json_path}")
    return out


# ── CLI ────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="评估 screen_double.py 信号的实际持有收益")
    parser.add_argument("--date", required=True, help="信号日 T（screen_double --date 的日期），如 2026-04-08")
    parser.add_argument("--hold-days", type=int, default=10, help="持有交易日数（默认10，即T+10收盘卖出）")
    parser.add_argument("--top-n", type=int, default=300, help="评估前N只（默认300）")
    args = parser.parse_args()
    validate_signal_date(args.date)

    print(f"\n{'='*60}")
    print(f"  eval_double.py  —  收益评估（持有{args.hold_days - 1}交易日）")
    print(f"  信号日: {args.date}  hold_days: {args.hold_days}")
    print(f"{'='*60}\n")

    t0 = time.time()
    eval_signal(args.date, args.top_n, args.hold_days)
    print(f"\n⏱ 用时 {time.time()-t0:.1f}秒")


if __name__ == "__main__":
    main()
