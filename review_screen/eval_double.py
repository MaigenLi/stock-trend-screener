#!/usr/bin/env python3
"""
eval_double.py — screen_double 信号评估（动态退出版）
===============================================================
用法:
  python eval_double.py --date 2026-04-08
  python eval_double.py --date 2026-04-08 --top-n 300

原理:
  screen_double.py --date T 产生的信号 = T日收盘后选出的股票
  买入 = T+1 日开盘
  卖出 = 动态退出（止损9.5% / 回撤6% / T+10锁利<10% / T+20硬上限）

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
    """解析 screen_double{_mode}_{date}.txt，返回 (code, name, signal_date, signal_close) 列表"""
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


# ── 动态退出参数（与 backtest_dynamic_exit.py 一致）─────────────────
STOP_LOSS_PCT    = 9.5
TRAILING_PCT     = 6.0
MIN_PROFIT_PCT   = 10.0
MAX_HOLD_DAYS    = 20
SL_RATIO   = 1 - STOP_LOSS_PCT   / 100   # 0.905
TRAIL_RATIO = 1 - TRAILING_PCT    / 100   # 0.94

# ── 收益评估（动态退出）────────────────────────────────
def calc_return(code, signal_date, names_cache):
    """
    T+1 开盘买入，动态退出（与 backtest_dynamic_exit.py 一致）：
      条件① 止损 9.5%：当日开盘价或收盘价 <= 买入价 * 0.905 → 以触发价卖出
      条件② 回撤 6%：当日收盘价 < 历史最高收盘价 * 0.94 → 以收盘价卖出
                  （T+2 仅记录最高收盘价，不检查回撤）
      条件③ T+10 锁利：T+10 日收盘时收益 < 10% → 以收盘价卖出
      条件④ T+20 硬上限：T+20 日收盘强制卖出
    返回 dict: code, name, signal_date, buy_date, buy_price,
              exit_date, exit_price, exit_reason, ret, hold_days,
              highest_close, highest_close_date, error
    """
    code = normalize_symbol(code)
    name = get_stock_name(code, names_cache) or code

    # 确定关键日期
    buy_date = next_n_trading_day(signal_date, 1)
    if buy_date is None:
        return dict(code=code, name=name, error="无法确定T+1买入日")

    sell_deadline = next_n_trading_day(signal_date, MAX_HOLD_DAYS)
    if sell_deadline is None:
        all_days = get_trading_days()
        sell_deadline = all_days[-1]

    t10_date = next_n_trading_day(signal_date, 10)
    if t10_date is None:
        all_days = get_trading_days()
        t10_date = all_days[-1]

    # 加载数据
    df = load_qfq_history(code, end_date=sell_deadline)
    if df is None or len(df) < 2:
        return dict(code=code, name=name, error="数据不足")

    df = df.sort_values("date").reset_index(drop=True)
    dates = df["date"].tolist()
    closes = df["close"].values
    opens  = df["open"].values  if "open"  in df.columns else None

    if opens is None:
        return dict(code=code, name=name, error="缺少open字段")

    buy_ts = pd.Timestamp(buy_date)
    try:
        buy_idx = dates.index(buy_ts)
    except ValueError:
        for i, d in enumerate(dates):
            if pd.Timestamp(d) >= buy_ts:
                buy_idx = i
                buy_date = str(d)[:10]
                break
        else:
            return dict(code=code, name=name, error=f"{buy_date}不在数据中")

    buy_price = float(opens[buy_idx])
    if buy_price <= 0:
        return dict(code=code, name=name, error=f"{buy_date}开盘价无效")

    stop_loss_line = buy_price * SL_RATIO
    sell_deadline_ts = pd.Timestamp(sell_deadline)
    t10_ts = pd.Timestamp(t10_date)

    highest_close = None
    highest_close_date = None
    exit_price = None
    exit_date = None
    exit_reason = None

    for i in range(buy_idx + 1, len(dates)):
        day_date = str(dates[i])[:10]
        day_open = float(opens[i])
        day_close = float(closes[i])
        day_dt = pd.Timestamp(dates[i])

        # 条件①：止损 9.5%（优先级最高）
        if day_open <= stop_loss_line:
            exit_price = round(day_open, 2)
            exit_date  = day_date
            exit_reason = f"止损-{STOP_LOSS_PCT}%（开盘）"
            break
        if day_close <= stop_loss_line:
            exit_price = round(day_close, 2)
            exit_date  = day_date
            exit_reason = f"止损-{STOP_LOSS_PCT}%（收盘）"
            break

        # 更新最高收盘价（从 T+2 开始）
        if highest_close is None or day_close > highest_close:
            highest_close = day_close
            highest_close_date = day_date

        # 条件②：回撤 6%（从 T+3 开始）
        if i >= buy_idx + 2 and highest_close is not None:
            if day_close < highest_close * TRAIL_RATIO:
                exit_price = round(day_close, 2)
                exit_date  = day_date
                exit_reason = f"回撤-{TRAILING_PCT}%（高{highest_close:.2f}→{day_close:.2f}）"
                break

        # 条件③：T+10 锁利（仅 T+10 当日）
        if day_date == t10_date:
            ret_t10 = (day_close / buy_price - 1) * 100
            if ret_t10 < MIN_PROFIT_PCT:
                exit_price = round(day_close, 2)
                exit_date  = day_date
                exit_reason = f"T+10锁利（收益{ret_t10:+.1f}%<{MIN_PROFIT_PCT}%）"
                break

        # 条件④：T+20 硬上限
        if day_dt >= sell_deadline_ts:
            exit_price = round(day_close, 2)
            exit_date  = day_date
            exit_reason = "T+20硬上限"
            break

    if exit_price is None:
        last_idx = len(dates) - 1
        exit_price = round(float(closes[last_idx]), 2)
        exit_date  = str(dates[last_idx])[:10]
        exit_reason = "数据末尾强制退出"

    if highest_close is None:
        highest_close = exit_price
        highest_close_date = exit_date

    ret = round((exit_price / buy_price - 1) * 100, 2)
    hold_days = (pd.Timestamp(exit_date) - pd.Timestamp(buy_date)).days

    return dict(
        code=code, name=name,
        signal_date=signal_date,
        buy_date=buy_date, buy_price=round(buy_price, 2),
        exit_date=exit_date, exit_price=exit_price,
        exit_reason=exit_reason,
        ret=ret, hold_days=hold_days,
        highest_close=round(highest_close, 2),
        highest_close_date=highest_close_date,
        error=None,
    )


def _ew(s):
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in str(s))

def _pad(s, w, right=False):
    e = _ew(s)
    return str(s) + " " * max(0, w - e) if right else " " * max(0, w - e) + str(s)


# ── 模式相关文件名工具 ─────────────────────────────
_CURRENT_MODE = "normal"

def _screen_double_filename(date, mode=None):
    m = mode or _CURRENT_MODE
    suffix = f"_{m}" if m != "normal" else ""
    return f"screen_double{suffix}_{date}"


# ── 报告 ────────────────────────────────────────────────
def eval_signal(signal_date: str, top_n: int = 300, mode: str = "normal",
                trend_level: int = 0, force_regenerate: bool = False):
    """
    评估 screen_double_{signal_date}.txt 的收益（动态退出，与 backtest_dynamic_exit.py 一致）。
    信号日 = T
    买入日 = T+1 开盘
    动态退出：止损9.5% / 回撤6% / T+10锁利<10% / T+20硬上限
    """
    global _CURRENT_MODE
    _CURRENT_MODE = mode
    signal_file = OUTPUT_DIR / f"{_screen_double_filename(signal_date, mode)}.txt"

    # 自动生成信号文件（如不存在，或强制重新生成）
    if force_regenerate or not signal_file.exists():
        if force_regenerate and signal_file.exists():
            signal_file.unlink()
        print(f"⚠️  自动运行 screen_double.py --date {signal_date} --mode {mode} --trend-level {trend_level} ...")
        screen_py = WORKSPACE / "stock_trend" / "review_screen" / "screen_double.py"
        cmd = [sys.executable, str(screen_py), "--date", signal_date, "--mode", mode]
        if trend_level > 0:
            cmd += ["--trend-level", str(trend_level)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ screen_double.py 运行失败:\n{result.stderr}")
            return
        print(f"✅ screen_double.py 完成")
        if not signal_file.exists():
            print(f"❌ 仍无信号文件: {signal_file}")
            return

    # 预加载名称缓存（供 calc_return 使用）
    names_cache = load_stock_names()

    signals = parse_screen_output(signal_file)
    if not signals:
        print(f"❌ {signal_file.name} 中未解析到股票")
        return

    print(f"📋 信号文件: {signal_file.name}")
    print(f"   信号日（T）: {signal_date}")
    print(f"   止损={STOP_LOSS_PCT}% | 回撤={TRAILING_PCT}% | T+10锁利≥{MIN_PROFIT_PCT}% | 上限T+{MAX_HOLD_DAYS}")
    if trend_level:
        print(f"   趋势过滤: P{trend_level}")
    print(f"   候选股票: {len(signals)} 只（取前 {min(top_n, len(signals))} 只评估）\n")

    results = []
    errors = []

    for code, name, sig_date, sig_close in signals[:top_n]:
        r = calc_return(code, sig_date, names_cache)
        if r.get("error"):
            errors.append((code, name, r["error"]))
            continue
        r["sig_close"] = sig_close
        results.append(r)

    if not results:
        print("❌ 没有任何有效结果")
        return

    results.sort(key=lambda x: -x["ret"])

    # 统计
    rets = [r["ret"] for r in results]
    win  = [r for r in results if r["ret"] > 0]
    gt20 = [r for r in results if r["ret"] > 20]
    gt10 = [r for r in results if r["ret"] > 10]
    avg  = sum(rets) / len(rets)
    median = float(np.median(rets))

    # 退出原因分布
    reason_counts = {}
    for r in results:
        base = r["exit_reason"].split("（")[0]
        reason_counts[base] = reason_counts.get(base, 0) + 1

    # 表格宽度
    W = [10, 9, 9, 9, 11, 9, 10, 27, 7]
    HDR = " ".join([
        _pad("代码",    W[0], True),
        _pad("名称",    W[1], True),
        _pad("买入价",  W[2]),
        _pad("卖出价",  W[3]),
        _pad("最高价",  W[4]),
        _pad("收益",    W[5]),
        _pad("持(自然日)", W[6]),
        _pad("退出原因",  W[7], True),
        _pad("买入日",  W[8], True),
    ])
    print("═" * (sum(W) + len(W) - 1))
    print(HDR)
    print("─" * (sum(W) + len(W) - 1))
    for r in results:
        tag = "🟢" if r["ret"] > 0 else "🔴"
        row = " ".join([
            _pad(r["code"],           W[0], True),
            _pad(r["name"],           W[1], True),
            _pad(f"{r['buy_price']:.2f}",  W[2]),
            _pad(f"{r['exit_price']:.2f}", W[3]),
            _pad(f"{r['highest_close']:.2f}", W[4]),
            _pad(f"{tag}{r['ret']:+.1f}%",   W[5]),
            _pad(f"{r['hold_days']}d",       W[6]),
            _pad(r["exit_reason"],          W[7], True),
            _pad(r["buy_date"],              W[8], True),
        ])
        print(row)
    print("─" * (sum(W) + len(W) - 1))

    print(f"\n  胜率: {len(win)}/{len(results)} = {len(win)/len(results)*100:.1f}%")
    print(f"  收益率>10%: {len(gt10)}/{len(results)} = {len(gt10)/len(results)*100:.1f}%")
    print(f"  收益率>20%: {len(gt20)}/{len(results)} = {len(gt20)/len(results)*100:.1f}%")
    print(f"  平均收益: {avg:+.2f}%  |  中位收益: {median:+.2f}%  |  标准差: {np.std(rets):.2f}%")
    print(f"  最大盈利: {max(rets):+.2f}%  |  最大亏损: {min(rets):+.2f}%")
    print(f"  平均持有时长: {np.mean([r['hold_days'] for r in results]):.1f} 自然日")
    if errors:
        print(f"\n  数据异常（{len(errors)} 只）: {[e[0] for e in errors[:5]]}")

    if reason_counts:
        print(f"\n  退出原因分布:")
        for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
            pct = count / len(results) * 100
            bar = "█" * int(pct / 2)
            print(f"    {reason:<20} {count:>4}只 ({pct:>5.1f}%) {bar}")

    # 保存结果（供 analyze_double_winners.py 使用）
    out = {
        "signal_date": signal_date,
        "mode": mode,
        "trend_level": trend_level,
        "strategy": {
            "stop_loss_pct": STOP_LOSS_PCT,
            "trailing_stop_pct": TRAILING_PCT,
            "min_profit_pct": MIN_PROFIT_PCT,
            "max_hold_days": MAX_HOLD_DAYS,
        },
        "total_signals": len(signals),
        "evaluated": len(results),
        "win_count": len(win),
        "win_rate": round(len(win)/len(results)*100, 1),
        "avg_return": round(avg, 2),
        "median_return": round(median, 2),
        "std_return": round(float(np.std(rets)), 2),
        "max_return": round(max(rets), 2),
        "min_return": round(min(rets), 2),
        "gt10_count": len(gt10),
        "gt10_rate": round(len(gt10)/len(results)*100, 1),
        "gt20_count": len(gt20),
        "gt20_rate": round(len(gt20)/len(results)*100, 1),
        "avg_hold_days": round(float(np.mean([r['hold_days'] for r in results])), 1),
        "exit_reason_dist": reason_counts,
        "results": results,
    }
    suffix = f"_{_CURRENT_MODE}" if _CURRENT_MODE != "normal" else ""
    json_path = OUTPUT_DIR / f"eval_double{suffix}_{signal_date}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n💾 JSON已保存: {json_path}")
    return out


# ── CLI ────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="评估 screen_double 信号的实际收益（动态退出）")
    parser.add_argument("--date", required=True, help="信号日 T，如 2026-04-08")
    parser.add_argument("--mode", default="normal", choices=["normal", "accelerated", "winner"],
                        help="screen_double 条件3模式（默认normal）")
    parser.add_argument("--trend-level", type=int, default=2,
                        help="趋势过滤严格度（默认2，与screen_double一致）\n"
                             "  0=无额外过滤 1=MA20向上+近20日涨幅>=8%%\n"
                             "  2=P1+近20日涨幅>=12%%+close<MA20*1.10+均线发散度>1%%")
    parser.add_argument("--top-n", type=int, default=300, help="评估前N只（默认300）")
    args = parser.parse_args()
    validate_signal_date(args.date)

    print(f"\n{'='*60}")
    print(f"  eval_double.py  —  收益评估（动态退出）")
    print(f"  信号日: {args.date}  模式: {args.mode}" + (f"  趋势过滤: P{args.trend_level}" if args.trend_level else ""))
    print(f"  止损={STOP_LOSS_PCT}% | 回撤={TRAILING_PCT}% | T+10锁利≥{MIN_PROFIT_PCT}% | 上限T+{MAX_HOLD_DAYS}")
    print(f"{'='*60}\n")

    t0 = time.time()
    # 非默认趋势过滤时强制重新生成信号文件（避免旧缓存冲突）
    force = args.trend_level != 2
    eval_signal(args.date, args.top_n, mode=args.mode,
                trend_level=args.trend_level, force_regenerate=force)
    print(f"\n⏱ 用时 {time.time()-t0:.1f}秒")


if __name__ == "__main__":
    main()
