#!/usr/bin/env python3
"""
backtest_dynamic_exit.py — 动态退出回测（基于 screen_double 信号）
=================================================================
买入规则：
  T日 = 信号日（screen_double 输出日期）
  T+1日 开盘价买入

卖出规则（动态退出，不固定持有天数）：
  逐日检查（优先级递减），满足任一条件即卖出：
    条件① 止损9.5%：当日开盘价或收盘价 ≤ 买入价 × 0.905 → 以触发价格卖出
    条件② 回撤6%：当日收盘价 < 历史最高收盘价 × 0.94 → 以收盘价卖出
                  （T+2仅记录最高收盘价，不检查回撤）
    条件③ T+10锁利：T+10日收盘时，收益 < 10% → 以收盘价卖出
    条件④ T+20硬上限：T+20日收盘强制卖出（无附加条件）

用法：
  # 单日评估
  python backtest_dynamic_exit.py --date 2026-04-08

  # 批量回测（读取已有 eval JSON）
  python backtest_dynamic_exit.py --start 2026-03-01 --end 2026-04-30

  # 批量回测（重新计算）
  python backtest_dynamic_exit.py --start 2026-03-01 --end 2026-04-30 --run

  # 指定 screen_double 参数
  python backtest_dynamic_exit.py --start 2026-03-01 --end 2026-04-30 --run --mode winner --gain20 10
"""
import sys
import json
import time
import argparse
import unicodedata
import subprocess
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

# ── 路径设置 ─────────────────────────────────────────────
WORKSPACE = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(WORKSPACE / "stock_trend" / "review_screen"))

from stock_trend.gain_turnover import load_qfq_history, normalize_symbol, get_stock_name, load_stock_names
from stock_trend.review_screen.date_utils import (
    get_trading_days,
    _is_trading_day,
)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

# ── 全局状态（由 main/run_batch 设置）─────────────────
_CURRENT_MODE = "normal"


def _screen_double_filename(date, mode=None):
    """返回 screen_double 的输出文件 stem（不含 .txt/.json）"""
    m = mode or _CURRENT_MODE
    suffix = f"_{m}" if m != "normal" else ""
    return f"screen_double{suffix}_{date}"


def _dynamic_exit_filename(date, mode=None):
    """返回动态退出评估的 JSON 文件名"""
    m = mode or _CURRENT_MODE
    suffix = f"_{m}" if m != "normal" else ""
    return f"dynamic_exit{suffix}_{date}"

# ── 策略参数（可CLI覆盖）─────────────────────────────────
STOP_LOSS_PCT = 8          # 止损线（%）
TRAILING_STOP_PCT = 6.0      # 回撤线（%）
MIN_PROFIT_PCT = 10.0        # T+10最低收益（%）
MAX_HOLD_DAYS = 20           # 最大持有交易日数
STOP_LOSS_RATIO = 1 - STOP_LOSS_PCT / 100      # 0.905
TRAILING_RATIO = 1 - TRAILING_STOP_PCT / 100     # 0.94


# ── 交易日历 ─────────────────────────────────────────────
def next_n_trading_day(date_str: str, n: int) -> str | None:
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
            return None
    result_idx = idx + n
    return days[result_idx] if result_idx < len(days) else None


def prev_trading_day(date_str: str) -> str | None:
    """返回 date_str 之前最近的交易日"""
    days = get_trading_days()
    for d in reversed(days):
        if d < date_str:
            return d
    return None


# ── 解析 screen_double 输出 ─────────────────────────────
def parse_screen_output(path: Path):
    """解析 screen_double_{date}.txt，(code, name, signal_date, signal_close) 列表"""
    import re
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


# ── 核心：动态退出评估 ──────────────────────────────────
def evaluate_dynamic_exit(code, signal_date, names_cache, end_date=None):
    """
    动态退出逻辑评估单只股票。

    参数:
      code: 股票代码
      signal_date: T日
      end_date: 数据截止日（默认 None = 使用最新数据）

    返回:
      dict with:
        code, name, signal_date,
        buy_date, buy_price,
        exit_date, exit_price, exit_reason,
        ret, hold_days,
        highest_close, highest_close_date,
        error
    """
    code = normalize_symbol(code)
    name = get_stock_name(code, names_cache) or code

    # 1. 确定买卖日期
    buy_date = next_n_trading_day(signal_date, 1)
    if buy_date is None:
        return {"code": code, "name": name, "error": "无法确定T+1买入日"}

    sell_deadline = next_n_trading_day(signal_date, MAX_HOLD_DAYS)
    if sell_deadline is None:
        # T+20 超出数据范围，使用最后一个交易日
        all_days = get_trading_days()
        sell_deadline = all_days[-1]

    t10_date = next_n_trading_day(signal_date, 10)
    if t10_date is None:
        # T+10 超出数据范围，使用最近可用的交易日
        all_days = get_trading_days()
        t10_date = all_days[-1]

    # 2. 加载数据（延伸到T+20之后，确保数据完整）
    data_end = end_date or sell_deadline
    df = load_qfq_history(code, end_date=data_end)
    if df is None or len(df) < 2:
        return {"code": code, "name": name, "error": "数据不足"}

    df = df.sort_values("date").reset_index(drop=True)
    dates = df["date"].tolist()
    closes = df["close"].values
    opens = df["open"].values if "open" in df.columns else None

    if opens is None:
        return {"code": code, "name": name, "error": "缺少open字段"}

    buy_ts = pd.Timestamp(buy_date)
    try:
        buy_idx = dates.index(buy_ts)
    except ValueError:
        # 尝试找下一个交易日
        for i, d in enumerate(dates):
            if pd.Timestamp(d) >= buy_ts:
                buy_idx = i
                buy_date = str(d)[:10]
                break
        else:
            return {"code": code, "name": name, "error": f"{buy_date}不在数据中"}

    buy_price = float(opens[buy_idx])
    if buy_price <= 0:
        return {"code": code, "name": name, "error": f"{buy_date}开盘价无效"}

    # 3. 逐日遍历，动态检查退出条件
    # i = buy_idx       → T+1日（买入日，不参与卖出）
    # i = buy_idx + 1   → T+2日（第一个可卖出日，仅检查止损+记录最高价）
    # i = buy_idx + 2   → T+3日（可检查回撤）
    # ...
    # i = buy_idx + 9   → T+10日（锁利检查）
    # i = buy_idx + 19  → T+20日（硬上限）
    sell_deadline_ts = pd.Timestamp(sell_deadline)
    t10_ts = pd.Timestamp(t10_date)
    stop_loss_line = buy_price * STOP_LOSS_RATIO

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

        # ── 条件①：止损 9.5%（优先级最高，每天检查）───
        # 先检查开盘价
        if day_open <= stop_loss_line:
            exit_price = round(day_open, 2)
            exit_date = day_date
            exit_reason = f"止损-{STOP_LOSS_PCT}%(开盘)"
            break
        # 再检查收盘价
        if day_close <= stop_loss_line:
            exit_price = round(day_close, 2)
            exit_date = day_date
            exit_reason = f"止损-{STOP_LOSS_PCT}%(收盘)"
            break

        # ── 更新最高收盘价（从 T+2 开始记录）─────────
        if highest_close is None or day_close > highest_close:
            highest_close = day_close
            highest_close_date = day_date

        # ── 条件②：回撤 6%（从 T+3 开始检查）─────────
        if i >= buy_idx + 2 and highest_close is not None:
            # T+2日 = buy_idx + 1, T+3日 = buy_idx + 2
            if day_close < highest_close * TRAILING_RATIO:
                exit_price = round(day_close, 2)
                exit_date = day_date
                exit_reason = f"回撤-{TRAILING_STOP_PCT}%(高{highest_close:.2f}→{day_close:.2f})"
                break

        # ── 条件③：T+10 锁利（仅 T+10 当日检查）─────
        # 注：用 t10_date 字符串精确匹配，而非 >= t10_ts，避免 T+11 仍触发
        if day_date == t10_date:
            ret_t10 = (day_close / buy_price - 1) * 100
            if ret_t10 < MIN_PROFIT_PCT:
                exit_price = round(day_close, 2)
                exit_date = day_date
                exit_reason = f"T+10锁利(收益{ret_t10:+.1f}%<{MIN_PROFIT_PCT}%)"
                break

        # ── 条件④：T+20 硬上限 ──────────────────────
        if day_dt >= sell_deadline_ts:
            exit_price = round(day_close, 2)
            exit_date = day_date
            exit_reason = "T+20硬上限"
            break

    # 4. 如果遍历完都没触发（数据不足），标记为未退出
    if exit_price is None:
        # 最后一天收盘强制退出
        last_idx = len(dates) - 1
        exit_price = round(float(closes[last_idx]), 2)
        exit_date = str(dates[last_idx])[:10]
        exit_reason = "强制退出"

    if highest_close is None:
        highest_close = exit_price
        highest_close_date = exit_date

    ret = round((exit_price / buy_price - 1) * 100, 2)
    hold_days = 0
    if buy_date and exit_date:
        # 计算自然日持有天数
        hold_days = (pd.Timestamp(exit_date) - pd.Timestamp(buy_date)).days

    return {
        "code": code,
        "name": name,
        "signal_date": signal_date,
        "buy_date": buy_date,
        "buy_price": buy_price,
        "exit_date": exit_date,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "ret": ret,
        "hold_days": hold_days,
        "highest_close": round(highest_close, 2),
        "highest_close_date": highest_close_date,
        "error": None,
    }


# ── 单日评估 ────────────────────────────────────────────
def eval_single_date(signal_date: str, top_n: int = 300,
                     mode: str = "normal", gain20: float = None,
                     trend_level: int = 0, names_cache=None,
                     force_regenerate: bool = False):
    """
    对单日信号运行动态退出评估。
    返回结果 dict，也保存 JSON。
    """
    signal_file = OUTPUT_DIR / f"{_screen_double_filename(signal_date, mode)}.txt"

    if names_cache is None:
        names_cache = load_stock_names()

    # 自动生成信号文件（如不存在，或强制重新生成）
    regenerate = force_regenerate or not signal_file.exists()
    if regenerate:
        if force_regenerate and signal_file.exists():
            signal_file.unlink()  # 删除旧文件，确保用新参数重新生成
        # winner 模式：调用 screen_double_winner.py（独立脚本，硬编码 WINNER+P2）
        # 其他模式：调用 screen_double.py（--mode 参数）
        if mode == "winner":
            print(f"   ⚠️  自动运行 screen_double_winner.py --date {signal_date} ...")
            screen_py = WORKSPACE / "stock_trend" / "review_screen" / "screen_double_winner.py"
            cmd = [sys.executable, str(screen_py), "--date", signal_date]
        else:
            print(f"   ⚠️  自动运行 screen_double.py --date {signal_date} ...")
            screen_py = WORKSPACE / "stock_trend" / "review_screen" / "screen_double.py"
            cmd = [sys.executable, str(screen_py), "--date", signal_date]
            if mode:
                cmd += ["--mode", mode]
            if gain20 is not None:
                cmd += ["--gain20", str(gain20)]
            if trend_level is not None and trend_level > 0:
                cmd += ["--trend-level", str(trend_level)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"   ❌ {screen_py.name} 运行失败: {result.stderr[-300:]}")
            return None
        print(f"   ✅ {screen_py.name} 完成")
        if not signal_file.exists():
            print(f"   ❌ 信号文件仍未生成")
            return None

    signals = parse_screen_output(signal_file)
    if not signals:
        print(f"   ❌ {signal_file.name} 中未解析到有效信号")
        return None

    print(f"   📋 {len(signals)} 只候选，评估前 {min(top_n, len(signals))} 只...")

    results = []
    errors = []

    for code, name, sig_date, sig_close in signals[:top_n]:
        r = evaluate_dynamic_exit(code, signal_date, names_cache)
        if r.get("error"):
            errors.append((code, name, r["error"]))
            continue
        results.append(r)

    if not results:
        print("   ❌ 无有效评估结果")
        return None

    # 统计
    rets = [r["ret"] for r in results]
    win = [r for r in results if r["ret"] > 0]
    gt20 = [r for r in results if r["ret"] > 20]
    gt10 = [r for r in results if r["ret"] > 10]
    avg = sum(rets) / len(rets)
    median = float(np.median(rets))

    # 退出原因分布
    reason_counts = defaultdict(int)
    for r in results:
        reason_counts[r["exit_reason"].split("(")[0]] += 1

    out = {
        "signal_date": signal_date,
        "total_signals": len(signals),
        "evaluated": len(results),
        "errors": len(errors),
        "win_count": len(win),
        "win_rate": round(len(win) / len(results) * 100, 1),
        "avg_return": round(avg, 2),
        "median_return": round(median, 2),
        "std_return": round(float(np.std(rets)), 2),
        "max_return": round(max(rets), 2),
        "min_return": round(min(rets), 2),
        "gt20_count": len(gt20),
        "gt20_rate": round(len(gt20) / len(results) * 100, 1),
        "gt10_count": len(gt10),
        "gt10_rate": round(len(gt10) / len(results) * 100, 1),
        "avg_hold_days": round(float(np.mean([r["hold_days"] for r in results])), 1),
        "exit_reason_dist": dict(reason_counts),
        "results": results,
    }

    # 保存 JSON
    json_path = OUTPUT_DIR / f"{_dynamic_exit_filename(signal_date, mode)}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    return out


# ── 表格输出 ────────────────────────────────────────────
def _ew(s):
    return sum(2 if unicodedata.east_asian_width(c) in ("W", "F") else 1 for c in str(s))


def _pad(s, w, right=False):
    e = _ew(s)
    return str(s) + " " * max(0, w - e) if right else " " * max(0, w - e) + str(s)


def print_single_date_report(out):
    """打印单日评估报告"""
    results = out["results"]
    rets = [r["ret"] for r in results]

    print(f"\n{'='*95}")
    print(f"  📊 动态退出评估  —  {out['signal_date']}")
    print(f"  买入: T+1开盘  |  止损: -{STOP_LOSS_PCT}%  |  回撤: -{TRAILING_STOP_PCT}%  |  T+10锁利: ≥{MIN_PROFIT_PCT}%  |  上限: T+{MAX_HOLD_DAYS}")
    print(f"  候选: {out['total_signals']}只  |  评估: {out['evaluated']}只  |  异常: {out['errors']}只")
    print(f"{'='*95}")

    # 顶栏
    W = [10, 9, 9, 9, 11, 9, 10, 27, 7]
    HDR = " ".join([
        _pad("代码", W[0], True), _pad("名称", W[1], True),
        _pad("买入价", W[2]), _pad("卖出价", W[3]),
        _pad("最高价", W[4]), _pad("收益", W[5]),
        _pad("持(自然日)", W[6]), _pad("退出原因", W[7], True),
        _pad("买入日", W[8], True),
    ])
    print(HDR)
    print("─" * (sum(W) + len(W) - 1))

    results_sorted = sorted(results, key=lambda x: -x["ret"])
    for r in results_sorted:
        tag = "🟢" if r["ret"] > 0 else "🔴"
        row = " ".join([
            _pad(r["code"], W[0], True),
            _pad(r["name"], W[1], True),
            _pad(f"{r['buy_price']:.2f}", W[2]),
            _pad(f"{r['exit_price']:.2f}", W[3]),
            _pad(f"{r['highest_close']:.2f}", W[4]),
            _pad(f"{tag}{r['ret']:+.1f}%", W[5]),
            _pad(f"{r['hold_days']}d", W[6]),
            _pad(r["exit_reason"], W[7], True),
            _pad(r["buy_date"], W[8], True),
        ])
        print(row)
    print("─" * (sum(W) + len(W) - 1))

    print(f"\n  胜率: {out['win_count']}/{out['evaluated']} = {out['win_rate']}%")
    print(f"  收益率>10%: {out['gt10_count']}/{out['evaluated']} = {out['gt10_rate']}%")
    print(f"  收益率>20%: {out['gt20_count']}/{out['evaluated']} = {out['gt20_rate']}%")
    print(f"  平均收益: {out['avg_return']:+.2f}%  |  中位收益: {out['median_return']:+.2f}%  |  标准差: {out['std_return']:.2f}%")
    print(f"  最大盈利: {out['max_return']:+.2f}%  |  最大亏损: {out['min_return']:+.2f}%")
    print(f"  平均持有时长: {out['avg_hold_days']} 自然日")

    # 退出原因分布
    if out.get("exit_reason_dist"):
        print(f"\n  退出原因分布:")
        for reason, count in sorted(out["exit_reason_dist"].items(), key=lambda x: -x[1]):
            pct = count / out["evaluated"] * 100
            bar = "█" * int(pct / 2)
            print(f"    {reason:<20} {count:>4}只 ({pct:>5.1f}%) {bar}")


# ── 批量回测 ────────────────────────────────────────────
def get_available_signal_dates(start: str, end: str, mode: str = "normal"):
    """返回区间内所有已有 screen_double_{mode?}_{date}.txt 的日期"""
    prefix = f"screen_double_{mode}_" if mode != "normal" else "screen_double_"
    dates = []
    for f in OUTPUT_DIR.glob(f"{prefix}*.txt"):
        stem = f.stem
        d = stem.replace(prefix, "")
        if start <= d <= end:
            dates.append(d)
    return sorted(dates)


def run_batch(args):
    global _CURRENT_MODE
    t0 = time.time()

    mode = getattr(args, "mode", "normal")
    _CURRENT_MODE = mode
    gain20 = getattr(args, "gain20", None)
    trend_level = getattr(args, "trend_level", 0)

    # 1. 确定日期列表
    if args.dates:
        dates = sorted(set(args.dates))
    elif args.start and args.end:
        if args.run:
            all_trading = [d for d in get_trading_days() if args.start <= d <= args.end]
            dates = all_trading
        else:
            dates = get_available_signal_dates(args.start, args.end, mode=mode)
    else:
        print("❌ 请指定 --start/--end 或 --dates")
        return

    # 过滤非交易日
    invalid = [d for d in dates if not _is_trading_day(d)]
    if invalid:
        for d in invalid:
            print(f"⚠️  {d} 非交易日，已跳过")
        dates = [d for d in dates if _is_trading_day(d)]
        if not dates:
            print("❌ 所有日期均非交易日")
            return

    print(f"\n{'='*70}")
    print(f"  backtest_dynamic_exit.py  —  批量动态退出回测")
    print(f"  日期范围: {dates[0]} → {dates[-1]}（共 {len(dates)} 个信号日）")
    print(f"  止损={STOP_LOSS_PCT}% | 回撤={TRAILING_STOP_PCT}% | T+10锁利≥{MIN_PROFIT_PCT}% | 上限T+{MAX_HOLD_DAYS}")
    print(f"{'='*70}\n")

    all_entries = []
    errors = []

    for i, date in enumerate(dates, 1):
        print(f"[{i}/{len(dates)}] {date} ", end="", flush=True)
        out = eval_single_date(date, top_n=args.top_n, mode=mode,
                                 gain20=gain20, trend_level=trend_level,
                                 force_regenerate=args.run)
        if out is None:
            print("❌ 失败")
            errors.append(date)
            continue
        print(f"✅ {out['evaluated']}只  win={out['win_rate']}%  avg={out['avg_return']:+.1f}%")
        all_entries.append(out)

    if not all_entries:
        print("❌ 无任何有效结果")
        return

    # 2. 全局汇总
    all_results = []
    for entry in all_entries:
        all_results.extend(entry["results"])

    rets = [r["ret"] for r in all_results]
    win = [r for r in all_results if r["ret"] > 0]
    gt20 = [r for r in all_results if r["ret"] > 20]
    gt10 = [r for r in all_results if r["ret"] > 10]

    # 退出原因全局分布
    reason_ret = defaultdict(list)
    for r in all_results:
        base_reason = r["exit_reason"].split("(")[0]
        reason_ret[base_reason].append(r["ret"])

    print(f"\n{'='*70}")
    print(f"📊 批量回测汇总（{len(dates)} 个信号日）")
    print(f"{'='*70}")
    print(f"   总交易: {len(all_results)} 次")
    print(f"   胜率: {len(win)}/{len(all_results)} = {len(win)/len(all_results)*100:.1f}%")
    print(f"   >10%: {len(gt10)}/{len(all_results)} = {len(gt10)/len(all_results)*100:.1f}%")
    print(f"   >20%: {len(gt20)}/{len(all_results)} = {len(gt20)/len(all_results)*100:.1f}%")
    print(f"   平均收益: {np.mean(rets):+.2f}%")
    print(f"   中位收益: {np.median(rets):+.2f}%")
    print(f"   标准差: {np.std(rets):.2f}%")
    print(f"   最大盈利: {max(rets):+.2f}%" + (f" ({max(all_results, key=lambda x: x['ret'])['code']})" if all_results else ""))
    print(f"   最大亏损: {min(rets):+.2f}%" + (f" ({min(all_results, key=lambda x: x['ret'])['code']})" if all_results else ""))
    print(f"   平均持有时长: {np.mean([r['hold_days'] for r in all_results]):.1f} 自然日")

    # 退出原因统计
    print(f"\n📊 退出原因全局分布")
    print(f"{'原因':<22}{'次数':>5}{'占比':>6}{'平均收益':>10}{'胜率':>8}")
    print("-" * 56)
    for reason in sorted(reason_ret.keys(), key=lambda k: -len(reason_ret[k])):
        rets_r = reason_ret[reason]
        cnt = len(rets_r)
        pct = cnt / len(all_results) * 100
        avg_r = np.mean(rets_r)
        win_r = sum(1 for r in rets_r if r > 0) / cnt * 100
        print(f"{reason:<24}{cnt:>6}{pct:>7.1f}%{avg_r:>+10.2f}%{win_r:>7.1f}%")

    # 3. 每日明细
    print(f"\n📅 每日收益明细")
    print(f"{'日期':<10}{'交易':>4}{'胜率':>4}{'平均':>5}{'中位':>7}{'最大':>7}{'最小':>10}{'>10%':>6}{'均持(天)':>10}")
    print("-" * 82)
    for entry in all_entries:
        r = entry["results"]
        rrets = [x["ret"] for x in r]
        holds = [x["hold_days"] for x in r]
        print(f"{entry['signal_date']:<10}"
              f"{len(r):>6}"
              f"{entry['win_rate']:>7.1f}%"
              f"{np.mean(rrets):>+8.2f}%"
              f"{np.median(rrets):>+8.2f}%"
              f"{max(rrets):>+8.2f}%"
              f"{min(rrets):>+8.2f}%"
              f"{entry['gt10_rate']:>5.1f}%"
              f"{np.mean(holds):>9.1f}")

    # 4. 保存汇总
    summary = {
        "args": vars(args),
        "strategy": {
            "stop_loss_pct": STOP_LOSS_PCT,
            "trailing_stop_pct": TRAILING_STOP_PCT,
            "min_profit_pct": MIN_PROFIT_PCT,
            "max_hold_days": MAX_HOLD_DAYS,
        },
        "stats": {
            "total_trades": len(all_results),
            "total_dates": len(all_entries),
            "win_rate": round(len(win) / len(all_results) * 100, 1),
            "avg_return": round(float(np.mean(rets)), 2),
            "median_return": round(float(np.median(rets)), 2),
            "std_return": round(float(np.std(rets)), 2),
            "max_return": round(max(rets), 2),
            "min_return": round(min(rets), 2),
            "gt10_rate": round(len(gt10) / len(all_results) * 100, 1),
            "gt20_rate": round(len(gt20) / len(all_results) * 100, 1),
            "avg_hold_days": round(float(np.mean([r["hold_days"] for r in all_results])), 1),
        },
        "exit_reason_dist": {k: {"count": len(v), "avg_ret": round(float(np.mean(v)), 2)}
                            for k, v in reason_ret.items()},
        "errors": errors,
    }
    out_path = OUTPUT_DIR / f"backtest_dynamic_exit{('_' + mode) if mode != 'normal' else ''}_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n💾 汇总已保存: {out_path}")
    print(f"⏱  总用时 {time.time() - t0:.1f} 秒")


# ── CLI ──────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="screen_double 动态退出回测",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --date 2026-04-08
  %(prog)s --start 2026-03-01 --end 2026-04-30
  %(prog)s --start 2026-03-01 --end 2026-04-30 --run
  %(prog)s --start 2026-03-01 --end 2026-04-30 --run --mode winner --gain20 10
        """,
    )
    parser.add_argument("--date", help="单日评估（信号日 T）")
    parser.add_argument("--start", help="批量起始日期")
    parser.add_argument("--end", help="批量结束日期")
    parser.add_argument("--dates", nargs="+", help="指定日期列表")
    parser.add_argument("--top-n", type=int, default=300, help="每信号日评估前N只（默认300）")
    parser.add_argument("--run", action="store_true", help="批量模式重新计算（默认只读已有结果）")
    parser.add_argument("--mode", default="winner", choices=["normal", "accelerated", "winner"],
                        help="screen_double 条件3模式（默认winner）")
    parser.add_argument("--gain20", type=float, default=None, help="20日涨幅最低门槛（pct）")
    parser.add_argument("--trend-level", type=int, default=2,
                        help="趋势过滤严格度（默认2，与screen_double一致）\n"
                             "  0=无额外过滤 1=MA20向上+近20日涨幅>=8%%\n"
                             "  2=P1+近20日涨幅>=12%%+close<MA20*1.10+均线发散度>1%%")
    args = parser.parse_args()

    if args.date:
        # 单日模式
        print(f"\n{'='*60}")
        print(f"  backtest_dynamic_exit.py  —  动态退出评估")
        print(f"  信号日: {args.date}")
        print(f"  止损={STOP_LOSS_PCT}% | 回撤={TRAILING_STOP_PCT}% | T+10锁利≥{MIN_PROFIT_PCT}% | 上限T+{MAX_HOLD_DAYS}")
        print(f"{'='*60}")
        t0 = time.time()
        out = eval_single_date(args.date, top_n=args.top_n,
                               mode=args.mode, gain20=args.gain20,
                               trend_level=args.trend_level)
        if out:
            print_single_date_report(out)
        print(f"\n⏱  用时 {time.time() - t0:.1f} 秒")
    else:
        # 批量模式
        run_batch(args)


if __name__ == "__main__":
    main()
