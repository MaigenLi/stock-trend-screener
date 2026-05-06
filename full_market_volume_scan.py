#!/usr/bin/env python3
"""
全市场放量实时监控
用法：python3 full_market_volume_scan.py [--top N] [--ratio R] [--interval S]

说明：
  实时扫描全市场放量股票，用今日实时累计量 vs 昨日全天量按时间比例估算
  量比 = 今日累计量 / (昨日全天量 × 已过分钟数/240)
  放量判定：量比 ≥ ratio（默认 1.5x）

  实时行情通过 mootdx 批量获取，终端原地刷新

用法示例：
  python3 full_market_volume_scan.py                        # 默认 Top10, 5分钟刷新
  python3 full_market_volume_scan.py --top 20 --ratio 2.0   # Top20, 2x放量阈值
  python3 full_market_volume_scan.py --once                 # 单次扫描（不复刷新）
"""

import argparse, re, sys, time
from pathlib import Path
from datetime import datetime, date as date_type, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

import pandas as pd

WORKSPACE = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(WORKSPACE))

from stock_trend import gain_turnover as gt

# ═══════════════════════════════════════════════════════════
# mootdx 常量
# ═══════════════════════════════════════════════════════════
MARKET_BY_CODE = {
    1: lambda c: c.startswith(("6", "9")),
    0: lambda c: c and c[0] in ("0", "1", "2", "3"),
}
TOTAL_MINUTES = 240
CLEAR_SCREEN = "\033[2J\033[H"
CLEAR_LINE   = "\033[2K"
_ansi_pat = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]|\x1b\[K")


def auto_market(code: str) -> int:
    for market, pred in MARKET_BY_CODE.items():
        if pred(code):
            return market
    return 1


# ═══════════════════════════════════════════════════════════
# 终端显示（与 mootdx_volume_monitor.py 完全相同）
# ═══════════════════════════════════════════════════════════
def cjk_width(s: str) -> int:
    w = 0
    for ch in s:
        cp = ord(ch)
        if cp > 0x1100 and (
            cp <= 0x115f or cp == 0x2329 or cp == 0x232a
            or (0x2e80 <= cp <= 0xa4cf and cp != 0x303f)
            or (0xac00 <= cp <= 0xd7a3)
            or (0xf900 <= cp <= 0xfaff)
            or (0xfe10 <= cp <= 0xfe19)
            or (0xfe30 <= cp <= 0xfe6f)
            or (0xff00 <= cp <= 0xff60)
            or (0xffe0 <= cp <= 0xffe6)
            or (0x20000 <= cp <= 0x2fffd)
            or (0x30000 <= cp <= 0x3fffd)
        ):
            w += 2
        else:
            w += 1
    return w


def pad_to_width(s: str, target: int, align: str = "<") -> str:
    sw = cjk_width(s)
    pad = target - sw
    if pad <= 0:
        return s
    if align == ">":
        return " " * pad + s
    return s + " " * pad


def strip_ansi(s: str) -> str:
    return _ansi_pat.sub("", s)


# 列定义
W_CODE  = 6
W_NAME  = 8
W_LAST  = 7
W_OPEN  = 7
W_PRICE = 8
W_VOL   = 12
W_RATIO = 8
W_FLAG  = 10

GAP_CODE_NAME = 2
GAP_NAME_LAST = 2
GAP_LAST_OPEN = 2
GAP_OPEN_PRICE = 2
GAP_PRICE_VOL = 3
GAP_VOL_RATIO  = 4
GAP_RATIO_FLAG = 4

SEP_TOTAL = (
    cjk_width("代码") + GAP_CODE_NAME + W_NAME + GAP_NAME_LAST
    + cjk_width("昨收盘") + GAP_LAST_OPEN + cjk_width("今开盘") + GAP_OPEN_PRICE
    + cjk_width("成交价") + GAP_PRICE_VOL + W_VOL + GAP_VOL_RATIO
    + W_RATIO + GAP_RATIO_FLAG + cjk_width("放量标识") + 4
)


def sep_line() -> str:
    return f"{CLEAR_LINE}{'─' * SEP_TOTAL}"


def hdr_line() -> str:
    return (
        f"{CLEAR_LINE}"
        f"{pad_to_width('代码', W_CODE)}{' ' * GAP_CODE_NAME}"
        f"{pad_to_width('名称', W_NAME)}{' ' * GAP_NAME_LAST}"
        f"{pad_to_width('昨收盘', W_LAST, '>')}{' ' * GAP_LAST_OPEN}"
        f"{pad_to_width('今开盘', W_OPEN, '>')}{' ' * GAP_OPEN_PRICE}"
        f"{pad_to_width('成交价', W_PRICE, '>')}{' ' * GAP_PRICE_VOL}"
        f"{pad_to_width('成交量', W_VOL, '>')}{' ' * GAP_VOL_RATIO}"
        f"{pad_to_width('量比', W_RATIO, '>')}{' ' * GAP_RATIO_FLAG}"
        f"{pad_to_width('标识', W_FLAG)}"
    )


def data_line(code: str, name: str, last_close, today_open,
              price, vol: int, ratio: float, is_breakout: bool,
              label: str = "") -> str:
    flag = "✅" if is_breakout else "  "
    verdict = " OK" if is_breakout else "NO "

    price_str = f"{price:>{W_PRICE}.2f}" if price else f"{'N/A':>{W_PRICE}}"
    vol_str   = f"{int(vol):>{W_VOL},}" if vol else f"{'N/A':>{W_VOL}}"
    ratio_str = f"{ratio:>{W_RATIO}.2f}x" if ratio else f"{'N/A':>{W_RATIO}}"
    last_str  = f"{last_close:>{W_LAST}.2f}" if last_close else f"{'N/A':>{W_LAST}}"
    open_str  = f"{today_open:>{W_OPEN}.2f}" if today_open else f"{'N/A':>{W_OPEN}}"

    chg_str = ""
    if price and last_close:
        pct = (price - last_close) / last_close * 100
        chg_str = f" ({pct:+.1f}%)"

    label_str = f" {label}" if label else ""

    return (
        f"{CLEAR_LINE}"
        f"{pad_to_width(code, W_CODE)}{' ' * GAP_CODE_NAME}"
        f"{pad_to_width(name, W_NAME)}{' ' * GAP_NAME_LAST}"
        f"{last_str}{' ' * GAP_LAST_OPEN}"
        f"{open_str}{' ' * GAP_OPEN_PRICE}"
        f"{price_str}{' ' * GAP_PRICE_VOL}"
        f"{vol_str}{' ' * GAP_VOL_RATIO}"
        f"{ratio_str}{' ' * GAP_RATIO_FLAG}"
        f"{flag} {verdict}{chg_str}{label_str}"
    )


# ═══════════════════════════════════════════════════════════
# 时间工具
# ═══════════════════════════════════════════════════════════
def current_minute_index(cur_time: datetime) -> int:
    total = cur_time.hour * 3600 + cur_time.minute * 60 + cur_time.second
    AM_START = 9 * 3600 + 30 * 60
    AM_END   = 11 * 3600 + 29 * 60
    PM_START = 13 * 3600
    PM_END   = 15 * 3600 - 1
    if total < AM_START:
        return -1
    if total <= AM_END:
        return (total - AM_START) // 60
    if total < PM_START:
        return -1
    if total <= PM_END:
        return (total - PM_START) // 60 + 120
    return TOTAL_MINUTES - 1


def minute_to_str(idx: int) -> str:
    if idx < 0:
        return "--:--"
    secs = (34200 + idx * 60) if idx < 120 else (46800 + (idx - 120) * 60)
    return f"{secs // 3600:02d}:{(secs % 3600) // 60:02d}"


# ═══════════════════════════════════════════════════════════
# 基线数据预加载（昨日全天成交量，来自 qfq 缓存）
# ═══════════════════════════════════════════════════════════
def find_last_trading_day(reference_date: date_type) -> date_type:
    """从 reference_date 往回找最近一个交易日（简单跳过周末和五一）"""
    from datetime import timedelta
    day = reference_date
    holidays_2026 = {
        date_type(2026, 5, 1), date_type(2026, 5, 2), date_type(2026, 5, 3),
        date_type(2026, 5, 4), date_type(2026, 5, 5),
    }
    for _ in range(14):
        if day.weekday() >= 5 or day in holidays_2026:
            day -= timedelta(days=1)
            continue
        return day
    return reference_date


def preload_baseline(codes: list, names_cache: dict) -> dict:
    """
    预加载所有股票的昨日数据：昨收、昨全天量
    返回 {code_lower: {"last_close": float, "yesterday_vol": int, "name": str}}
    """
    t0 = time.time()
    baseline = {}
    ref_date = datetime.now().date()
    last_trading = find_last_trading_day(ref_date)
    today_str = ref_date.strftime("%Y-%m-%d")
    print(f"📅 基准交易日: {last_trading}  |  今日: {today_str}")

    for code in codes:
        try:
            c = gt.normalize_prefixed(code)
            df = gt.load_qfq_history(c, refresh=False)
            if df is None or len(df) < 2:
                continue
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            latest_date = str(latest.get("date", ""))[:10]

            # 取昨日（上一交易日）的 bar
            if latest_date >= today_str:
                yesterday_bar = prev
            else:
                yesterday_bar = latest

            # 去前缀，与 mootdx 返回格式对齐（如 sh600000 → 600000）
            raw_code = code.lower()
            for pfx in ("sh", "sz", "bj"):
                if raw_code.startswith(pfx):
                    raw_code = raw_code[2:]
                    break

            baseline[raw_code] = {
                "last_close": float(yesterday_bar["close"]),
                "yesterday_vol": int(yesterday_bar["volume"]),
                "name": names_cache.get(code, ""),
            }
        except Exception:
            continue

    print(f"✅ 基线预加载: {len(baseline)} 只, 耗时 {time.time()-t0:.1f}s")
    return baseline


# ═══════════════════════════════════════════════════════════
# 实时行情获取（mootdx 批量）
# ═══════════════════════════════════════════════════════════
def get_mootdx_client():
    from mootdx.quotes import Quotes
    return Quotes.factory(market="std")


BATCH_SIZE = 80  # mootdx get_security_quotes 单次上限


def fetch_all_realtime(all_codes: list) -> dict:
    """
    通过 mootdx 分块获取全市场实时行情（单次上限 80 只）
    返回 {code_lower: {"price":, "open":, "vol":, "amount":, "last_close":}}
    """
    client = get_mootdx_client()
    quotes = {}
    total = len(all_codes)

    for start in range(0, total, BATCH_SIZE):
        chunk = all_codes[start:start + BATCH_SIZE]
        batch = [(auto_market(c), c) for c in chunk]
        try:
            result = client.client.get_security_quotes(batch)
            for q in (result or []):
                code = q.get("code", "")
                quotes[code.lower()] = {
                    "price": float(q.get("price") or 0),
                    "open": float(q.get("open") or 0),
                    "vol": int(q.get("vol") or 0),
                    "amount": float(q.get("amount") or 0),
                    "last_close": float(q.get("last_close") or 0),
                }
        except Exception as e:
            # 单批次失败不影响其他批次
            pass

    return quotes


# ═══════════════════════════════════════════════════════════
# 渲染
# ═══════════════════════════════════════════════════════════
def render_top(top_data: list, cur_time: datetime, cur_idx: int,
               args, breakout_count: int, total_scanned: int,
               cycle: int, cycle_elapsed: float):
    """终端原地刷新 Top N 表格"""
    lines = []
    now_str = cur_time.strftime("%H:%M:%S")
    minute_label = minute_to_str(cur_idx)
    day_fraction = min(cur_idx + 1, TOTAL_MINUTES) / TOTAL_MINUTES * 100 if cur_idx >= 0 else 0

    # 标题行
    title1 = (
        f"全市场放量监控  |  {now_str} [{minute_label}]  "
        f"进度 {day_fraction:.0f}%  |  "
        f"有效 {total_scanned}只  |  "
        f"放量 {breakout_count}只  |  "
        f"阈值 ≥{args.ratio}x  |  "
        f"第{cycle}轮  {cycle_elapsed:.1f}s"
    )
    pad = SEP_TOTAL - cjk_width(title1)
    if pad > 0:
        title1 += " " * (pad // 2)

    lines.append(title1)
    lines.append(sep_line())
    lines.append(hdr_line())
    lines.append(sep_line())

    for r in top_data:
        lines.append(data_line(
            r["code"], r["name"],
            r["last_close"], r["open"],
            r["price"], r["vol"],
            r["vol_ratio"], r["vol_ratio"] >= args.ratio,
        ))

    lines.append(sep_line())

    # 汇总行
    footer = (
        f"总计：{total_scanned} 只  |  "
        f"量OK：{breakout_count}  |  "
        f"量NO：{total_scanned - breakout_count}  |  "
        f"按 Ctrl+C 退出"
    )
    pad = SEP_TOTAL - cjk_width(strip_ansi(footer))
    if pad > 0:
        footer += " " * pad
    lines.append(f"{CLEAR_LINE}{footer}")

    sys.stdout.write(CLEAR_SCREEN)
    sys.stdout.write("\n".join(lines) + "\n")
    sys.stdout.flush()


# ═══════════════════════════════════════════════════════════
# 单次扫描（--once 模式）
# ═══════════════════════════════════════════════════════════
def scan_once(args, baseline: dict):
    """单次全市场扫描并输出"""
    t0 = time.time()
    print("🚀 获取实时行情...")
    quotes = fetch_all_realtime(list(baseline.keys()))
    if not quotes:
        print("❌ 无法获取实时行情")
        return

    cur_time = datetime.now()
    cur_idx = current_minute_index(cur_time)
    day_frac = min(cur_idx + 1, TOTAL_MINUTES) / TOTAL_MINUTES if cur_idx >= 0 else 1.0
    if cur_idx < 0:
        day_frac = 1.0  # 盘前/盘后用全天量

    results = []
    for code_lower, bl in baseline.items():
        q = quotes.get(code_lower)
        if not q or bl["yesterday_vol"] <= 0:
            continue
        today_vol = q["vol"]
        if today_vol <= 0:
            continue
        expected_vol = bl["yesterday_vol"] * day_frac
        if expected_vol <= 0:
            continue
        vol_ratio = today_vol / expected_vol
        results.append({
            "code": code_lower,
            "name": bl["name"],
            "last_close": q.get("last_close", bl["last_close"]) or bl["last_close"],
            "open": q["open"],
            "price": q["price"],
            "vol": today_vol,
            "vol_ratio": round(vol_ratio, 2),
        })

    results.sort(key=lambda r: r["vol_ratio"], reverse=True)
    breakouts = [r for r in results if r["vol_ratio"] >= args.ratio]
    display = breakouts[:args.top] if breakouts else results[:args.top]

    render_top(display, cur_time, cur_idx, args,
               len(breakouts), len(results), 1, time.time() - t0)
    print(f"\n⏱️  总耗时: {time.time()-t0:.1f}s")


# ═══════════════════════════════════════════════════════════
# 实时监控循环
# ═══════════════════════════════════════════════════════════
def watch_loop(args, baseline: dict):
    """实时监控循环，每 interval 秒刷新全市场 Top N"""
    all_codes_lower = list(baseline.keys())
    cycle = 0

    print(f"🔭 全市场放量监控启动  |  Top{args.top}  |  阈值≥{args.ratio}x  |  刷新间隔 {args.interval}s")
    print(f"📋 股票基数: {len(baseline)} 只")
    print(f"⏰ 开始时间: {datetime.now().strftime('%H:%M:%S')}")
    print()

    try:
        while True:
            cycle += 1
            cycle_t0 = time.time()
            cur_time = datetime.now()
            cur_idx = current_minute_index(cur_time)

            # 交易时间检查
            if cur_idx < 0:
                h, m = cur_time.hour, cur_time.minute
                is_lunch = (h == 11 and m >= 30) or (h == 12)
                status = "午休" if is_lunch else "非交易时段"
                print(f"\r{' ' * 60}", end="")
                print(f"\r⏸️  {status} ({cur_time.strftime('%H:%M:%S')}) — "
                      f"等待中...  Ctrl+C 退出", end="", flush=True)
                time.sleep(10)
                continue

            # 获取实时行情
            quotes = fetch_all_realtime(all_codes_lower)
            if not quotes:
                time.sleep(5)
                continue

            # 计算量比
            day_frac = (cur_idx + 1) / TOTAL_MINUTES
            results = []
            for code_lower, bl in baseline.items():
                q = quotes.get(code_lower)
                if not q or bl["yesterday_vol"] <= 0:
                    continue
                today_vol = q["vol"]
                if today_vol <= 0:
                    continue
                expected_vol = bl["yesterday_vol"] * day_frac
                if expected_vol <= 0:
                    continue
                vol_ratio = today_vol / expected_vol
                results.append({
                    "code": code_lower,
                    "name": bl["name"],
                    "last_close": q.get("last_close", bl["last_close"]) or bl["last_close"],
                    "open": q["open"],
                    "price": q["price"],
                    "vol": today_vol,
                    "vol_ratio": round(vol_ratio, 2),
                })

            results.sort(key=lambda r: r["vol_ratio"], reverse=True)
            breakouts = [r for r in results if r["vol_ratio"] >= args.ratio]
            display = breakouts[:args.top] if breakouts else results[:args.top]

            cycle_elapsed = time.time() - cycle_t0
            render_top(display, cur_time, cur_idx, args,
                       len(breakouts), len(results), cycle, cycle_elapsed)

            # 精确等待
            elapsed = time.time() - cycle_t0
            if elapsed < args.interval:
                time.sleep(args.interval - elapsed)

    except KeyboardInterrupt:
        print(f"\n\n⏹️  监控已停止（Ctrl+C）。共 {cycle} 轮。\n", flush=True)


# ═══════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="全市场放量实时监控")
    parser.add_argument("--top", "-n", type=int, default=10, help="显示前 N 只（默认10）")
    parser.add_argument("--ratio", "-r", type=float, default=1.5, help="放量阈值（量比，默认1.5x）")
    parser.add_argument("--interval", "-i", type=int, default=300, help="刷新间隔秒数（默认300=5分钟）")
    parser.add_argument("--once", action="store_true", help="单次扫描（不复刷新）")
    args = parser.parse_args()

    # ── 预加载全市场代码 & 昨日基线 ────────────────────────
    t0 = time.time()
    all_codes = gt.get_all_stock_codes()
    names_cache = gt.load_stock_names()
    print(f"📋 全市场股票: {len(all_codes)} 只")
    baseline = preload_baseline(all_codes, names_cache)
    if not baseline:
        print("❌ 基线数据为空，请先运行 cache_qfq_daily.py --refresh")
        return
    print(f"⏱️  启动耗时: {time.time()-t0:.1f}s\n")

    if args.once:
        scan_once(args, baseline)
    else:
        watch_loop(args, baseline)


if __name__ == "__main__":
    main()
