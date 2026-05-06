#!/usr/bin/env python3
"""
全市场放量实时监控（分钟级精度）
用法：python3 full_market_volume_scan.py [--top N] [--ratio R] [--interval S]

与 mootdx_volume_monitor.py 同架构：
  昨日分时逐分钟累计 vs 今日实时累计量，同分钟点精确对比
  量比 = 今日到N分钟累计量 / 昨日到N分钟累计量
  放量判定：量比 ≥ ratio（默认 1.5x）

用法示例：
  python3 full_market_volume_scan.py                        # Top10, 5分钟刷新
  python3 full_market_volume_scan.py --top 20 --ratio 2.0   # Top20, 2x阈值
  python3 full_market_volume_scan.py --once                 # 单次扫描
"""

import argparse, re, sys, time
from pathlib import Path
from datetime import datetime, date as date_type, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed, Future as _Future
import threading
import json

import pandas as pd

WORKSPACE = Path(__file__).parent.parent.parent.resolve()  # stock_trend/mootdx/ → workspace/
sys.path.insert(0, str(WORKSPACE))

from stock_trend import gain_turnover as gt

# ═══════════════════════════════════════════════════════════
# 常量
# ═══════════════════════════════════════════════════════════
MARKET_BY_CODE = {
    1: lambda c: c.startswith(("6", "9")),
    0: lambda c: c and c[0] in ("0", "1", "2", "3"),
}
TOTAL_MINUTES = 240
BATCH_SIZE = 80
CLEAR_SCREEN = "\033[2J\033[H"
CLEAR_LINE   = "\033[2K"
_ansi_pat = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]|\x1b\[K")

_DEFAULT_PRELOAD_WORKERS = 8


def auto_market(code: str) -> int:
    for market, pred in MARKET_BY_CODE.items():
        if pred(code):
            return market
    return 1


# ═══════════════════════════════════════════════════════════
# 终端显示（与 mootdx_volume_monitor 完全一致）
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


W_CODE  = 6; W_NAME  = 8; W_LAST  = 7; W_OPEN  = 7
W_PRICE = 8; W_VOL   = 12; W_RATIO = 8; W_FLAG  = 10
GAP_CODE_NAME = 2; GAP_NAME_LAST = 2; GAP_LAST_OPEN = 2
GAP_OPEN_PRICE = 2; GAP_PRICE_VOL = 3; GAP_VOL_RATIO = 4; GAP_RATIO_FLAG = 4

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
              price, vol: int, ratio: float, is_breakout: bool) -> str:
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
    return (
        f"{CLEAR_LINE}"
        f"{pad_to_width(code, W_CODE)}{' ' * GAP_CODE_NAME}"
        f"{pad_to_width(name, W_NAME)}{' ' * GAP_NAME_LAST}"
        f"{last_str}{' ' * GAP_LAST_OPEN}"
        f"{open_str}{' ' * GAP_OPEN_PRICE}"
        f"{price_str}{' ' * GAP_PRICE_VOL}"
        f"{vol_str}{' ' * GAP_VOL_RATIO}"
        f"{ratio_str}{' ' * GAP_RATIO_FLAG}"
        f"{flag} {verdict}{chg_str}"
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
# 寻找上一交易日
# ═══════════════════════════════════════════════════════════
_holidays_2026 = {
    date_type(2026, 5, 1), date_type(2026, 5, 2), date_type(2026, 5, 3),
    date_type(2026, 5, 4), date_type(2026, 5, 5),
}


def find_last_trading_day(reference_date: date_type) -> date_type:
    day = reference_date - timedelta(days=1)
    for _ in range(14):
        if day.weekday() >= 5 or day in _holidays_2026:
            day -= timedelta(days=1)
            continue
        return day
    return reference_date


# ═══════════════════════════════════════════════════════════
# 昨日分时数据预加载（分钟精度，与 mootdx_volume_monitor 相同）
# ═══════════════════════════════════════════════════════════
def _fetch_yesterday_minute(code: str, yesterday_date: date_type,
                            client) -> list | None:
    """
    获取单只股票昨日的 240 分钟累计量数组
    返回 list[int]（长度 240, 每元素为到该分钟为止的累计量），失败返回 None
    """
    mkt = auto_market(code)
    date_int = (yesterday_date.year * 10000
                + yesterday_date.month * 100
                + yesterday_date.day)
    try:
        bars = client.get_history_minute_time_data(mkt, code, date_int)
        if not bars or len(bars) < 200:  # 至少200分钟有效
            return None
        cum = []
        running = 0
        for b in bars:
            running += int(b.get("vol") or 0)
            cum.append(running)
        return cum
    except Exception:
        return None


def preload_yesterday_bars(all_codes: list, names_cache: dict,
                           yesterday_date: date_type,
                           max_workers: int = _DEFAULT_PRELOAD_WORKERS) -> dict:
    """
    预加载全市场昨日分时累计量（并行，mootdx）
    返回 {code: {"cum": [240 ints], "name": str, "last_close": float, "yesterday_total": int}}
    失败股票不在字典中（后续回退到 qfq 日线近似）
    """
    from mootdx.quotes import Quotes

    t0 = time.time()
    result = {}
    total = len(all_codes)
    done = [0]
    lock = threading.Lock()

    # 也加载 qfq 日线基线（昨收 + 备用量）
    today_str = datetime.now().strftime("%Y-%m-%d")
    daily_fallback = {}

    def _worker(code):
        nonlocal daily_fallback
        raw = code.lower()
        for pfx in ("sh", "sz", "bj"):
            if raw.startswith(pfx):
                raw = raw[2:]
                break
        c = gt.normalize_prefixed(code)
        client = Quotes.factory(market="std")
        cum = _fetch_yesterday_minute(raw, yesterday_date, client.client)
        if cum:
            # 昨日总成交量（最后一分钟累计）
            yesterday_total = cum[-1] if cum else 0
            # 昨收从 qfq 缓存取
            last_close = 0.0
            try:
                df = gt.load_qfq_history(c, refresh=False)
                if df is not None and len(df) >= 1:
                    last_date = str(df["date"].iloc[-1])[:10]
                    if last_date >= today_str and len(df) >= 2:
                        last_close = float(df["close"].iloc[-2])
                    else:
                        last_close = float(df["close"].iloc[-1])
            except Exception:
                pass
            with lock:
                result[raw] = {
                    "cum": cum,
                    "name": names_cache.get(code, ""),
                    "last_close": last_close,
                    "yesterday_total": yesterday_total,
                }

    # 并行执行（每线程独立 mootdx client）
    workers = min(max_workers, len(all_codes))
    print(f"📥 预加载昨日 {yesterday_date} 分时数据（{len(all_codes)} 只, {workers} 线程）...")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_worker, code): code for code in all_codes}
        for future in as_completed(futures):
            done[0] += 1
            if done[0] % 500 == 0 or done[0] == total:
                elapsed = time.time() - t0
                eta = elapsed / done[0] * (total - done[0])
                print(f"  进度: {done[0]}/{total} ({done[0]/total*100:.0f}%) "
                      f"耗时={elapsed:.0f}s ETA={eta:.0f}s 已获取={len(result)}只",
                      end="\r", flush=True)

    print(f"\n✅ 昨日分时预加载: {len(result)} 只（{total - len(result)} 只待回退）, "
          f"耗时 {time.time()-t0:.1f}s")
    return result


# ═══════════════════════════════════════════════════════════
# 实时行情获取（mootdx 分块批量）
# ═══════════════════════════════════════════════════════════
def get_mootdx_client():
    from mootdx.quotes import Quotes
    return Quotes.factory(market="std")


def fetch_all_realtime(all_codes: list) -> dict:
    """
    通过 mootdx 分块获取全市场实时行情
    返回 {code: {"price": float, "open": float, "vol": int, "last_close": float}}
    """
    client = get_mootdx_client()
    quotes = {}
    total = len(all_codes)
    for start in range(0, total, BATCH_SIZE):
        chunk = all_codes[start:start + BATCH_SIZE]
        batch = [(auto_market(c), c) for c in chunk]
        try:
            bresult = client.client.get_security_quotes(batch)
            for q in (bresult or []):
                code = q.get("code", "")
                quotes[code.lower()] = {
                    "price": float(q.get("price") or 0),
                    "open": float(q.get("open") or 0),
                    "vol": int(q.get("vol") or 0),
                    "last_close": float(q.get("last_close") or 0),
                }
        except Exception:
            pass
    return quotes


# ═══════════════════════════════════════════════════════════
# 量比计算（分钟精度 + 日线回退）
# ═══════════════════════════════════════════════════════════
def compute_ratio_from_minute(code_raw: str, quote: dict, cur_idx: int,
                              yesterday_data: dict | None) -> float | None:
    """
    计算量比（精确到当前分钟）
    优先用昨日分时累计数组做同分钟对比
    回退到日线比例估算
    返回量比（float），数据不足返回 None
    """
    today_vol = quote.get("vol", 0)
    if today_vol <= 0:
        return None

    # ── 分钟精度 ──────────────────────────────────────
    if yesterday_data is not None and "cum" in yesterday_data:
        cum = yesterday_data["cum"]
        cap = min(cur_idx, len(cum) - 1)
        if cap >= 0:
            yes_cum = cum[cap]
            if yes_cum > 0:
                return today_vol / yes_cum

    # ── 回退：日线比例 ────────────────────────────────
    yesterday_total = yesterday_data.get("yesterday_total", 0) if yesterday_data else 0
    if yesterday_total > 0:
        day_frac = (cur_idx + 1) / TOTAL_MINUTES if cur_idx >= 0 else 1.0
        expected = yesterday_total * day_frac
        if expected > 0:
            return today_vol / expected

    return None


# ═══════════════════════════════════════════════════════════
# 单轮扫描
# ═══════════════════════════════════════════════════════════
def scan_cycle(quotes: dict, yesterday: dict, cur_time: datetime,
               cur_idx: int, ratio_threshold: float) -> list[dict]:
    """一轮全市场扫描，返回按量比排序的结果列表"""
    results = []
    for code_raw, q in quotes.items():
        if q["vol"] <= 0:
            continue
        yd = yesterday.get(code_raw)
        r = compute_ratio_from_minute(code_raw, q, cur_idx, yd)
        if r is None:
            continue
        last_close = q.get("last_close", 0) or (yd.get("last_close", 0) if yd else 0)
        results.append({
            "code": code_raw,
            "name": yd["name"] if yd else "",
            "last_close": last_close,
            "open": q["open"],
            "price": q["price"],
            "vol": q["vol"],
            "vol_ratio": round(r, 2),
        })
    results.sort(key=lambda x: x["vol_ratio"], reverse=True)
    return results


# ═══════════════════════════════════════════════════════════
# 渲染
# ═══════════════════════════════════════════════════════════
def render_top(top_data: list, cur_time: datetime, cur_idx: int,
               args, breakout_count: int, total_scanned: int,
               cycle: int, cycle_elapsed: float, yesterday: dict):
    lines = []
    now_str = cur_time.strftime("%H:%M:%S")
    minute_label = minute_to_str(cur_idx)
    coverage = len(yesterday)

    title1 = (
        f"全市场放量监控 [分钟级]  |  {now_str} [{minute_label}]  |  "
        f"有效 {total_scanned}只  |  "
        f"分钟数据 {coverage}只  |  "
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
            r["code"], r["name"], r["last_close"], r["open"],
            r["price"], r["vol"], r["vol_ratio"],
            r["vol_ratio"] >= args.ratio,
        ))
    lines.append(sep_line())
    footer = (
        f"总计：{total_scanned} 只  |  量OK：{breakout_count}  |  "
        f"量NO：{total_scanned - breakout_count}  |  "
        f"昨日 ▶ 今日 同分钟累计对比  |  Ctrl+C 退出"
    )
    pad = SEP_TOTAL - cjk_width(strip_ansi(footer))
    if pad > 0:
        footer += " " * pad
    lines.append(f"{CLEAR_LINE}{footer}")
    sys.stdout.write(CLEAR_SCREEN)
    sys.stdout.write("\n".join(lines) + "\n")
    sys.stdout.flush()


# ═══════════════════════════════════════════════════════════
# --once 模式
# ═══════════════════════════════════════════════════════════
def scan_once(args, yesterday: dict, all_codes: list):
    t0 = time.time()
    print("🚀 获取实时行情...")
    quotes = fetch_all_realtime(all_codes)
    if not quotes:
        print("❌ 无法获取实时行情")
        return
    cur_time = datetime.now()
    cur_idx = current_minute_index(cur_time)
    results = scan_cycle(quotes, yesterday, cur_time, cur_idx, args.ratio)
    breakouts = [r for r in results if r["vol_ratio"] >= args.ratio]
    display = breakouts[:args.top] if breakouts else results[:args.top]
    render_top(display, cur_time, cur_idx, args,
               len(breakouts), len(results), 1,
               time.time() - t0, yesterday)
    print(f"\n⏱️  总耗时: {time.time()-t0:.1f}s")


# ═══════════════════════════════════════════════════════════
# 监控循环
# ═══════════════════════════════════════════════════════════
def watch_loop(args, yesterday: dict, all_codes: list):
    cycle = 0
    print(f"🔭 全市场放量监控 [分钟级精度]  |  Top{args.top}  |  阈值≥{args.ratio}x  |  "
          f"刷新间隔 {args.interval}s")
    print(f"📋 股票: {len(all_codes)} 只  |  昨日分时数据: {len(yesterday)} 只")
    print(f"⏰ 开始时间: {datetime.now().strftime('%H:%M:%S')}")
    print()

    try:
        while True:
            cycle += 1
            cycle_t0 = time.time()
            cur_time = datetime.now()
            cur_idx = current_minute_index(cur_time)

            if cur_idx < 0:
                h, m = cur_time.hour, cur_time.minute
                is_lunch = (h == 11 and m >= 30) or (h == 12)
                status = "午休" if is_lunch else "非交易时段"
                print(f"\r{' ' * 60}", end="")
                print(f"\r⏸️  {status} ({cur_time.strftime('%H:%M:%S')}) — "
                      f"等待中...  Ctrl+C 退出", end="", flush=True)
                time.sleep(10)
                continue

            quotes = fetch_all_realtime(all_codes)
            if not quotes:
                time.sleep(5)
                continue

            results = scan_cycle(quotes, yesterday, cur_time, cur_idx, args.ratio)
            breakouts = [r for r in results if r["vol_ratio"] >= args.ratio]
            display = breakouts[:args.top] if breakouts else results[:args.top]

            cycle_elapsed = time.time() - cycle_t0
            render_top(display, cur_time, cur_idx, args,
                       len(breakouts), len(results), cycle,
                       cycle_elapsed, yesterday)

            elapsed = time.time() - cycle_t0
            if elapsed < args.interval:
                time.sleep(args.interval - elapsed)

    except KeyboardInterrupt:
        print(f"\n\n⏹️  监控已停止。共 {cycle} 轮。\n", flush=True)


# ═══════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="全市场放量实时监控（分钟级精度）")
    parser.add_argument("--top", "-n", type=int, default=10, help="显示前 N 只（默认10）")
    parser.add_argument("--ratio", "-r", type=float, default=1.5, help="放量阈值（默认1.5x）")
    parser.add_argument("--interval", "-i", type=int, default=300, help="刷新间隔（秒，默认300）")
    parser.add_argument("--once", action="store_true", help="单次扫描（不复刷新）")
    parser.add_argument("--preload-workers", type=int, default=_DEFAULT_PRELOAD_WORKERS,
                        help=f"预加载线程数（默认{_DEFAULT_PRELOAD_WORKERS}）")
    args = parser.parse_args()

    # ── 确定上一交易日 ───────────────────────────────────
    last_trading = find_last_trading_day(datetime.now().date())
    print(f"📅 上一交易日: {last_trading}  |  今日: {datetime.now().strftime('%Y-%m-%d')}")

    # ── 加载代码 ─────────────────────────────────────────
    t0 = time.time()
    all_codes = gt.get_all_stock_codes()
    names_cache = gt.load_stock_names()
    print(f"📋 全市场股票: {len(all_codes)} 只")

    # ── 预加载昨日分时 ───────────────────────────────────
    yesterday = preload_yesterday_bars(all_codes, names_cache, last_trading,
                                       max_workers=args.preload_workers)

    if not yesterday:
        print("❌ 昨日分时数据为空，退出")
        return

    # ── 提取无前缀代码列表（与 mootdx 对齐）──────────────
    raw_codes = []
    for code in all_codes:
        raw = code.lower()
        for pfx in ("sh", "sz", "bj"):
            if raw.startswith(pfx):
                raw = raw[2:]
                break
        raw_codes.append(raw)

    print(f"⏱️  启动耗时: {time.time()-t0:.1f}s\n")

    if args.once:
        scan_once(args, yesterday, raw_codes)
    else:
        watch_loop(args, yesterday, raw_codes)


if __name__ == "__main__":
    main()
