#!/usr/bin/env python3
"""
全市场放量扫描脚本
用法：python3 full_market_volume_scan.py [--top N] [--threshold RATIO] [--avg-days N]

说明：
  基于 qfq 日线缓存扫描全市场放量股票，按量比排序，显示 Top N
  量比 = 最新日成交量 / N日均量（默认5日）
  放量判定：量比 ≥ threshold（默认 2.0）

输出格式：与 mootdx/mootdx_volume_monitor.py 终端表格一致
"""

import argparse, re, sys, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd

WORKSPACE = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(WORKSPACE))

from stock_trend import gain_turnover as gt


# ═══════════════════════════════════════════════════════════
# 终端显示宽度计算（与 mootdx_volume_monitor.py 完全相同）
# ═══════════════════════════════════════════════════════════
_ansi_pat = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]|\x1b\[K")


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


# ═══════════════════════════════════════════════════════════
# 列定义（与 mootdx_volume_monitor.py 完全相同）
# ═══════════════════════════════════════════════════════════
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
    cjk_width("代码") + GAP_CODE_NAME
    + W_NAME + GAP_NAME_LAST
    + cjk_width("昨收盘") + GAP_LAST_OPEN
    + cjk_width("今开盘") + GAP_OPEN_PRICE
    + cjk_width("成交价") + GAP_PRICE_VOL
    + W_VOL + GAP_VOL_RATIO
    + W_RATIO + GAP_RATIO_FLAG
    + cjk_width("放量标识") + 4
)


def sep_line() -> str:
    return f"{'─' * SEP_TOTAL}"


def hdr_line() -> str:
    return (
        f"{pad_to_width('代码', W_CODE)}{' ' * GAP_CODE_NAME}"
        f"{pad_to_width('名称', W_NAME)}{' ' * GAP_NAME_LAST}"
        f"{pad_to_width('昨收盘', W_LAST, '>')}{' ' * GAP_LAST_OPEN}"
        f"{pad_to_width('今开盘', W_OPEN, '>')}{' ' * GAP_OPEN_PRICE}"
        f"{pad_to_width('成交价', W_PRICE, '>')}{' ' * GAP_PRICE_VOL}"
        f"{pad_to_width('成交量', W_VOL, '>')}{' ' * GAP_VOL_RATIO}"
        f"{pad_to_width('量比', W_RATIO, '>')}{' ' * GAP_RATIO_FLAG}"
        f"{pad_to_width('标识', W_FLAG)}"
    )


def data_line(code: str, name: str, last_close: float, today_open: float,
              price, vol: int, ratio: float, is_breakout: bool,
              date_str: str = "") -> str:
    flag = "✅" if is_breakout else "  "
    verdict = " OK" if is_breakout else "NO "

    price_str = f"{price:>{W_PRICE}.2f}" if price else f"{'N/A':>{W_PRICE}}"
    vol_str   = f"{int(vol):>{W_VOL},}" if vol else f"{'N/A':>{W_VOL}}"
    ratio_str = f"{ratio:>{W_RATIO}.2f}x" if ratio else f"{'N/A':>{W_RATIO}}"

    chg_str = ""
    if price and last_close:
        pct = (price - last_close) / last_close * 100
        chg_str = f" ({pct:+.1f}%)"

    return (
        f"{pad_to_width(code, W_CODE)}{' ' * GAP_CODE_NAME}"
        f"{pad_to_width(name, W_NAME)}{' ' * GAP_NAME_LAST}"
        f"{last_close:>{W_LAST}.2f}{' ' * GAP_LAST_OPEN}"
        f"{today_open:>{W_OPEN}.2f}{' ' * GAP_OPEN_PRICE}"
        f"{price_str}{' ' * GAP_PRICE_VOL}"
        f"{vol_str}{' ' * GAP_VOL_RATIO}"
        f"{ratio_str}{' ' * GAP_RATIO_FLAG}"
        f"{flag} {verdict}{chg_str}"
    )


# ═══════════════════════════════════════════════════════════
# 扫描逻辑
# ═══════════════════════════════════════════════════════════
def scan_one(code: str, names_cache: dict, avg_days: int = 5):
    """扫描单只股票的放量情况，返回 dict 或 None"""
    try:
        c = gt.normalize_prefixed(code)
        df = gt.load_qfq_history(c, refresh=False)
        if df is None or len(df) < avg_days + 2:
            return None

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        # N 日均量（不含最新日）
        avg_vol = float(df["volume"].iloc[-(avg_days + 1):-1].mean())
        if avg_vol <= 0:
            return None

        vol_ratio = float(latest["volume"]) / avg_vol

        return {
            "code": code,
            "name": names_cache.get(code, ""),
            "last_close": float(prev["close"]),
            "open": float(latest["open"]),
            "price": float(latest["close"]),
            "volume": int(latest["volume"]),
            "vol_ratio": round(vol_ratio, 2),
            "data_date": str(latest.get("date", ""))[:10],
        }
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="全市场放量扫描")
    parser.add_argument("--top", "-n", type=int, default=10, help="显示前 N 只（默认10）")
    parser.add_argument("--threshold", "-r", type=float, default=2.0, help="放量阈值（量比，默认2.0x）")
    parser.add_argument("--avg-days", "-a", type=int, default=5, help="量比参考的日均线周期（默认5日）")
    parser.add_argument("--workers", "-w", type=int, default=16, help="并行线程数（默认16）")
    parser.add_argument("--min-vol", type=float, default=0, help="最低成交量过滤（手，默认0=不过滤）")
    parser.add_argument("--no-breakout-only", action="store_true", help="不过滤放量，显示全部 TopN（默认仅放量）")
    args = parser.parse_args()

    # ── 加载全市场代码 ────────────────────────────────────
    t0 = time.time()
    all_codes = gt.get_all_stock_codes()
    names_cache = gt.load_stock_names()
    print(f"📋 全市场股票: {len(all_codes)} 只")
    print(f"📊 放量阈值: {args.threshold}x  |  均量周期: {args.avg_days}日  |  显示 Top{args.top}")
    print(f"🚀 开始扫描（workers={args.workers}）...")

    # ── 并行扫描 ──────────────────────────────────────────
    results = []
    done = [0]
    total = len(all_codes)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(scan_one, code, names_cache, args.avg_days): code
            for code in all_codes
        }
        for future in as_completed(futures):
            done[0] += 1
            if done[0] % 1000 == 0 or done[0] == total:
                elapsed = time.time() - t0
                eta = elapsed / done[0] * (total - done[0])
                print(f"  进度: {done[0]}/{total} ({done[0]/total*100:.0f}%) "
                      f"耗时={elapsed:.0f}s ETA={eta:.0f}s", end="\r", flush=True)
            result = future.result()
            if result is not None:
                results.append(result)

    print(f"\n✅ 扫描完成: {len(results)} 只有效股票, 耗时 {time.time()-t0:.1f}s")

    if not results:
        print("⚠️ 无有效数据，请先运行 cache_qfq_daily.py --refresh")
        return

    # ── 过滤 & 排序 ───────────────────────────────────────
    # 成交量过滤
    if args.min_vol > 0:
        before = len(results)
        results = [r for r in results if r["volume"] >= args.min_vol]
        print(f"📉 成交量过滤 (≥{args.min_vol:,}手): {before} → {len(results)}")

    # 排序
    results.sort(key=lambda r: r["vol_ratio"], reverse=True)

    # 提取放量/非放量
    breakouts = [r for r in results if r["vol_ratio"] >= args.threshold]
    others = [r for r in results if r["vol_ratio"] < args.threshold]

    if not args.no_breakout_only:
        display = breakouts[:args.top]
        display_mode = "放量"
    else:
        display = results[:args.top]
        display_mode = "全部"

    if not display:
        print(f"\n⚠️ 无放量股票（阈值≥{args.threshold}x），量比 Top10：")
        display = results[:10]
        display_mode = "Top10"

    # ── 渲染表格 ──────────────────────────────────────────
    data_date = display[0]["data_date"] if display else "N/A"
    title = (
        f" 最新数据日：{data_date}  |  阈值：≥{args.threshold}x  |  "
        f"放量：{len(breakouts)}只 / 有效：{len(results)}只  |  "
        f"显示：{display_mode} Top{min(args.top, len(display))}"
    )
    pad = SEP_TOTAL - len(title)
    title += " " * max(0, pad)

    lines = []
    lines.append(title)
    lines.append(sep_line())
    lines.append(hdr_line())
    lines.append(sep_line())
    for r in display:
        lines.append(data_line(
            r["code"], r["name"],
            r["last_close"], r["open"],
            r["price"], r["volume"],
            r["vol_ratio"], r["vol_ratio"] >= args.threshold,
            r["data_date"],
        ))
    lines.append(sep_line())

    # 汇总行
    footer = (
        f"合计：{len(results)} 只  |  "
        f"量OK：{len(breakouts)}  |  "
        f"量NO：{len(results) - len(breakouts)}"
    )
    pad = SEP_TOTAL - len(strip_ansi(footer))
    lines.append(f"{footer}{' ' * max(0, pad)}")

    print("\n".join(lines))
    print(f"\n⏱️  总耗时: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
