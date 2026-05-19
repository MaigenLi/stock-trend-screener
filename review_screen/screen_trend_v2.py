#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
screen_trend.py — 六条件选股器
============================================================
6条选股条件（同时满足）：
  1. MA5>MA10>MA20 且 MA5>MA60, MA10>MA60（多头排列，均线在MA60上方）
  2. MA5上升 且 MA10上升 且 MA20上升（均线多头扩散）
  3. 20日涨幅在 15%~35% 之间
  4. 近5日平均换手率 ≥ 5%
  5. 流通市值 > 30亿
  6. 当日涨幅在 -2%~8%

使用方法：
  python screen_trend.py                    # 全市场扫描（默认最新日期）
  python screen_trend.py --date 2026-05-08   # 指定信号日扫描
  python screen_trend.py --top 30           # 显示前30只
  python screen_trend.py --min-turnover 3    # 换手率门槛3%
  python screen_trend.py --code sh603629     # 单票分析
"""

import sys
import argparse
from pathlib import Path

WORKSPACE = Path.home() / ".openclaw/workspace"
sys.path.insert(0, str(WORKSPACE / "stock_trend"))

import numpy as np
import pandas as pd
from gain_turnover import (
    load_qfq_history, get_all_stock_codes,
    rolling_mean, load_stock_names,
)

# ── 参数 ──────────────────────────────────────────────────
TURNOVER_LEN  = 5     # 近5日
TURNOVER_MIN  = 5.0   # 换手率下限%%
GAIN20_MIN    = 15.0  # 20日涨幅下限%%
GAIN20_MAX    = 35.0  # 20日涨幅上限%%
MKT_CAP_MIN   = 30.0  # 流通市值下限（亿元）
GAIN_DAY_MIN  = -4.0  # 当日涨幅下限%%
GAIN_DAY_MAX  = 8.0   # 当日涨幅上限%%
MA_LEN        = 20    # 计算MA20需要
MA_LEN_60     = 60    # 计算MA60需要


# ══════════════════════════════════════════════════════════════
# 核心检测
# ══════════════════════════════════════════════════════════════

def check_ma4(df: pd.DataFrame,
              turnover_min: float = TURNOVER_MIN,
              signal_date: str = None,
             ) -> dict | None:
    """
    6条条件逐条核验（默认取最新日期，或指定 signal_date）：
      ① MA5>MA10>MA20 且 MA5>MA60, MA10>MA60
      ② MA5上升 且 MA10上升 且 MA20上升
      ③ 近5日平均换手率 ≥ turnover_min
      ④ 20日涨幅在 15%~35%
      ⑤ 流通市值 > MKT_CAP_MIN（亿元）
      ⑥ 当日涨幅在 GAIN_DAY_MIN%~GAIN_DAY_MAX%
    """
    n = len(df)
    if n < 65:
        return None

    # ── 确定分析索引 ──────────────────────────────
    if signal_date:
        date_map = {str(d)[:10]: idx for idx, d in enumerate(df["date"].values)}
        if signal_date not in date_map:
            return None
        i = date_map[signal_date]
    else:
        i = n - 1   # 默认最新日期

    if i < 64:
        return None

    close    = df["close"].values.astype(float)
    turnover = df["true_turnover"].values.astype(float)
    dates    = df["date"].values
    outs     = df["outstanding_share"].values.astype(float) if "outstanding_share" in df.columns else None

    # ── 计算均线 ──────────────────────────────
    ma5  = rolling_mean(close, 5).astype(float)
    ma10 = rolling_mean(close, 10).astype(float)
    ma20 = rolling_mean(close, MA_LEN).astype(float)
    ma60 = rolling_mean(close, MA_LEN_60).astype(float)

    # ── 条件①：MA5>MA10>MA20 且 MA5>MA60, MA10>MA60 ─────────
    if not (ma5[i] > ma10[i] > ma20[i] and ma5[i] > ma60[i] and ma10[i] > ma60[i]):
        return None

    # ── 条件②：MA5上升 且 MA10上升 且 MA20上升 ──
    ma5_rise   = ma5[i]   > ma5[i-1]
    ma10_rise  = ma10[i]  > ma10[i-1]
    ma20_rise  = ma20[i]  > ma20[i-1]
    if not (ma5_rise and ma10_rise and ma20_rise):
        return None

    # ── 条件④：20日涨幅在 15%~35% ──────────
    if i < 20:
        return None
    gain20 = (close[i] / close[i - 20] - 1) * 100.0
    if not (GAIN20_MIN <= gain20 <= GAIN20_MAX):
        return None

    # ── 条件③：近5日平均换手率 ≥ turnover_min ──
    if i < TURNOVER_LEN:
        return None
    turnover_avg = np.mean(turnover[i - TURNOVER_LEN + 1 : i + 1])
    if turnover_avg < turnover_min:
        return None

    # ── 条件⑤：流通市值 > MKT_CAP_MIN ───────────────────────
    if outs is not None and not np.isnan(outs[i]) and outs[i] > 0:
        mktcap = close[i] * outs[i] / 1e8
        if mktcap <= MKT_CAP_MIN:
            return None
    else:
        return None   # 无流通股本数据则跳过

    # ── 条件⑥：当日涨幅在 GAIN_DAY_MIN%~GAIN_DAY_MAX% ─────
    if i < 1:
        return None
    gain_day = (close[i] / close[i - 1] - 1) * 100.0
    if not (GAIN_DAY_MIN < gain_day < GAIN_DAY_MAX):
        return None

    # ── 计算辅助指标（用于展示）──────────────
    ma5_chg5d  = (ma5[i]  - ma5[i-5])  / ma5[i-5]  * 100.0 if i >= 5  else 0.0
    ma10_chg5d = (ma10[i] - ma10[i-5]) / ma10[i-5] * 100.0 if i >= 5  else 0.0
    ma20_chg5d = (ma20[i] - ma20[i-5]) / ma20[i-5] * 100.0 if i >= 5  else 0.0

    turnover_today = turnover[i]

    vol5_avg  = np.mean(df["volume"].values[i-4:i+1])
    vol5_prev = np.mean(df["volume"].values[i-9:i-4]) if i >= 10 else np.mean(df["volume"].values[max(0,i-9):i])
    vol_ratio = (vol5_avg / vol5_prev) if vol5_prev > 0 else 0.0

    spread = (ma5[i]/ma10[i] - 1)*100 + (ma10[i]/ma20[i] - 1)*100

    return {
        "date":         str(dates[i])[:10],
        "close":        round(close[i], 2),
        "ma5":          round(ma5[i], 2),
        "ma10":         round(ma10[i], 2),
        "ma20":         round(ma20[i], 2),
        "ma5_rise":     ma5_rise,
        "ma10_rise":    ma10_rise,
        "ma20_rise":    ma20_rise,
        "ma5_chg5d":    round(ma5_chg5d, 2),
        "ma10_chg5d":   round(ma10_chg5d, 2),
        "ma20_chg5d":   round(ma20_chg5d, 2),
        "gain20":        round(gain20, 1),
        "turnover_avg":  round(turnover_avg, 2),
        "turnover_today": round(turnover_today, 2),
        "vol_ratio":     round(vol_ratio, 2),
        "spread":        round(spread, 2),
    }


# ══════════════════════════════════════════════════════════════
# 批量扫描
# ══════════════════════════════════════════════════════════════

def scan(codes: list, names: dict, turnover_min: float, top: int, signal_date: str = None) -> list:
    results = []
    for idx, code in enumerate(codes):
        try:
            df = load_qfq_history(code, end_date=signal_date)
            r = check_ma4(df, turnover_min=turnover_min, signal_date=signal_date)
            if r:
                r["code"] = code
                name = names.get(code, code)
                if "ST" in name.upper() or "*ST" in name or "S*ST" in name:
                    continue
                r["name"] = name
                results.append(r)
        except Exception:
            pass

        if (idx + 1) % 1000 == 0:
            sys.__stdout__.write(f"  已扫描 {idx+1} 只 ... 当前满足 {len(results)} 只 ")
            sys.__stdout__.flush()

    total = len(results)
    results.sort(key=lambda x: -x["gain20"])
    sys.__stdout__.write("\n  \u2713 扫描完毕，共 %d 只满足条件（取前 %d 只）\n\n" % (total, top))\nsys.__stdout__.flush()
    return results[:top]


# ══════════════════════════════════════════════════════════════
# 输出
# ══════════════════════════════════════════════════════════════

def print_banner():
    print("""
╔═════════════════════════════════════════════════════════╗
║              六条件选股器                                ║
╠═════════════════════════════════════════════════════════╣
║  ① MA5>MA10>MA20 且 MA5>MA60, MA10>MA60                 ║
║  ② MA5上升 且 MA10上升 且 MA20上升                       ║
║  ③ 近5日平均换手率 ≥ 5.0%%                               ║
║  ④ 20日涨幅在 15%~35%%                                   ║
║  ⑤ 流通市值 > 30亿                                       ║
║  ⑥ 当日涨幅 -2%%~8%%                                     ║
╚═════════════════════════════════════════════════════════╝
    """)


def print_results(results: list, turnover_min: float):
    if not results:
        print("未找到符合条件的股票")
        return

    print(f"\n{'═' * 100}")
    print(f"  六条件选股 — 找到 {len(results)} 只（换手率≥{turnover_min}%）")
    print(f"{'═' * 100}")
    print(f"{'#':>3} {'代码':<10} {'名称':<6} {'日期':<6} {'收盘':>7} "
          f"{'20日涨幅':>8} {'均换手':>7} {'换手今':>7} {'量比':>5} "
          f"{'多头'}")
    print(f"{'─' * 100}")

    for rank, r in enumerate(results, 1):
        print(f"{rank:3d} {r['code']:<10} {r['name']:<6} {r['date']:<10} {r['close']:>7.2f} "
              f"{r['gain20']:>+7.1f}% {r['turnover_avg']:>7.2f}% {r['turnover_today']:>6.2f}% "
              f"{r['vol_ratio']:>5.2f}x  ✓")

    print()
    print_detail(results[:10], turnover_min)


def print_detail(results: list, turnover_min: float):
    if not results:
        return
    print("=== 逐条核验 ===\n")
    for rank, r in enumerate(results, 1):
        print(f"【{rank}. {r['code']} {r['name']}】  收盘 {r['close']}")
        print(f"   ① MA5>{r['ma5']:.2f} > MA10>{r['ma10']:.2f} > MA20>{r['ma20']:.2f}  →  {'✓' if r['ma5']>r['ma10']>r['ma20'] else '✗'}")
        print(f"   ② MA5上升{'↑' if r['ma5_rise'] else '↓'}  MA10上升{'↑' if r['ma10_rise'] else '↓'}  MA20上升{'↑' if r['ma20_rise'] else '↓'}  →  {'✓' if r['ma5_rise'] and r['ma10_rise'] and r['ma20_rise'] else '✗'}")
        print(f"   ③ 近5日均换手率 {r['turnover_avg']:.2f}% (需≥{turnover_min}%)  →  {'✓' if r['turnover_avg'] >= turnover_min else '✗'}")
        print(f"   ④ 20日涨幅 {r['gain20']:+.1f}% (需15%~35%)  →  {'✓' if 15<=r['gain20']<=35 else '✗'}")
        print(f"   MA5近5日变化 {r['ma5_chg5d']:+.2f}%  MA10近5日 {r['ma10_chg5d']:+.2f}%  MA20近5日 {r['ma20_chg5d']:+.2f}%")
        print(f"   均线发散度 {r['spread']:.2f}%")
        print()


# ══════════════════════════════════════════════════════════════
# 主程序
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="六条件选股器（同时满足以下全部条件）\n\n"
                        "条件①：MA5>MA10>MA20 且 MA5>MA60, MA10>MA60（多头排列，均线在MA60上方）\n"
                        "条件②：MA5上升 且 MA10上升 且 MA20上升（均线扩散）\n"
                        "条件③：近5日平均换手率 ≥ X%%（默认 5.0%%）\n"
                        "条件④：20日涨幅在 15%%~35%% 之间\n"
                        "条件⑤：流通市值 > MKT_CAP_MIN亿（默认30亿）\n"
                        "条件⑥：当日涨幅在 GAIN_DAY_MIN%%~GAIN_DAY_MAX%%（默认 -2%%~8%%）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--top",          type=int,   default=300,      help="显示前N只（默认300）")
    parser.add_argument("--min-turnover", type=float, default=TURNOVER_MIN, help="换手率门槛%%（需≥此值，默认5.0%%）")
    parser.add_argument("--date",         type=str,   default=None,       help="信号日（格式YYYY-MM-DD，默认最新缓存日）")
    parser.add_argument("--output",     type=str,   default=None,       help="输出文件路径（默认 output/screen_trend_YYYY-MM-DD.txt）")
    parser.add_argument("--code",        type=str,   default=None,   help="单票分析")
    args = parser.parse_args()

    # ── 输出重定向到文件 ───────────────────────────────
    default_output = str(Path("output") / f"screen_trend_{args.date or 'latest'}.txt")
    out_path = Path(args.output) if args.output else Path(default_output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(out_path, "w", encoding="utf-8")
    _orig = sys.stdout

    class _Tee:
        def write(self, s):
            _orig.write(s)
            log_file.write(s)
            log_file.flush()
        def flush(self):
            _orig.flush()
            log_file.flush()

    sys.stdout = _Tee()

    print_banner()

    sys.__stdout__.write("加载股票名称 ... ")
    sys.__stdout__.flush()
    names = load_stock_names()
    sys.__stdout__.write(f"完成 ({len(names)} 只)\n\n")\nsys.__stdout__.flush()

    if args.code:
        print(f"=== 单票分析: {args.code} ===")
        df = load_qfq_history(args.code, end_date=args.date)
        r = check_ma4(df, turnover_min=args.min_turnover, signal_date=args.date)
        if r:
            r["code"] = args.code
            r["name"] = names.get(args.code, args.code)
            print_results([r], args.min_turnover)
        else:
            print(f"  {args.code} 不满足六条件")
        sys.stdout = sys.__stdout__
        log_file.close()
        print(f"\n结果已写入: {out_path}")
        return

    date_hint = f"（信号日 {args.date}）" if args.date else ""
    sys.__stdout__.write(f"全市场扫描 {date_hint}（换手率≥{args.min_turnover}%）...\n")
sys.__stdout__.flush()
    codes = get_all_stock_codes()
    sys.__stdout__.write(f"股票数量: {len(codes)}\n\n")\nsys.__stdout__.flush()

    results = scan(codes, names, turnover_min=args.min_turnover, top=args.top, signal_date=args.date)
    print_results(results, args.min_turnover)

    sys.stdout = sys.__stdout__
    log_file.close()
    print(f"\n结果已写入: {out_path}")


if __name__ == "__main__":
    main()
