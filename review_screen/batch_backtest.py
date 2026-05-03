#!/usr/bin/env python3
"""
batch_backtest.py — screen_double 批量回测 + 跨日期共性验证
==============================================================
任务：
1. 对多个信号日期批量运行收益评估
2. 汇总所有日期的收益分布、胜率、共性特征
3. 验证任务B推断出的规律在多日期中是否稳定

用法：
  python batch_backtest.py --start 2026-03-01 --end 2026-04-30
  python batch_backtest.py --dates 2026-04-08 2026-04-14 2026-04-24
  python batch_backtest.py --start 2026-03-01 --end 2026-04-30 --run
"""

import sys
import json
import time
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

WORKSPACE = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(WORKSPACE / "stock_trend" / "review_screen"))

from stock_trend.review_screen.date_utils import validate_signal_date, _is_trading_day, _prev_trading_day, _next_trading_day, get_trading_days
from stock_trend.gain_turnover import normalize_symbol, get_stock_name
import stock_trend.review_screen.screen_double as sd
import stock_trend.review_screen.eval_double as ed
import stock_trend.review_screen.analyze_double_winners as adw

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


# ── 交易日历 ─────────────────────────────────────────────
def get_available_signal_dates(start: str, end: str):
    """返回区间内所有有 screen_double_*.txt 的日期（升序）"""
    dates = []
    for f in OUTPUT_DIR.glob("screen_double_*.txt"):
        d = f.stem.replace("screen_double_", "")
        if start <= d <= end:
            dates.append(d)
    return sorted(dates)


# ── 特征收集工具（批量复用 analyze_double_winners 的提取逻辑）────
def batch_extract_features(codes_dates, signal_date):
    """codes_dates: [(code, name, future_ret, ...)]"""
    sd.preload()
    sd.load_sector_map()
    feats = []
    for item in codes_dates:
        code = normalize_symbol(item["code"])
        feat = adw.extract_t_features(code, signal_date)
        if feat:
            feat["future_ret"] = item.get("ret")
            feats.append(feat)
    return feats


# ── 全局汇总统计 ─────────────────────────────────────────
def aggregate_returns(all_results):
    """all_results: [{date: str, results: [(code, ret, ...)]}]"""
    flat = []
    for entry in all_results:
        date = entry["date"]
        for r in entry["results"]:
            flat.append({"date": date, **r})

    if not flat:
        return {}

    rets = [float(r["ret"]) for r in flat if r.get("ret") is not None]
    win = [r for r in flat if float(r.get("ret", 0)) > 0]
    loss = [r for r in flat if float(r.get("ret", 0)) <= 0]
    gt20 = [r for r in flat if float(r.get("ret", 0)) > 20]
    gt10 = [r for r in flat if float(r.get("ret", 0)) > 10]

    return {
        "total_trades": len(flat),
        "total_dates": len(all_results),
        "win_count": len(win),
        "loss_count": len(loss),
        "win_rate": round(len(win) / len(flat) * 100, 1) if flat else 0,
        "avg_return": round(float(np.mean(rets)), 2) if rets else 0,
        "median_return": round(float(np.median(rets)), 2) if rets else 0,
        "std_return": round(float(np.std(rets)), 2) if rets else 0,
        "max_return": round(float(max(rets)), 2) if rets else 0,
        "min_return": round(float(min(rets)), 2) if rets else 0,
        "gt20_count": len(gt20),
        "gt20_rate": round(len(gt20) / len(flat) * 100, 1) if flat else 0,
        "gt10_count": len(gt10),
        "gt10_rate": round(len(gt10) / len(flat) * 100, 1) if flat else 0,
        "best_trade": max(flat, key=lambda x: float(x.get("ret", 0))) if flat else None,
        "worst_trade": min(flat, key=lambda x: float(x.get("ret", 0))) if flat else None,
        "flat": flat,
    }


# ── 跨日期共性特征验证 ───────────────────────────────────
def cross_date_pattern_analysis(all_winner_feats_by_date):
    """
    all_winner_feats_by_date: [{date: str, winners: [feat_dict]}]
    合并所有日期的赢家特征，统计哪些模式在多日期中重复出现。
    """
    # 布尔特征跨日期出现率
    bool_features = [
        "normal_path", "accelerated_path", "price_above_ma10", "price_above_ma20",
        "price_above_ma60", "in_hot_sector", "ma5_above_ma60", "ma10_above_ma60",
        "ma20_above_ma60",
    ]
    # 数值特征区间
    numeric_features = [
        "rsi", "gain3", "gain5", "gain10", "avg_turn5", "avg_turn10",
        "dist_ma20", "vol_ratio_5_20", "pos_in_20d", "l2_score",
    ]

    total_dates = len(all_winner_feats_by_date)
    if total_dates == 0:
        return {}

    # 布尔：某特征在多少个日期的赢家组中 ≥60% 出现
    bool_cross_date = {}
    for feat in bool_features:
        appearances = 0
        details = []
        for entry in all_winner_feats_by_date:
            wfeats = entry.get("winners", [])
            if not wfeats:
                continue
            ratio = sum(1 for f in wfeats if f.get(feat)) / len(wfeats)
            if ratio >= 0.60:
                appearances += 1
            details.append({"date": entry["date"], "ratio": round(ratio * 100, 1)})
        bool_cross_date[feat] = {
            "appearances": appearances,
            "total_dates": total_dates,
            "hit_rate": round(appearances / total_dates * 100, 1),
            "details": details,
        }

    # 数值：各日期赢家中位数的均值，判断偏移方向是否一致
    numeric_cross_date = {}
    for feat in numeric_features:
        median_per_date = []
        for entry in all_winner_feats_by_date:
            wfeats = entry.get("winners", [])
            vals = [f.get(feat) for f in wfeats if f.get(feat) is not None]
            if len(vals) >= 3:
                median_per_date.append(float(np.median(vals)))
        if len(median_per_date) >= total_dates * 0.5:  # 至少一半日期有数据
            mean_of_medians = round(float(np.mean(median_per_date)), 2)
            std_of_medians = round(float(np.std(median_per_date)), 2)
            numeric_cross_date[feat] = {
                "mean_of_medians": mean_of_medians,
                "std_of_medians": std_of_medians,
                "dates_with_data": len(median_per_date),
                "total_dates": total_dates,
            }

    return {"bool": bool_cross_date, "numeric": numeric_cross_date}


# ── 主流程 ────────────────────────────────────────────────
def _ensure_signal_file(date, mode=None, gain20=None, turnover=None):
    """确保 screen_double_{date}.txt 存在，不存在则自动运行（带参数）。"""
    import subprocess, sys
    from pathlib import Path
    sig_file = OUTPUT_DIR / f"screen_double_{date}.txt"
    if sig_file.exists():
        return True
    screen_py = WORKSPACE / "stock_trend" / "review_screen" / "screen_double.py"
    cmd = [sys.executable, str(screen_py), "--date", date]
    if mode:
        cmd += ["--mode", mode]
    if gain20 is not None:
        cmd += ["--gain20", str(gain20)]
    if turnover is not None:
        cmd += ["--turnover", str(turnover)]
    print(f"   ⚠️  信号文件不存在，自动运行 screen_double.py {' '.join(cmd[2::2])} ...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"   ❌ screen_double.py 运行失败: {result.stderr[-300:]}")
        return False
    print(f"   ✅ screen_double.py 完成")
    return sig_file.exists()

def run_batch(args):
    t0 = time.time()

    # 提取 screen_double 参数（用于自动生成信号文件）
    mode = getattr(args, 'mode', None)
    gain20 = getattr(args, 'gain20', None)
    turnover = getattr(args, 'turnover', None)

    # 1. 确定日期列表
    if args.dates:
        dates = sorted(set(args.dates))
    elif args.start and args.end:
        if args.run:
            # --run 模式：以所有交易日为基准，信号文件缺失则自动生成
            all_trading = [d for d in get_trading_days() if args.start <= d <= args.end]
            existing = set(get_available_signal_dates(args.start, args.end))
            missing = [d for d in all_trading if d not in existing]
            if missing:
                print(f"⚠️  区间 [{args.start} ~ {args.end}] 内 {len(missing)} 个交易日缺少信号文件，将逐个生成...")
            dates = all_trading
        else:
            # 非 --run 模式：只用已有信号文件的日期
            dates = get_available_signal_dates(args.start, args.end)
    else:
        print("❌ 请指定 --start/--end 或 --dates")
        return

    if not dates:
        # 列出所有已存在的信号文件日期
        all_dates = sorted([
            f.stem.replace("screen_double_", "")
            for f in OUTPUT_DIR.glob("screen_double_*.txt")
        ])
        range_str = f"[{args.start} ~ {args.end}]" if args.start and args.end else str(args.dates)
        print(f"❌ 区间 {range_str} 内没有信号文件")
        if all_dates:
            print(f"   已有的信号文件（共 {len(all_dates)} 个）:")
            print(f"   {all_dates[0]} ~ {all_dates[-1]}")
        else:
            print("   尚未生成任何信号文件，请先运行 screen_double.py")
        return

    # 验证每个日期是否为有效交易日（非交易日自动跳过）
    invalid = [d for d in dates if not _is_trading_day(d)]
    if invalid:
        for d in invalid:
            prev = _prev_trading_day(d)
            nxt = _next_trading_day(d)
            print(f"⚠️  {d} 非交易日，已跳过（前一: {prev}, 后一: {nxt}）")
        dates = [d for d in dates if _is_trading_day(d)]
        if not dates:
            print("❌ 所有日期均非交易日")
            return

    print(f"\n{'='*70}")
    print("  batch_backtest.py  —  批量回测")
    print(f"  日期范围: {dates[0]} → {dates[-1]}（共 {len(dates)} 个信号日）")
    print(f"{'='*70}\n")

    # 2. 预加载缓存（只做一次）
    sd.preload()
    sd.load_sector_map()

    # 3. 遍历每个日期
    all_results = []      # [{date, results}]
    all_winner_feats = {}  # date -> [feat_dict]
    errors = []

    for i, date in enumerate(dates, 1):
        json_path = OUTPUT_DIR / f"eval_double_{date}.json"

        if args.run:
            # 确保信号文件存在
            if not _ensure_signal_file(date, mode=mode, gain20=gain20, turnover=turnover):
                errors.append(date)
                continue
            print(f"[{i}/{len(dates)}] 🔄 运行 eval_double --date {date} --hold-days {args.hold_days}")
            try:
                entry = ed.eval_signal(date, top_n=args.top_n, hold_days=args.hold_days)
                if entry:
                    all_results.append({"date": date, **entry})
                else:
                    errors.append(date)
            except Exception as e:
                print(f"   ❌ {date}: {e}")
                errors.append(date)
                continue
        else:
            # 读现有结果
            if not json_path.exists():
                print(f"[{i}/{len(dates)}] ⏭  {date} 无回测结果，跳过（加 --run 重新计算）")
                errors.append(date)
                continue
            with open(json_path, "r", encoding="utf-8") as f:
                entry = json.load(f)
            all_results.append({"date": date, **entry})
            print(f"[{i}/{len(dates)}] ✅ 读取 {date}，{entry['evaluated']} 只")

    if not all_results:
        print("❌ 没有任何有效结果")
        return

    # 4. 全局收益统计
    stats = aggregate_returns(all_results)
    flat = stats.get("flat", [])

    print(f"\n{'='*70}")
    print("📊 批量回测汇总")
    print(f"{'='*70}")
    print(f"   总交易次数: {stats['total_trades']}")
    print(f"   信号日数量: {stats['total_dates']}")
    print(f"   胜率: {stats['win_count']}/{stats['total_trades']} = {stats['win_rate']}%")
    print(f"   收益率>10%: {stats['gt10_count']}/{stats['total_trades']} = {stats['gt10_rate']}%")
    print(f"   收益率>20%: {stats['gt20_count']}/{stats['total_trades']} = {stats['gt20_rate']}%")
    print(f"   平均收益: {stats['avg_return']:+.2f}%")
    print(f"   中位收益: {stats['median_return']:+.2f}%")
    print(f"   标准差: {stats['std_return']:.2f}%")
    print(f"   最大盈利: {stats['max_return']:+.2f}%  ({stats['best_trade']['code'] if stats['best_trade'] else '-'})")
    print(f"   最大亏损: {stats['min_return']:+.2f}%  ({stats['worst_trade']['code'] if stats['worst_trade'] else '-'})")

    # 5. 每日收益分布
    print(f"\n📅 每日收益明细")
    print(f"{'日期':<14}{'交易数':>6}{'胜率':>8}{'平均收益':>10}{'>20%':>6}{'>10%':>6}")
    print("-" * 62)
    for entry in all_results:
        date = entry["date"]
        results = entry["results"]
        rets = [float(r.get("ret", 0)) for r in results]
        wins = sum(1 for r in rets if r > 0)
        gt20 = sum(1 for r in rets if r > 20)
        gt10 = sum(1 for r in rets if r > 10)
        avg = np.mean(rets) if rets else 0
        print(f"{date:<14}{len(results):>6}{wins/len(results)*100:>7.1f}%{avg:>+10.2f}%{gt20:>6}{gt10:>6}")

    # 6. 批量提取赢家特征（收益 > threshold）
    threshold = args.threshold
    print(f"\n🏆 共性特征分析（收益 > {threshold:.1f}%）")
    print("-" * 50)

    winner_feats_by_date = []
    for entry in all_results:
        date = entry["date"]
        winners = [r for r in entry["results"] if float(r.get("ret", 0)) > threshold]
        if not winners:
            continue

        # 提取赢家特征
        feats = batch_extract_features(winners, date)
        winner_feats_by_date.append({"date": date, "winners": feats})

    if winner_feats_by_date:
        cross = cross_date_pattern_analysis(winner_feats_by_date)
        bool_cross = cross.get("bool", {})
        numeric_cross = cross.get("numeric", {})

        # 布尔特征跨日期稳定性
        stable_bools = [(k, v) for k, v in bool_cross.items() if v["hit_rate"] >= 50]
        stable_bools.sort(key=lambda x: -x[1]["hit_rate"])

        print(f"\n✅ 布尔特征跨日期稳定性（出现在≥50%日期的赢家组中）")
        print(f"{'特征':<20}{'出现率':>8}{'涉及日期':>10}")
        print("-" * 40)
        for feat, info in stable_bools:
            print(f"{feat:<20}{info['hit_rate']:>7.1f}%{info['appearances']:>10}/{info['total_dates']}")

        # 数值特征跨日期稳定性
        print(f"\n📐 数值特征跨日期中位数汇总")
        print(f"{'特征':<18}{'赢家中位数均值':>16}{'波动范围':>14}{'数据日期数':>12}")
        print("-" * 62)
        for feat, info in sorted(numeric_cross.items(), key=lambda x: x[1]["mean_of_medians"]):
            print(f"{feat:<18}{info['mean_of_medians']:>16.2f}{info['std_of_medians']:>14.2f}{info['dates_with_data']:>12}/{info['total_dates']}")

        # 汇总共性结论
        print(f"\n🔍 跨日期共性结论（出现率≥60%的布尔特征）")
        high_conf = [(k, v) for k, v in bool_cross.items() if v["hit_rate"] >= 60]
        high_conf.sort(key=lambda x: -x[1]["hit_rate"])
        if high_conf:
            for feat, info in high_conf:
                print(f"   · {feat}: 赢家组中约 {info['hit_rate']}% 满足该条件（在 {info['appearances']}/{info['total_dates']} 个日期）")
        else:
            print("   （当前日期样本较少，未出现高置信度共性）")

    # 7. 保存批量结果
    out = {
        "args": vars(args),
        "stats": {k: v for k, v in stats.items() if k != "flat"},
        "all_results": [{"date": e["date"], "results": e["results"]} for e in all_results],
        "winner_feats_by_date": winner_feats_by_date,
        "errors": errors,
    }
    out_path = OUTPUT_DIR / "batch_backtest_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"\n💾 汇总已保存: {out_path}")
    print(f"⏱ 总用时 {time.time() - t0:.1f}秒")


# ── CLI ──────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="screen_double 批量回测 + 跨日期共性验证")
    parser.add_argument("--start", help="起始日期（包含），如 2026-03-01")
    parser.add_argument("--end", help="结束日期（包含），如 2026-04-30")
    parser.add_argument("--dates", nargs="+", help="指定日期列表，如 2026-04-08 2026-04-14")
    parser.add_argument("--top-n", type=int, default=300, help="每个信号日评估前N只")
    parser.add_argument("--threshold", type=float, default=20.0, help="共性分析收益阈值")
    parser.add_argument("--hold-days", type=int, default=10, help="持有交易日数（默认10）")
    parser.add_argument("--run", action="store_true", help="重新运行评估（不加则只读现有JSON）")
    parser.add_argument("--mode", default="normal", choices=["normal", "accelerated", "winner"],
                        help="screen_double 条件3模式（默认 normal）")
    parser.add_argument("--gain20", type=float, default=None,
                        help="screen_double 20日涨幅最低门槛%%（默认不设限）")
    parser.add_argument("--turnover", type=float, default=None,
                        help="screen_double 5日均换手率最低门槛%%（默认按市值规则）")
    args = parser.parse_args()
    run_batch(args)


if __name__ == "__main__":
    main()