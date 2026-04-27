#!/usr/bin/env python3
"""
T+1 赢家特征分析 v3（每日独立分析版）
=========================================
核心理念：先看信号质量，再验证结果。

每天流程：
  T日  →  运行 screen_strat1_v2 取 top 信号
        →  分析这批信号的 T日 特征分布（不看T+1）
        →  记录"信号质量指标"
        →  找出其中 T+1 涨幅≥5% 的股票
        →  记录这些赢家的 T日 特征

最终目标：
  学习赢家的 T日 特征，找出优化筛选条件的规律。

用法：
  python analyze_T1_winners.py --start 2025-12-18 --end 2026-04-23
  python analyze_T1_winners.py --start 2025-12-18 --end 2026-04-23 --min-gain 5.0 --top-n 80
"""
import sys
import json
import time
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

WORKSPACE = Path.home() / ".openclaw" / "workspace"

sys.path.insert(0, str(WORKSPACE / "stock_trend"))
from gain_turnover import (
    load_qfq_history,
    get_all_stock_codes,
    load_stock_names,
    compute_rsi_scalar,
    normalize_symbol,
)

sys.path.insert(0, str(WORKSPACE / "stock_trend" / "review_screen"))
from screen_strat1_v2 import preload, check_base, score_stock, load_hot_sectors

# ─────────────────────────────────────────────────────────────
def get_trading_dates(code="sh000001", start="", end="") -> list:
    df = load_qfq_history(code, end_date=end)
    dates = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d").tolist()
    dates = sorted(set(dates))
    if start: dates = [d for d in dates if d >= start]
    if end:   dates = [d for d in dates if d <= end]
    return dates

def get_next_date(dates: list, cur: str) -> str | None:
    try:
        i = dates.index(cur)
        return dates[i+1] if i+1 < len(dates) else None
    except ValueError:
        return None

# ─────────────────────────────────────────────────────────────
def collect_T_day_features(code: str, sig_date: str) -> dict | None:
    """收集股票在信号日T的关键特征（用于预测）。"""
    try:
        df = load_qfq_history(code, end_date=sig_date)
        df = df.sort_values("date").tail(80).reset_index(drop=True)
        if len(df) < 30:
            return None
        close = df["close"].values
        high  = df["high"].values
        low   = df["low"].values
        vol   = df["volume"].values
        op    = df["open"].values
        i = len(close) - 1

        ma5  = float(np.mean(close[i-4:i+1]))
        ma10 = float(np.mean(close[i-9:i+1]))
        ma20 = float(np.mean(close[i-19:i+1]))
        ma60 = float(np.mean(close[max(0,i-59):i+1])) if i >= 60 else ma20

        # MA20方向（5日）
        if i >= 25:
            ma20_now  = float(np.mean(close[i-19:i+1]))
            ma20_5ago = float(np.mean(close[i-24:i-4]))
            ma20_dir = (ma20_now/ma20_5ago - 1)*100 if ma20_5ago > 0 else 0.0
        else:
            ma20_dir = 0.0

        # 换手率
        turnover = float(df["turnover"].iloc[i]) if "turnover" in df.columns else 0.0

        # 量比
        vol5  = float(np.mean(vol[i-4:i+1]))
        vol20 = float(np.mean(vol[i-19:i+1]))
        vol_ratio = vol5 / vol20 if vol20 > 0 else 0.0

        # RSI
        rsi = compute_rsi_scalar(close, 14)

        # 涨幅
        gain1  = (close[i]/close[i-1]-1)*100  if i>=1 and close[i-1]>0 else 0.0
        gain3  = (close[i]/close[i-3]-1)*100  if i>=3 and close[i-3]>0 else 0.0
        gain5  = (close[i]/close[i-5]-1)*100  if i>=5 and close[i-5]>0 else 0.0
        gain10 = (close[i]/close[i-10]-1)*100 if i>=10 and close[i-10]>0 else 0.0
        gain20 = (close[i]/close[i-20]-1)*100 if i>=20 and close[i-20]>0 else 0.0

        # 偏离MA20
        dist_ma20 = (close[i]-ma20)/ma20*100 if ma20>0 else 0.0

        # 阳线实体
        entity = (close[i]-op[i])/op[i]*100 if op[i]>0 else 0.0

        # 下影线
        body = abs(close[i]-op[i])
        lower_shadow = min(op[i], close[i]) - low[i]
        lower_r = lower_shadow/body if body>0 else 0.0

        # 波动率
        if i >= 5:
            dret = np.diff(close[i-5:i+1]) / close[i-5:i]
            vol_cv = float(np.std(dret))
        else:
            vol_cv = 0.0

        # 波段位置：当前价在近20日区间的位置
        high20 = float(np.max(high[i-19:i+1]))
        low20  = float(np.min(low[i-19:i+1]))
        pos_in_band = (close[i]-low20)/(high20-low20)*100 if high20>low20 else 50.0

        return {
            "code": code,
            "close": float(close[i]),
            "ma5": round(ma5,2), "ma10": round(ma10,2),
            "ma20": round(ma20,2), "ma60": round(ma60,2),
            "ma5_ma10_spread": round((ma5/ma10-1)*100, 3),
            "ma10_ma20_spread": round((ma10/ma20-1)*100, 3),
            "ma20_ma60_spread": round((ma20/ma60-1)*100, 3),
            "ma20_dir_5d": round(ma20_dir, 3),
            "price_above_ma5": bool(close[i] > ma5),
            "price_above_ma20": bool(close[i] > ma20),
            "price_above_ma60": bool(close[i] > ma60),
            "turnover": round(turnover, 2),
            "vol_ratio_5d": round(vol_ratio, 3),
            "vol_cv": round(vol_cv, 4),
            "rsi": round(float(rsi), 1),
            "gain1": round(gain1, 2), "gain3": round(gain3, 2),
            "gain5": round(gain5, 2), "gain10": round(gain10, 2),
            "gain20": round(gain20, 2),
            "dist_ma20": round(dist_ma20, 2),
            "entity_pct": round(entity, 2),
            "lower_ratio": round(lower_r, 2),
            "pos_in_band": round(pos_in_band, 1),
        }
    except:
        return None

# ─────────────────────────────────────────────────────────────
def run(start: str, end: str, min_gain: float = 5.0, top_n: int = 80):
    t0 = time.time()
    print(f"\n🔍 T+1赢家每日独立分析 v3")
    print(f"   信号日: {start} → {end}")
    print(f"   T+1涨幅阈值: ≥{min_gain}%")
    print(f"   每日最多信号: {top_n}")

    preload()
    codes = get_all_stock_codes()
    names = load_stock_names()
    all_dates = get_trading_dates(start=start, end=end)
    print(f"   股票: {len(codes)} 只 | 交易日: {len(all_dates)} 天")

    # ── 每日分析结果 ─────────────────────────────────────
    daily_results = []  # 每天一条记录

    for sig_date in all_dates:
        hot = load_hot_sectors()

        # T日筛选
        base_signals = []
        for code in codes:
            sig = check_base(code, sig_date)
            if sig:
                base_signals.append(sig)

        final_signals = []
        for sig in base_signals:
            ok, sc, reasons = score_stock(sig, hot)
            if ok:
                sig["_score"] = sc
                final_signals.append(sig)

        final_signals.sort(key=lambda x: (-x["_score"], -x["rsi"], -x["gain5d"]))
        top_signals = final_signals[:top_n]

        # ── Step 1：收集这批 top 信号在 T日 的特征 ─────
        # 不看 T+1，先记录信号质量
        signal_chars = []
        for sig in top_signals:
            chars = collect_T_day_features(sig["code"], sig_date)
            if chars:
                chars["_strat_score"] = sig["_score"]
                signal_chars.append(chars)

        # ── Step 2：计算 T日 信号质量指标（不看T+1） ─────
        def dist_stats(lst, field):
            vals = [c[field] for c in lst if field in c]
            if not vals: return {}
            a = np.array(vals, dtype=float)
            return {"mean": round(float(np.mean(a)),2),
                    "median": round(float(np.median(a)),2),
                    "std": round(float(np.std(a)),2)}

        sig_rsi    = dist_stats(signal_chars, "rsi")
        sig_turn   = dist_stats(signal_chars, "turnover")
        sig_vr     = dist_stats(signal_chars, "vol_ratio_5d")
        sig_gain5  = dist_stats(signal_chars, "gain5")
        sig_gain20 = dist_stats(signal_chars, "gain20")
        sig_dist   = dist_stats(signal_chars, "dist_ma20")

        # 均线多头排列比例（仅统计）
        pct_above_ma5  = sum(1 for c in signal_chars if c["price_above_ma5"])  / len(signal_chars)*100 if signal_chars else 0
        pct_above_ma20 = sum(1 for c in signal_chars if c["price_above_ma20"]) / len(signal_chars)*100 if signal_chars else 0
        pct_ma5_gt_ma10 = sum(1 for c in signal_chars if c["ma5_ma10_spread"]>0) / len(signal_chars)*100 if signal_chars else 0
        pct_ma10_gt_ma20 = sum(1 for c in signal_chars if c["ma10_ma20_spread"]>0) / len(signal_chars)*100 if signal_chars else 0

        # ── Step 3：找 T+1 涨幅 ≥ min_gain 的赢家 ──────
        t1_date = get_next_date(all_dates, sig_date)
        winners = []
        all_T1_gains = []

        for chars in signal_chars:
            code = chars["code"]
            try:
                df_c = load_qfq_history(code, end_date=t1_date)
                df_c = df_c.sort_values("date")
                df_c["date"] = pd.to_datetime(df_c["date"]).dt.strftime("%Y-%m-%d")
                pmap = dict(zip(df_c["date"], df_c["close"]))
                p0 = pmap.get(sig_date)
                p1 = pmap.get(t1_date)
                if p0 and p0 > 0 and p1:
                    t1g = (p1/p0 - 1)*100
                    all_T1_gains.append(t1g)
                    if t1g >= min_gain:
                        chars2 = dict(chars)
                        chars2["T1_gain"] = round(t1g, 2)
                        winners.append(chars2)
            except:
                pass

        winner_count = len(winners)
        winner_rate = winner_count / len(signal_chars)*100 if signal_chars else 0
        avg_T1_gain = float(np.mean(all_T1_gains)) if all_T1_gains else 0.0

        # ── Step 4：赢家的 T日 特征 ───────────────────
        if winners:
            w_rsi   = dist_stats(winners, "rsi")
            w_turn  = dist_stats(winners, "turnover")
            w_vr    = dist_stats(winners, "vol_ratio_5d")
            w_gain5 = dist_stats(winners, "gain5")
            w_gain20= dist_stats(winners, "gain20")
            w_dist  = dist_stats(winners, "dist_ma20")
            w_pos   = dist_stats(winners, "pos_in_band")
            w_ma20_dir = dist_stats(winners, "ma20_dir_5d")
        else:
            w_rsi = w_turn = w_vr = w_gain5 = w_gain20 = w_dist = w_pos = w_ma20_dir = {}

        day_record = {
            "signal_date": sig_date,
            "T1_date": t1_date,
            "signal_count": len(signal_chars),
            "winner_count": winner_count,
            "winner_rate_pct": round(winner_rate, 2),
            "avg_T1_gain_all": round(avg_T1_gain, 2),

            # 全部信号的质量（先验）
            "signal_rsi_mean":    sig_rsi.get("mean"),
            "signal_rsi_std":     sig_rsi.get("std"),
            "signal_turnover_mean": sig_turn.get("mean"),
            "signal_vol_ratio_mean": sig_vr.get("mean"),
            "signal_gain5_mean": sig_gain5.get("mean"),
            "signal_gain20_mean":sig_gain20.get("mean"),
            "signal_dist_mean":   sig_dist.get("mean"),
            "pct_above_ma5":  round(pct_above_ma5, 1),
            "pct_above_ma20": round(pct_above_ma20, 1),
            "pct_ma5_gt_ma10": round(pct_ma5_gt_ma10, 1),
            "pct_ma10_gt_ma20": round(pct_ma10_gt_ma20, 1),

            # 赢家的 T日 特征（后验）
            "winner_rsi_mean":    w_rsi.get("mean"),
            "winner_rsi_std":     w_rsi.get("std"),
            "winner_turnover_mean": w_turn.get("mean"),
            "winner_vol_ratio_mean": w_vr.get("mean"),
            "winner_gain5_mean":  w_gain5.get("mean"),
            "winner_gain20_mean": w_gain20.get("mean"),
            "winner_dist_mean":   w_dist.get("mean"),
            "winner_pos_in_band_mean": w_pos.get("mean"),
            "winner_ma20_dir_mean": w_ma20_dir.get("mean"),

            # 每日赢家的完整T日特征列表（用于共性学习）
            "winners": [{k:v for k,v in w.items() if not k.startswith("_")}
                        for w in winners],
        }
        daily_results.append(day_record)

        # 进度输出
        winner_flag = "✅" if winner_count > 0 else "  "
        print(f"  {sig_date} → {t1_date}: "
              f"信号{len(signal_chars)}只 "
              f"赢{winner_count}只({winner_rate:.0f}%) "
              f"信号RSI={sig_rsi.get('mean','?')} "
              f"赢RSI={w_rsi.get('mean','?')} "
              f"赢T+1={avg_T1_gain:+.1f}%", flush=True)

    # ── 跨日学习 ──────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  跨日共性学习")
    print(f"{'='*70}")

    # 只用有赢家的日子（更真实）
    winner_days = [d for d in daily_results if d["winner_count"] > 0]
    print(f"\n  有赢家的交易日: {len(winner_days)} / {len(daily_results)} 天")

    if not winner_days:
        print("  ⚠️ 无赢家数据，无法分析")
        return

    # 对比：赢家的T日特征 vs 全部信号的T日特征
    all_chars_flat = []
    for d in winner_days:
        all_chars_flat.extend(d["winners"])

    print(f"  赢家样本总数: {len(all_chars_flat)} 只")

    # 计算赢家的特征区间
    num_fields = ["rsi","turnover","vol_ratio_5d","gain5","gain20",
                  "dist_ma20","pos_in_band","ma20_dir_5d","entity_pct","lower_ratio"]
    bin_fields = ["price_above_ma5","price_above_ma20","price_above_ma60"]

    print(f"\n  {'赢家T日特征':<25} {'均值':>8} {'中位数':>8} {'P10':>8} {'P90':>8}")
    print(f"  {'-'*65}")
    for f in num_fields:
        vals = [c[f] for c in all_chars_flat if f in c]
        if not vals: continue
        a = np.array(vals, dtype=float)
        print(f"  {f:<25} {np.mean(a):>8.2f} {np.median(a):>8.2f} "
              f"{np.percentile(a,10):>8.2f} {np.percentile(a,90):>8.2f}")

    print(f"\n  {'赢家T日布尔特征':<25} {'占比':>8}")
    print(f"  {'-'*40}")
    for f in bin_fields:
        pct = sum(1 for c in all_chars_flat if c.get(f)) / len(all_chars_flat)*100
        print(f"  {f:<25} {pct:>7.1f}%")

    # 字段名映射：比较用的field名 -> daily_results里的key
    _key_map = {
        "turnover": ("winner_turnover_mean", "signal_turnover_mean"),
        "vol_ratio_5d": ("winner_vol_ratio_mean", "signal_vol_ratio_mean"),
    }
    _fields = ["rsi","turnover","vol_ratio_5d","gain5","gain20","dist_ma20"]

    print(f"\n  {'赢家 vs 信号均值对比':<30} {'赢家均值':>8} {'信号均值':>8} {'差异':>8}")
    print(f"  {'-'*60}")
    for f in _fields:
        wkey, skey = _key_map.get(f, (f"winner_{f}_mean", f"signal_{f}_mean"))
        wv = np.mean([d[wkey] for d in winner_days if d.get(wkey)]) or 0
        sv = np.mean([d[skey] for d in daily_results if d.get(skey)]) or 0
        diff = wv - sv
        arrow = "↑" if diff > 0 else "↓"
        print(f"  {f:<30} {wv:>8.2f} {sv:>8.2f} {diff:>+7.2f}{arrow}")

    # ── 信号质量能否预测赢家率？ ─────────────────────────
    print(f"\n  每日赢家率分布:")
    rates = [d["winner_rate_pct"] for d in daily_results if d["winner_count"]>0]
    print(f"    均值={np.mean(rates):.1f}%  中位数={np.median(rates):.1f}%  "
          f"最大={np.max(rates):.1f}%  最小={np.min(rates):.1f}%")

    # ── 哪些天赢家多？ ──────────────────────────────────
    print(f"\n  赢家最多的5天:")
    top_days = sorted(winner_days, key=lambda d: -d["winner_count"])[:5]
    for d in top_days:
        print(f"    {d['signal_date']}: {d['winner_count']}只  "
              f"信号RSI={d['signal_rsi_mean']}  "
              f"赢RSI={d['winner_rsi_mean']}  "
              f"5日涨={d['winner_gain5_mean']}%")

    # ── 赢家典型T日特征画像 ─────────────────────────────
    print(f"\n  📋 赢家T日典型画像（按频率最高的值）:")
    for f in ["rsi","turnover","vol_ratio_5d","gain5","dist_ma20"]:
        vals = [c[f] for c in all_chars_flat if f in c]
        if not vals: continue
        a = np.array(vals)
        p25, p75 = np.percentile(a,25), np.percentile(a,75)
        print(f"    {f}: P25={p25:.1f}  中位数={np.median(a):.1f}  P75={p75:.1f}")

    # ── 保存 ──────────────────────────────────────────────
    out_path = WORKSPACE / "stock_reports" / f"T1_daily_v3_{start}_{end}.json"
    with open(out_path,"w",encoding="utf-8") as f:
        json.dump({
            "params": {"start":start,"end":end,"min_gain":min_gain,"top_n":top_n},
            "daily_results": daily_results,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n💾 已保存: {out_path}")
    print(f"⏱ {time.time()-t0:.0f}秒")

# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2025-12-18")
    p.add_argument("--end",   default="2026-04-23")
    p.add_argument("--min-gain", type=float, default=5.0)
    p.add_argument("--top-n", type=int, default=80)
    a = p.parse_args()
    run(a.start, a.end, a.min_gain, a.top_n)
