#!/usr/bin/env python3
"""
analyze_double_winners.py — screen_double T+10赢家特征分析
=============================================================
任务B：
1. 读取 eval_double_{T}.json
2. 筛选收益 > threshold 的赢家股票
3. 提取这些赢家在 T 日及以前的技术特征
4. 与全样本组对比，找出较明显的共性特征

用法：
  python analyze_double_winners.py --date 2026-04-08
  python analyze_double_winners.py --date 2026-04-08 --threshold 20
  python analyze_double_winners.py --date 2026-04-08 --threshold 20 --top-feature-n 12
"""

import sys
import json
import time
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

WORKSPACE = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(WORKSPACE))

from stock_trend.review_screen.date_utils import validate_signal_date
from stock_trend.gain_turnover import normalize_symbol, load_stock_names, get_stock_name
from stock_trend.review_screen import screen_double as sd

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


# ── 工具函数 ─────────────────────────────────────────────
def pct(v, n=1):
    return round(float(v) * 100, n)


def mean_or_none(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return float(np.mean(vals))


def median_or_none(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return float(np.median(vals))


def ratio_true(items, key):
    vals = [bool(x.get(key, False)) for x in items if key in x]
    if not vals:
        return None
    return sum(vals) / len(vals)


_names_cache = None  # lazy load

def _get_names_cache():
    global _names_cache
    if _names_cache is None:
        _names_cache = load_stock_names()
    return _names_cache

# ── 读取回测结果 ─────────────────────────────────────────
def load_eval_json(signal_date: str):
    path = OUTPUT_DIR / f"eval_double_{signal_date}.json"
    if not path.exists():
        raise FileNotFoundError(f"评估文件不存在: {path}\n请先运行: python eval_double.py --date {signal_date}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── 特征提取（只用T日及以前数据）──────────────────────────
def extract_t_features(code: str, signal_date: str):
    """提取单只股票在 T 日的技术特征，只使用 T 日及以前数据。"""
    try:
        base = sd.check_base(code, signal_date)
        if base is None:
            return None

        code = normalize_symbol(code)
        df = sd._price.get(code)
        if df is None or df.empty:
            return None
        il = df["date"].tolist()
        try:
            idx = il.index(signal_date)
        except Exception:
            return None
        if idx < 65:
            return None

        hist = df.iloc[:idx + 1].copy().reset_index(drop=True)
        closes = hist["close"].values.astype(float)
        opens = hist["open"].values.astype(float) if "open" in hist.columns else None
        highs = hist["high"].values.astype(float) if "high" in hist.columns else None
        lows = hist["low"].values.astype(float) if "low" in hist.columns else None
        vols = hist["volume"].values.astype(float) if "volume" in hist.columns else None
        if "true_turnover" in hist.columns and hist["true_turnover"].notna().any():
            turns = hist["true_turnover"].values.astype(float)
        elif "turnover" in hist.columns:
            turns = hist["turnover"].values.astype(float)
        else:
            turns = np.zeros(len(hist), dtype=float)

        t = len(hist) - 1
        close_t = float(closes[t])
        ma5 = sd.calc_ma(closes, 5)
        ma10 = sd.calc_ma(closes, 10)
        ma20 = sd.calc_ma(closes, 20)
        ma60 = sd.calc_ma(closes, 60)
        d5 = sd.ma_direction(closes, 5)
        d10 = sd.ma_direction(closes, 10)
        d20 = sd.ma_direction(closes, 20)
        d60 = sd.ma_direction(closes, 60)
        macd, dif, dea = sd.calc_macd(closes)
        rsi = sd.calc_rsi(closes)

        gain1 = (close_t / closes[t-1] - 1) * 100 if t >= 1 and closes[t-1] > 0 else None
        gain3 = (close_t / closes[t-3] - 1) * 100 if t >= 3 and closes[t-3] > 0 else None
        gain5 = (close_t / closes[t-5] - 1) * 100 if t >= 5 and closes[t-5] > 0 else None
        gain10 = (close_t / closes[t-10] - 1) * 100 if t >= 10 and closes[t-10] > 0 else None
        gain20 = (close_t / closes[t-20] - 1) * 100 if t >= 20 and closes[t-20] > 0 else None

        avg_turn5 = float(np.mean(turns[t-4:t+1])) if t >= 4 else float(turns[t])
        avg_turn10 = float(np.mean(turns[t-9:t+1])) if t >= 9 else avg_turn5
        turn_today = float(turns[t])
        vol5 = float(np.mean(vols[t-4:t+1])) if vols is not None and t >= 4 else None
        vol20 = float(np.mean(vols[t-19:t+1])) if vols is not None and t >= 19 else None
        vol_ratio_5_20 = (vol5 / vol20) if (vol5 is not None and vol20 not in (None, 0)) else None

        body_pct = None
        if opens is not None and opens[t] > 0:
            body_pct = (close_t / float(opens[t]) - 1) * 100

        amp_today = None
        if highs is not None and lows is not None and close_t > 0:
            amp_today = (float(highs[t]) - float(lows[t])) / close_t * 100

        high20 = float(np.max(highs[t-19:t+1])) if highs is not None and t >= 19 else None
        low20 = float(np.min(lows[t-19:t+1])) if lows is not None and t >= 19 else None
        pos_in_20d = None
        if high20 is not None and low20 is not None and high20 > low20:
            pos_in_20d = (close_t - low20) / (high20 - low20) * 100

        above_ma5_days = 0
        for i in range(max(0, t - 4), t + 1):
            ma5_i = sd.calc_ma(closes[:i+1], 5)
            if ma5_i is not None and closes[i] > ma5_i:
                above_ma5_days += 1

        sector = sd.get_sector(code)
        hot = sd.load_hot_sectors()
        in_hot_sector = bool(sector and sector in hot)

        normal_path = (d5 == 1 and d10 == 1 and d20 == 1 and d60 == 1)
        gain_today = (close_t / closes[t-1] - 1) * 100 if t >= 1 and closes[t-1] > 0 else 0.0
        accelerated_path = (d10 == 1 and d20 == 1 and d60 == 1 and gain_today > 5.0 and turn_today >= 10.0)

        # 二层分数也带出来，方便赢家组 vs 全样本组对比
        market_pool = sd._build_market_rps_pool(list(sd._price.keys()), signal_date)
        enriched = [dict(base)]
        sd._inject_rps(enriched, market_pool)
        hot_sectors = sd.load_hot_sectors()
        ok, score, reasons = sd.score_stock(enriched[0], hot_sectors)

        return {
            "code": code,
            "name": get_stock_name(code, _get_names_cache()),
            "signal_date": signal_date,
            "close": round(close_t, 2),
            "ma5": round(float(ma5), 2),
            "ma10": round(float(ma10), 2),
            "ma20": round(float(ma20), 2),
            "ma60": round(float(ma60), 2),
            "ma5_above_ma60": bool(ma5 > ma60),
            "ma10_above_ma60": bool(ma10 > ma60),
            "ma20_above_ma60": bool(ma20 > ma60),
            "d5": int(d5), "d10": int(d10), "d20": int(d20), "d60": int(d60),
            "normal_path": bool(normal_path),
            "accelerated_path": bool(accelerated_path),
            "price_above_ma10": bool(close_t > ma10),
            "price_above_ma20": bool(close_t > ma20),
            "price_above_ma60": bool(close_t > ma60),
            "dist_ma20": round(float(base.get("dist_ma20", 0.0)), 2),
            "dist_ma60": round((close_t - ma60) / ma60 * 100, 2) if ma60 else None,
            "rsi": round(float(rsi), 2) if rsi is not None else None,
            "dif": round(float(dif), 4) if dif is not None else None,
            "dea": round(float(dea), 4) if dea is not None else None,
            "macd": round(float(macd), 4) if macd is not None else None,
            "gain1": round(gain1, 2) if gain1 is not None else None,
            "gain3": round(gain3, 2) if gain3 is not None else None,
            "gain5": round(gain5, 2) if gain5 is not None else None,
            "gain10": round(gain10, 2) if gain10 is not None else None,
            "gain20": round(gain20, 2) if gain20 is not None else None,
            "turn_today": round(turn_today, 2),
            "avg_turn5": round(avg_turn5, 2),
            "avg_turn10": round(avg_turn10, 2),
            "vol_ratio_5_20": round(vol_ratio_5_20, 3) if vol_ratio_5_20 is not None else None,
            "body_pct": round(body_pct, 2) if body_pct is not None else None,
            "amp_today": round(amp_today, 2) if amp_today is not None else None,
            "pos_in_20d": round(pos_in_20d, 2) if pos_in_20d is not None else None,
            "above_ma5_days_5d": int(above_ma5_days),
            "market_cap": float(base.get("market_cap", 0.0)) if base.get("market_cap") is not None else None,
            "sector": sector,
            "in_hot_sector": in_hot_sector,
            "l2_score": round(float(score), 2) if ok else None,
            "l2_reasons": reasons,
        }
    except Exception:
        return None


# ── 共性分析 ─────────────────────────────────────────────
def summarize_numeric(all_feats, winner_feats, key):
    all_vals = [x.get(key) for x in all_feats if x.get(key) is not None]
    win_vals = [x.get(key) for x in winner_feats if x.get(key) is not None]
    if not all_vals or not win_vals:
        return None
    return {
        "feature": key,
        "all_mean": round(float(np.mean(all_vals)), 2),
        "all_median": round(float(np.median(all_vals)), 2),
        "winner_mean": round(float(np.mean(win_vals)), 2),
        "winner_median": round(float(np.median(win_vals)), 2),
        "delta_mean": round(float(np.mean(win_vals) - np.mean(all_vals)), 2),
    }


def summarize_bool(all_feats, winner_feats, key):
    all_ratio = ratio_true(all_feats, key)
    win_ratio = ratio_true(winner_feats, key)
    if all_ratio is None or win_ratio is None:
        return None
    return {
        "feature": key,
        "all_ratio": round(all_ratio * 100, 1),
        "winner_ratio": round(win_ratio * 100, 1),
        "delta_ratio": round((win_ratio - all_ratio) * 100, 1),
    }


def infer_patterns(all_feats, winner_feats):
    patterns = []
    if not winner_feats:
        return patterns

    bool_keys = [
        "normal_path", "accelerated_path", "price_above_ma10", "price_above_ma20",
        "price_above_ma60", "in_hot_sector", "ma5_above_ma60", "ma10_above_ma60", "ma20_above_ma60",
    ]
    for key in bool_keys:
        info = summarize_bool(all_feats, winner_feats, key)
        if info and info["winner_ratio"] >= 60 and info["delta_ratio"] >= 10:
            patterns.append({
                "type": "bool",
                "feature": key,
                "winner_ratio": info["winner_ratio"],
                "all_ratio": info["all_ratio"],
                "delta_ratio": info["delta_ratio"],
                "note": f"赢家组该特征占比 {info['winner_ratio']}%，较全样本高 {info['delta_ratio']}pct",
            })

    # 数值特征区间共性
    numeric_keys = [
        "rsi", "gain1", "gain3", "gain5", "gain10", "gain20", "avg_turn5", "avg_turn10",
        "dist_ma20", "dist_ma60", "vol_ratio_5_20", "body_pct", "amp_today", "pos_in_20d", "l2_score",
    ]
    for key in numeric_keys:
        vals = [x.get(key) for x in winner_feats if x.get(key) is not None]
        if len(vals) < 3:
            continue
        q25 = float(np.percentile(vals, 25))
        q75 = float(np.percentile(vals, 75))
        all_info = summarize_numeric(all_feats, winner_feats, key)
        if all_info is None:
            continue
        # 只保留和全样本均值差异相对明显的特征
        if abs(all_info["delta_mean"]) >= max(1.0, abs(all_info["all_mean"]) * 0.15):
            patterns.append({
                "type": "range",
                "feature": key,
                "winner_iqr": [round(q25, 2), round(q75, 2)],
                "winner_mean": all_info["winner_mean"],
                "all_mean": all_info["all_mean"],
                "delta_mean": all_info["delta_mean"],
                "note": f"赢家组 {key} 中位区间约 {q25:.2f}~{q75:.2f}，均值较全样本偏移 {all_info['delta_mean']:+.2f}",
            })

    # 按差异强度排序
    def score(p):
        if p["type"] == "bool":
            return abs(p["delta_ratio"])
        return abs(p["delta_mean"])

    patterns.sort(key=score, reverse=True)
    return patterns


def run(signal_date: str, threshold: float = 20.0, top_feature_n: int = 12):
    t0 = time.time()
    print(f"\n{'='*70}")
    print("  analyze_double_winners.py  —  赢家特征分析")
    print(f"  信号日: {signal_date} | 收益阈值: > {threshold:.1f}%")
    print(f"{'='*70}\n")

    data = load_eval_json(signal_date)
    results = data.get("results", [])
    if not results:
        print("❌ 回测结果为空")
        return

    winners = [r for r in results if float(r.get("ret", 0)) > threshold]
    if not winners:
        print(f"❌ 没有收益 > {threshold:.1f}% 的股票")
        return

    sd.preload()
    sd.load_sector_map()

    all_feats = []
    for r in results:
        feat = extract_t_features(r["code"], signal_date)
        if feat:
            feat["future_ret"] = r["ret"]
            feat["buy_date"] = r.get("buy_date")
            feat["sell_date"] = r.get("sell_date")
            all_feats.append(feat)

    winner_codes = {r["code"] for r in winners}

    winner_codes = {r["code"] for r in winners}
    winner_feats = [f for f in all_feats if f["code"] in winner_codes]

    if not winner_feats:
        print("❌ 赢家特征提取失败（all_feats为空）")
        return

    print(f"✅ 成功提取特征: 全样本 {len(all_feats)} 只，赢家 {len(winner_feats)} 只\n")

    # 数值统计
    numeric_keys = [
        "rsi", "gain1", "gain3", "gain5", "gain10", "gain20", "avg_turn5",
        "avg_turn10", "dist_ma20", "dist_ma60", "vol_ratio_5_20", "body_pct",
        "amp_today", "pos_in_20d", "l2_score", "market_cap"
    ]
    bool_keys = [
        "normal_path", "accelerated_path", "price_above_ma10", "price_above_ma20",
        "price_above_ma60", "in_hot_sector", "ma5_above_ma60", "ma10_above_ma60", "ma20_above_ma60",
    ]

    numeric_summary = [x for x in (summarize_numeric(all_feats, winner_feats, k) for k in numeric_keys) if x]
    bool_summary = [x for x in (summarize_bool(all_feats, winner_feats, k) for k in bool_keys) if x]
    patterns = infer_patterns(all_feats, winner_feats)

    print("📊 赢家组 vs 全样本组（数值特征）")
    print(f"{'特征':<13}{'全样本均值':>10}{'赢家均值':>8}{'差值':>6}{'赢家中位数':>10}")
    print("-" * 70)
    for row in sorted(numeric_summary, key=lambda x: abs(x['delta_mean']), reverse=True)[:top_feature_n]:
        print(f"{row['feature']:<18}{row['all_mean']:>12.2f}{row['winner_mean']:>12.2f}{row['delta_mean']:>10.2f}{row['winner_median']:>12.2f}")

    print("\n📊 赢家组 vs 全样本组（布尔特征）")
    print(f"{'特征':<13}{'全样本占比':>10}{'赢家占比':>8}{'差值pct':>10}")
    print("-" * 60)
    for row in sorted(bool_summary, key=lambda x: abs(x['delta_ratio']), reverse=True):
        print(f"{row['feature']:<18}{row['all_ratio']:>11.1f}%{row['winner_ratio']:>11.1f}%{row['delta_ratio']:>10.1f}")

    print("\n🔍 推断出的共性特征")
    if not patterns:
        print("   暂未找到明显共性（可能样本过少或差异不显著）")
    else:
        for i, p in enumerate(patterns[:top_feature_n], 1):
            print(f"   {i}. {p['feature']}: {p['note']}")

    print("\n🏆 赢家明细")
    print(f"{'代码':<10}{'名称':<8}{'收益':>8}{'RSI':>8}{'5日涨%':>8}{'5日换手':>8}{'L2分':>6}{'路径':>8}")
    print("-" * 80)
    for f in sorted(winner_feats, key=lambda x: -x.get("future_ret", 0)):
        path = "加速" if f.get("accelerated_path") else ("正常" if f.get("normal_path") else "-")
        print(f"{f['code']:<10}{f['name'][:8]:<8}{f['future_ret']:>7.2f}%{(f.get('rsi') or 0):>8.1f}{(f.get('gain5') or 0):>10.2f}{(f.get('avg_turn5') or 0):>10.2f}{(f.get('l2_score') or 0):>8.1f}{path:>8}")

    out = {
        "signal_date": signal_date,
        "threshold": threshold,
        "total": len(results),
        "winner_count": len(winners),
        "winner_codes": [normalize_symbol(r["code"]) for r in winners],
        "numeric_summary": numeric_summary,
        "bool_summary": bool_summary,
        "patterns": patterns,
        "winner_features": winner_feats,
    }
    out_path = OUTPUT_DIR / f"analyze_double_winners_{signal_date}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n💾 JSON已保存: {out_path}")
    print(f"\n⏱ 用时 {time.time() - t0:.1f}秒")
    return out


# ── CLI ────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="分析 screen_double T+10 赢家在 T 日之前的共性特征")
    parser.add_argument("--date", required=True, help="信号日 T，如 2026-04-08")
    parser.add_argument("--threshold", type=float, default=20.0, help="赢家收益阈值，默认>20%%")
    parser.add_argument("--top-feature-n", type=int, default=12, help="输出前N个差异特征")
    args = parser.parse_args()
    validate_signal_date(args.date)
    run(args.date, args.threshold, args.top_feature_n)


if __name__ == "__main__":
    main()
