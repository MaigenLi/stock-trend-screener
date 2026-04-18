#!/usr/bin/env python3
"""
Triple Screen T+1 命中率回测（固定版）
======================================
Phase1: 加载Step1缓存（已有）
Phase2: 预计算trend评分（已有）
Phase3: 对每组参数直接跑完整TripleScreen（不用缓存score）
Phase4: 纯内存网格搜索（T+1涨幅验证）
"""

import sys, json, time, gc
from pathlib import Path
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd

WORKSPACE = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(WORKSPACE))

from stock_trend import gain_turnover as gt
from stock_trend import trend_strong_screen as tss
from stock_trend import rps_strong_screen as rps

CACHE_DIR   = Path.home() / ".openclaw/workspace/.cache/qfq_daily"
REPORTS_DIR = Path.home() / "stock_reports"
START_DATE  = "2025-01-02"
END_DATE    = "2026-04-16"
SAMPLE_INTERVAL = 5
MIN_T1_GAIN = 3.0
HIT_TARGET  = 60.0

PARAM_GRID = {
    "rps_composite": [75, 80],
    "rps20_min":     [70, 75],
    "rsi_low":       [45, 50],
    "rsi_high":      [80, 85],
    "max_ret5":      [20, 25, 30],
    "min_turnover":  [2.0, 3.0],
    "trend_score":   [20, 30],
}

# ── 加载Step1缓存 ──────────────────────────────────────
def load_step1_cache():
    with open(REPORTS_DIR / "backtest_step1_cache.json") as f:
        raw = json.load(f)
    return {d: pd.DataFrame(v["data"], columns=v["cols"]) for d, v in raw.items()}

# ── 批量读T+1涨幅 ──────────────────────────────────────
def batch_t1_gains(codes: list[str], trade_date: str) -> dict[str, float]:
    gains = {}
    for code in codes:
        pure = code[-6:]
        f = CACHE_DIR / f"{pure}_qfq.csv"
        if not f.exists():
            continue
        try:
            df = pd.read_csv(f, usecols=["date", "close"])
            df["date"] = df["date"].astype(str).str[:10]
            df = df.sort_values("date")
            sig = df[df["date"] == trade_date]
            post = df[df["date"] > trade_date]
            if sig.empty or post.empty:
                continue
            p0 = float(sig.iloc[0]["close"])
            p1 = float(post.iloc[0]["close"])
            if p0 > 0:
                gains[code.lower()] = (p1 / p0 - 1) * 100.0
        except:
            continue
    return gains

# ── 预计算趋势评分 ────────────────────────────────────
def preload_trend_scores(step1_cache: dict[str, pd.DataFrame], sample_days: list[str]) -> dict:
    """返回 {(day, code): trend_score}"""
    trend_cache: dict[str, float] = {}
    t0 = time.time()
    for i, day in enumerate(sample_days):
        df = step1_cache[day]
        codes = df["code"].str.lower().tolist()
        if not codes:
            continue
        target_date = datetime.strptime(day, "%Y-%m-%d")
        results = tss.scan_market(codes, top_n=0, score_threshold=0,
                                   max_workers=8, target_date=target_date)
        for item in results:
            if isinstance(item, tuple) and len(item) >= 4:
                c = item[0].lower()
                s = float(item[2]) if item[2] else 0
                trend_cache[(day, c)] = s
        if (i + 1) % 10 == 0:
            eta = (time.time()-t0)/(i+1)*(len(sample_days)-i-1)
            print(f"   [{i+1}/{len(sample_days)}] {day} | ETA:{eta:.0f}s")
    print(f"   趋势缓存完成: {len(trend_cache)}条 | 耗时:{time.time()-t0:.0f}s")
    return trend_cache

# ── 一次性T+1涨幅计算 ─────────────────────────────────
def compute_all_t1_gains(step1_cache: dict, sample_days: list[str]) -> dict[str, float]:
    """
    对所有采样日和候选股票预计算T+1涨幅。
    key = f"{day}|{code}" → T+1涨幅
    """
    t1_path = REPORTS_DIR / "backtest_t1_gains.json"
    if t1_path.exists():
        with open(t1_path) as f:
            return json.load(f)

    t0 = time.time()
    all_t1: dict[str, float] = {}

    for i, day in enumerate(sample_days):
        df = step1_cache[day]
        codes = df["code"].str.lower().tolist()
        if not codes:
            continue
        gains = batch_t1_gains(codes, day)
        for code, g in gains.items():
            all_t1[f"{day}|{code}"] = g
        if (i + 1) % 10 == 0:
            eta = (time.time()-t0)/(i+1)*(len(sample_days)-i-1)
            print(f"   T+1进度 [{i+1}/{len(sample_days)}] {day} | {len(all_t1)}条 | ETA:{eta:.0f}s")

    with open(t1_path, "w") as f:
        json.dump(all_t1, f)
    print(f"   T+1涨幅缓存完成: {len(all_t1)}条 | 耗时:{time.time()-t0:.0f}s")
    return all_t1

# ── 完整TripleScreen（subprocess方式）──
def run_triple_screen_for_params(
    params: dict,
    sample_days: list[str],
    step1_cache: dict,
    trend_cache: dict,
    t1_gains: dict,
    max_workers: int = 8,
) -> dict:
    """
    对一组参数，跑所有采样日的完整TripleScreen，返回命中率统计。
    """
    total_selected = 0
    total_hit = 0
    daily_stats = []

    for day in sample_days:
        df = step1_cache.get(day)
        if df is None or df.empty:
            continue

        # Step1 筛选
        mask = (
            (df["composite"] >= params["rps_composite"]) &
            (df["rsi"] >= params["rsi_low"]) &
            (df["rsi"] <= params["rsi_high"]) &
            (df["ret20_rps"] >= params["rps20_min"]) &
            (df["ret20"] <= 40) &
            (df["ret20"] >= -10) &
            (df["ret5"] <= params["max_ret5"]) &
            (df["avg_turnover_5"] >= params["min_turnover"])
        )
        step1_codes = df[mask]["code"].str.lower().tolist()
        if not step1_codes:
            continue

        # Step2 trend
        step2_codes = [
            c for c in step1_codes
            if trend_cache.get((day, c), 0) >= params["trend_score"]
        ]
        if not step2_codes:
            continue

        # Step3 gain_turnover（直接调用，不过缓存）
        target_date = datetime.strptime(day, "%Y-%m-%d")
        config = gt.StrategyConfig(
            signal_days=3, min_gain=2.0, max_gain=10.0, quality_days=20,
            check_fundamental=False, sector_bonus=False, check_volume_surge=False,
        )
        from stock_trend.gain_turnover_screen import screen_market
        try:
            gain_results = screen_market(
                codes=step2_codes,
                config=config,
                target_date=target_date,
                top_n=len(step2_codes),
                max_workers=max_workers,
                refresh_cache=False,
            )
        except Exception as e:
            print(f"   ⚠️ screen_market error on {day}: {e}")
            continue

        if not gain_results:
            continue

        # T+1验证（查预计算的t1_gains）
        checked = 0
        hits = 0
        for r in gain_results:
            key = f"{day}|{r.code.lower()}"
            g = t1_gains.get(key)
            if g is None:
                continue
            checked += 1
            total_selected += 1
            if g > MIN_T1_GAIN:
                hits += 1
                total_hit += 1

        daily_stats.append({"date": day, "selected": len(gain_results), "checked": checked, "hits": hits})

    hit_rate = total_hit / total_selected * 100 if total_selected > 0 else 0.0
    return {
        "params": params,
        "total_selected": total_selected,
        "total_hit": total_hit,
        "hit_rate": round(hit_rate, 2),
        "daily_stats": daily_stats,
    }

# ── 主流程 ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("📊 Triple Screen T+1 命中率回测")
    print(f"   区间: {START_DATE} → {END_DATE} | 每{SAMPLE_INTERVAL}交易日")
    print(f"   T+1阈值: >{MIN_T1_GAIN}% | 目标: ≥{HIT_TARGET}%")
    print("=" * 60)

    # Step1缓存
    print("\n📦 加载Step1缓存...")
    step1_cache = load_step1_cache()
    sample_days = sorted(step1_cache.keys())
    print(f"   {len(sample_days)} 天: {sample_days[0]} ... {sample_days[-1]}")

    # Phase2: 趋势缓存
    print(f"\n⚙️  Phase 2: 预计算趋势评分...")
    trend_cache = preload_trend_scores(step1_cache, sample_days)

    # Phase3: T+1涨幅缓存
    print(f"\n⚙️  Phase 3: 预计算T+1涨幅...")
    t1_gains = compute_all_t1_gains(step1_cache, sample_days)
    print(f"   T+1涨幅缓存: {len(t1_gains)} 条")

    # Phase4: 网格搜索
    keys = list(PARAM_GRID.keys())
    combos = list(product(*PARAM_GRID.values()))
    print(f"\n🔍 Phase 4: 网格搜索 {len(combos)} 组参数 × {len(sample_days)} 天")

    all_results = []
    t0 = time.time()

    for idx, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        result = run_triple_screen_for_params(
            params, sample_days, step1_cache, trend_cache, t1_gains, max_workers=8
        )
        result["idx"] = idx + 1
        all_results.append(result)

        hit_mark = "✅" if result["hit_rate"] >= HIT_TARGET else "❌"
        print(f"  [{idx+1:3}/{len(combos)}] {hit_mark} 命中率={result['hit_rate']:.1f}%({result['total_hit']}/{result['total_selected']}) "
              f"| RPS综≥{params['rps_composite']} RSI[{params['rsi_low']},{params['rsi_high']}] "
              f"RPS20≥{params['rps20_min']} 近5日≤{params['max_ret5']}% "
              f"换手≥{params['min_turnover']}% 趋势≥{params['trend_score']}")

        if (idx + 1) % 10 == 0:
            gc.collect()

    all_results.sort(key=lambda x: -x["hit_rate"])

    # ── 报告 ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("📊 回测结果")
    print("=" * 70)

    print(f"\n{'#':<4} {'RPS综':>6} {'RSI下':>5} {'RSI上':>5} {'RPS20':>6} {'近5日':>6} {'换手%':>6} {'趋势':>5} | {'命中率':>7} {'命中':>5} {'信号':>5} {'达标':>4}")
    for r in all_results[:15]:
        p = r["params"]
        mark = "✅" if r["hit_rate"] >= HIT_TARGET else "❌"
        print(f"{r['idx']:<4} {p['rps_composite']:>6.0f} {p['rsi_low']:>5.0f} {p['rsi_high']:>5.0f} "
              f"{p['rps20_min']:>6.0f} {p['max_ret5']:>6.0f} {p['min_turnover']:>6.1f} {p['trend_score']:>5.0f} | "
              f"{r['hit_rate']:>6.1f}% {r['total_hit']:>5} {r['total_selected']:>5} {mark:>4}")

    best = all_results[0]
    bp = best["params"]
    print(f"\n🏆 最优参数:")
    print(f"   Step1: RPS综合≥{bp['rps_composite']:.0f}  RPS20≥{bp['rps20_min']:.0f}")
    print(f"   Step1: RSI∈[{bp['rsi_low']:.0f}, {bp['rsi_high']:.0f}]")
    print(f"   Step1: 近5日涨幅≤{bp['max_ret5']:.0f}%  换手率≥{bp['min_turnover']:.1f}%")
    print(f"   Step2: 趋势评分≥{bp['trend_score']:.0f}")
    print(f"\n   命中率: {best['hit_rate']:.1f}% ({best['total_hit']}/{best['total_selected']})")

    # 每日明细
    if best["daily_stats"]:
        print(f"\n每日明细:")
        print(f"{'日期':<12} {'选出':>5} {'验证':>5} {'命中':>5} {'命中率':>8}")
        for ds in best["daily_stats"]:
            rate = ds["hits"] / ds["checked"] * 100 if ds["checked"] > 0 else 0
            print(f"{ds['date']:<12} {ds['selected']:>5} {ds['checked']:>5} {ds['hits']:>5} {rate:>7.0f}%")

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = REPORTS_DIR / f"backtest_T1_{ts}.txt"
    out_path.write_text(f"""
================================================================
Triple Screen T+1 命中率回测报告
区间: {START_DATE} → {END_DATE} | 每{SAMPLE_INTERVAL}交易日 | T+1阈值: >{MIN_T1_GAIN}% | 目标: ≥{HIT_TARGET}%

最优参数:
  RPS综合≥{bp['rps_composite']:.0f}  RPS20≥{bp['rps20_min']:.0f}
  RSI∈[{bp['rsi_low']:.0f}, {bp['rsi_high']:.0f}]
  近5日涨幅≤{bp['max_ret5']:.0f}%  换手率≥{bp['min_turnover']:.1f}%
  趋势评分≥{bp['trend_score']:.0f}

命中率: {best['hit_rate']:.1f}% ({best['total_hit']}/{best['total_selected']})
================================================================
""".strip(), encoding="utf-8")
    print(f"\n💾 报告: {out_path}")

    json_path = REPORTS_DIR / f"backtest_results_{ts}.json"
    json_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"💾 JSON: {json_path}")
    print(f"\n总耗时: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
