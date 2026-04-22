#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
参数优化（训练集 / 验证集分离版）
基于 triple_screen 三步量化选股系统
"""

import sys, time, itertools, argparse, re
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

WORKSPACE = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(WORKSPACE))

import rps_strong_screen as rps
import trend_strong_screen as tss
import gain_turnover as gt
from stock_trend import gain_turnover_screen


# ═══════════════════════════════════════════════════════════════
# 参数空间（可调整）
# ═══════════════════════════════════════════════════════════════

PARAM_GRID = {
    "rps_composite":     [70.0, 75.0, 80.0],
    "rsi_low":           [45.0, 50.0, 55.0],
    "rsi_high":          [78.0, 80.0, 82.0],
    "rps20_min":         [70.0, 75.0],
    "max_ret20":         [30.0, 40.0, 50.0],
    "max_ret5":          [20.0, 30.0],
    "ret3_min":          [2.0,  3.0,  5.0],
    "min_turnover":      [2.0,  3.0],
    "trend_score":       [25.0, 30.0, 35.0],
    "gain_days":         [3,    5],
    "gain_min":          [2.0,  3.0],
    "gain_max":          [8.0,  12.0],
    "quality_days":      [15,   20],
}


# ═══════════════════════════════════════════════════════════════
# 数据 & 缓存
# ═══════════════════════════════════════════════════════════════

CACHE_DIR = WORKSPACE / ".cache/qfq_daily"


def get_trade_dates(start: str, end: str) -> list:
    df = tss.get_index_kline("sh000001")
    if df is None or df.empty:
        return []
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df[(df["date"] >= start) & (df["date"] <= end)]
    return sorted(df["date"].tolist())


def get_kline_from_cache(code: str) -> pd.DataFrame:
    c = re.sub(r"^(sz|sh|bj)", "", code.lower())
    p = CACHE_DIR / f"{c}_qfq.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, dtype={"date": str})
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df.sort_values("date").reset_index(drop=True)


def _fetch_future_ak(code: str, after_date: str) -> pd.DataFrame:
    try:
        prefix = "sh" if code.startswith("sh") else "sz"
        c6 = code.lower().replace("sz", "").replace("sh", "")
        df = gt.ak.stock_zh_a_daily(symbol=f"{prefix}{c6}", adjust="qfq")
        if df is None or df.empty:
            return pd.DataFrame()
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df = df[df["date"] > after_date]
        return df[["date", "close"]].rename(columns={"close": "close_ak"})
    except Exception:
        return pd.DataFrame()


def get_forward_return(code: str, buy_date: str, hold: int) -> float:
    df = get_kline_from_cache(code)
    if df.empty:
        return np.nan
    df2 = df[df["date"] >= buy_date].copy().reset_index(drop=True)
    if len(df2) < hold + 1:
        future = _fetch_future_ak(code, df2.iloc[-1]["date"] if len(df2) else buy_date)
        if not future.empty:
            df2 = pd.concat([df2, future], ignore_index=True)
    if len(df2) < hold + 1:
        return np.nan
    buy = float(df2.iloc[0]["close"])
    sell_raw = df2.iloc[hold]["close"]
    if "close_ak" in df2.columns and pd.notna(df2.iloc[hold].get("close_ak")):
        sell = float(df2.iloc[hold]["close_ak"])
    else:
        sell = float(sell_raw) if pd.notna(sell_raw) else np.nan
    if buy == 0 or np.isnan(buy) or np.isnan(sell):
        return np.nan
    return (sell / buy - 1) * 100


# ═══════════════════════════════════════════════════════════════
# 三步筛选（内联 triple_screen 逻辑）
# ═══════════════════════════════════════════════════════════════

def screen_once(target: datetime, params: dict, top_n: int = 0) -> list:
    t0 = time.time()

    # ── Step1: RPS ──────────────────────────────────────
    all_codes = rps.get_all_stock_codes()
    df_all = rps.scan_rps(all_codes, top_n=len(all_codes),
                           max_workers=params.get("workers", 8),
                           target_date=target)

    if "data_date" in df_all.columns:
        ts = target.strftime("%Y-%m-%d")
        df_all = df_all[df_all["data_date"] == ts]

    if df_all.empty:
        return []

    df = df_all[
        (df_all["composite"]  >= params["rps_composite"]) &
        (df_all["ret20_rps"]  >= params["rps20_min"]) &
        (df_all["rsi"]        >= params["rsi_low"]) &
        (df_all["rsi"]        <= params["rsi_high"]) &
        (df_all["ret20"]      <= params["max_ret20"]) &
        (df_all["ret20"]      >= -10) &
        (df_all["ret5"]       <= params["max_ret5"]) &
        (df_all["ret3"]       >= params["ret3_min"]) &
        (df_all["avg_turnover_5"] >= params["min_turnover"])
    ].copy()

    if df.empty:
        return []

    # ── Step2: trend ────────────────────────────────────
    step1_codes = df["code"].str.lower().tolist()
    raw = tss.scan_market(
        codes=step1_codes,
        top_n=top_n,
        score_threshold=params["trend_score"],
        max_workers=params.get("workers", 8),
        target_date=target,
    )
    step2_codes = [r[0] for r in raw if isinstance(r, tuple) and len(r) >= 4]
    if not step2_codes:
        return []

    # ── Step3: gain_turnover ────────────────────────────
    config = gt.StrategyConfig(
        signal_days=params["gain_days"],
        min_gain=params["gain_min"],
        max_gain=params["gain_max"],
        quality_days=params["quality_days"],
        check_fundamental=False,
        sector_bonus=False,
        check_volume_surge=True,
        min_turnover=params.get("min_turnover", 2.0),
        score_threshold=params.get("score_threshold_step3", 40.0),
    )

    results = gain_turnover_screen.screen_market(
        codes=step2_codes,
        config=config,
        target_date=target,
        top_n=len(step2_codes),
        max_workers=params.get("workers", 8),
        refresh_cache=False,
    )
    return results


# ═══════════════════════════════════════════════════════════════
# 评分函数
# ═══════════════════════════════════════════════════════════════

def score_strategy(df: pd.DataFrame, hold: int = 3) -> float:
    """综合评分：均值 * 0.6 + 胜率 * 100 * 0.4"""
    col = f"ret_{hold}d"
    df = df.dropna(subset=[col])
    if len(df) < 10:
        return -999
    win_rate = (df[col] > 0).mean()
    avg_ret  = df[col].mean()
    return avg_ret * 0.6 + win_rate * 100 * 0.4


# ═══════════════════════════════════════════════════════════════
# 回测单组参数
# ═══════════════════════════════════════════════════════════════

HOLD_DAYS = [1, 3, 5]
MAX_HOLD  = max(HOLD_DAYS)


def run_backtest_for_params(dates: list, params: dict, hold: int = 3) -> tuple:
    """
    在指定日期列表上回测，返回 (score, n_samples, records_df)
    """
    records = []
    for d in dates:
        target = datetime.strptime(d, "%Y-%m-%d")
        try:
            results = screen_once(target, params)
        except Exception:
            continue
        for r in results:
            ret = get_forward_return(r.code, d, hold)
            records.append({
                "date":   d,
                "code":   r.code,
                "name":   r.name,
                "score":  r.score,
                "rsi":    r.rsi14,
                "close":  r.close,
                f"ret_{hold}d": ret,
            })

    if not records:
        return -999, 0, pd.DataFrame()

    df = pd.DataFrame(records)
    sc = score_strategy(df, hold)
    return sc, len(df), df


# ═══════════════════════════════════════════════════════════════
# 参数组合枚举（带剪枝）
# ═══════════════════════════════════════════════════════════════

def enumerate_params():
    keys   = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="参数优化（训练/验证分离）— triple_screen 版")
    p.add_argument("--start",  type=str, help="开始日期 YYYY-MM-DD")
    p.add_argument("--end",    type=str, help="结束日期 YYYY-MM-DD")
    p.add_argument("--days",   type=int, default=60,   help="默认往前取60天")
    p.add_argument("--split",  type=float, default=0.7, help="训练集比例（默认0.7）")
    p.add_argument("--hold",   type=int, default=3,    help="评分持有天数（默认3）")
    p.add_argument("--top",    type=int, default=30,   help="每个信号日最多买几只（默认30）")
    p.add_argument("--workers", type=int, default=6,   help="并行线程数（默认6）")
    p.add_argument("--max-combos", type=int, default=0, help="最多跑几组组合（0=不限）")
    return p.parse_args()


def resolve_dates(args):
    end   = datetime.now()
    start = end - timedelta(days=args.days)
    return (args.start or start.strftime("%Y-%m-%d"),
            args.end   or end.strftime("%Y-%m-%d"))


# ═══════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════

def optimize(start_date: str, end_date: str, split_ratio: float, hold: int,
             top_n: int, workers: int, max_combos: int):
    print("\n" + "=" * 70)
    print("📊 参数优化（训练 / 验证分离）— triple_screen 三步选股系统")
    print("=" * 70)

    # ── 准备交易日 ──────────────────────────────────────
    dates = get_trade_dates(start_date, end_date)
    if not dates:
        print("❌ 无交易日数据")
        return

    # 确保缓存足够覆盖 T+MAX_HOLD
    cache_end = get_kline_from_cache("sz002859")["date"].max()
    try:
        all_td = get_trade_dates(start_date, cache_end)
        end_idx = all_td.index(dates[-1])
        actual_end_idx = max(0, end_idx - MAX_HOLD)
        actual_end = all_td[actual_end_idx]
        dates = [d for d in dates if d <= actual_end]
    except (ValueError, IndexError):
        pass

    if len(dates) < 20:
        print(f"❌ 有效交易日不足（{len(dates)} < 20）")
        return

    # ── 切分 ────────────────────────────────────────────
    split_idx    = int(len(dates) * split_ratio)
    train_dates  = dates[:split_idx]
    test_dates   = dates[split_idx:]

    print(f"\n📅 总区间:   {dates[0]} → {dates[-1]}（{len(dates)} 个交易日）")
    print(f"📂 训练集:  {train_dates[0]} → {train_dates[-1]}（{len(train_dates)} 天）")
    print(f"📂 验证集:  {test_dates[0]} → {test_dates[-1]}（{len(test_dates)} 天）")
    print(f"📌 持有期:  T+{hold}（评分指标）")
    print(f"📌 信号日上限: {top_n} 只")
    print(f"📌 并行:    {workers} 线程")
    print(f"📌 参数空间: {len(list(enumerate_params()))} 种组合")
    print()

    # ── 搜索 ────────────────────────────────────────────
    results = []
    total   = 0
    t0      = time.time()

    for params in enumerate_params():
        if max_combos > 0 and total >= max_combos:
            break
        total += 1
        params["workers"] = workers

        print(f"[{total}] {params}", end="", flush=True)

        try:
            train_sc, train_n, _ = run_backtest_for_params(train_dates, params, hold)
        except Exception as e:
            print(f" → 训练集失败: {e}")
            continue

        print(f" → Train样本={train_n} 得分={train_sc:.2f}", end="")

        if train_sc < 0:
            print(" → 淘汰（负分）")
            continue

        try:
            test_sc, test_n, _ = run_backtest_for_params(test_dates, params, hold)
        except Exception:
            test_sc, test_n = -999, 0

        gap = test_sc - train_sc
        print(f" → Test样本={test_n} 得分={test_sc:.2f} gap={gap:+.2f}")

        results.append({
            "params":        str(params),
            "train_score":   round(train_sc, 4),
            "test_score":    round(test_sc, 4),
            "gap":           round(gap, 4),
            "train_samples": train_n,
            "test_samples":  test_n,
            **params,
        })

    # ── 输出 ────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n耗时 {elapsed:.0f}s / {total} 组\n")

    if not results:
        print("⚠️ 无有效结果")
        return

    df = pd.DataFrame(results)
    df = df.sort_values("test_score", ascending=False)

    print("=" * 70)
    print("🏆 最优参数（验证集 Top10）:")
    print("=" * 70)
    top = df.head(10)
    for _, row in top.iterrows():
        print(f"\n  验证集得分: {row['test_score']:+.4f}  (训练={row['train_score']:+.4f} gap={row['gap']:+.4f})")
        print(f"  样本: 训练={row['train_samples']} 验证={row['test_samples']}")
        print(f"  参数: {row['params']}")

    # 保存
    out = WORKSPACE / "param_opt_triple_result.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n💾 已保存: {out}")

    # 打印最优推荐
    best = df.iloc[0]
    print("\n" + "=" * 70)
    print("✅ 推荐参数（验证集最优）:")
    print("=" * 70)
    for k, v in sorted(best.items()):
        if k in PARAM_GRID:
            print(f"  {k}: {v}")


# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    args = parse_args()
    start_date, end_date = resolve_dates(args)
    optimize(start_date, end_date, args.split, args.hold,
             args.top, args.workers, args.max_combos)
