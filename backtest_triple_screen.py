#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回测脚本（基于 triple_screen.py 三步量化选股系统）

功能：
- 调用 triple_screen 三步筛选逻辑
- 用本地 QFQ 缓存计算 T+1/T+3/T+5 收益
- 输出统计结果（胜率、均值、夏普比、月度明细）

用法：
  python backtest_triple_screen.py --days 10          # 最近10天
  python backtest_triple_screen.py --start 2026-03-01 --end 2026-04-17
  python backtest_triple_screen.py --days 30 --rps-composite 80
"""

import sys, time, argparse, re
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

# ── 默认参数（与 triple_screen 一致）─────────────────────────
DEFAULT_RPS_COMPOSITE = 75.0
DEFAULT_RSI_LOW = 50.0
DEFAULT_RSI_HIGH = 82.0
DEFAULT_RPS20_MIN = 75.0
DEFAULT_MAX_RET20 = 40.0
DEFAULT_MAX_RET5 = 30.0
DEFAULT_RET3_MIN = 3.0
DEFAULT_MIN_TURNOVER = 2.0
DEFAULT_TREND_SCORE = 30.0
DEFAULT_GAIN_DAYS = 3
DEFAULT_GAIN_MIN = 2.0
DEFAULT_GAIN_MAX = 10.0
DEFAULT_QUALITY_DAYS = 20
DEFAULT_WORKERS = 8
HOLD_DAYS = [1, 3, 5]

CACHE_DIR = WORKSPACE / ".cache/qfq_daily"


# ═══════════════════════════════════════════════════════════════
# 数据获取（本地缓存，无网络）
# ═══════════════════════════════════════════════════════════════

def get_trade_dates(start: str, end: str) -> list:
    """获取交易日列表"""
    df = tss.get_index_kline("sh000001")
    if df is None or df.empty:
        return []
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df[(df["date"] >= start) & (df["date"] <= end)]
    return sorted(df["date"].tolist())


def get_kline_from_cache(code: str) -> pd.DataFrame:
    """从本地 QFQ 缓存读取日线"""
    c = re.sub(r"^(sz|sh|bj)", "", code.lower())
    p = CACHE_DIR / f"{c}_qfq.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, dtype={"date": str})
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df.sort_values("date").reset_index(drop=True)


def _fetch_future_ak(code: str, after_date: str) -> pd.DataFrame:
    """用 AkShare 补充本地缓存之后的数据（一次性网络请求）"""
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
    """计算 T+hold 日收益率（本地缓存 + AkShare 补充未来数据）"""
    df = get_kline_from_cache(code)
    if df.empty:
        return np.nan

    df2 = df[df["date"] >= buy_date].copy().reset_index(drop=True)
    if len(df2) < hold + 1:
        # 补充未来数据
        future = _fetch_future_ak(code, df2.iloc[-1]["date"] if len(df2) else buy_date)
        if not future.empty:
            df2 = pd.concat([df2, future], ignore_index=True)

    if len(df2) < hold + 1:
        return np.nan

    buy = float(df2.iloc[0]["close"])
    sell_raw = df2.iloc[hold]["close"]
    # 如果该行是从 AkShare 补充的，取 close_ak
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

def screen_once(target: datetime, params: dict) -> list:
    """执行一次完整三步筛选，返回结果列表"""
    t0 = time.time()

    # ── Step1: RPS ──────────────────────────────────────
    all_codes = rps.get_all_stock_codes()
    df_all = rps.scan_rps(all_codes, top_n=len(all_codes),
                           max_workers=params["workers"],
                           target_date=target)

    if "data_date" in df_all.columns:
        ts = target.strftime("%Y-%m-%d")
        df_all = df_all[df_all["data_date"] == ts]

    if df_all.empty:
        return []

    df = df_all[
        (df_all["composite"] >= params["rps_composite"]) &
        (df_all["ret20_rps"] >= params["rps20_min"]) &
        (df_all["rsi"] >= params["rsi_low"]) &
        (df_all["rsi"] <= params["rsi_high"]) &
        (df_all["ret20"] <= params["max_ret20"]) &
        (df_all["ret20"] >= -10) &
        (df_all["ret5"] <= params["max_ret5"]) &
        (df_all["ret3"] >= params["ret3_min"]) &
        (df_all["avg_turnover_5"] >= params["min_turnover"])
    ].copy()

    if df.empty:
        return []

    # ── Step2: trend ────────────────────────────────────
    step1_codes = df["code"].str.lower().tolist()

    raw = tss.scan_market(
        codes=step1_codes,
        top_n=0,
        score_threshold=params["trend_score"],
        max_workers=params["workers"],
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
        min_turnover=params["min_turnover"],
        score_threshold=params.get("score_threshold_step3", 40.0),
    )

    results = gain_turnover_screen.screen_market(
        codes=step2_codes,
        config=config,
        target_date=target,
        top_n=len(step2_codes),
        max_workers=params["workers"],
        refresh_cache=False,
    )

    print(f"  {target.strftime('%Y-%m-%d')} → {len(results)} 只 | {time.time()-t0:.1f}s")
    return results


# ═══════════════════════════════════════════════════════════════
# 回测主循环
# ═══════════════════════════════════════════════════════════════

def run_backtest(start_date: str, end_date: str, params: dict):
    # 确定本地缓存的最早未来日期（买股日期不能太接近缓存最新日）
    cache_end = get_kline_from_cache("sz002859")["date"].max()
    max_hold = max(HOLD_DAYS)  # 5
    # 取 cache 最新日之前 max_hold 个交易日作为实际截止
    all_trade_dates = get_trade_dates(start_date, cache_end)
    # 找到 cache_end 在交易日列表中的位置，向前回溯 max_hold 天
    try:
        end_idx = len(all_trade_dates) - 1
        actual_end_idx = max(0, end_idx - max_hold)  # buy on date[i] → T+max_hold sell on date[i+max_hold]
        actual_end = all_trade_dates[actual_end_idx]
    except ValueError:
        actual_end = end_date
    if actual_end < start_date:
        print(f"❌ 缓存最新日 {cache_end} 不足以覆盖 T+{max_hold} 回测，请先更新缓存")
        return

    dates = get_trade_dates(start_date, actual_end)
    if not dates:
        print("❌ 无交易日数据")
        return

    print(f"\n📊 回测区间: {start_date} → {actual_end}（{len(dates)} 个交易日）")
    print(f"   买股截止日: {actual_end}（缓存最新: {cache_end}，确保 T+{max_hold} 数据充足）")
    print(f"   RPS≥{params['rps_composite']} RSI[{params['rsi_low']},{params['rsi_high']}] "
          f"窗口[{params['gain_min']},{params['gain_max']}]% 趋势≥{params['trend_score']}")

    all_records = []
    t0_total = time.time()

    for i, d in enumerate(dates):
        target = datetime.strptime(d, "%Y-%m-%d")

        try:
            results = screen_once(target, params)
        except Exception as e:
            print(f"  ❌ {d} 筛选失败: {e}")
            continue

        for r in results:
            rec = {
                "date": d,
                "code": r.code,
                "name": r.name,
                "window_gain": r.total_gain_window,
                "score": r.score,
                "rsi": r.rsi14,
                "close": r.close,
            }
            for h in HOLD_DAYS:
                rec[f"ret_{h}d"] = get_forward_return(r.code, d, h)
            all_records.append(rec)

        if (i + 1) % 5 == 0:
            elapsed = time.time() - t0_total
            eta = elapsed / (i + 1) * (len(dates) - i - 1)
            print(f"  进度 {i+1}/{len(dates)} 已选 {len(all_records)} 只 | 耗时 {elapsed:.0f}s ETA {eta:.0f}s")

    if not all_records:
        print("\n⚠️ 无样本")
        return

    df = pd.DataFrame(all_records)

    # ── 统计 ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("📈 回测结果")
    print("=" * 70)

    for h in HOLD_DAYS:
        col = f"ret_{h}d"
        valid = df[col].dropna()
        if valid.empty:
            continue
        win = (valid > 0).mean()
        avg = valid.mean()
        med = valid.median()
        std = valid.std()
        max_w = valid.max()
        max_l = valid.min()
        sr = (avg / std) if std > 0.01 else 0

        print(f"\n  T+{h}（持有{h}个交易日）：")
        print(f"    样本数: {len(valid)}  胜率: {win:.2%}  均值: {avg:+.2f}%  中位数: {med:+.2f}%")
        print(f"    标准差: {std:.2f}  最大盈利: {max_w:+.2f}%  最大亏损: {max_l:+.2f}%")
        print(f"    夏普比（日）: {sr:.3f}")

    # ── 月度 ─────────────────────────────────────────
    print("\n  📅 月度明细：")
    df["month"] = df["date"].str[:7]
    for m, grp in df.groupby("month"):
        print(f"    {m}: {len(grp)} 只", end="")
        for h in HOLD_DAYS:
            v = grp[f"ret_{h}d"].dropna()
            if not v.empty:
                print(f"  T+{h}胜率={v[v>0].shape[0]/len(v):.0%} 均值={v.mean():+.2f}%", end="")
        print()

    # ── 保存 ─────────────────────────────────────────
    out = WORKSPACE / f"backtest_triple_{start_date}_{end_date}.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n💾 已保存: {out}")


# ═══════════════════════════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="回测 triple_screen 三步选股系统")
    p.add_argument("--start", type=str, default=None)
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--days", type=int, default=10)
    p.add_argument("--rps-composite", type=float, default=DEFAULT_RPS_COMPOSITE)
    p.add_argument("--rsi-low", type=float, default=DEFAULT_RSI_LOW)
    p.add_argument("--rsi-high", type=float, default=DEFAULT_RSI_HIGH)
    p.add_argument("--rps20-min", type=float, default=DEFAULT_RPS20_MIN)
    p.add_argument("--max-ret20", type=float, default=DEFAULT_MAX_RET20)
    p.add_argument("--max-ret5", type=float, default=DEFAULT_MAX_RET5)
    p.add_argument("--ret3-min", type=float, default=DEFAULT_RET3_MIN)
    p.add_argument("--min-turnover", type=float, default=DEFAULT_MIN_TURNOVER)
    p.add_argument("--trend-score", type=float, default=DEFAULT_TREND_SCORE)
    p.add_argument("--gain-days", type=int, default=DEFAULT_GAIN_DAYS)
    p.add_argument("--min-gain", type=float, default=DEFAULT_GAIN_MIN)
    p.add_argument("--max-gain", type=float, default=DEFAULT_GAIN_MAX)
    p.add_argument("--quality-days", type=int, default=DEFAULT_QUALITY_DAYS)
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    p.add_argument("--score-threshold-step3", type=float, default=40.0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    end_date = args.end or datetime.now().strftime("%Y-%m-%d")
    start_date = args.start or (
        datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=args.days)
    ).strftime("%Y-%m-%d")

    # 规范化参数 key（argparse dest 用下划线，代码内部用下划线/驼峰混用）
    KEY_MAP = {
        "min_gain": "gain_min",
        "max_gain": "gain_max",
        "gain_days": "gain_days",
        "score_threshold_step3": "score_threshold_step3",
    }
    raw = {k: v for k, v in vars(args).items() if k not in ("start", "end", "days")}
    params = {KEY_MAP.get(k, k): v for k, v in raw.items()}

    run_backtest(start_date, end_date, params)
