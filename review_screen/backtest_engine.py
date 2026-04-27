#!/usr/bin/env python3
"""
回测引擎 — screen_trend_filter 真实胜率验证
==============================================

对历史每个交易日 T：
  1. 用 T 日指标运行筛选逻辑（as-of T 日）
  2. T+1 开盘价买入
  3. 持有 N 天后收盘卖出（或止损）
  4. 记录收益率

用法：
    python backtest_engine.py --start 2025-01-01 --end 2026-04-23
    python backtest_engine.py --start 2025-01-01 --end 2025-09-30 --out train_results.json
    python backtest_engine.py --start 2025-10-01 --end 2026-04-23 --out val_results.json
    python backtest_engine.py --combined --analyze  # 合并分析
"""

import sys
import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

import numpy as np

WORKSPACE = Path(__file__).parent.parent.parent.resolve()
CACHE_DIR = WORKSPACE / ".cache"
INDICATORS_DIR = CACHE_DIR / "indicators"
QFQ_DIR = CACHE_DIR / "qfq_daily"

# ── 筛选阈值（与 screen_trend_filter.py 保持一致）─────────────
TREND_FILTER = {
    "gain20_min": 13.8,
    "wave_quality_min": 4.0,
    "ma20_60_sep_min": 3.5,
}

BUY_READY = {
    "min_gain1": 0.3,
    "max_gain1": 8.0,
    "max_rsi": 80.0,
    "max_ma5_dist": 8.0,
    "limit_pct": 9.5,
}

COLUMNS = ["date", "open", "high", "low", "close", "volume", "amount",
           "turnover", "outstanding_share", "true_turnover"]


# ── 数据加载（惰性 + LRU缓存）──────────────────────────────
_MAX_CACHE = 200  # 最多缓存200只股票的指标
_stock_indicators_cache = {}  # code -> {date: row}
_stock_indicators_order = []   # 访问顺序（用于LRU）


def load_indicators_for_stock(code: str) -> dict | None:
    """按需加载单只股票的指标（带LRU缓存）"""
    global _stock_indicators_cache, _stock_indicators_order

    if code in _stock_indicators_cache:
        # 移到末尾（most recently used）
        _stock_indicators_order.remove(code)
        _stock_indicators_order.append(code)
        return _stock_indicators_cache[code]

    path = INDICATORS_DIR / f"{code}_indicators.json"
    if not path.exists():
        return None

    try:
        with open(path) as fh:
            data = json.load(fh)
        index = {}
        for row in data:
            index[row["date"]] = row

        # LRU淘汰
        if len(_stock_indicators_cache) >= _MAX_CACHE:
            oldest = _stock_indicators_order.pop(0)
            del _stock_indicators_cache[oldest]

        _stock_indicators_cache[code] = index
        _stock_indicators_order.append(code)
        return index
    except Exception:
        return None


def get_indicators(code: str, date: str) -> dict | None:
    """获取某只股票在指定日期的指标"""
    index = load_indicators_for_stock(code)
    if index is None:
        return None
    return index.get(date)


def preload_indicators(codes: list[str] = None):
    """预热指定股票的指标（批量回测前调用，不传参数则无操作）"""
    # 惰性加载，不需要预热
    pass


def load_qfq(code: str):
    """加载单只股票的前复权数据"""
    if code in _qiq_cache:
        return _qiq_cache[code]

    path = QFQ_DIR / f"{code}_qfq.csv"
    if not path.exists():
        return None

    import pandas as pd
    df = pd.read_csv(path, usecols=COLUMNS)
    df = df.sort_values("date").reset_index(drop=True)
    _qiq_cache[code] = df
    return df


_qiq_cache = {}


def get_indicators(code: str, date: str) -> dict | None:
    """获取某只股票在指定日期的指标（惰性加载）"""
    index = load_indicators_for_stock(code)
    if index is None:
        return None
    return index.get(date)


def get_price_on_date(code: str, date: str) -> dict | None:
    """获取某只股票在指定日期的价格（从原始CSV）"""
    df = load_qfq(code)
    if df is None:
        return None
    row = df[df["date"] == date]
    if row.empty:
        return None
    r = row.iloc[0]
    return {
        "open": float(r["open"]),
        "high": float(r["high"]),
        "low": float(r["low"]),
        "close": float(r["close"]),
        "volume": float(r["volume"]),
    }


def get_trading_dates(start: str, end: str, lookback: int = 0) -> list[str]:
    """
    获取 start 到 end 之间所有有交易的日期。
    lookback: 额外向前看多少天（用于预热）
    """
    df = load_qfq("000001")  # 用平安银行作日期基准
    if df is None:
        return []
    # 向前多看 lookback 天
    import pandas as pd
    start_idx = df[df["date"] >= start].index[0] - lookback
    start_idx = max(0, start_idx)
    start_actual = df.iloc[start_idx]["date"]
    dates = df[(df["date"] >= start_actual) & (df["date"] <= end)]["date"].tolist()
    return sorted(dates)


def get_next_trading_date(date: str, offset: int = 1) -> str | None:
    """获取 offset 个交易日后的日期"""
    df = load_qfq("000001")
    if df is None:
        return None
    dates = df["date"].tolist()
    try:
        idx = dates.index(date)
        target_idx = idx + offset
        if 0 <= target_idx < len(dates):
            return dates[target_idx]
    except ValueError:
        pass
    return None


# ── 筛选逻辑（与 screen_trend_filter.py 完全一致）────────────

def apply_trend_filter(ind: dict) -> bool:
    """趋势三条件"""
    gain20 = ind.get("gain20", 0)
    wave_quality = ind.get("wave_quality", 0)
    ma_sep = ind.get("ma_sep", 0)
    return (
        gain20 >= TREND_FILTER["gain20_min"]
        and wave_quality >= TREND_FILTER["wave_quality_min"]
        and ma_sep >= TREND_FILTER["ma20_60_sep_min"]
    )


def apply_buy_ready(ind: dict) -> tuple[bool, str]:
    """买入准备度过滤"""
    gain1 = ind.get("gain1", 0)
    rsi = ind.get("rsi", 50)
    ma5_dist = ind.get("ma5_distance_pct", 0)
    vol_ratio = ind.get("vol_ratio", 1.0)
    vol_up_vs_down = ind.get("vol_up_vs_down", 1.0)

    if gain1 >= BUY_READY["limit_pct"]:
        return False, "涨停"
    if gain1 < BUY_READY["min_gain1"]:
        return False, f"涨幅{gain1:.1f}%过弱"
    if gain1 > BUY_READY["max_gain1"]:
        return False, f"涨幅{gain1:.1f}%过高"
    if rsi >= BUY_READY["max_rsi"]:
        return False, f"RSI={rsi:.0f}过热"
    if abs(ma5_dist) > BUY_READY["max_ma5_dist"]:
        return False, f"距MA5±{ma5_dist:.0f}%过远"

    return True, "买入就绪"


def compute_score(ind: dict) -> float:
    """综合趋势评分（简化版，与 screen_trend_filter 评分逻辑一致）"""
    score = 50.0
    score += min(ind.get("gain20", 0) * 0.5, 20)  # 20日涨幅加分
    score += min(ind.get("wave_quality", 0) * 2, 15)  # 波段质量
    score += min(ind.get("ma_sep", 0) * 1.5, 10)  # MA分离
    # RSI惩罚
    rsi = ind.get("rsi", 50)
    if rsi > 85:
        score -= 5
    elif rsi > 80:
        score -= 2
    # 量能软扣分
    if ind.get("vol_ratio", 1) < 0.70:
        score -= 3
    if ind.get("vol_up_vs_down", 1) < 0.90:
        score -= 2
    return score


# ── 回测核心 ─────────────────────────────────────────────────

def run_backtest(
    signal_date: str,
    hold_days: int = 5,
    stop_loss_pct: float = 8.0,
) -> list[dict]:
    """
    对单个信号日运行回测

    Returns:
        list of dict{
            "code", "name", "signal_date",
            "entry_price"（T+1开盘价）,
            "exit_price"（卖出收盘价）,
            "hold_days_actual",
            "pnl_pct"（盈亏%）,
            "stopped"（是否止损）,
            "signal_score",
            "reason"（拒绝原因或通过）
        }
    """
    results = []
    codes = [f.stem.replace("_indicators", "") for f in INDICATORS_DIR.glob("*_indicators.json")]

    for code in codes:
        ind = get_indicators(code, signal_date)
        if not ind:
            continue

        # 第一步：基础趋势过滤（趋势三条件）
        if not apply_trend_filter(ind):
            continue

        # 第二步：买入准备度
        ready, reason = apply_buy_ready(ind)
        if not ready:
            continue

        # 第三步：获取 T+1 开盘价（买入价）
        entry = get_price_on_date(code, signal_date)
        if not entry or entry["open"] <= 0:
            continue

        entry_price = entry["open"]
        stop_ref = ind.get("stop_loss_ref", entry_price * 0.97)

        # 第四步：持有 N 天后卖出（或止损）
        exit_price = None
        hold_days_actual = hold_days
        stopped = False

        for d in range(1, hold_days + 1):
            next_date = get_next_trading_date(signal_date, d)
            if not next_date:
                break

            price_data = get_price_on_date(code, next_date)
            if not price_data:
                break

            close_price = price_data["close"]
            daily_high = price_data["high"]
            daily_low = price_data["low"]

            # 止损检查（用当日最低价，模拟盘中触及止损线）
            if daily_low <= stop_ref * (1 - stop_loss_pct / 100):
                exit_price = stop_ref * (1 - stop_loss_pct / 100)  # 以止损价成交
                hold_days_actual = d
                stopped = True
                break

            # 持有到期
            if d == hold_days:
                exit_price = close_price

        if exit_price is None:
            continue

        pnl_pct = (exit_price - entry_price) / entry_price * 100

        results.append({
            "code": code,
            "signal_date": signal_date,
            "entry_price": round(entry_price, 3),
            "exit_price": round(exit_price, 3),
            "hold_days_actual": hold_days_actual,
            "pnl_pct": round(pnl_pct, 3),
            "stopped": stopped,
            "stop_ref": round(stop_ref, 3),
            "signal_score": round(compute_score(ind), 1),
            "reason": reason,
            "gain20": ind.get("gain20", 0),
            "wave_quality": ind.get("wave_quality", 0),
            "ma_sep": ind.get("ma_sep", 0),
            "rsi": ind.get("rsi", 0),
            "vol_ratio": ind.get("vol_ratio", 0),
        })

    return results


def run_full_backtest(
    start_date: str,
    end_date: str,
    hold_days: int = 5,
    skip_interval: int = 4,
) -> list[dict]:
    """
    全量回测

    Args:
        start_date: 信号开始日期
        end_date: 信号结束日期
        hold_days: 持有天数
        skip_interval: 每隔几天出一批信号（避免同日信号过多，默认4天）
    """
    preload_indicators()

    trading_dates = get_trading_dates(start_date, end_date)
    # 跳过前60天（指标预热）
    warmup = 60  # 指标预热天数（需要足够历史数据计算MA60等）
    trading_dates = get_trading_dates(start_date, end_date, lookback=warmup)
    if len(trading_dates) <= warmup:
        print(f"⚠️  日期范围太短（需要>{warmup}天）")
        return []

    # 去掉前 warmup 天（最古老的日期，用于预热）
    trading_dates = trading_dates[warmup:]

    # 每隔 skip_interval 天出一批信号
    signal_dates = trading_dates[::skip_interval]

    print(f"📊 回测区间: {signal_dates[0]} → {signal_dates[-1]}")
    print(f"   持有期: {hold_days} 天  信号间隔: {skip_interval} 天")
    print(f"   预计信号批次数: {len(signal_dates)}")

    all_results = []
    done = 0

    for sd in signal_dates:
        batch = run_backtest(sd, hold_days=hold_days)
        for r in batch:
            r["hold_days_requested"] = hold_days
        all_results.extend(batch)
        done += 1

        if done % 20 == 0 or done == len(signal_dates):
            print(f"  进度: {done}/{len(signal_dates)}  累计信号: {len(all_results)}")

    return all_results


# ── 统计分析 ─────────────────────────────────────────────────

def analyze_results(results: list[dict]) -> dict:
    """统计分析回测结果"""
    if not results:
        return {"error": "无回测结果"}

    pnls = [r["pnl_pct"] for r in results]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    # 分持仓天数
    by_hold = defaultdict(list)
    for r in results:
        by_hold[r["hold_days_actual"]].append(r["pnl_pct"])

    # 分止损与否
    stopped = [r["pnl_pct"] for r in results if r["stopped"]]
    not_stopped = [r["pnl_pct"] for r in results if not r["stopped"]]

    # 分信号强度
    strong = [r for r in results if r["signal_score"] >= 70]
    medium = [r for r in results if 60 <= r["signal_score"] < 70]
    weak = [r for r in results if r["signal_score"] < 60]

    return {
        "total_signals": len(results),
        "win_count": len(wins),
        "win_rate": round(len(wins) / len(results) * 100, 1),
        "avg_pnl": round(np.mean(pnls), 3),
        "median_pnl": round(np.median(pnls), 3),
        "std_pnl": round(np.std(pnls), 3),
        "max_pnl": round(max(pnls), 2),
        "min_pnl": round(min(pnls), 2),
        "avg_win": round(np.mean(wins), 3) if wins else 0,
        "avg_loss": round(np.mean(losses), 3) if losses else 0,
        "profit_factor": round(abs(np.sum(wins) / np.sum(losses)), 2) if losses else float("inf"),

        # 分持仓天数
        "by_hold_days": {
            f"T+{k}天": {
                "count": len(v),
                "win_rate": round(len([x for x in v if x > 0]) / len(v) * 100, 1) if v else 0,
                "avg_pnl": round(np.mean(v), 3) if v else 0,
            }
            for k, v in sorted(by_hold.items())
        },

        # 止损统计
        "stopped_count": len(stopped),
        "stopped_avg_pnl": round(np.mean(stopped), 3) if stopped else 0,
        "not_stopped_count": len(not_stopped),
        "not_stopped_avg_pnl": round(np.mean(not_stopped), 3) if not_stopped else 0,

        # 分信号强度
        "strong_signals": {
            "count": len(strong),
            "win_rate": round(len([x for x in strong if x["pnl_pct"] > 0]) / len(strong) * 100, 1) if strong else 0,
            "avg_pnl": round(np.mean([x["pnl_pct"] for x in strong]), 3) if strong else 0,
        },
        "medium_signals": {
            "count": len(medium),
            "win_rate": round(len([x for x in medium if x["pnl_pct"] > 0]) / len(medium) * 100, 1) if medium else 0,
            "avg_pnl": round(np.mean([x["pnl_pct"] for x in medium]), 3) if medium else 0,
        },
        "weak_signals": {
            "count": len(weak),
            "win_rate": round(len([x for x in weak if x["pnl_pct"] > 0]) / len(weak) * 100, 1) if weak else 0,
            "avg_pnl": round(np.mean([x["pnl_pct"] for x in weak]), 3) if weak else 0,
        },

        # Top5 / Bottom5
        "top5_avg": round(np.mean(sorted(pnls, reverse=True)[:5]), 2),
        "bottom5_avg": round(np.mean(sorted(pnls)[:5]), 2),
    }


def print_report(stats: dict, label: str = ""):
    """打印统计报告"""
    print(f"\n{'=' * 60}")
    print(f"📊 回测统计 {label}")
    print(f"{'=' * 60}")
    print(f"  总信号数:   {stats['total_signals']}")
    print(f"  胜率:       {stats['win_rate']}%")
    print(f"  平均收益:   {stats['avg_pnl']:+.3f}%")
    print(f"  中位数收益: {stats['median_pnl']:+.3f}%")
    print(f"  标准差:     {stats['std_pnl']}")
    print(f"  最大盈利:   {stats['max_pnl']:+.2f}%")
    print(f"  最大亏损:   {stats['min_pnl']:+.2f}%")
    print(f"  平均盈利:   {stats['avg_win']:+.3f}%")
    print(f"  平均亏损:   {stats['avg_loss']:+.3f}%")
    print(f"  盈亏比:     {stats['profit_factor']}")

    print(f"\n📅 分持仓天数:")
    for k, v in stats["by_hold_days"].items():
        print(f"  {k}: {v['count']}笔  胜率{v['win_rate']}%  均值{v['avg_pnl']:+.3f}%")

    print(f"\n🛡️  止损统计:")
    print(f"  止损次数: {stats['stopped_count']}  平均收益: {stats['stopped_avg_pnl']:+.3f}%")
    print(f"  非止损:   {stats['not_stopped_count']}  平均收益: {stats['not_stopped_avg_pnl']:+.3f}%")

    print(f"\n🏆 分信号强度:")
    print(f"  🟢强信号: {stats['strong_signals']['count']}笔  胜率{stats['strong_signals']['win_rate']}%  均值{stats['strong_signals']['avg_pnl']:+.3f}%")
    print(f"  🔵中信号: {stats['medium_signals']['count']}笔  胜率{stats['medium_signals']['win_rate']}%  均值{stats['medium_signals']['avg_pnl']:+.3f}%")
    print(f"  🟡弱信号: {stats['weak_signals']['count']}笔  胜率{stats['weak_signals']['win_rate']}%  均值{stats['weak_signals']['avg_pnl']:+.3f}%")

    print(f"\n  Top5均值: {stats['top5_avg']:+.2f}%")
    print(f"  Bottom5均值: {stats['bottom5_avg']:+.2f}%")


# ── 主程序 ───────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="回测引擎 — screen_trend_filter 胜率验证")
    parser.add_argument("--start", type=str, default="2025-01-01", help="信号开始日期")
    parser.add_argument("--end", type=str, default="2026-04-23", help="信号结束日期")
    parser.add_argument("--hold", type=int, default=5, help="持有天数")
    parser.add_argument("--interval", type=int, default=4, help="信号间隔天数")
    parser.add_argument("--out", type=str, default=None, help="结果输出文件")
    parser.add_argument("--analyze", action="store_true", help="分析已有结果文件")
    parser.add_argument("--combined", action="store_true", help="合并训练+验证结果分析")
    args = parser.parse_args()

    if args.analyze:
        # 分析已有结果
        files = []
        if Path("train_results.json").exists():
            files.append(("训练集", "train_results.json"))
        if Path("val_results.json").exists():
            files.append(("验证集", "val_results.json"))

        for label, path in files:
            with open(path) as f:
                results = json.load(f)
            stats = analyze_results(results)
            print_report(stats, label)

        combined = []
        for _, path in files:
            with open(path) as f:
                combined.extend(json.load(f))
        if combined:
            stats = analyze_results(combined)
            print_report(stats, "全部")
        return

    # 运行回测
    print(f"\n🚀 开始回测: {args.start} → {args.end}")
    print(f"   持有期: {args.hold}天  信号间隔: {args.interval}天")

    results = run_full_backtest(
        start_date=args.start,
        end_date=args.end,
        hold_days=args.hold,
        skip_interval=args.interval,
    )

    if not results:
        print("⚠️  无回测结果")
        return

    stats = analyze_results(results)
    print_report(stats)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n💾 结果已写入: {args.out}")

    return stats


if __name__ == "__main__":
    main()
