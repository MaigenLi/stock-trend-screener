#!/usr/bin/env python3
"""
信号监控器回测与参数寻优

对 signal_monitor 的评分体系做真实历史回测：
  - 用 gain_turnover.load_qfq_history() 获取前复权 K 线
  - 每天模拟 T 日收盘后用历史数据评分
  - 记录 T+1/T+3/T+5 前向收益
  - 网格搜索最优参数组合

目标：找到能区分"买入后上涨"和"卖出后下跌"的参数权重
"""

import argparse
import itertools
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# 确保能导入 gain_turnover
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gain_turnover import load_qfq_history, get_stock_name

# ═══════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════

@dataclass
class SignalWeights:
    """评分权重参数（待寻优）"""
    # 价格 vs 均线（±分）
    w_price_vs_ma: float = 4.0
    # 均线多头排列（+分）
    w_full_align: float = 12.0    # MA5>MA10>MA20>MA60
    w_partial_align: float = 6.0   # MA5>MA10>MA20
    # 均线空头排列（-分）
    w_full_misalign: float = -12.0
    w_partial_misalign: float = -6.0
    # 均线方向（±分）
    w_direction: float = 3.0       # 每条均线方向
    # RSI 买/卖阈值
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    w_rsi_bonus: float = 5.0
    w_rsi_penalty: float = -3.0
    # 前一交易日涨跌对今日预期（动量延续因子）
    w_momentum: float = 2.0        # 前日涨 >0, 今日加分
    # MACD 柱状图贡献
    w_macd_bullish: float = 3.0
    w_macd_bearish: float = -2.0
    # 评分基线（不参与参数寻优，固定50）
    baseline: float = 50.0


DEFAULT_WEIGHTS = SignalWeights()


# ═══════════════════════════════════════════════════════════════
# 指标计算（纯向量化，无未来数据）
# ═══════════════════════════════════════════════════════════════

def ema_vec(data: np.ndarray, period: int) -> np.ndarray:
    """EMA 向量化"""
    result = np.full(len(data), np.nan)
    if len(data) < period:
        return result
    multiplier = 2.0 / (period + 1.0)
    result[period - 1] = np.mean(data[:period])
    for i in range(period, len(data)):
        result[i] = (data[i] - result[i - 1]) * multiplier + result[i - 1]
    return result


def calc_indicators_vectorized(close: np.ndarray, high: np.ndarray,
                               low: np.ndarray, volume: np.ndarray,
                               min_bars: int = 120) -> dict:
    """对全量K线计算指标数组（无未来数据）"""
    n = len(close)
    if n < min_bars:
        return {}

    ma5 = np.array([np.nan] * n)
    ma10 = np.array([np.nan] * n)
    ma20 = np.array([np.nan] * n)
    ma60 = np.array([np.nan] * n)

    for i in range(n):
        if i >= 4:
            ma5[i] = np.mean(close[i - 4:i + 1])
        if i >= 9:
            ma10[i] = np.mean(close[i - 9:i + 1])
        if i >= 19:
            ma20[i] = np.mean(close[i - 19:i + 1])
        if i >= 59:
            ma60[i] = np.mean(close[i - 59:i + 1])

    # RSI (Wilder smoothing)
    rsi = np.full(n, np.nan)
    if n > 14:
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        avg_gain = np.full(n - 1, np.nan)
        avg_loss = np.full(n - 1, np.nan)
        avg_gain[13] = np.mean(gain[:14])
        avg_loss[13] = np.mean(loss[:14])
        for i in range(14, n - 1):
            avg_gain[i] = (avg_gain[i - 1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i - 1] * 13 + loss[i]) / 14
        for i in range(n):
            j = i - 1
            if j >= 13 and avg_loss[j] > 0:
                rs = avg_gain[j] / avg_loss[j]
                rsi[i] = 100.0 - 100.0 / (1.0 + rs)
            elif j >= 13:
                rsi[i] = 100.0

    # MACD
    ema12 = ema_vec(close, 12)
    ema26 = ema_vec(close, 26)
    dif = ema12 - ema26
    dea = ema_vec(dif, 9)
    macd_hist = 2.0 * (dif - dea)

    return {
        "ma5": ma5, "ma10": ma10, "ma20": ma20, "ma60": ma60,
        "rsi": rsi, "dif": dif, "dea": dea, "macd_hist": macd_hist,
        "close": close, "volume": volume,
    }


# ═══════════════════════════════════════════════════════════════
# 评分函数（在 T 日收盘后调用，只看 T 日及之前的数据）
# ═══════════════════════════════════════════════════════════════

def score_signal_at(idx: int, ind: dict, w: SignalWeights) -> float:
    """在 idx 位置评分（T日收盘），不使用 idx 之后的任何数据"""
    close_t = ind["close"][idx]
    ma5_t = ind["ma5"][idx]
    ma10_t = ind["ma10"][idx]
    ma20_t = ind["ma20"][idx]
    ma60_t = ind["ma60"][idx]
    rsi_t = ind["rsi"][idx]
    macd_t = ind["macd_hist"][idx]

    if np.isnan(ma60_t):
        return 0.0

    score = w.baseline

    # 1. 价格 vs 均线
    for ma in [ma5_t, ma10_t, ma20_t, ma60_t]:
        if np.isnan(ma):
            continue
        if close_t > ma:
            score += w.w_price_vs_ma
        else:
            score -= w.w_price_vs_ma

    # 2. 均线排列
    if not any(np.isnan(x) for x in [ma5_t, ma10_t, ma20_t, ma60_t]):
        if ma5_t > ma10_t > ma20_t > ma60_t:
            score += w.w_full_align
        elif ma5_t > ma10_t > ma20_t:
            score += w.w_partial_align
        elif ma5_t < ma10_t < ma20_t < ma60_t:
            score += w.w_full_misalign
        elif ma5_t < ma10_t < ma20_t:
            score += w.w_partial_misalign

    # 3. 均线方向（与5日前比较）
    for i, ma_arr in enumerate([
        (ma5_t, ind["ma5"]), (ma10_t, ind["ma10"]),
        (ma20_t, ind["ma20"]), (ma60_t, ind["ma60"]),
    ]):
        cur_val, arr = ma_arr
        prev_idx = max(0, idx - 5)
        prev_val = arr[prev_idx] if prev_idx < len(arr) else np.nan
        if not np.isnan(cur_val) and not np.isnan(prev_val):
            if cur_val > prev_val * 1.001:
                score += w.w_direction
            elif cur_val < prev_val * 0.999:
                score -= w.w_direction

    # 4. RSI
    if not np.isnan(rsi_t):
        if rsi_t < w.rsi_oversold:
            score += w.w_rsi_bonus
        elif rsi_t > w.rsi_overbought:
            score += w.w_rsi_penalty

    # 5. 前日涨跌动量
    if idx > 0:
        prev_close = ind["close"][idx - 1]
        if close_t > prev_close * 1.001:
            score += w.w_momentum
        elif close_t < prev_close * 0.999:
            score -= w.w_momentum

    # 6. MACD
    if not np.isnan(macd_t):
        if macd_t > 0:
            score += w.w_macd_bullish
        else:
            score += w.w_macd_bearish

    return max(min(score, 100.0), 0.0)


# ═══════════════════════════════════════════════════════════════
# 标签分配（百分位制，与 signal_monitor 一致）
# ═══════════════════════════════════════════════════════════════

def assign_label(score: float, all_scores: list[float], idx: int,
                 close_t: float, prev_close: float,
                 macd_hist: float) -> str:
    """在 all_scores 中分配标签"""
    n = len(all_scores)
    if n < 5:
        if score >= 60:
            return "买入"
        elif score <= 30:
            return "卖出"
        return "观望"

    sorted_scores = sorted(all_scores)
    p80 = sorted_scores[max(0, min(n - 1, int(n * 0.80)))]
    p60 = sorted_scores[max(0, min(n - 1, int(n * 0.60)))]
    p30 = sorted_scores[max(0, min(n - 1, int(n * 0.30)))]

    # 安全垫
    change_pct = (close_t - prev_close) / prev_close * 100 if prev_close > 0 else 0
    safe = (change_pct > 0 or macd_hist > 0)

    if score >= p80:
        return "买入"
    elif score >= p60:
        return "关注"
    elif score >= p30 or safe:
        return "观望"
    return "卖出"


# ═══════════════════════════════════════════════════════════════
# 回测引擎
# ═══════════════════════════════════════════════════════════════

@dataclass
class BacktestSignal:
    date: str
    code: str
    score: float
    label: str
    close: float
    ret_1d: float | None = None  # T+1 收益率
    ret_3d: float | None = None  # T+1到T+3 累积收益率
    ret_5d: float | None = None  # T+1到T+5 累积收益率


def backtest_stock(code: str, df: pd.DataFrame, w: SignalWeights,
                   start_idx: int = 120) -> list[BacktestSignal]:
    """对单只股票回测，返回每一天的信号"""
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    volume = df["volume"].values
    dates = df.index.astype(str).tolist() if hasattr(df.index, 'astype') else [str(d) for d in df.index]

    ind = calc_indicators_vectorized(close, high, low, volume, min_bars=120)
    if not ind:
        return []

    n = len(close)
    results = []

    for i in range(start_idx, n - 5):  # 留出至少5天前向收益
        if np.isnan(ind["ma60"][i]):
            continue

        score = score_signal_at(i, ind, w)
        if score == 0:
            continue

        # 当天所有股票评分列表（单股回测时，标签用绝对阈值）
        if score >= 80:
            label = "买入"
        elif score >= 65:
            label = "关注"
        elif score >= 40:
            label = "观望"
        else:
            label = "卖出"

        close_t = close[i]
        close_t1 = close[i + 1] if i + 1 < n else close_t
        close_t3 = close[i + 3] if i + 3 < n else close_t
        close_t5 = close[i + 5] if i + 5 < n else close_t

        ret_1d = (close_t1 - close_t) / close_t * 100 if close_t > 0 else 0
        ret_3d = (close_t3 - close_t) / close_t * 100 if close_t > 0 else 0
        ret_5d = (close_t5 - close_t) / close_t * 100 if close_t > 0 else 0

        results.append(BacktestSignal(
            date=dates[i] if i < len(dates) else str(i),
            code=code,
            score=score,
            label=label,
            close=close_t,
            ret_1d=ret_1d,
            ret_3d=ret_3d,
            ret_5d=ret_5d,
        ))

    return results


# ═══════════════════════════════════════════════════════════════
# 评价指标
# ═══════════════════════════════════════════════════════════════

def evaluate_weights(signals: list[BacktestSignal]) -> dict:
    """评估一组权重参数的表现"""
    buy_signals = [s for s in signals if s.label == "买入"]
    sell_signals = [s for s in signals if s.label == "卖出"]

    def stats(sig_list: list, tag: str) -> dict:
        if not sig_list:
            return {f"{tag}_count": 0}
        r1 = [s.ret_1d for s in sig_list if s.ret_1d is not None]
        r3 = [s.ret_3d for s in sig_list if s.ret_3d is not None]
        r5 = [s.ret_5d for s in sig_list if s.ret_5d is not None]
        return {
            f"{tag}_count": len(sig_list),
            f"{tag}_ret1d_mean": np.mean(r1) if r1 else 0,
            f"{tag}_ret1d_win": sum(1 for r in r1 if r > 0) / len(r1) if r1 else 0,
            f"{tag}_ret3d_mean": np.mean(r3) if r3 else 0,
            f"{tag}_ret3d_win": sum(1 for r in r3 if r > 0) / len(r3) if r3 else 0,
            f"{tag}_ret5d_mean": np.mean(r5) if r5 else 0,
            f"{tag}_ret5d_win": sum(1 for r in r5 if r > 0) / len(r5) if r5 else 0,
        }

    result = {}
    result.update(stats(buy_signals, "buy"))
    result.update(stats(sell_signals, "sell"))

    # 关键指标：买入信号的平均收益
    result["buy_ret1d_mean"] = result.get("buy_ret1d_mean", 0)
    result["buy_ret3d_mean"] = result.get("buy_ret3d_mean", 0)
    result["buy_ret5d_mean"] = result.get("buy_ret5d_mean", 0)

    # 区分度指标：买入收益 - 卖出收益
    if result.get("buy_count", 0) > 0 and result.get("sell_count", 0) > 0:
        result["separation_1d"] = result.get("buy_ret1d_mean", 0) - result.get("sell_ret1d_mean", 0)
        result["separation_3d"] = result.get("buy_ret3d_mean", 0) - result.get("sell_ret3d_mean", 0)
        result["separation_5d"] = result.get("buy_ret5d_mean", 0) - result.get("sell_ret5d_mean", 0)
    else:
        result["separation_1d"] = 0
        result["separation_3d"] = 0
        result["separation_5d"] = 0

    return result


# ═══════════════════════════════════════════════════════════════
# 参数寻优
# ═══════════════════════════════════════════════════════════════

def load_watchlist(filepath: str) -> list[str]:
    """从 EBK 或 txt 加载股票代码"""
    codes = []
    path = Path(filepath)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent.parent / path
    if not path.exists():
        path = Path.home() / ".openclaw/workspace" / filepath
    if not path.exists():
        print(f"警告：文件不存在 {filepath}", file=sys.stderr)
        return codes
    raw = path.read_text(encoding="utf-8", errors="ignore")
    for line in raw.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("{"):
            try:
                obj = json.loads(line)
                codes.append(obj.get("code", obj.get("symbol", "")))
            except Exception:
                pass
        else:
            # 纯文本代码：支持6位(600862)或7位(1600744=上海+code)
            for token in line.replace(",", " ").split():
                token = token.strip()
                if token.isdigit():
                    if len(token) == 7:
                        token = token[1:]  # 去掉市场前缀
                    if len(token) == 6:
                        codes.append(token)
    return codes


def param_grid_search(codes: list[str], date_start: str | None = None,
                      date_end: str | None = None, max_stocks: int = 0,
                      n_samples: int = 200):
    """随机搜索最优参数"""
    # 参数空间（缩小到5只验证">5只</缩小到5只验证"）
    import random
    random.seed(42)

    param_space = {
        "w_price_vs_ma": [1.0, 2.0, 3.0, 4.0, 5.0],
        "w_full_align": [6.0, 8.0, 10.0, 12.0, 15.0],
        "w_partial_align": [3.0, 5.0, 7.0],
        "w_full_misalign": [-6.0, -8.0, -10.0, -12.0, -15.0],
        "w_partial_misalign": [-3.0, -5.0, -7.0],
        "w_direction": [1.0, 2.0, 3.0, 4.0],
        "rsi_oversold": [20.0, 25.0, 30.0, 35.0],
        "rsi_overbought": [65.0, 70.0, 75.0, 80.0],
        "w_rsi_bonus": [2.0, 4.0, 6.0],
        "w_rsi_penalty": [-1.0, -3.0, -5.0],
        "w_momentum": [0.5, 1.0, 2.0, 3.0],
        "w_macd_bullish": [1.0, 2.0, 3.0, 5.0],
        "w_macd_bearish": [-0.5, -1.0, -2.0, -3.0],
    }

    keys = list(param_space.keys())

    # 生成随机样本
    samples = []
    for _ in range(n_samples):
        combo = tuple(random.choice(param_space[k]) for k in keys)
        samples.append(combo)

    # 去掉重复
    samples = list(dict.fromkeys(samples))
    total = len(samples)
    print(f"随机采样: {total} 个参数组合")
    print(f"股票数量: {len(codes)}")
    if date_start:
        print(f"回测区间: {date_start} ~ {date_end or '最新'}")
    print()

    # 预加载所有股票数据
    all_hist = {}
    loaded = 0
    for code in codes:
        df = load_qfq_history(code, end_date=date_end)
        if df is not None and len(df) >= 130:
            if date_start:
                df = df[df.index >= date_start]
            if len(df) >= 130:
                all_hist[code] = df
                loaded += 1
        if max_stocks > 0 and loaded >= max_stocks:
            break

    print(f"成功加载 {loaded}/{len(codes)} 只股票历史数据\n")

    # ── 断点续跑 ──
    ckpt_path = Path.home() / ".openclaw/workspace/stock_trend/mootdx/.backtest_checkpoint.json"
    evaluated = set()
    best = None
    best_score = -999
    if ckpt_path.exists():
        ckpt = json.loads(ckpt_path.read_text())
        evaluated = set(tuple(c) if isinstance(c, list) else c for c in ckpt.get("evaluated", []))
        if ckpt.get("best"):
            best = ckpt["best"]
            best_score = ckpt["best"]["opt_score"]
        print(f"📂 续跑: 已评估 {len(evaluated)}/{total} 组合")

    best = best or {}
    best_score = best.get("opt_score", -999) if isinstance(best, dict) else -999
    start_time = time.time()

    for combo_idx, combo in enumerate(samples):
        # 跳过已评估的组合
        if combo in evaluated:
            continue
        w = SignalWeights(
            w_price_vs_ma=combo[0],
            w_full_align=combo[1],
            w_partial_align=combo[2],
            w_full_misalign=combo[3],
            w_partial_misalign=combo[4],
            w_direction=combo[5],
            rsi_oversold=combo[6],
            rsi_overbought=combo[7],
            w_rsi_bonus=combo[8],
            w_rsi_penalty=combo[9],
            w_momentum=combo[10],
            w_macd_bullish=combo[11],
            w_macd_bearish=combo[12],
        )

        all_signals = []
        for code, df in all_hist.items():
            sigs = backtest_stock(code, df, w)
            all_signals.extend(sigs)

        if not all_signals:
            continue

        ev = evaluate_weights(all_signals)
        evaluated.add(combo)

        # 优化目标：买入信号 T+3 平均收益
        buy_count = ev.get("buy_count", 0)
        buy_ret3d = ev.get("buy_ret3d_mean", 0)
        buy_win3 = ev.get("buy_ret3d_win", 0)
        separation = ev.get("separation_3d", 0)

        opt_score = buy_ret3d * 0.4 + buy_win3 * 0.3 + separation * 0.3

        n_done = len(evaluated)
        if n_done % max(1, total // 20) == 0:
            elapsed = time.time() - start_time
            print(f"  [{n_done}/{total}] {n_done/total*100:.0f}% "
                  f"({elapsed:.0f}s) best={best_score:.2f}")

        if buy_count >= 10 and buy_win3 >= 0.45 and opt_score > best_score:
            best_score = opt_score
            best = {
                "combo_idx": combo_idx,
                "weights": {
                    "w_price_vs_ma": w.w_price_vs_ma,
                    "w_full_align": w.w_full_align,
                    "w_partial_align": w.w_partial_align,
                    "w_full_misalign": w.w_full_misalign,
                    "w_partial_misalign": w.w_partial_misalign,
                    "w_direction": w.w_direction,
                    "rsi_oversold": w.rsi_oversold,
                    "rsi_overbought": w.rsi_overbought,
                    "w_rsi_bonus": w.w_rsi_bonus,
                    "w_rsi_penalty": w.w_rsi_penalty,
                    "w_momentum": w.w_momentum,
                    "w_macd_bullish": w.w_macd_bullish,
                    "w_macd_bearish": w.w_macd_bearish,
                    "baseline": w.baseline,
                },
                "total_signals": len(all_signals),
                "eval": {k: round(v, 4) if isinstance(v, float) else v
                         for k, v in ev.items()},
                "opt_score": round(opt_score, 4),
            }

        # 每5个组合保存一次进度
        if n_done % 5 == 0:
            ckpt_path.write_text(json.dumps({
                "evaluated": [list(c) for c in evaluated],
                "best": best,
                "n_done": n_done,
                "total": total,
            }, indent=2, ensure_ascii=False))

    # 最终保存
    ckpt_path.write_text(json.dumps({
        "evaluated": [list(c) for c in evaluated],
        "best": best,
        "n_done": len(evaluated),
        "total": total,
        "completed": True,
    }, indent=2, ensure_ascii=False))

    elapsed = time.time() - start_time
    print(f"\n完成！耗时 {elapsed:.0f}s ({elapsed/60:.1f}min)")

    return best, all_hist


# ═══════════════════════════════════════════════════════════════
# 报告
# ═══════════════════════════════════════════════════════════════

def print_report(best: dict):
    print("\n" + "=" * 60)
    print(" 最优参数")
    print("=" * 60)
    for k, v in best["weights"].items():
        print(f"  {k:25s} = {v}")
    print()
    print(f"  总信号数: {best['total_signals']}")
    print(f"  优化评分: {best['opt_score']:.4f}")
    print()

    ev = best["eval"]
    print("=" * 60)
    print(" 回测指标")
    print("=" * 60)

    for tag, cn in [("buy", "🟢买入"), ("sell", "🔴卖出")]:
        c = ev.get(f"{tag}_count", 0)
        if c == 0:
            print(f"\n  {cn}: 无信号")
            continue

        r1m = ev.get(f"{tag}_ret1d_mean", 0)
        r1w = ev.get(f"{tag}_ret1d_win", 0)
        r3m = ev.get(f"{tag}_ret3d_mean", 0)
        r3w = ev.get(f"{tag}_ret3d_win", 0)
        r5m = ev.get(f"{tag}_ret5d_mean", 0)
        r5w = ev.get(f"{tag}_ret5d_win", 0)

        print(f"\n  {cn}: {c} 个信号")
        print(f"    T+1 平均收益: {r1m:+.2f}%  胜率: {r1w:.1%}")
        print(f"    T+3 平均收益: {r3m:+.2f}%  胜率: {r3w:.1%}")
        print(f"    T+5 平均收益: {r5m:+.2f}%  胜率: {r5w:.1%}")

    print(f"\n  区分度 (买入-卖出):")
    print(f"    T+1: {ev.get('separation_1d', 0):+.2f}%")
    print(f"    T+3: {ev.get('separation_3d', 0):+.2f}%")
    print(f"    T+5: {ev.get('separation_5d', 0):+.2f}%")


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="信号监控器回测与参数寻优")
    parser.add_argument("--file", "-f", default="output/watchlist.EBK",
                        help="股票列表文件")
    parser.add_argument("--start", "-s", default=None,
                        help="回测起始日期 (YYYY-MM-DD)")
    parser.add_argument("--end", "-e", default=None,
                        help="回测结束日期 (YYYY-MM-DD)")
    parser.add_argument("--max-stocks", "-n", type=int, default=30,
                        help="最多回测股票数（默认30）")
    parser.add_argument("--samples", type=int, default=200,
                        help="随机采样参数组合数（默认200）")
    parser.add_argument("--weights", "-w", default=None,
                        help="使用指定权重JSON文件（跳过寻优）")
    parser.add_argument("--output", "-o",
                        default=str(Path.home() / ".openclaw/workspace/stock_trend/mootdx/best_weights.json"),
                        help="输出最优权重JSON文件")
    args = parser.parse_args()

    codes = load_watchlist(args.file)
    if not codes:
        print("错误：无股票代码", file=sys.stderr)
        sys.exit(1)

    print(f"股票池: {len(codes)} 只 (回测 {min(args.max_stocks, len(codes))} 只)\n")

    if args.weights:
        # 使用指定权重
        w_data = json.loads(Path(args.weights).read_text())
        w = SignalWeights(**w_data)
        print("使用指定权重，跳过寻优\n")

        all_signals = []
        for code in codes[:args.max_stocks]:
            df = load_qfq_history(code, end_date=args.end)
            if df is not None and len(df) >= 130:
                if args.start:
                    df = df[df.index >= args.start]
                sigs = backtest_stock(code, df, w)
                all_signals.extend(sigs)

        ev = evaluate_weights(all_signals)
        print("=" * 60)
        print(" 回测结果")
        print("=" * 60)
        for tag, cn in [("buy", "买入"), ("sell", "卖出")]:
            c = ev.get(f"{tag}_count", 0)
            if c == 0:
                continue
            print(f"\n  {cn}: {c} 个信号")
            print(f"    T+1 均收益: {ev.get(f'{tag}_ret1d_mean',0):+.2f}%  胜率: {ev.get(f'{tag}_ret1d_win',0):.1%}")
            print(f"    T+3 均收益: {ev.get(f'{tag}_ret3d_mean',0):+.2f}%  胜率: {ev.get(f'{tag}_ret3d_win',0):.1%}")
            print(f"    T+5 均收益: {ev.get(f'{tag}_ret5d_mean',0):+.2f}%  胜率: {ev.get(f'{tag}_ret5d_win',0):.1%}")
        return

    # 参数寻优
    best, _ = param_grid_search(
        codes, date_start=args.start, date_end=args.end,
        max_stocks=args.max_stocks, n_samples=args.samples,
    )

    if best is None:
        print("未找到有效参数", file=sys.stderr)
        sys.exit(1)

    print_report(best)

    if args.output:
        Path(args.output).write_text(
            json.dumps(best["weights"], indent=2, ensure_ascii=False)
        )
        print(f"\n权重已保存到 {args.output}")


if __name__ == "__main__":
    main()
