#!/usr/bin/env python3
"""
信号监控器回测 V2 — 积分制评分

将每个交易因子拆成独立的积分项，每项有明确的加减分规则。
13维线性权重 → 7大因子积分制，每项0~N分，总分0~100。

因子设计：
  1. 趋势排列 (max 25分): MA多头排列+25, 部分+15, 空头-25, 部分-15
  2. 价格位置 (max 20分): 价>MA5/10/20/60 各+5, 反之-5
  3. 成交量   (max 15分): 量比>2.0→+15, >1.5→+10, <0.5→-10
  4. 短期动量 (max 15分): 5日涨跌>5%→+15, >2%→+10, <-5%→-15
  5. MACD     (max 15分): 金叉→+15, 红柱→+8, 死叉→-15, 绿柱→-8
  6. RSI      (max  5分): 超卖(<30)→+5, 超买(>70)→-5
  7. 开盘跳空 (max  5分): 高开>2%→+5, 低开<-2%→-5

改为对称设计，多空同等对待。
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gain_turnover import load_qfq_history


# ═══════════════════════════════════════════════════════════════
# 积分制评分（V2：对称、透明、每项明确）
# ═══════════════════════════════════════════════════════════════

@dataclass
class ScoreBreakdown:
    """每项因子的得分明细"""
    trend_align: float = 0.0       # 趋势排列 ±25
    price_position: float = 0.0    # 价格位置 ±20
    volume_factor: float = 0.0     # 成交量 ±15
    momentum: float = 0.0          # 短期动量 ±15
    macd_factor: float = 0.0       # MACD ±15
    rsi_factor: float = 0.0        # RSI ±5
    gap_factor: float = 0.0        # 开盘跳空 ±5
    total: float = 0.0

    def to_list(self):
        return [
            f"排列{self.trend_align:+.0f}",
            f"价位{self.price_position:+.0f}",
            f"量{self.volume_factor:+.0f}",
            f"动量{self.momentum:+.0f}",
            f"MACD{self.macd_factor:+.0f}",
            f"RSI{self.rsi_factor:+.0f}",
            f"跳空{self.gap_factor:+.0f}",
        ]


def ema_vec(data: np.ndarray, period: int) -> np.ndarray:
    result = np.full(len(data), np.nan)
    if len(data) < period:
        return result
    m = 2.0 / (period + 1.0)
    result[period - 1] = np.mean(data[:period])
    for i in range(period, len(data)):
        result[i] = (data[i] - result[i - 1]) * m + result[i - 1]
    return result


def calc_all_indicators(close, open_, high, low, volume, min_bars=120):
    """一次计算所有指标，避免重复"""
    n = len(close)
    if n < min_bars:
        return None

    # MA
    def rolling_mean(arr, w):
        out = np.full(n, np.nan)
        for i in range(w - 1, n):
            out[i] = np.mean(arr[i - w + 1:i + 1])
        return out

    ma5 = rolling_mean(close, 5)
    ma10 = rolling_mean(close, 10)
    ma20 = rolling_mean(close, 20)
    ma60 = rolling_mean(close, 60)
    vol_ma5 = rolling_mean(volume, 5)

    # RSI
    rsi = np.full(n, np.nan)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    ag = np.full(n - 1, np.nan)
    al = np.full(n - 1, np.nan)
    ag[13] = np.mean(gain[:14])
    al[13] = np.mean(loss[:14])
    for i in range(14, n - 1):
        ag[i] = (ag[i - 1] * 13 + gain[i]) / 14
        al[i] = (al[i - 1] * 13 + loss[i]) / 14
    for i in range(n):
        j = i - 1
        if j >= 13 and al[j] > 0:
            rsi[i] = 100.0 - 100.0 / (1.0 + ag[j] / al[j])
        elif j >= 13:
            rsi[i] = 100.0

    # MACD
    ema12 = ema_vec(close, 12)
    ema26 = ema_vec(close, 26)
    dif = ema12 - ema26
    dea = ema_vec(dif, 9)
    macd = 2.0 * (dif - dea)

    return {
        "ma5": ma5, "ma10": ma10, "ma20": ma20, "ma60": ma60,
        "vol_ma5": vol_ma5,
        "rsi": rsi, "macd": macd, "dif": dif, "dea": dea,
        "close": close, "open": open_, "high": high, "low": low,
        "volume": volume,
    }


def score_credit_system(idx: int, ind: dict) -> ScoreBreakdown:
    """积分制评分：每项因子独立计分，多空对称"""
    sb = ScoreBreakdown()
    c = ind["close"][idx]
    o = ind["open"][idx]
    v = ind["volume"][idx]

    ma5 = ind["ma5"][idx]
    ma10 = ind["ma10"][idx]
    ma20 = ind["ma20"][idx]
    ma60 = ind["ma60"][idx]
    vma5 = ind["vol_ma5"][idx]
    rsi_v = ind["rsi"][idx]
    macd_v = ind["macd"][idx]

    if np.isnan(ma60) or np.isnan(ma20):
        return sb

    # ── 1. 趋势排列 (max ±25) ──
    if not any(np.isnan(x) for x in [ma5, ma10, ma20, ma60]):
        if ma5 > ma10 > ma20 > ma60:
            sb.trend_align = 25
        elif ma5 > ma10 > ma20:
            sb.trend_align = 15
        elif ma5 > ma10:
            sb.trend_align = 5
        elif ma5 < ma10 < ma20 < ma60:
            sb.trend_align = -25
        elif ma5 < ma10 < ma20:
            sb.trend_align = -15
        elif ma5 < ma10:
            sb.trend_align = -5

    # ── 2. 价格位置 (max ±20) ──
    for ma in [ma5, ma10, ma20, ma60]:
        if not np.isnan(ma):
            if c > ma:
                sb.price_position += 5
            else:
                sb.price_position -= 5

    # ── 3. 成交量 (max ±15) ──
    if not np.isnan(vma5) and vma5 > 0:
        vol_ratio = v / vma5
        if vol_ratio > 2.0:
            sb.volume_factor = 15
        elif vol_ratio > 1.5:
            sb.volume_factor = 10
        elif vol_ratio > 1.2:
            sb.volume_factor = 5
        elif vol_ratio < 0.5:
            sb.volume_factor = -15
        elif vol_ratio < 0.7:
            sb.volume_factor = -10
        elif vol_ratio < 0.9:
            sb.volume_factor = -5

    # ── 4. 短期动量 (max ±15) ──
    if idx >= 5:
        ret_5d = (c - ind["close"][idx - 5]) / ind["close"][idx - 5] * 100
        if ret_5d > 5:
            sb.momentum = 15
        elif ret_5d > 2:
            sb.momentum = 10
        elif ret_5d > 0:
            sb.momentum = 5
        elif ret_5d < -5:
            sb.momentum = -15
        elif ret_5d < -2:
            sb.momentum = -10
        elif ret_5d < 0:
            sb.momentum = -5

    # ── 5. MACD (max ±15) ──
    # 金叉/死叉判断
    if idx >= 1:
        prev_dif = ind["dif"][idx - 1]
        prev_dea = ind["dea"][idx - 1]
        cur_dif = ind["dif"][idx]
        cur_dea = ind["dea"][idx]
        if not np.isnan(prev_dif) and not np.isnan(cur_dif):
            if prev_dif < prev_dea and cur_dif > cur_dea:
                sb.macd_factor = 15  # 金叉
            elif prev_dif > prev_dea and cur_dif < cur_dea:
                sb.macd_factor = -15  # 死叉
            elif cur_dif > cur_dea:
                sb.macd_factor = 8   # 多头
            else:
                sb.macd_factor = -8  # 空头

    # ── 6. RSI (max ±5) ──
    if not np.isnan(rsi_v):
        if rsi_v < 30:
            sb.rsi_factor = 5
        elif rsi_v > 70:
            sb.rsi_factor = -5

    # ── 7. 开盘跳空 (max ±5) ──
    if idx >= 1:
        prev_c = ind["close"][idx - 1]
        if prev_c > 0:
            gap_pct = (o - prev_c) / prev_c * 100
            if gap_pct > 2:
                sb.gap_factor = 5
            elif gap_pct < -2:
                sb.gap_factor = -5

    sb.total = (sb.trend_align + sb.price_position + sb.volume_factor
                + sb.momentum + sb.macd_factor + sb.rsi_factor + sb.gap_factor)
    return sb


# ═══════════════════════════════════════════════════════════════
# 回测
# ═══════════════════════════════════════════════════════════════

@dataclass
class Signal2:
    code: str
    date: str
    score: float
    label: str
    breakdown: ScoreBreakdown
    ret_1d: float = 0
    ret_3d: float = 0
    ret_5d: float = 0


def backtest_all(codes: list[str], start_date=None, end_date=None):
    all_signals = []
    for i, code in enumerate(codes):
        df = load_qfq_history(code, end_date=end_date)
        if df is None or len(df) < 130:
            continue
        if start_date:
            df = df[df.index >= start_date]

        close = df["close"].values
        open_ = df["open"].values
        high = df["high"].values
        low = df["low"].values
        volume = df["volume"].values
        dates = [str(d)[:10] for d in df.index]

        ind = calc_all_indicators(close, open_, high, low, volume)
        if ind is None:
            continue

        n = len(close)
        scores_today = []

        # 第一遍：计算所有评分
        for t in range(130, n - 5):
            if np.isnan(ind["ma60"][t]):
                continue
            sb = score_credit_system(t, ind)
            scores_today.append((t, sb))

        if not scores_today:
            continue

        # 百分位分标签（按当天全市场）
        # 单股回测不用百分位，用绝对阈值
        for t, sb in scores_today:
            if sb.total >= 60:
                label = "买入"
            elif sb.total >= 40:
                label = "关注"
            elif sb.total >= 20:
                label = "观望"
            else:
                label = "卖出"

            ct = close[t]
            r1 = (close[t + 1] - ct) / ct * 100 if t + 1 < n else 0
            r3 = (close[t + 3] - ct) / ct * 100 if t + 3 < n else 0
            r5 = (close[t + 5] - ct) / ct * 100 if t + 5 < n else 0

            all_signals.append(Signal2(
                code=code, date=dates[t], score=sb.total,
                label=label, breakdown=sb,
                ret_1d=r1, ret_3d=r3, ret_5d=r5,
            ))

    return all_signals


def evaluate(signals):
    for tag in ["买入", "关注", "观望", "卖出"]:
        ss = [s for s in signals if s.label == tag]
        if not ss:
            print(f"\n  {tag}: 无信号")
            continue
        r1 = [s.ret_1d for s in ss]
        r3 = [s.ret_3d for s in ss]
        r5 = [s.ret_5d for s in ss]
        print(f"\n  {tag}: {len(ss)} 信号")
        print(f"    T+1: {np.mean(r1):+.2f}% 胜率{sum(1 for r in r1 if r>0)/len(r1):.1%}")
        print(f"    T+3: {np.mean(r3):+.2f}% 胜率{sum(1 for r in r3 if r>0)/len(r3):.1%}")
        print(f"    T+5: {np.mean(r5):+.2f}% 胜率{sum(1 for r in r5 if r>0)/len(r5):.1%}")

    # 区分度
    buys = [s for s in signals if s.label == "买入"]
    sells = [s for s in signals if s.label == "卖出"]
    if buys and sells:
        br1 = np.mean([s.ret_1d for s in buys])
        sr1 = np.mean([s.ret_1d for s in sells])
        print(f"\n  区分度 T+1: {br1-sr1:+.2f}%  T+3: "
              f"{np.mean([s.ret_3d for s in buys])-np.mean([s.ret_3d for s in sells]):+.2f}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", "-f", default="output/watchlist.EBK")
    parser.add_argument("--max-stocks", "-n", type=int, default=30)
    parser.add_argument("--start", "-s", default=None)
    parser.add_argument("--end", "-e", default=None)
    args = parser.parse_args()

    # 加载代码
    path = Path(args.file)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent.parent / path
    raw = path.read_text(encoding="utf-8", errors="ignore")
    codes = []
    for line in raw.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        for tok in line.replace(",", " ").split():
            if tok.isdigit():
                if len(tok) == 7:
                    tok = tok[1:]
                if len(tok) == 6:
                    codes.append(tok)

    codes = list(dict.fromkeys(codes))[:args.max_stocks]
    print(f"回测 {len(codes)} 只股票")
    if args.start:
        print(f"区间: {args.start} ~ {args.end or '最新'}")

    signals = backtest_all(codes, args.start, args.end)
    print(f"\n总信号: {len(signals)}")
    print("=" * 50)
    evaluate(signals)

    # 显示买入信号的因子分解（前5个例子）
    buys = [s for s in signals if s.label == "买入"][:5]
    if buys:
        print("\n" + "=" * 50)
        print("买入信号因子分解示例:")
        for s in buys:
            parts = s.breakdown.to_list()
            print(f"  {s.code} {s.date} 总分={s.score:+.0f}  {' '.join(parts)}")


if __name__ == "__main__":
    main()
