#!/usr/bin/env python3
"""
策略1选股系统
==============

信号条件（全部基于T日及之前数据，无未来数据）：
  1. 收盘 > MA5
  2. MA5上涨, MA10上涨, MA20上涨, MA60上涨
  3. MA5 > MA10 > MA20 > MA60（均线多头排列）
  4. MACD > 0, DIF > 0, DEA > 0
  5. 最近5日涨幅 > 5%
  6. 5日均换手率按市值分档过滤

市值换手率门槛：
  >= 500亿 → ≥ 1.0%
  >= 100亿 → ≥ 3.0%
  >= 30亿  → ≥ 5.0%
  < 30亿  → ≥ 10.0%

使用方法：
  python screen_strat1.py --date 2026-04-24
  python screen_strat1.py --date 2026-04-24 --top-n 20
"""

import sys
import json
import time
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

WORKSPACE = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(WORKSPACE))

from stock_trend.review_screen.screen import load_stock_names

QFQ_DIR = WORKSPACE / ".cache" / "qfq_daily"

# ── 换手率门槛（按市值分档）───────────────────────────────
def min_turnover_by_cap(market_cap):
    """根据市值返回最低换手率门槛（%）"""
    if market_cap >= 500:
        return 1.0
    elif market_cap >= 100:
        return 3.0
    elif market_cap >= 30:
        return 5.0
    else:
        return 10.0


# ── 数据加载 ─────────────────────────────────────────────
_price = {}

def preload():
    global _price
    print("📂 加载数据...", flush=True)
    for f in QFQ_DIR.glob("*_qfq.csv"):
        code = f.stem.replace("_qfq", "")
        try:
            df = pd.read_csv(f, usecols=["date","open","close","high","low","volume","amount","turnover","true_turnover"])
            df = df.sort_values("date").reset_index(drop=True)
            _price[code] = df
        except:
            pass
    print(f"✅ {len(_price)}只股票已加载", flush=True)


def get_dates():
    for df in list(_price.values())[:1]:
        return sorted(df["date"].tolist())
    return []


def next_date(date, offset=1):
    dates = get_dates()
    try:
        idx = dates.index(date)
        if 0 <= idx + offset < len(dates):
            return dates[idx + offset]
    except:
        pass
    return None


# ── 指标计算 ─────────────────────────────────────────────
def calc_ma(closes, period):
    if len(closes) < period:
        return None
    return float(np.mean(closes[-period:]))


def calc_macd(closes, fast=12, slow=26, signal=9):
    """计算MACD/DIF/DEA（标准通达信算法）"""
    n = len(closes)
    if n < slow + signal:
        return None, None, None
    ef = calc_ema(closes, fast)
    es = calc_ema(closes, slow)
    if ef is None or es is None:
        return None, None, None
    dif = ef - es
    # 构建DIF序列（每个时点的DIF）
    dif_series = []
    for i in range(signal, n):
        ef_i = calc_ema(closes[:i+1], fast)
        es_i = calc_ema(closes[:i+1], slow)
        if ef_i is not None and es_i is not None:
            dif_series.append(ef_i - es_i)
    if len(dif_series) < signal:
        return None, None, None
    dea = calc_ema(dif_series, signal)
    macd = (dif - dea) * 2 if dea is not None else None
    return macd, dif, dea


def calc_ema(closes, period):
    if len(closes) < period:
        return None
    alpha = 2.0 / (period + 1)
    ema = float(closes[0])
    for c in closes[1:]:
        ema = alpha * c + (1 - alpha) * ema
    return ema


def ma_direction(closes, period):
    """判断均线方向：上涨=1，下跌=-1，不足=0"""
    if len(closes) < period + 5:
        return 0
    ma_now = calc_ma(closes, period)
    ma_5d_ago = calc_ma(closes[:-5], period) if len(closes) >= 5 else None
    if ma_now is None or ma_5d_ago is None:
        return 0
    if ma_now > ma_5d_ago * 1.001:  # 微弱上涨也算上涨
        return 1
    elif ma_now < ma_5d_ago * 0.999:
        return -1
    return 0


# ── 核心分析 ─────────────────────────────────────────────
def analyze_stock(code, signal_date):
    """检查是否满足策略1全部条件"""
    df = _price.get(code)
    if df is None:
        return None
    il = df["date"].tolist()
    try:
        idx = il.index(signal_date)
    except:
        return None
    if idx < 65:  # 至少需要足够历史算MA60
        return None

    # 提取数据：T-65 到 T（取够均线和MACD的历史）
    window = df.iloc[idx - 65:idx + 1]
    closes = window["close"].values
    volumes = window["volume"].values
    amounts = window["amount"].values
    # turnover/true_turnover 字段直接就是换手率%（如0.3=0.3%）
    turnovers = window["turnover"].values if "turnover" in window.columns else np.zeros(len(window))
    T_pos = len(closes) - 1
    close_T = closes[T_pos]

    # ── 基础数据 ────────────────────────────────────────
    ma5_now = calc_ma(closes, 5)
    ma10_now = calc_ma(closes, 10)
    ma20_now = calc_ma(closes, 20)
    ma60_now = calc_ma(closes, 60)
    if None in [ma5_now, ma10_now, ma20_now, ma60_now]:
        return None

    # ── 条件1：收盘 > MA5 ───────────────────────────────
    if close_T <= ma5_now:
        return None

    # ── 条件2：MA5/10/20/60全部上涨 ───────────────────
    dir5 = ma_direction(closes, 5)
    dir10 = ma_direction(closes, 10)
    dir20 = ma_direction(closes, 20)
    dir60 = ma_direction(closes, 60)
    if not (dir5 == 1 and dir10 == 1 and dir20 == 1 and dir60 == 1):
        return None

    # ── 条件3：均线多头排列 ─────────────────────────────
    if not (ma5_now > ma10_now > ma20_now > ma60_now):
        return None

    # ── 条件4：MACD > 0, DIF > 0, DEA > 0 ─────────────
    macd, dif, dea = calc_macd(closes)
    if macd is None or dif is None or dea is None:
        return None
    if not (macd > 0 and dif > 0 and dea > 0):
        return None

    # ── 条件5：最近5日涨幅 > 5% ───────────────────────
    if T_pos < 5:
        return None
    close_5d_ago = closes[T_pos - 5]
    gain5d = (close_T / close_5d_ago - 1) * 100
    if gain5d <= 5.0:
        return None

    # ── 条件6：换手率（市值分档）───────────────────────
    market_cap = 0.0
    cap_path = Path.home() / "stock_code" / "results" / "all_stock_names_final.json"
    if cap_path.exists():
        try:
            names_data = json.load(open(cap_path))
            if code in names_data:
                market_cap = float(names_data[code].get("market_cap", 0))
        except:
            pass

    # 计算5日均换手率（直接用CSV的turnover字段，单位%）
    avg_turnover_5 = float(np.mean(turnovers[T_pos - 4:T_pos + 1])) if T_pos >= 4 else float(turnovers[T_pos])

    threshold = min_turnover_by_cap(market_cap)
    if avg_turnover_5 < threshold:
        return None

    # ── 信号汇总 ───────────────────────────────────────
    return {
        "code": code,
        "signal_date": signal_date,
        "close": round(close_T, 2),
        "ma5": round(ma5_now, 2),
        "ma10": round(ma10_now, 2),
        "ma20": round(ma20_now, 2),
        "ma60": round(ma60_now, 2),
        "macd": round(macd, 4),
        "dif": round(dif, 4),
        "dea": round(dea, 4),
        "gain5d": round(gain5d, 2),
        "avg_turnover_5": round(avg_turnover_5, 2),
        "market_cap": round(market_cap, 1),
        "threshold": threshold,
        "dir5": dir5, "dir10": dir10, "dir20": dir20, "dir60": dir60,
    }


# ── 扫描主函数 ─────────────────────────────────────────
def screen_strat1(target_date, top_n=10):
    print(f"📊 策略1扫描: {target_date}", flush=True)
    start = time.time()
    if not _price:
        preload()

    codes = [f.stem.replace("_qfq", "") for f in QFQ_DIR.glob("*_qfq.csv")]
    print(f"   全市场 {len(codes)} 只", flush=True)

    signals = []
    for code in codes:
        result = analyze_stock(code, target_date)
        if result:
            signals.append(result)

    print(f"   完成: {time.time() - start:.1f}秒", flush=True)

    if not signals:
        print(f"\n⚠️  无策略1信号（{target_date}）")
        return []

    # 排序：5日涨幅优先（涨得越多越优先）
    signals.sort(key=lambda x: -x["gain5d"])

    names = load_stock_names()
    for r in signals:
        r["name"] = names.get(r["code"], r["code"])

    print(f"\n🏆 策略1信号（{len(signals)} 只）")
    print("=" * 110)
    print(f"{'代码':<12} {'名称':<8} {'收盘':>7} {'MA5':>7} {'MA10':>7} {'MA20':>7} {'MA60':>7} "
          f"{'5日涨':>7} {'换手':>6} {'市值(亿)':>9}")
    print("-" * 110)
    for r in signals[:top_n]:
        print(f"{r['code']:<12} {r['name']:<8} {r['close']:>7.2f} "
              f"{r['ma5']:>7.2f} {r['ma10']:>7.2f} {r['ma20']:>7.2f} {r['ma60']:>7.2f} "
              f"{r['gain5d']:>+6.1f}% {r.get('avg_turnover_5', 0):>5.1f}% {r.get('market_cap', 0):>8.0f}")

    # MACD详情
    print(f"\n{'代码':<12} {'MACD':>9} {'DIF':>9} {'DEA':>9} {'多头方向':>12}")
    print("-" * 55)
    for r in signals[:top_n]:
        dirs = f"MA5{'↑' if r['dir5']==1 else '↓'} MA10{'↑' if r['dir10']==1 else '↓'} MA20{'↑' if r['dir20']==1 else '↓'} MA60{'↑' if r['dir60']==1 else '↓'}"
        print(f"{r['code']:<12} {r['macd']:>9.4f} {r['dif']:>9.4f} {r['dea']:>9.4f} {dirs}")

    out = Path.home() / "stock_reports" / f"strat1_{target_date}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"date": target_date, "strategy": "策略1", "signals": signals}, f, ensure_ascii=False, indent=2)
    print(f"\n💾 已保存: {out}")
    return signals


# ── 入口 ────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="策略1选股系统")
    parser.add_argument("--date", type=str, required=True, help="信号日期 YYYY-MM-DD")
    parser.add_argument("--top-n", type=int, default=10, help="显示TOP N")
    args = parser.parse_args()

    try:
        datetime.strptime(args.date, "%Y-%m-%d")
    except ValueError:
        print(f"❌ 日期格式错误: {args.date}")
        sys.exit(1)

    preload()
    screen_strat1(args.date, top_n=args.top_n)


if __name__ == "__main__":
    main()
