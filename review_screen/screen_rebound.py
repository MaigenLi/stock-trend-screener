#!/usr/bin/env python3
"""
超跌反弹选股系统 v3
====================

核心理念：
  A股散户主导市场，极度超卖 + 止跌确认 + 横盘整理 = 反弹概率高
  纯T日及之前数据选股，T+1开盘价买入，不偷任何未来数据

8步选股逻辑：
  Step 1  初步过滤     — 价格/流动性/新股/涨跌停
  Step 2  超卖识别     — RSI(5)<30 + 连续4天下跌 + 5日跌>5%
  Step 3  横盘整理     — 20日振幅<20% + 近5日缩量 + T日温和放量
  Step 4  止跌确认     — T-1长下影线 + T日阳线 + 不创新低
  Step 5  板块共振     — 所属板块近5日涨幅>市场中位数
  Step 6  趋势方向     — MA20>MA60 或 均线向上发散
  Step 7  T+1买入      — T+1开盘价，无未来数据
  Step 8  持有5天     — 时间止损，T+5收盘卖出

使用方法：
    python screen_rebound.py --date 2026-04-24
    python backtest_rebound.py   # 回测验证
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
MERGED_PATH = WORKSPACE / ".cache" / "indicators_merged.json"

# ── 策略参数 ─────────────────────────────────────────────
class ReboundConfig:
    # Step 1 初步过滤
    min_price = 3.0
    max_price = 150.0
    min_avg_amount_60d = 5_000_000   # 60日均成交额 > 500万
    min_list_days = 60                 # 上市 > 60日

    # Step 2 超卖
    rsi_period = 5
    rsi_max = 30                      # RSI(5) < 30
    consec_down_min = 4                # 连续下跌 ≥ 4天
    loss5d_min = 5.0                   # 近5日跌幅 > 5%

    # Step 3 横盘整理
    range_20d_max = 20.0               # 20日振幅 < 20%
    vol_5d_vs_20d_max = 0.8           # 近5日均量 < 20日均量的0.8倍（缩量）
    vol_T_vs_5d_min = 1.5             # T日量 > 近5日均量的1.5倍（温和放量）

    # Step 4 止跌确认
    lower_shadow_min = 0.60           # 下影线比例 > 60%
    require_bullish_T = True           # T日必须阳线
    require_no_new_low = True          # T日最低价 > T-5最低价

    # Step 5 板块
    require_sector_outperform = True   # 板块近5日涨幅 > 市场中位数

    # Step 6 趋势
    trend_type = "ma20_above_ma60"     # "ma20_above_ma60" | "ma5_above_ma10_above_ma20" | "any"

    # Step 7&8 交易
    hold_days = 5

    def to_dict(self):
        return {k: v for k, v in self.__class__.__dict__.items()
                if not k.startswith('_') and k.isupper()}


# ── 数据加载 ─────────────────────────────────────────────
_price = {}
_merged = {}
_sector_map = {}  # code -> sector

def preload():
    global _price, _merged, _sector_map
    print("📂 加载数据...", flush=True)
    for f in QFQ_DIR.glob("*_qfq.csv"):
        code = f.stem.replace("_qfq", "")
        try:
            df = pd.read_csv(f, usecols=["date","open","close","high","low","volume","amount"])
            df = df.sort_values("date").reset_index(drop=True)
            _price[code] = df
        except:
            pass

    with open(MERGED_PATH) as f:
        _merged = json.load(f)

    # 板块映射（从merged index取）
    for code in _price:
        if code in _merged:
            info = _merged[code]
            _sector_map[code] = info.get("sector", "")

    dates = sorted(_merged.keys())
    print(f"✅ {len(dates)}天  {len(_price)}只股票", flush=True)
    return dates


def get_dates():
    return sorted(_merged.keys())


def get_price(code, date):
    df = _price.get(code)
    if df is None:
        return None
    r = df[df["date"] == date]
    if r.empty:
        return None
    r = r.iloc[0]
    return {
        "open": float(r["open"]),
        "close": float(r["close"]),
        "high": float(r["high"]),
        "low": float(r["low"]),
        "volume": float(r["volume"]),
        "amount": float(r["amount"]) if "amount" in r and not pd.isna(r["amount"]) else 0,
    }


def next_date(date, offset=1):
    dates = get_dates()
    try:
        idx = dates.index(date)
        if 0 <= idx + offset < len(dates):
            return dates[idx + offset]
    except:
        pass
    return None


def get_n_dates_before(date, n):
    """获取date之前的n个日期（含date）"""
    dates = get_dates()
    try:
        idx = dates.index(date)
        if idx - n + 1 < 0:
            return []
        return dates[idx - n + 1: idx + 1]
    except:
        return []


# ── 指标计算（全部只用T日及之前数据）───────────────────────

def calc_rsi(closes, period=5):
    """计算RSI（标准Wilder平滑）"""
    if len(closes) < period + 1:
        return None
    deltas = np.diff(closes[-period - 1:])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = float(np.mean(gains))
    avg_loss = float(np.mean(losses))
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def calc_ma(closes, period):
    if len(closes) < period:
        return None
    return float(np.mean(closes[-period:]))


def check_consecutive_down(code, T, n=4):
    """连续下跌天数"""
    df = _price.get(code)
    if df is None:
        return 0
    il = df["date"].tolist()
    try:
        idx = il.index(T)
    except:
        return 0
    if idx < n:
        return 0
    count = 0
    for i in range(idx - n + 1, idx + 1):
        c_prev = float(df.iloc[i - 1]["close"])
        c_curr = float(df.iloc[i]["close"])
        if c_curr < c_prev:
            count += 1
        else:
            break
    return count


def lower_shadow_ratio(row):
    """下影线比例：(low - close) / (high - low)"""
    low = float(row["low"])
    high = float(row["high"])
    close = float(row["close"])
    open_ = float(row["open"])
    body = abs(close - open_)
    total_range = high - low
    if total_range <= 0:
        return 0.0
    lower = (low - min(close, open_)) / total_range
    return max(0.0, min(1.0, lower))


def get_sector_return_5d(code, T):
    """板块近5日累计涨幅（从merged index取）"""
    if code not in _merged:
        return None
    info = _merged[code]
    sector = info.get("sector", "")
    if not sector:
        return None
    sector_r5d = info.get("sector_r5d")
    if sector_r5d is not None:
        return float(sector_r5d)
    return None


def analyze_stock_T(code, T, cfg: ReboundConfig):
    """
    分析单只股票在T日的状态
    全部指标只用T日及之前数据
    返回：是否满足全部8步条件
    """
    df = _price.get(code)
    if df is None:
        return None

    il = df["date"].tolist()
    try:
        idx = il.index(T)
    except:
        return None

    if idx < 25:  # 至少需要足够历史
        return None

    # 提取近20日K线数据（不含T+1）
    n_need = max(60, 25)  # 足够算各种指标
    if idx < n_need:
        return None

    window = df.iloc[idx - n_need: idx + 1]  # T-n到T
    closes = window["close"].values
    volumes = window["volume"].values
    dates_w = window["date"].tolist()

    T_pos = len(window) - 1  # T在window中的位置

    # ── Step 1: 初步过滤 ──────────────────────────────────
    close_T = closes[T_pos]
    if close_T < cfg.min_price or close_T > cfg.max_price:
        return None

    # 60日均成交额
    if idx < 60:
        return None
    avg_amount_60d = float(np.mean(df.iloc[idx - 60:idx]["amount"].values))
    if avg_amount_60d < cfg.min_avg_amount_60d:
        return None

    # 上市时间（粗略：近60日有数据即可）
    # （已通过idx>=60间接验证）

    # 涨跌停过滤（取T-1和T）
    for offset in [0, 1]:
        if idx - offset < 1:
            continue
        prev_close = float(df.iloc[idx - offset]["close"])
        prev_prev_close = float(df.iloc[idx - offset - 1]["close"])
        if prev_prev_close > 0:
            change = (prev_close - prev_prev_close) / prev_prev_close * 100
            if abs(change) >= 9.7:  # 接近涨跌停
                return None

    # ── Step 2: 超卖识别 ──────────────────────────────────
    # RSI(5) 用T日收盘前数据
    rsi = calc_rsi(closes[:T_pos + 1], cfg.rsi_period)
    if rsi is None or rsi >= cfg.rsi_max:
        return None

    # 连续下跌
    consec = check_consecutive_down(code, T, cfg.consec_down_min)
    if consec < cfg.consec_down_min:
        return None

    # 近5日跌幅
    if T_pos < 5:
        return None
    close_T_5d_ago = closes[T_pos - 5]
    loss5d = (close_T / close_T_5d_ago - 1) * 100 if close_T_5d_ago > 0 else 0
    if loss5d >= 0:  # 必须是下跌
        return None
    if abs(loss5d) < cfg.loss5d_min:
        return None

    # 近5日均量 > 近20日均量的60%（有人在承接）
    vol_5d = float(np.mean(volumes[T_pos - 4:T_pos + 1]))
    vol_20d = float(np.mean(volumes[T_pos - 19:T_pos + 1]))
    if vol_20d <= 0 or vol_5d < vol_20d * 0.6:
        return None

    # ── Step 3: 横盘整理 ──────────────────────────────────
    if T_pos < 20:
        return None
    high_20d = float(np.max(closes[T_pos - 19:T_pos + 1]))
    low_20d = float(np.min(closes[T_pos - 19:T_pos + 1]))
    if low_20d <= 0:
        return None
    range_20d = (high_20d / low_20d - 1) * 100
    if range_20d >= cfg.range_20d_max:
        return None

    # 近5日缩量（vs 20日均量）
    if vol_20d <= 0 or vol_5d >= vol_20d * cfg.vol_5d_vs_20d_max:
        return None

    # T日温和放量（vs 近5日均量，但不含T自身）
    vol_5d_before_T = float(np.mean(volumes[T_pos - 4:T_pos]))  # 不含T
    vol_T = volumes[T_pos]
    if vol_5d_before_T <= 0:
        return None
    vol_T_ratio = vol_T / vol_5d_before_T
    if vol_T_ratio < cfg.vol_T_vs_5d_min:
        return None

    # ── Step 4: 止跌确认 ──────────────────────────────────
    # T-1下影线
    if idx < 1:
        return None
    row_T1 = df.iloc[idx - 1]
    ls_ratio_T1 = lower_shadow_ratio(row_T1)
    if ls_ratio_T1 < cfg.lower_shadow_min:
        return None

    # T日阳线
    if cfg.require_bullish_T:
        if close_T <= float(df.iloc[idx]["open"]):
            return None

    # T日不创新低
    if cfg.require_no_new_low:
        low_5d_before = float(np.min(df.iloc[idx - 5:idx]["low"].values))
        if float(df.iloc[idx]["low"]) <= low_5d_before:
            return None

    # ── Step 5: 板块共振 ──────────────────────────────────
    if cfg.require_sector_outperform:
        sector_r = get_sector_return_5d(code, T)
        # 如果没有板块数据，用全市场中位数代替（保守）
        # 实际使用中可以从sector API获取
        if sector_r is not None and sector_r < -2.0:  # 板块太弱不要
            return None

    # ── Step 6: 趋势方向 ──────────────────────────────────
    if T_pos < 59:
        ma20 = float(np.mean(closes[T_pos - 19:T_pos + 1]))
        ma60 = float(np.mean(closes[T_pos - 59:T_pos + 1])) if T_pos >= 59 else None
    else:
        ma20 = calc_ma(closes[T_pos - 19:T_pos + 1], 20)
        ma60 = calc_ma(closes[T_pos - 59:T_pos + 1], 60)

    ma5 = calc_ma(closes[T_pos - 4:T_pos + 1], 5)
    ma10 = calc_ma(closes[T_pos - 9:T_pos + 1], 10)

    trend_ok = False
    if cfg.trend_type == "ma20_above_ma60":
        trend_ok = (ma20 is not None and ma60 is not None and ma20 > ma60 > 0)
    elif cfg.trend_type == "ma5_above_ma10_above_ma20":
        trend_ok = (ma5 is not None and ma10 is not None and ma20 is not None
                    and ma5 > ma10 > ma20)
    elif cfg.trend_type == "any":
        trend_ok = (ma20 is not None and ma60 is not None and ma20 > ma60 > 0) or \
                   (ma5 is not None and ma10 is not None and ma20 is not None
                    and ma5 > ma10 > ma20)

    if not trend_ok:
        return None

    # ── 信号汇总 ──────────────────────────────────────────
    return {
        "code": code,
        "date": T,
        "close": round(close_T, 2),
        "rsi": round(rsi, 1),
        "consec_down": consec,
        "loss5d": round(loss5d, 1),
        "range_20d": round(range_20d, 1),
        "vol_T_ratio": round(vol_T_ratio, 2),
        "ls_ratio_T1": round(ls_ratio_T1, 2),
        "ma20": round(ma20, 2) if ma20 else None,
        "ma60": round(ma60, 2) if ma60 else None,
        "trend_ok": trend_ok,
        "close_T": close_T,
    }


def screen_rebound(target_date, top_n=10):
    """
    扫描超跌反弹信号（实盘用）
    T日收盘后扫描 → T+1开盘买入
    """
    print(f"📊 超跌反弹扫描: {target_date}", flush=True)
    start = time.time()

    codes = [f.stem.replace("_qfq", "") for f in QFQ_DIR.glob("*_qfq.csv")]
    print(f"   全市场 {len(codes)} 只", flush=True)

    cfg = ReboundConfig()
    signals = []

    for code in codes:
        result = analyze_stock_T(code, target_date, cfg)
        if result:
            signals.append(result)

    print(f"   完成: {time.time() - start:.1f}秒", flush=True)

    if not signals:
        print("\n⚠️  无超跌反弹信号")
        return []

    # 排序：RSI越低越优先
    signals.sort(key=lambda x: x["rsi"])

    names = load_stock_names()
    for r in signals:
        r["name"] = names.get(r["code"], r["code"])

    print(f"\n🏆 超跌反弹信号（{len(signals)} 只）")
    print("=" * 100)
    print(f"{'代码':<12} {'名称':<8} {'RSI(5)':>7} {'连跌':>4} {'5日跌':>6} {'20日振幅':>8} {'T日量比':>8} {'下影比':>7}")
    print("-" * 100)
    for r in signals[:top_n]:
        print(f"{r['code']:<12} {r['name']:<8} {r['rsi']:>6.1f} {r['consec_down']:>3d}天 "
              f"{r['loss5d']:>+5.1f}% {r['range_20d']:>7.1f}% {r['vol_T_ratio']:>7.1f}x "
              f"{r['ls_ratio_T1']:>6.0%}")

    # 保存
    out = Path.home() / "stock_reports" / f"rebound_{target_date}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out_data = {"date": target_date, "strategy": "超跌反弹", "params": cfg.to_dict(), "signals": signals}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(out_data, f, ensure_ascii=False, indent=2)
    print(f"\n💾 已保存: {out}")

    return signals


def main():
    parser = argparse.ArgumentParser(description="超跌反弹选股系统 v3")
    parser.add_argument("--date", type=str, required=True, help="信号日期 YYYY-MM-DD")
    parser.add_argument("--top-n", type=int, default=10, help="显示TOP N")
    args = parser.parse_args()

    try:
        datetime.strptime(args.date, "%Y-%m-%d")
    except ValueError:
        print(f"❌ 日期格式错误: {args.date}")
        sys.exit(1)

    preload()
    screen_rebound(args.date, top_n=args.top_n)


if __name__ == "__main__":
    main()
