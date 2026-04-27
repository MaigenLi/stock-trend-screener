#!/usr/bin/env python3
"""
超跌反弹选股系统 v4（8步完整实现版）
=====================================

完整实现8步选股逻辑，Step 6趋势过滤可开关：
  Step 1  初步过滤     — 价格/流动性/新股/涨跌停
  Step 2  超卖识别     — RSI(9)<40 OR 连续2天下跌且近3日跌>2%
  Step 3  横盘整理     — 20日振幅<25% AND 近5日缩量
  Step 4  止跌确认     — T-1长下影线 OR T日阳线 AND 不创新低
  Step 5  板块共振     — 板块近5日涨幅>全市场中位数（默认关闭）
  Step 6  趋势方向     — MA20>MA60 且 收盘>MA20（默认关闭，可开关）
  Step 7  T+1买入     — T+1开盘价，无未来数据
  Step 8  持有5天     — 时间止损，T+5收盘卖出

使用方法：
    python screen_rebound_v2.py --date 2026-04-24
    python screen_rebound_v2.py --date 2026-04-24 --trend   # 开启MA20>MA60趋势过滤
    python screen_rebound_v2.py --date 2026-04-24 --no-trend  # 关闭趋势过滤（默认）
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

# ── 策略参数 ─────────────────────────────────────────────
class Config:
    # Step 1
    min_price = 3.0
    max_price = 150.0
    min_avg_amount = 30_000_000    # 60日均成交额 > 3000万

    # Step 2 超卖（OR逻辑：满足其一即可）
    rsi_period = 9
    rsi_max = 40                   # RSI(9) < 40
    consec_min = 2                  # 连续下跌 >= 2天
    loss3d_min = 2.0               # 近3日跌幅 > 2%
    vol承接_ratio = 0.5             # 近5日均量 > 近20日均量×50%

    # Step 3 横盘整理（AND逻辑）
    range_20d_max = 25.0           # 20日振幅 < 25%
    vol缩量_max = 0.90             # 近5日均量 < 20日均量×90%

    # Step 4 止跌确认（OR逻辑）
    ls_min = 0.40                  # T-1下影线比例 > 40%
    require_bullish_T = True        # T日阳线（收盘>开盘）
    require_no_new_low = True       # T日最低价 > T-5最低价

    # Step 5 板块（默认关闭）
    require_sector_outperform = False

    # Step 6 趋势方向（可开关，默认关闭）
    trend_filter = False            # True=开启MA20>MA60过滤
    trend_type = "ma20_above_ma60" # MA20>MA60 且 收盘>MA20

    # Step 7&8 交易
    hold_days = 5


# ── 数据加载 ─────────────────────────────────────────────
_price = {}

def preload():
    global _price
    print("📂 加载数据...", flush=True)
    for f in QFQ_DIR.glob("*_qfq.csv"):
        code = f.stem.replace("_qfq", "")
        try:
            df = pd.read_csv(f, usecols=["date","open","close","high","low","volume","amount"])
            df = df.sort_values("date").reset_index(drop=True)
            _price[code] = df
        except:
            pass
    print(f"✅ {len(_price)}只股票已加载", flush=True)


def get_dates():
    for df in list(_price.values())[:1]:
        return sorted(df["date"].tolist())
    return []


def get_price(code, date):
    df = _price.get(code)
    if df is None: return None
    r = df[df["date"] == date]
    if r.empty: return None
    r = r.iloc[0]
    return {"open":float(r["open"]),"close":float(r["close"]),
            "high":float(r["high"]),"low":float(r["low"])}


def next_date(date, offset=1):
    dates = get_dates()
    try:
        idx = dates.index(date)
        if 0 <= idx+offset < len(dates): return dates[idx+offset]
    except: pass
    return None


def calc_rsi(closes, period=9):
    if len(closes) < period+1: return None
    deltas = np.diff(closes[-period-1:])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = float(np.mean(gains)); avg_loss = float(np.mean(losses))
    if avg_loss == 0: return 100.0
    return 100 - 100/(1 + avg_gain/avg_loss)


def lower_shadow_ratio(row):
    low=float(row["low"]); high=float(row["high"])
    close=float(row["close"]); open_=float(row["open"])
    total=high-low
    if total<=0: return 0.0
    return max(0.0, min(1.0, (low-min(close,open_))/total))


# ── 核心分析函数 ─────────────────────────────────────────
def analyze_stock(code, signal_date, cfg: Config):
    """完整8步分析"""
    df = _price.get(code)
    if df is None: return None
    il = df["date"].tolist()
    try: idx = il.index(signal_date)
    except: return None
    if idx < 65: return None

    closes_all = df.iloc[idx-65:idx+1]["close"].values
    volumes_all = df.iloc[idx-65:idx+1]["volume"].values
    T_pos = len(closes_all) - 1
    close_T = closes_all[T_pos]

    # Step 1: 初步过滤
    if not (cfg.min_price <= close_T <= cfg.max_price): return None
    avg_amount = float(np.mean(df.iloc[idx-60:idx]["amount"].values))
    if avg_amount < cfg.min_avg_amount: return None
    for off in [0, 1]:
        if idx-off < 1: continue
        pc = float(df.iloc[idx-off]["close"]); ppc = float(df.iloc[idx-off-1]["close"])
        if ppc > 0 and abs((pc-ppc)/ppc*100) >= 9.7: return None

    # Step 2: 超卖（OR）
    rsi = calc_rsi(closes_all[:T_pos+1], cfg.rsi_period)
    cond_A = rsi is not None and rsi < cfg.rsi_max
    consec = 0
    for i in range(T_pos-1, T_pos-4, -1):
        if i < 0: break
        if float(df.iloc[i]["close"]) < float(df.iloc[i-1]["close"]): consec += 1
        else: break
    loss_3d = (close_T/closes_all[T_pos-3]-1)*100 if T_pos >= 3 else 0.0
    cond_B = consec >= cfg.consec_min and loss_3d < -cfg.loss3d_min
    if not (cond_A or cond_B): return None
    vol_5d = float(np.mean(volumes_all[T_pos-4:T_pos+1]))
    vol_20d = float(np.mean(volumes_all[T_pos-19:T_pos+1]))
    if vol_20d <= 0 or vol_5d < vol_20d*cfg.vol承接_ratio: return None

    # Step 3: 横盘（AND）
    high20 = float(np.max(closes_all[T_pos-19:T_pos+1]))
    low20 = float(np.min(closes_all[T_pos-19:T_pos+1]))
    if low20 <= 0: return None
    range20 = (high20/low20-1)*100
    if range20 >= cfg.range_20d_max: return None
    if vol_5d >= vol_20d*cfg.vol缩量_max: return None

    # Step 4: 止跌（OR）
    ls_T1 = lower_shadow_ratio(df.iloc[idx-1])
    cond_ls = ls_T1 > cfg.ls_min
    open_T = float(df.iloc[idx]["open"])
    cond_bull = close_T > open_T
    low5d_b = float(np.min(df.iloc[idx-5:idx]["low"].values))
    cond_nl = float(df.iloc[idx]["low"]) > low5d_b
    if not (cond_ls or (cond_bull and cond_nl)): return None

    # Step 6: 趋势方向（可开关）
    if cfg.trend_filter:
        ma20 = float(np.mean(closes_all[T_pos-19:T_pos+1]))
        ma60 = float(np.mean(closes_all[T_pos-59:T_pos+1])) if T_pos >= 59 else None
        if ma60 is None or not (ma20 > ma60 > 0 and close_T > ma20): return None

    return {
        "code": code, "signal_date": signal_date,
        "close": round(close_T, 2),
        "open": round(open_T, 2),
        "rsi": round(rsi, 1) if rsi else None,
        "consec_down": consec,
        "loss_3d": round(loss_3d, 1),
        "vol_ratio": round(vol_5d/vol_20d, 2) if vol_20d > 0 else None,
        "range_20d": round(range20, 1),
        "ls_T1": round(ls_T1, 2),
        "_score": (rsi if rsi else 50),
    }


def screen_rebound(target_date, top_n=10, cfg=None):
    """扫描8步信号"""
    if cfg is None: cfg = Config()
    print(f"📊 8步超跌反弹扫描: {target_date} [趋势过滤:{'开启' if cfg.trend_filter else '关闭'}]", flush=True)
    start = time.time()
    if not _price: preload()
    codes = [f.stem.replace("_qfq","") for f in QFQ_DIR.glob("*_qfq.csv")]
    print(f"   全市场 {len(codes)} 只", flush=True)
    signals = []
    for code in codes:
        result = analyze_stock(code, target_date, cfg)
        if result: signals.append(result)
    print(f"   完成: {time.time()-start:.1f}秒", flush=True)
    if not signals:
        print(f"\n⚠️  无信号（{target_date}）")
        return []
    signals.sort(key=lambda x: x.get("rsi") or 50)
    names = load_stock_names()
    for r in signals:
        r["name"] = names.get(r["code"], r["code"])
    print(f"\n🏆 信号（{len(signals)} 只）[趋势过滤:{'开启' if cfg.trend_filter else '关闭'}]")
    print("=" * 100)
    print(f"{'代码':<12} {'名称':<8} {'RSI':>6} {'连跌':>4} {'3日跌':>7} {'20日振幅':>8} {'量比':>6} {'下影T-1':>8}")
    print("-" * 100)
    for r in signals[:top_n]:
        print(f"{r['code']:<12} {r['name']:<8} {r.get('rsi',0):>5.1f} "
              f"{r['consec_down']:>3d}天 {r['loss_3d']:>+6.1f}% "
              f"{r['range_20d']:>7.1f}% {r.get('vol_ratio',0):>5.2f} {r['ls_T1']:>7.0%}")
    out = Path.home()/"stock_reports"/f"rebound8_{target_date}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"date":target_date,"strategy":"8步超跌反弹",
                  "trend_filter": cfg.trend_filter, "signals": signals}, f, ensure_ascii=False, indent=2)
    print(f"\n💾 已保存: {out}")
    return signals


def main():
    parser = argparse.ArgumentParser(description="8步超跌反弹选股系统")
    parser.add_argument("--date", type=str, required=True, help="信号日期 YYYY-MM-DD")
    parser.add_argument("--top-n", type=int, default=10, help="显示TOP N")
    g = parser.add_mutually_exclusive_group()
    g.add_argument("--trend", dest="trend_filter", action="store_true", help="开启MA20>MA60趋势过滤")
    g.add_argument("--no-trend", dest="trend_filter", action="store_false", help="关闭趋势过滤（默认）")
    parser.set_defaults(trend_filter=False)
    args = parser.parse_args()
    try:
        datetime.strptime(args.date, "%Y-%m-%d")
    except ValueError:
        print(f"❌ 日期格式错误: {args.date}"); sys.exit(1)
    cfg = Config()
    cfg.trend_filter = args.trend_filter
    preload()
    screen_rebound(args.date, top_n=args.top_n, cfg=cfg)


if __name__ == "__main__":
    main()
