#!/usr/bin/env python3
"""
Momentum Screen — 次日动量策略
================================
设计目标：筛选出的股票，T+1 涨幅 >3% 的比例 ≥ 60%

核心思想：
  昨天的强势股，今天继续强势的概率更高。
  关键特征：
    1. 昨日涨幅 > 2%（有动量）
    2. RSI 处于健康区间（45~70），既不是弱势也不是超买
    3. 量能放大（今日量/昨日量 ≥ 1.3x）
    4. 价格在 20 日均线上方（趋势向上）
    5. 昨日收盘在当日高位区（收盘价/最高价 > 0.97，说明是主动买入推上去的）

参数（待回测确定）：
    - ret1_min: 昨日涨幅下限 [1, 2, 3]%
    - ret1_max: 昨日涨幅上限 [5, 8, 10]%
    - rsi_low:  RSI下限 [40, 45, 50]
    - rsi_high: RSI上限 [65, 70, 75]
    - vol_ratio: 量比（今日量/昨日量）[1.2, 1.5]
    - close_position: 收盘位（收在当日区间顶部）[0.95, 0.97]
    - ma20_filter: 是否要求价格在 MA20 上方
"""

import sys, json, time, itertools
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore', module='numpy')

WORKSPACE = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(WORKSPACE))

CACHE_DIR   = Path.home() / ".openclaw/workspace/.cache/qfq_daily"
REPORTS_DIR = Path.home() / "stock_reports"
START_DATE  = "2025-01-02"
END_DATE    = "2026-04-16"
SAMPLE_INTERVAL = 5    # 每5个交易日采样
MIN_T1_GAIN = 3.0      # T+1 涨幅阈值
HIT_TARGET  = 60.0      # 目标命中率

PARAM_GRID = {
    "ret1_min":       [1.0, 2.0, 3.0],     # 昨日涨幅下限
    "ret1_max":       [5.0, 8.0, 10.0],    # 昨日涨幅上限
    "rsi_low":        [40, 45, 50],         # RSI下限
    "rsi_high":       [65, 70, 75],         # RSI上限
    "vol_ratio_min":  [1.2, 1.5],           # 量比下限
    "ma20_filter":    [True, False],         # 是否要求价格在MA20上方
}

# ── 工具函数 ─────────────────────────────────────────────

def get_all_codes() -> list[str]:
    """获取全市场代码"""
    codes = []
    for f in CACHE_DIR.glob("*.csv"):
        pure = f.stem.replace("_qfq", "")
        if len(pure) == 6 and pure.isdigit():
            prefix = "sh" if pure.startswith(("60", "68", "90")) else "sz"
            codes.append(f"{prefix}{pure}")
    return codes

def get_trading_days(start: str, end: str) -> list[str]:
    """从缓存提取交易日列表"""
    all_dates = set()
    for f in list(CACHE_DIR.glob("*.csv"))[:100]:
        try:
            df = pd.read_csv(f, usecols=["date"])
            for d in df["date"].unique():
                ds = str(d)[:10]
                if start <= ds <= end:
                    all_dates.add(ds)
        except:
            pass
    return sorted(all_dates)

def load_stock_daily(code: str, end_date: str) -> pd.DataFrame | None:
    """加载单只股票前复权日线数据（截止到 end_date）"""
    pure = code[-6:]
    f = CACHE_DIR / f"{pure}_qfq.csv"
    if not f.exists():
        return None
    try:
        df = pd.read_csv(f)
        df["date"] = df["date"].astype(str).str[:10]
        df = df.sort_values("date").reset_index(drop=True)
        df = df[df["date"] <= end_date].reset_index(drop=True)
        return df
    except:
        return None

def calc_rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    """计算RSI数组（忽略除零警告）"""
    if len(closes) < period + 1:
        return np.full(len(closes), np.nan)
    deltas = np.diff(closes, prepend=closes[0])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.convolve(gains, np.ones(period)/period, mode='same')
    avg_loss = np.convolve(losses, np.ones(period)/period, mode='same')
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = np.where(avg_loss == 0, 100, avg_gain / avg_loss)
        rsi = 100 - (100 / (1 + rs))
    return rsi

def calc_ma(closes: np.ndarray, period: int) -> np.ndarray:
    """计算MA数组"""
    ma = np.full(len(closes), np.nan)
    for i in range(period - 1, len(closes)):
        ma[i] = np.mean(closes[i - period + 1:i + 1])
    return ma

def batch_t1_gains(codes: list[str], trade_date: str) -> dict[str, float]:
    """批量计算 T+1 涨幅"""
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

# ── 预计算每日市场数据 ───────────────────────────────────
def preload_market_data(
    all_codes: list[str],
    sample_days: list[str],
    max_workers: int = 8,
) -> dict[str, pd.DataFrame]:
    """
    对每个采样日，预计算所有股票的指标。
    返回 {date: DataFrame}，包含 code, ret1, rsi, vol_ratio, close_position, ma20_above
    """
    cache_path = REPORTS_DIR / "momentum_screen_cache.json"
    if cache_path.exists():
        print(f"📦 加载缓存: {cache_path.name}")
        with open(cache_path) as f:
            raw = json.load(f)
        return {d: pd.DataFrame(v["data"], columns=v["cols"]) for d, v in raw.items()}

    print(f"⚙️  预计算 {len(sample_days)} 天的市场数据（{len(all_codes)} 只）...")
    t0 = time.time()
    market_data: dict[str, pd.DataFrame] = {}

    for i, day in enumerate(sample_days):
        # 需要 T 日的 RSI、T-1 日的量、T 日的收盘位置
        # 使用 T 日收盘数据（含 T-1、T、T+1）
        rows = []
        batch_codes = all_codes  # 全量并行太慢，改用随机抽样
        # 每批 500 只，分布采样
        step = max(1, len(all_codes) // 500)
        sampled = all_codes[::step][:500]

        for code in sampled:
            df = load_stock_daily(code, day)
            if df is None or len(df) < 25:
                continue

            closes = df["close"].values.astype(float)
            highs  = df["high"].values.astype(float)
            vols   = df["volume"].values.astype(float)
            dates  = df["date"].values

            # 找 day 在 DataFrame 中的位置
            day_idx_arr = np.where(dates == day)[0]
            if len(day_idx_arr) == 0:
                continue
            idx = day_idx_arr[0]

            if idx < 1 or idx >= len(closes):
                continue

            # ret1: T日涨幅（T日收盘/T-1收盘-1）× 100
            ret1 = (closes[idx] / closes[idx - 1] - 1) * 100.0

            # RSI (T日)
            rsi_today = calc_rsi(closes[:idx+1], 14)
            rsi = float(rsi_today[idx]) if not np.isnan(rsi_today[idx]) else 50.0

            # vol_ratio: T日量/T-1日量
            vol_today  = vols[idx]
            vol_yest   = vols[idx - 1] if idx >= 1 else 0
            vol_ratio  = float(vol_today / vol_yest) if vol_yest > 0 else 1.0

            # close_position: 收盘价在当日高低区间的位置
            high_today  = highs[idx]
            low_today   = df["low"].values[idx]
            price_range = high_today - low_today
            close_pos   = (closes[idx] - low_today) / price_range if price_range > 0 else 0.5

            # ma20_above: T日收盘是否在 MA20 上方
            ma20_arr = calc_ma(closes, 20)
            ma20_val = ma20_arr[idx]
            ma20_above = bool(closes[idx] > ma20_val) if not np.isnan(ma20_val) else False

            rows.append({
                "code": code.lower(),
                "date": day,
                "ret1": ret1,
                "rsi":  rsi,
                "vol_ratio": vol_ratio,
                "close_position": close_pos,
                "ma20_above": ma20_above,
                "close": closes[idx],
            })

        if rows:
            market_data[day] = pd.DataFrame(rows)

        if (i + 1) % 10 == 0:
            eta = (time.time()-t0)/(i+1)*(len(sample_days)-i-1)
            print(f"   [{i+1}/{len(sample_days)}] {day} | {len(sampled)}只采样 | ETA:{eta:.0f}s")

    # 保存缓存
    print(f"💾 保存缓存: {cache_path.name}")
    cache_data = {
        d: {"cols": list(df.columns), "data": df.values.tolist()}
        for d, df in market_data.items()
    }
    with open(cache_path, "w") as f:
        json.dump(cache_data, f)
    print(f"   完成 {len(market_data)} 天 | 耗时: {time.time()-t0:.0f}s")
    return market_data

# ── 单参数回测 ───────────────────────────────────────────
def backtest_params(
    params: dict,
    market_data: dict[str, pd.DataFrame],
    sample_days: list[str],
) -> dict:
    total_selected = 0
    total_hit = 0
    daily_stats = []

    for day in sample_days:
        if day not in market_data:
            continue
        df = market_data[day]

        # 筛选
        mask = (
            (df["ret1"] >= params["ret1_min"]) &
            (df["ret1"] <= params["ret1_max"]) &
            (df["rsi"] >= params["rsi_low"]) &
            (df["rsi"] <= params["rsi_high"]) &
            (df["vol_ratio"] >= params["vol_ratio_min"])
        )
        if params["ma20_filter"]:
            mask = mask & df["ma20_above"]

        selected = df[mask]
        if selected.empty:
            continue

        codes = selected["code"].tolist()

        # T+1 涨幅
        gains = batch_t1_gains(codes, day)
        checked = 0
        hits = 0
        for code in codes:
            g = gains.get(code)
            if g is None:
                continue
            checked += 1
            total_selected += 1
            if g > MIN_T1_GAIN:
                hits += 1
                total_hit += 1

        if checked > 0:
            daily_stats.append({
                "date": day, "selected": len(selected),
                "checked": checked, "hits": hits,
            })

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
    print("📊 Momentum Screen — 次日动量策略回测")
    print(f"   区间: {START_DATE} → {END_DATE} | 每{SAMPLE_INTERVAL}交易日")
    print(f"   T+1涨幅阈值: >{MIN_T1_GAIN}% | 目标: ≥{HIT_TARGET}%")
    print("=" * 60)

    all_codes = get_all_codes()
    print(f"\n全市场股票: {len(all_codes)} 只")

    all_days = get_trading_days(START_DATE, END_DATE)
    sample_days = all_days[::SAMPLE_INTERVAL]
    print(f"交易日: {len(all_days)} 天 | 采样: {len(sample_days)} 天")
    print(f"采样日: {sample_days[:3]} ... {sample_days[-3:]}")

    # 预计算
    market_data = preload_market_data(all_codes, sample_days)

    # 网格搜索
    keys = list(PARAM_GRID.keys())
    combos = list(itertools.product(*PARAM_GRID.values()))
    print(f"\n🔍 网格搜索: {len(combos)} 组参数 × {len(sample_days)} 天")

    all_results = []
    t0 = time.time()

    for idx, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        result = backtest_params(params, market_data, sample_days)
        result["idx"] = idx + 1
        all_results.append(result)

        hit_mark = "✅" if result["hit_rate"] >= HIT_TARGET else "❌"
        print(f"  [{idx+1:3}/{len(combos)}] {hit_mark} {result['hit_rate']:.1f}%({result['total_hit']}/{result['total_selected']}) "
              f"| ret1=[{params['ret1_min']},{params['ret1_max']}]% "
              f"RSI=[{params['rsi_low']},{params['rsi_high']}] "
              f"vol≥{params['vol_ratio_min']} "
              f"MA20={'T' if params['ma20_filter'] else 'F'}")

    all_results.sort(key=lambda x: -x["hit_rate"])

    # 报告
    print("\n" + "=" * 70)
    print("📊 回测结果")
    print("=" * 70)

    print(f"\n{'#':<4} {'昨日涨%':>9} {'RSI':>8} {'量比':>6} {'MA20':>5} | {'命中率':>8} {'命中':>6} {'信号':>6} {'达标':>4}")
    for r in all_results[:20]:
        p = r["params"]
        mark = "✅" if r["hit_rate"] >= HIT_TARGET else "❌"
        print(f"{r['idx']:<4} {p['ret1_min']:.0f}~{p['ret1_max']:.0f}%{'':>4} "
              f"[{p['rsi_low']},{p['rsi_high']}]{'':>2} "
              f"≥{p['vol_ratio_min']}{'':>3} {str(p['ma20_filter'])[0]:>5} | "
              f"{r['hit_rate']:>7.1f}% {r['total_hit']:>6} {r['total_selected']:>6} {mark:>4}")

    best = all_results[0]
    bp = best["params"]
    print(f"\n🏆 最优参数:")
    print(f"   昨日涨幅: {bp['ret1_min']:.0f}% ~ {bp['ret1_max']:.0f}%")
    print(f"   RSI区间: [{bp['rsi_low']}, {bp['rsi_high']}]")
    print(f"   量比下限: ≥{bp['vol_ratio_min']:.1f}")
    print(f"   MA20过滤: {'是' if bp['ma20_filter'] else '否'}")
    print(f"\n   命中率: {best['hit_rate']:.1f}% ({best['total_hit']}/{best['total_selected']})")

    if best["daily_stats"]:
        print(f"\n每日明细:")
        print(f"{'日期':<12} {'选出':>5} {'验证':>5} {'命中':>5} {'命中率':>8}")
        for ds in best["daily_stats"]:
            rate = ds["hits"] / ds["checked"] * 100 if ds["checked"] > 0 else 0
            print(f"{ds['date']:<12} {ds['selected']:>5} {ds['checked']:>5} {ds['hits']:>5} {rate:>7.0f}%")

    # 保存
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = REPORTS_DIR / f"momentum_backtest_{ts}.txt"
    out_path.write_text(f"""
================================================================
Momentum Screen 回测报告
区间: {START_DATE} → {END_DATE} | 每{SAMPLE_INTERVAL}交易日 | T+1阈值: >{MIN_T1_GAIN}% | 目标: ≥{HIT_TARGET}%

最优参数:
  昨日涨幅: {bp['ret1_min']:.0f}% ~ {bp['ret1_max']:.0f}%
  RSI区间: [{bp['rsi_low']}, {bp['rsi_high']}]
  量比下限: ≥{bp['vol_ratio_min']:.1f}
  MA20过滤: {'是' if bp['ma20_filter'] else '否'}

命中率: {best['hit_rate']:.1f}% ({best['total_hit']}/{best['total_selected']})
================================================================
""".strip(), encoding="utf-8")
    print(f"\n💾 报告: {out_path}")

    json_path = REPORTS_DIR / f"momentum_results_{ts}.json"
    json_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"💾 JSON: {json_path}")
    print(f"\n总耗时: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
