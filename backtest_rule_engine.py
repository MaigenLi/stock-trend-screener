#!/usr/bin/env python3
"""
规则引擎回测 | 2025-07-08 ~ 2026-04-08（9个月）
用法: python backtest_rule_engine.py
"""

import json, struct, time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── 配置 ─────────────────────────────────────────────────
TDX_DIR            = str(Path.home() / "stock_data" / "vipdoc")
STOCK_CODES_FILE   = Path.home() / "stock_code" / "results" / "stock_codes.txt"
BACKTEST_START     = pd.Timestamp("2025-07-08")
BACKTEST_END       = pd.Timestamp("2026-04-08")
TOP_N              = 10       # 每日最多持有N只
HOLDING_DAYS       = [1, 3, 5]
MAX_WORKERS        = 20

# ── 规则参数 ─────────────────────────────────────────────
MIN_PRICE          = 3.0
MAX_PRICE          = 200.0
MIN_3D_CHANGE      = 3.0
MAX_3D_CHANGE      = 30.0
MAX_10D_CHANGE     = 40.0
MIN_UP_DAYS        = 2       # 近3日中至少几天上涨
MIN_VOL_RATIO      = 1.0
MIN_SCORE          = 70.0


def load_codes() -> list:
    return [
        l.strip() for l in open(STOCK_CODES_FILE)
        if l.strip() and not l.startswith('#') and l.strip().startswith(('sh', 'sz'))
    ]


def read_tdx(code: str):
    """返回 {date: {close, volume}} dict，按date排序"""
    try:
        market = 'sh' if code.startswith('sh') else 'sz'
        num = code[2:]
        path = f"{TDX_DIR}/{market}/lday/{market}{num}.day"
        with open(path, 'rb') as f:
            raw = f.read()
        rows = {}
        for i in range(0, len(raw), 32):
            d = raw[i:i+32]
            if len(d) < 32:
                continue
            di = struct.unpack('I', d[0:4])[0]
            c  = struct.unpack('I', d[16:20])[0] / 100.0
            v  = struct.unpack('I', d[24:28])[0]
            rows[pd.Timestamp(str(di))] = {'close': c, 'volume': v}
        return code, rows
    except Exception:
        return code, None


def build_indicators(df_dict: dict) -> dict:
    """
    df_dict: {date: {'close', 'volume'}}
    返回带指标的 dict，只保留 lookback_days 内的数据
    """
    if not df_dict:
        return {}

    # 转 DataFrame
    dates = sorted(df_dict.keys())
    data = [{'date': d, 'close': df_dict[d]['close'], 'volume': df_dict[d]['volume']} for d in dates]
    df = pd.DataFrame(data).set_index('date')

    # 指标
    df['ma5']   = df['close'].rolling(5).mean()
    df['ma10']  = df['close'].rolling(10).mean()
    df['ma20']  = df['close'].rolling(20).mean()
    df['vol_ma5'] = df['volume'].rolling(5).mean()
    df['vol_ratio'] = df['volume'] / df['vol_ma5']

    df['close_3d']   = df['close'].shift(3)
    df['three_chg']  = (df['close'] - df['close_3d']) / df['close_3d'] * 100

    df['close_10d']  = df['close'].shift(10)
    df['ten_chg']    = (df['close'] - df['close_10d']) / df['close_10d'] * 100

    up = (df['close'] > df['close'].shift(1)).astype(int)
    df['up_days_3']  = up.rolling(3).sum()
    df['avg_vr_3']   = df['vol_ratio'].rolling(3).mean()

    # 粗评分
    s  = pd.Series(0.0, index=df.index)
    s += (df['close'] > df['ma5'])   * 25
    s += (df['ma5']  > df['ma10'])  * 20
    s += (df['ma10'] > df['ma20'])  * 15
    s += (df['three_chg'] >= 5)     * 15
    s += (df['avg_vr_3']  >= 1.5)   * 15
    s += (df['up_days_3'] >= 2)     * 10
    df['score'] = s

    return df


def passes(row) -> bool:
    """判断一行是否通过规则"""
    if any(pd.isna(row[k]) for k in ['ma5', 'ma10', 'ma20', 'avg_vr_3']):
        return False
    p = row['close']
    if p < MIN_PRICE or p > MAX_PRICE:
        return False
    # 均线多头：MA5 > MA10 > MA20（3链，不含MA60）
    if not (row['ma5'] > row['ma10'] > row['ma20']):
        return False
    if p <= row['ma5']:
        return False
    three = row['three_chg']
    if three < MIN_3D_CHANGE or three > MAX_3D_CHANGE:
        return False
    ten = row.get('ten_chg')
    if not pd.isna(ten) and ten > MAX_10D_CHANGE:
        return False
    if row['up_days_3'] < MIN_UP_DAYS:
        return False
    if row['avg_vr_3'] < MIN_VOL_RATIO:
        return False
    if row['score'] < MIN_SCORE:
        return False
    return True


def run():
    print("=" * 65)
    print("📊 规则引擎回测 | 2025-07-08 ~ 2026-04-08（9个月）")
    print("=" * 65)

    # 1. 加载代码
    print("\n▶️  加载股票列表...")
    codes = load_codes()
    print(f"  共 {len(codes)} 只")

    # 2. 读取Tdx数据（并行）
    print("\n▶️  读取通达信日线数据...")
    t0 = time.time()
    raw_map = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(read_tdx, c): c for c in codes}
        for future in as_completed(futures):
            code, data = future.result()
            if data:
                raw_map[code] = data
    print(f"  读取完成 {len(raw_map)} 只，耗时 {time.time()-t0:.1f}s")

    # 3. 构建指标
    print("\n▶️  计算技术指标...")
    t0 = time.time()
    ind_map = {}   # {code: DataFrame with indicators}
    for code, df_dict in raw_map.items():
        df = build_indicators(df_dict)
        if len(df) >= 60:
            ind_map[code] = df
    print(f"  有效股票: {len(ind_map)} 只，耗时 {time.time()-t0:.1f}s")

    # 4. 确定交易日（取各股票共有日期）
    print("\n▶️  确定交易日...")
    all_dates = set()
    for df in ind_map.values():
        all_dates.update(df.index.tolist())
    trading_dates = sorted(
        d for d in all_dates
        if BACKTEST_START <= d <= BACKTEST_END
    )
    print(f"  回测交易日: {len(trading_dates)} 天 "
          f"({trading_dates[0].date()} ~ {trading_dates[-1].date()})")

    # 5. 建立 date -> index 的映射（用于快速切仓）
    date_to_idx = {d: i for i, d in enumerate(trading_dates)}

    # 6. 每日筛选（用最近可交易日，不强求精确对日）
    print("\n▶️  每日筛选...")
    t0 = time.time()

    # 为每只股票建立所有可用日期的索引
    stock_avail = {}
    for code, df in ind_map.items():
        avail = sorted(df.index.tolist())
        stock_avail[code] = avail

    daily_picks = {}  # {date_idx: [code, ...]}

    for di, date in enumerate(trading_dates):
        picks = []
        for code, df in ind_map.items():
            avail = stock_avail[code]
            # 找 <= date 的最近可用日
            idx = len(avail) - 1
            while idx >= 0 and avail[idx] > date:
                idx -= 1
            if idx < 0:
                continue
            row = df.iloc[idx]
            if passes(row):
                picks.append(code)
        daily_picks[di] = picks[:TOP_N]

        if (di + 1) % 40 == 0 or di == len(trading_dates) - 1:
            elapsed = time.time() - t0
            eta = elapsed / (di + 1) * (len(trading_dates) - di - 1)
            print(f"  {di+1:>4}/{len(trading_dates)} | "
                  f"今日筛选 {len(daily_picks[di]):>3} 只 | ETA {eta:.0f}s")

    print(f"\n  总耗时 {time.time()-t0:.1f}s")

    # 7. 建立收盘价查询
    print("\n▶️  建立收盘价索引...")
    close_idx = {}  # (code, date_idx) -> close
    for code, df in ind_map.items():
        avail = stock_avail[code]
        avail_idx = {d: i for i, d in enumerate(avail)}
        for i, row in df.iterrows():
            di = avail_idx.get(i)
            if di is not None:
                close_idx[(code, di)] = row['close']

    # 8. 计算收益
    print("\n▶️  计算持有期收益...")

    results = {}  # {hold: stat_dict}
    bench_rets = defaultdict(list)  # {hold: [ret, ...]}

    for hold in HOLDING_DAYS:
        strat_rets = []
        wins, total = 0, 0

        for di, date in enumerate(trading_dates):
            codes_held = daily_picks.get(di, [])
            sell_di = di + hold
            if sell_di >= len(trading_dates):
                continue

            for code in codes_held:
                bp = close_idx.get((code, di))
                sp = close_idx.get((code, sell_di))
                if bp and sp and bp > 0:
                    ret = (sp - bp) / bp * 100
                    strat_rets.append(ret)
                    if ret > 0:
                        wins += 1
                    total += 1

            # 基准（全市场等权）
            for code in ind_map:
                bp = close_idx.get((code, di))
                sp = close_idx.get((code, sell_di))
                if bp and sp and bp > 0:
                    bench_rets[hold].append((sp - bp) / bp * 100)

        if strat_rets:
            results[hold] = {
                'mean_return':   np.mean(strat_rets),
                'median_return': np.median(strat_rets),
                'std_return':    np.std(strat_rets),
                'win_rate':      wins / total * 100,
                'total_return':  (np.prod([1 + r/100 for r in strat_rets]) - 1) * 100,
                'n_trades':      total,
            }
        else:
            results[hold] = None

    # 9. 打印报告
    print("\n" + "=" * 65)
    print("📊 回测结果")
    print("=" * 65)
    print(f"\n回测区间: {BACKTEST_START.date()} ~ {BACKTEST_END.date()}  |  交易日: {len(trading_dates)}天")
    print(f"规则: MA5>MA10>MA20 + 价格>MA5 + 三天涨幅3~30% + 近3日涨≥2天 + 量比≥1.0 + 粗评分≥70")

    hdr = f"{'指标':<24}" + "".join(f"{'持有'+str(h)+'天':>14}" for h in HOLDING_DAYS)
    print(f"\n{'─'*65}")
    print(hdr)
    print(f"{'─'*65}")

    for label, key in [
        ("策略平均收益率(%)",    "mean_return"),
        ("中位数收益率(%)",      "median_return"),
        ("标准差(%)",            "std_return"),
        ("胜率(%)",              "win_rate"),
        ("复合总收益率(%)",      "total_return"),
        ("交易次数",             "n_trades"),
    ]:
        vals = []
        for h in HOLDING_DAYS:
            v = results.get(h)
            vals.append(f"{v[key]:>14.2f}" if v else f"{'N/A':>14}")
        print(f"{label:<24}" + "".join(vals))

    print(f"{'─'*65}")
    bvals = []
    for h in HOLDING_DAYS:
        r = bench_rets.get(h, [])
        bvals.append(f"{np.mean(r):>14.2f}" if r else f"{'N/A':>14}")
    print(f"{'基准平均收益(%)':<24}" + "".join(bvals))

    print(f"{'─'*65}")
    avals = []
    for h in HOLDING_DAYS:
        s = results.get(h)
        b = np.mean(bench_rets.get(h, []))
        if s and b:
            avals.append(f"{s['mean_return']-b:>+14.2f}")
        else:
            avals.append(f"{'N/A':>14}")
    print(f"{'超额收益(%)':<24}" + "".join(avals))

    # 统计
    total_picks = sum(len(v) for v in daily_picks.values())
    days_picked = sum(1 for v in daily_picks.values() if v)
    avg_picks   = total_picks / len(daily_picks) if daily_picks else 0
    print(f"\n每日平均筛选: {avg_picks:.1f} 只  |  有选股天数: {days_picked}/{len(daily_picks)}")

    # 10. 保存
    out_dir = Path.home() / ".openclaw" / "workspace" / "stock_trend" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    save = {
        "backtest_period": [str(BACKTEST_START.date()), str(BACKTEST_END.date())],
        "rules": {
            "MA_filter": "MA5 > MA10 > MA20 + price > MA5",
            "min_3day_change": MIN_3D_CHANGE,
            "max_3day_change": MAX_3D_CHANGE,
            "max_10day_change": MAX_10D_CHANGE,
            "min_up_days_3": MIN_UP_DAYS,
            "min_vol_ratio": MIN_VOL_RATIO,
            "min_score": MIN_SCORE,
        },
        "trading_days": len(trading_dates),
        "avg_picks_per_day": round(avg_picks, 2),
        "days_with_picks": days_picked,
        "results_by_holding": {str(k): v for k, v in results.items() if v},
        "benchmark_by_holding": {str(k): round(float(np.mean(v)), 4) for k, v in bench_rets.items() if v},
        "generated_at": datetime.now().isoformat(),
    }
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(save, f, ensure_ascii=False, indent=2)
    print(f"\n💾 结果已保存: {out_file}")
    print("=" * 65)

    return results, bench_rets


if __name__ == "__main__":
    run()
