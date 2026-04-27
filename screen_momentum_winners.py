#!/usr/bin/env python3
"""
动量赢家筛选器 — screen_momentum_winners.py
============================================
基于 T+1 涨幅≥5% 赢家的 853 只样本共性分析设计的选股策略。

核心发现（2025-12 至 2026-04，82 个交易日验证）：
  ⚠️ 均线多头排列（ma5>ma10>ma20>ma60）仅 20% 赢家满足
  ✅ 真正的共性：价格已在 MA5/MA20 上方 + 中期趋势确立 + 量能温和放大

筛选条件（严格版）：
  1. 价格 > MA5  且 价格 > MA20
  2. MA20 趋势向上（5日内 MA20 上涨 > 0.5%）
  3. RSI ∈ [62, 72]
  4. 换手率 ≥ 6%
  5. 近5日均量 / 近20日均量 ≥ 1.1
  6. 5日涨幅 ∈ [4%, 22%]
  7. 20日涨幅 > 15%
  8. 价格偏离 MA20 ∈ [8%, 40%]

用法：
  python screen_momentum_winners.py --date 2026-04-24
  python screen_momentum_winners.py --start 2026-04-18 --end 2026-04-24 --min-gain 3.0
  python screen_momentum_winners.py --date 2026-04-24 --output json
"""
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

WORKSPACE = Path.home() / ".openclaw" / "workspace"
QFQ_DIR   = WORKSPACE / ".cache" / "qfq_daily"

sys.path.insert(0, str(WORKSPACE / "stock_trend"))
from gain_turnover import (
    load_qfq_history,
    get_all_stock_codes,
    load_stock_names,
    get_stock_name,
    compute_rsi_scalar,
    normalize_symbol,
)

# ── 参数 ────────────────────────────────────────────────────
class Config:
    # RSI
    rsi_min = 62.0
    rsi_max = 72.0
    # 量能
    min_turnover = 6.0        # 最低换手率 %
    min_vol_ratio = 1.1        # 5日均量/20日均量
    # 涨幅
    gain5_min = 4.0           # 5日涨幅下限 %
    gain5_max = 22.0          # 5日涨幅上限 %
    gain20_min = 15.0         # 20日涨幅下限 %
    # 偏离
    dist_ma20_min = 8.0       # 价格偏离MA20下限 %
    dist_ma20_max = 40.0      # 价格偏离MA20上限 %
    # 均线趋势
    ma20_dir_min = 0.5        # MA20 5日必须涨超 %


# ── 数据缓存 ──────────────────────────────────────────────
_price_cache = {}

def preload():
    """预加载所有CSV到内存（与screen_strat1_v2一致）。"""
    global _price_cache
    _price_cache.clear()
    for f in QFQ_DIR.glob("*_qfq.csv"):
        code = normalize_symbol(f.stem.replace("_qfq", ""))
        try:
            df = pd.read_csv(f)
            df = df.sort_values("date").reset_index(drop=True)
            _price_cache[code] = df
        except:
            pass
    return len(_price_cache)

def get_df(code: str) -> pd.DataFrame | None:
    code = normalize_symbol(code)
    return _price_cache.get(code)

def get_dates(df: pd.DataFrame) -> list:
    return pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d").tolist()

def find_idx(df: pd.DataFrame, target_date: str) -> int | None:
    dates = get_dates(df)
    try:
        return dates.index(target_date)
    except ValueError:
        return None

# ── 指标计算 ───────────────────────────────────────────────
def calc_metrics(code: str, target_date: str) -> dict | None:
    """
    计算单只股票在 target_date 的全部筛选指标。
    失败返回 None。
    """
    df = get_df(code)
    if df is None:
        return None
    idx = find_idx(df, target_date)
    if idx is None or idx < 25:
        return None

    window = df.iloc[max(0, idx-80):idx+1].copy()
    close = window["close"].values
    high  = window["high"].values
    low   = window["low"].values
    vol   = window["volume"].values
    op    = window["open"].values

    i = len(close) - 1  # 最新一根

    # 均线
    ma5  = float(np.mean(close[i-4:i+1]))
    ma10 = float(np.mean(close[i-9:i+1]))
    ma20 = float(np.mean(close[i-19:i+1]))
    ma60 = float(np.mean(close[max(0,i-59):i+1])) if i >= 60 else ma20

    # MA20趋势（5日）
    if i >= 5:
        ma20_now  = float(np.mean(close[i-19:i+1]))
        ma20_5ago = float(np.mean(close[i-24:i-4])) if i >= 25 else float(np.mean(close[:i-19])) if i > 20 else None
        ma20_dir_pct = (ma20_now/ma20_5ago - 1)*100 if ma20_5ago and ma20_5ago > 0 else 0.0
    else:
        ma20_dir_pct = 0.0

    # 换手率
    if "turnover" in window.columns:
        turnover = float(window["turnover"].iloc[i])
    else:
        turnover = 0.0

    # 量比
    vol5  = float(np.mean(vol[i-4:i+1]))
    vol20 = float(np.mean(vol[i-19:i+1]))
    vol_ratio = vol5 / vol20 if vol20 > 0 else 0.0

    # RSI
    rsi = compute_rsi_scalar(close, 14)

    # 涨幅
    gain1  = (close[i]/close[i-1]-1)*100   if i>=1 and close[i-1]>0 else 0.0
    gain3  = (close[i]/close[i-3]-1)*100   if i>=3 and close[i-3]>0 else 0.0
    gain5  = (close[i]/close[i-5]-1)*100   if i>=5 and close[i-5]>0 else 0.0
    gain20 = (close[i]/close[i-20]-1)*100  if i>=20 and close[i-20]>0 else 0.0

    # 偏离MA20
    dist_ma20 = (close[i] - ma20) / ma20 * 100.0 if ma20 > 0 else 0.0

    # 阳线实体
    entity_pct = (close[i]-op[i])/op[i]*100 if op[i]>0 else 0.0

    # 波段高低（用于参考）
    recent_high = float(np.max(high[max(0,i-20):i+1]))
    recent_low  = float(np.min(low[max(0,i-20):i+1]))

    return {
        "code": code,
        "close": float(close[i]),
        "ma5": round(ma5, 2),
        "ma10": round(ma10, 2),
        "ma20": round(ma20, 2),
        "ma60": round(ma60, 2),
        "price_above_ma5":  close[i] > ma5,
        "price_above_ma20": close[i] > ma20,
        "ma20_up_pct": round(ma20_dir_pct, 2),
        "turnover": round(turnover, 2),
        "vol_ratio": round(vol_ratio, 3),
        "rsi": round(float(rsi), 1),
        "gain1": round(gain1, 2),
        "gain3": round(gain3, 2),
        "gain5": round(gain5, 2),
        "gain20": round(gain20, 2),
        "dist_ma20": round(dist_ma20, 2),
        "entity_pct": round(entity_pct, 2),
        "recent_high": round(recent_high, 2),
        "recent_low":  round(recent_low, 2),
    }


def check_filters(m: dict) -> tuple[bool, str]:
    """严格过滤。返回 (通过, 拒绝原因)。"""
    reasons = []

    if not m["price_above_ma5"]:
        return False, "价格不在MA5上方"
    if not m["price_above_ma20"]:
        return False, "价格不在MA20上方"
    if m["ma20_up_pct"] < Config.ma20_dir_min:
        return False, f"MA20趋势向下(5日{Config.ma20_dir_min}%)"
    if not (Config.rsi_min <= m["rsi"] <= Config.rsi_max):
        return False, f"RSI={m['rsi']}不在[{Config.rsi_min},{Config.rsi_max}]"
    if m["turnover"] < Config.min_turnover:
        return False, f"换手={m['turnover']}%<{Config.min_turnover}%"
    if m["vol_ratio"] < Config.min_vol_ratio:
        return False, f"量比={m['vol_ratio']}<{Config.min_vol_ratio}"
    if not (Config.gain5_min <= m["gain5"] <= Config.gain5_max):
        return False, f"5日涨={m['gain5']}%超出[{Config.gain5_min},{Config.gain5_max}]%"
    if m["gain20"] <= Config.gain20_min:
        return False, f"20日涨={m['gain20']}%<={Config.gain20_min}%"
    if not (Config.dist_ma20_min <= m["dist_ma20"] <= Config.dist_ma20_max):
        return False, f"偏离MA20={m['dist_ma20']}%超出[{Config.dist_ma20_min},{Config.dist_ma20_max}]%"

    return True, "通过"


def score_stock(m: dict) -> tuple[int, list]:
    """对通过筛选的股票评分。"""
    score = 0
    reasons = []

    # RSI 健康加分（最优区间63-69中部加更多）
    rsi = m["rsi"]
    if 64 <= rsi <= 68:
        score += 25; reasons.append(f"RSI={rsi}(黄金区间)")
    elif 62 <= rsi < 64 or 68 < rsi <= 72:
        score += 15; reasons.append(f"RSI={rsi}(次优)")
    else:
        score += 5; reasons.append(f"RSI={rsi}(边界)")

    # 换手率加分
    if m["turnover"] >= 15:
        score += 20; reasons.append(f"换手={m['turnover']}%🔥")
    elif m["turnover"] >= 10:
        score += 15; reasons.append(f"换手={m['turnover']}%较活跃")
    elif m["turnover"] >= 6:
        score += 10; reasons.append(f"换手={m['turnover']}%")

    # 量比加分
    vr = m["vol_ratio"]
    if vr >= 1.5:
        score += 15; reasons.append(f"量比={vr}(明显放大)")
    elif vr >= 1.2:
        score += 10; reasons.append(f"量比={vr}(温和放大)")
    elif vr >= 1.1:
        score += 5; reasons.append(f"量比={vr}(微幅放大)")

    # 5日涨幅加分（最优区间 8-15%）
    g5 = m["gain5"]
    if 8 <= g5 <= 15:
        score += 20; reasons.append(f"5日涨={g5}%(最优区间)")
    elif 5 <= g5 < 8:
        score += 15; reasons.append(f"5日涨={g5}%")
    elif 15 < g5 <= 22:
        score += 10; reasons.append(f"5日涨={g5}%(偏热)")

    # 20日涨幅加分
    g20 = m["gain20"]
    if g20 >= 30:
        score += 10; reasons.append(f"20日涨={g20}%🔥")
    elif g20 >= 20:
        score += 5; reasons.append(f"20日涨={g20}%")

    # 偏离MA20适中加分（最优区间 15-30%）
    dist = m["dist_ma20"]
    if 12 <= dist <= 28:
        score += 10; reasons.append(f"偏离MA20={dist}%(健康)")
    elif 8 <= dist < 12 or 28 < dist <= 40:
        score += 5; reasons.append(f"偏离MA20={dist}%")

    return score, reasons


# ── 表格输出 ───────────────────────────────────────────────
_COLS = [
    ("代码",   9, ">"), ("名称",   7, "<"),
    ("评分",   5, ">"), ("收盘",   8, ">"),
    ("RSI",    5, ">"), ("换手%",  7, ">"),
    ("量比",   5, ">"), ("5日%",  7, ">"),
    ("20日%", 7, ">"), ("偏离%",  7, ">"),
    ("MA5",   7, ">"), ("MA20",  7, ">"),
    ("MA20趋势", 9, ">"), ("阳线%", 7, ">"),
    ("拒绝原因", 20, "<"),
]

def _vw(s):
    import unicodedata
    return sum(2 if unicodedata.east_asian_width(c) in ("W","F") else 1 for c in str(s))

def _pr(s, w): return str(s) + " " * max(0, w - _vw(s))
def _pl(s, w): return " " * max(0, w - _vw(s)) + str(s)

def _row(vals):
    parts = []
    for (lbl, w, al), v in zip(_COLS, vals):
        parts.append(_pl(v, w) if al == ">" else _pr(v, w))
    return " ".join(parts)

def _header():
    return _row([l for l, _, _ in _COLS])


# ── 单日扫描 ─────────────────────────────────────────────
def screen_date(target_date: str, codes: list, names: dict,
                show_rejected: bool = False) -> list:
    """
    扫描指定日期。
    返回通过筛选的股票列表（含评分和拒绝原因）。
    """
    results = []
    rejected = []

    for code in codes:
        m = calc_metrics(code, target_date)
        if m is None:
            continue
        passed, reason = check_filters(m)
        if passed:
            score, reasons = score_stock(m)
            m["score"] = score
            m["reasons"] = reasons
            results.append(m)
        elif show_rejected:
            m["reject_reason"] = reason
            rejected.append(m)

    results.sort(key=lambda x: -x["score"])
    return results, rejected


# ── 多日扫描 ─────────────────────────────────────────────
def screen_range(start: str, end: str, min_gain: float = 5.0,
                 top_n: int = 80) -> dict:
    """
    扫描日期区间，找出 T+1 日涨幅 ≥ min_gain 的股票。
    返回 {date: [result_chars]}。
    """
    codes = get_all_stock_codes()
    df_ref = get_df("sh000001")
    dates = pd.to_datetime(df_ref["date"]).dt.strftime("%Y-%m-%d").tolist()
    dates = sorted(set(dates))
    in_range = [d for d in dates if start <= d <= end]

    print(f"  范围: {start} → {end}，共 {len(in_range)} 个交易日", flush=True)

    day_winners = {}
    for sig_date in in_range:
        results, _ = screen_date(sig_date, codes, {}, show_rejected=False)

        # 计算 T+1 涨幅
        t1_idx = dates.index(sig_date) + 1 if sig_date in dates else None
        if t1_idx is None or t1_idx >= len(dates):
            continue
        t1_date = dates[t1_idx]

        winners = []
        for m in results[:top_n]:
            code = m["code"]
            df_c = get_df(code)
            if df_c is None:
                continue
            dates_c = get_dates(df_c)
            if sig_date not in dates_c or t1_date not in dates_c:
                continue
            p0 = float(df_c[df_c["date"]==sig_date]["close"].iloc[0])
            p1 = float(df_c[df_c["date"]==t1_date]["close"].iloc[0])
            if p0 <= 0:
                continue
            t1g = (p1/p0 - 1)*100
            if t1g >= min_gain:
                m2 = dict(m)
                m2["T1_gain"] = round(t1g, 2)
                winners.append(m2)

        day_winners[sig_date] = winners
        print(f"  {sig_date}: 通过{len(results)}只, T+1≥{min_gain}%赢{len(winners)}只", flush=True)

    return day_winners


# ── 主入口 ────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="动量赢家筛选器")
    parser.add_argument("--date", default="")
    parser.add_argument("--start", default="")
    parser.add_argument("--end",   default="")
    parser.add_argument("--min-gain", type=float, default=5.0,
                        help="T+1涨幅阈值（多日模式）")
    parser.add_argument("--top-n", type=int, default=80,
                        help="每日最多取信号数")
    parser.add_argument("--show-rejected", action="store_true",
                        help="同时显示被拒绝的股票")
    parser.add_argument("--output", choices=["txt","json","both"], default="txt")
    parser.add_argument("--rs", type=float, dest="rsi_min", default=62.0,
                        help=f"RSI下限 (默认{Config.rsi_min})")
    parser.add_argument("--re", type=float, dest="rsi_max", default=72.0,
                        help=f"RSI上限 (默认{Config.rsi_max})")
    parser.add_argument("--tv", type=float, dest="turnover_min", default=6.0,
                        help=f"最低换手率%% (默认{Config.min_turnover})")
    args = parser.parse_args()

    Config.rsi_min = args.rsi_min
    Config.rsi_max = args.rsi_max
    Config.min_turnover = args.turnover_min

    t0 = time.time()
    n = preload()
    print(f"\n📂 已加载 {n} 只数据", flush=True)

    if args.date:
        # 单日模式
        codes = get_all_stock_codes()
        names = load_stock_names()
        results, rejected = screen_date(args.date, codes, names,
                                        show_rejected=args.show_rejected)
        print(f"\n{'='*100}")
        print(f"  动量赢家筛选 {args.date}  |  通过 {len(results)} 只")
        print(f"{'='*100}")
        print(_header(), flush=True)
        print("-"*100, flush=True)
        for m in results[:60]:
            name = names.get(m["code"], m["code"])[:5]
            row = _row([
                m["code"], name,
                str(m["score"]),
                f"{m['close']:.2f}",
                f"{m['rsi']:.0f}",
                f"{m['turnover']:.1f}",
                f"{m['vol_ratio']:.2f}",
                f"{m['gain5']:+.1f}",
                f"{m['gain20']:+.1f}",
                f"{m['dist_ma20']:+.1f}",
                f"{m['ma5']:.2f}",
                f"{m['ma20']:.2f}",
                f"{m['ma20_up_pct']:+.2f}%",
                f"{m['entity_pct']:+.1f}%",
                m["reasons"][0] if m["reasons"] else "",
            ])
            print(row, flush=True)
        print("="*100, flush=True)

        if args.output in ("json","both"):
            out = WORKSPACE/"stock_reports"/f"momentum_{args.date}.json"
            with open(out,"w",encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\n💾 JSON已保存: {out}", flush=True)

    elif args.start and args.end:
        # 多日T+1分析模式
        day_winners = screen_range(args.start, args.end,
                                   min_gain=args.min_gain, top_n=args.top_n)
        # 汇总统计
        all_w = []
        for wlist in day_winners.values():
            all_w.extend(wlist)
        print(f"\n{'='*70}")
        print(f"  多日汇总: {args.start} → {args.end}  |  T+1≥{args.min_gain}%赢{len(all_w)}只")
        print(f"{'='*70}")
        if all_w:
            print(f"  RSI均值={np.mean([w['rsi'] for w in all_w]):.1f}  "
                  f"换手均值={np.mean([w['turnover'] for w in all_w]):.1f}%  "
                  f"量比均值={np.mean([w['vol_ratio'] for w in all_w]):.2f}  "
                  f"5日涨={np.mean([w['gain5'] for w in all_w]):.1f}%  "
                  f"20日涨={np.mean([w['gain20'] for w in all_w]):.1f}%")
            top5 = sorted(all_w, key=lambda x: -x["score"])[:10]
            print(f"\n  综合评分Top10:")
            for w in top5:
                name = load_stock_names().get(w["code"], w["code"])[:6]
                print(f"    {w['code']} {name:<6} score={w['score']}  "
                      f"RSI={w['rsi']}  换={w['turnover']:.1f}%  "
                      f"T+1={w.get('T1_gain',0):+.1f}%  "
                      f"5日={w['gain5']:+.1f}%  20日={w['gain20']:+.1f}%")
        print(f"\n⏱ {time.time()-t0:.0f}秒", flush=True)
    else:
        print("请指定 --date 或 --start/--end")
        parser.print_help()


if __name__ == "__main__":
    main()
