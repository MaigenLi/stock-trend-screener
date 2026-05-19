#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
盘后主力行为评分系统
功能：
    1. 计算波动率压缩指标（复用 gain_turnover 的 vol_compression_scalar）
    2. 检测主力行为：
        - 缩量洗盘（价格在MA20上方，但近5日均量<近20日均量×0.7 且 振幅收缩）
        - 假跌破（今日最低<MA20×0.98，但收盘>MA20）
        - 放量回封（今日高开低走但尾盘回封涨停，放量）
        - 龙头换手（换手率>20% 且 股价不跌）
    3. 结合板块热点、RPS、涨幅等综合评分
输出：
    信号字典 + 总评分 + 各条件原因
依赖：
    gain_turnover.py（统一数据接口）
"""

import sys
from pathlib import Path

WORKSPACE = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(WORKSPACE))

import numpy as np
import pandas as pd
from stock_trend import gain_turnover as gt

# ── 可调参数 ────────────────────────────────────────────────
L1_MIN_BARS   = 65    # 最低K线数量
BOLL_WINDOW   = 20    # 布林带/振幅均线窗口
VOL_WINDOW    = 20    # 成交量均线窗口
BB_WIDTH_MIN  = 3.0   # 布林收口阈值（%）
VOL_SHRINK    = 0.7    # 缩量阈值（5日均量 < 0.7 × 20日均量 → 缩量洗盘）
AMP_SHRINK    = 0.7    # 振幅收缩阈值
VOL_RESEAL    = 2.0    # 放量回封：今日量 > 20日均量 × 此倍数
LEADER_TURNOVER = 20.0 # 龙头换手率门槛（%）
SCORE_COMPRESSION = 10
SCORE_WASHOUT     = 10
SCORE_FALSE_BREAK = 8
SCORE_RESEAL      = 15
SCORE_LEADER      = 10
SCORE_HOT_SECTOR  = 5
SCORE_RPS         = 5

# ── 工具 ────────────────────────────────────────────────────
def rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """简单移动平均，返回等长数组（前方为 nan）。"""
    n = len(arr)
    out = np.full(n, np.nan, dtype=float)
    if n < window:
        return out
    out[window - 1:] = np.convolve(arr, np.ones(window) / window, mode="valid")
    return out


def _limit_up_pct(code: str) -> float:
    """根据代码返回涨停板涨幅（10%/20%/30%）。"""
    pure = gt.normalize_symbol(code)
    if pure.startswith(("30", "301")):          # 创业板
        return 0.20
    if pure.startswith(("68", "430", "830", "87", "920")):  # 科创板/北交所
        return 0.20
    return 0.10                                   # 主板


def _true_range(high: np.ndarray, low: np.ndarray,
                close: np.ndarray) -> np.ndarray:
    """TR = max(H-L, |H-PC|, |L-PC|)，返回等长数组（首个为 nan）。"""
    n = len(high)
    tr = np.full(n, np.nan)
    if n < 2:
        return tr
    pc = close[:-1]
    tr[1:] = np.maximum(high[1:] - low[1:],
                        np.maximum(np.abs(high[1:] - pc),
                                   np.abs(low[1:] - pc)))
    return tr


# ── 主力行为检测 ────────────────────────────────────────────
def detect_washout(high: np.ndarray, low: np.ndarray,
                   close: np.ndarray, volume: np.ndarray) -> bool:
    """
    缩量洗盘：
      收盘 > MA20（趋势未坏）
      近5日均量 < 0.7 × 近20日均量（量能萎缩）
      近5日振幅 < 0.7 × 近20日振幅均值（波动收敛）
    """
    n = len(close)
    if n < 21:
        return False

    ma20 = rolling_mean(close, 20)
    ma20_last = ma20[-1]
    if np.isnan(ma20_last) or close[-1] <= ma20_last:
        return False

    vol5  = float(np.mean(volume[-5:]))   if n >= 5  else np.nan
    vol20 = float(np.mean(volume[-20:])) if n >= 20 else np.nan
    amp5  = float(np.mean((high[-5:] - low[-5:]) / close[-5:])) if n >= 5 else np.nan
    amp20 = float(np.mean((high[-20:] - low[-20:]) / close[-20:])) if n >= 20 else np.nan

    if np.isnan(vol5) or np.isnan(vol20) or np.isnan(amp5) or np.isnan(amp20):
        return False

    return (vol5 < vol20 * VOL_SHRINK) and (amp5 < amp20 * AMP_SHRINK)


def detect_false_break(high: np.ndarray, low: np.ndarray,
                       close: np.ndarray) -> bool:
    """
    假跌破：
      今日最低 < MA20 × 0.98（盘中跌破均线2%）
      收盘 > MA20（收回均线之上）
    """
    n = len(close)
    if n < 21:
        return False

    ma20 = rolling_mean(close, 20)
    ma20_last = ma20[-1]
    if np.isnan(ma20_last):
        return False

    today_low  = low[-1]
    today_close = close[-1]

    return (today_low < ma20_last * 0.98) and (today_close > ma20_last)


def detect_reseal(high: np.ndarray, low: np.ndarray,
                  open_: np.ndarray, close: np.ndarray,
                  volume: np.ndarray, code: str) -> bool:
    """
    放量回封：
      昨日收盘 × (1 + limit_up_pct) 为涨停价
      今日最高价 ≥ 涨停价 × 0.995（摸到涨停）
      今日收盘价 ≥ 涨停价 × 0.995（封住涨停）
      今日开盘价 < 涨停价 × 0.98（高开低走后再封，说明抛压被消化）
      今日成交量 > 20日均量 × VOL_RESEAL（放量确认）
    """
    n = len(close)
    if n < 21:
        return False

    prev_close = close[-2]
    limit_pct  = _limit_up_pct(code)
    limit_up   = prev_close * (1.0 + limit_pct)

    vol20_avg = float(np.mean(volume[-20:])) if n >= 20 else np.nan
    if np.isnan(vol20_avg) or vol20_avg <= 0:
        return False

    h, o, c, v = high[-1], open_[-1], close[-1], volume[-1]

    touched  = h >= limit_up * 0.995
    sealed   = c >= limit_up * 0.995
    opened_low = o < limit_up * 0.98     # 高开后低走（消化抛压）
    vol_surge = v > vol20_avg * VOL_RESEAL

    return touched and sealed and opened_low and vol_surge


def detect_leader_turnover(true_turnover: np.ndarray,
                            close: np.ndarray) -> bool:
    """
    龙头换手：
      当日真实换手率 > LEADER_TURNOVER（默认20%）
      收盘价不跌（今日收盘 ≥ 昨日收盘）
    """
    n = len(close)
    if n < 2:
        return False

    turnover_today = true_turnover[-1]
    if np.isnan(turnover_today) or turnover_today < LEADER_TURNOVER:
        return False

    return close[-1] >= close[-2]


# ── 综合评分 ────────────────────────────────────────────────
def score_stock(df: pd.DataFrame,
                code: str,
                rps: float = 0.0,
                sector_hot: bool = False,
                hot_sectors: dict = None,
                stock_sector_map: dict = None) -> dict:
    """
    盘后主力行为综合评分。

    参数：
        df              : load_qfq_history 返回的 DataFrame（当日K线已含）
        code            : 股票代码（用于判断涨停板幅度）
        rps             : RPS 值（0=不加分）
        sector_hot      : 是否在热点板块（True=直接指定，否则按 stock_sector_map 查）
        hot_sectors     : {板块名: 涨跌幅} 热板块字典（来自 get_top_sectors）
        stock_sector_map: {code: 板块名} 映射字典
    返回：
        dict，包含评分结果、各子条件布尔值、加分原因列表
    """
    if df is None or len(df) < L1_MIN_BARS:
        return {"score": 0, "reasons": ["数据不足"]}

    close   = df["close"].values.astype(float)
    high    = df["high"].values.astype(float)
    low     = df["low"].values.astype(float)
    open_   = df["open"].values.astype(float)
    volume  = df["volume"].values.astype(float)
    # 优先 true_turnover（真实换手率），fallback 到 turnover
    true_turnover = df["true_turnover"].values.astype(float) \
        if "true_turnover" in df.columns else df["turnover"].values.astype(float)

    n = len(close)

    # ── 1. 波动率压缩（复用 gain_turnover 的实现）──────────
    vc = gt.vol_compression_scalar(high, low, close, volume)
    vol_signal  = vc["squeeze_count"] >= 3   # 4项中触发≥3项视为压缩信号
    vol_details = {k: vc[k] for k in
                   ("squeeze_count", "atr_shrink", "bb_squeeze",
                    "amp_shrink", "vol_squeeze")}

    # ── 2. 主力行为检测 ─────────────────────────────────
    washout    = detect_washout(high, low, close, volume)
    false_break = detect_false_break(high, low, close)
    reseal     = detect_reseal(high, low, open_, close, volume, code)
    leader     = detect_leader_turnover(true_turnover, close)

    # ── 3. 板块热点判断 ─────────────────────────────────
    if sector_hot:
        is_hot = True
    elif hot_sectors and stock_sector_map:
        sector = stock_sector_map.get(code, "")
        is_hot = sector in hot_sectors
    else:
        is_hot = False

    # ── 4. 评分汇总 ─────────────────────────────────────
    score = 0
    reasons = []

    if vol_signal:
        score += SCORE_COMPRESSION
        reasons.append(f"波动率压缩({vc['squeeze_count']}项)+{SCORE_COMPRESSION}")

    if washout:
        score += SCORE_WASHOUT
        reasons.append(f"缩量洗盘+{SCORE_WASHOUT}")

    if false_break:
        score += SCORE_FALSE_BREAK
        reasons.append(f"假跌破MA20+{SCORE_FALSE_BREAK}")

    if reseal:
        score += SCORE_RESEAL
        reasons.append(f"放量回封涨停+{SCORE_RESEAL}")

    if leader:
        score += SCORE_LEADER
        reasons.append(f"龙头高换手+{SCORE_LEADER}")

    if rps > 0:
        score += SCORE_RPS
        reasons.append(f"RPS={rps:.0f}+{SCORE_RPS}")

    if is_hot:
        score += SCORE_HOT_SECTOR
        reasons.append(f"热点板块+{SCORE_HOT_SECTOR}")

    return {
        "score":          score,
        "reasons":        reasons,
        # 各子条件详情
        "vol_compression": vol_signal,
        "vol_details":    vol_details,
        "washout":        washout,
        "false_break":    false_break,
        "reseal":         reseal,
        "leader":         leader,
        "hot_sector":     is_hot,
        "close_today":    round(float(close[-1]), 2),
        "true_turnover_today": round(float(true_turnover[-1]), 2)
                              if not np.isnan(true_turnover[-1]) else None,
    }


# ── 全市场扫描 ──────────────────────────────────────────────
def scan_market(codes: list[str],
                names: dict[str, str],
                top_n: int = 20,
                min_score: int = 10,
                rps_dict: dict = None,
                hot_sectors: dict = None,
                stock_sector_map: dict = None) -> list[dict]:
    """
    全市场主力行为扫描（盘后用）。

    参数：
        codes             : 股票代码列表
        names             : {code: name} 名称缓存
        top_n             : 每个信号类型最多取 top_n 只
        min_score         : 最低评分门槛
        rps_dict          : {code: rps} 字典（可选）
        hot_sectors       : {sector_name: pct} 今日热点板块
        stock_sector_map  : {code: sector_name}
    返回：
        按 score 降序排列的信号列表
    """
    results = []
    for code in codes:
        df = gt.load_qfq_history(code)
        if df is None or len(df) < L1_MIN_BARS:
            continue
        rps  = rps_dict.get(code, 0.0) if rps_dict else 0.0
        res  = score_stock(df, code, rps=rps,
                            sector_hot=False,
                            hot_sectors=hot_sectors,
                            stock_sector_map=stock_sector_map)
        if res["score"] < min_score:
            continue
        results.append({
            "code":  code,
            "name":  names.get(code, names.get(gt.normalize_symbol(code), "未知")),
            **res,
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


# ── 命令行入口 ──────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="盘后主力行为评分系统")
    parser.add_argument("--code", "-c", help="单只股票代码（如 600862）")
    parser.add_argument("--scan", "-s", action="store_true", help="全市场扫描")
    parser.add_argument("--top", "-n", type=int, default=20, help="每类型最多显示 N 只")
    parser.add_argument("--min-score", "-m", type=int, default=10, help="最低评分门槛")
    parser.add_argument("--end", "-e", default=None, help="截止日期 YYYY-MM-DD")
    args = parser.parse_args()

    if args.code:
        # ── 单票分析 ──────────────────────────────────
        df = gt.load_qfq_history(args.code, end_date=args.end)
        if df is None:
            print(f"❌ 无法加载 {args.code} 的K线数据")
            sys.exit(1)

        hot_secs = gt.get_top_sectors(n=15) if args.scan else set()
        sec_map  = gt.get_stock_sector_map() if args.scan else {}

        res = score_stock(df.tail(L1_MIN_BARS), args.code,
                          rps=0, hot_sectors=hot_secs,
                          stock_sector_map=sec_map)
        name = gt.get_stock_name(args.code, gt.load_stock_names())

        print(f"\n{'='*60}")
        print(f"  {name} ({args.code})  主力行为评分")
        print(f"{'='*60}")
        print(f"  总评分   : {res['score']} 分")
        print(f"  今日收盘 : {res['close_today']}")
        print(f"  真实换手 : {res['true_turnover_today']}%")
        print()
        print(f"  各维度：")
        for key in ("vol_compression", "washout", "false_break",
                    "reseal", "leader", "hot_sector"):
            val = res.get(key)
            label = {"vol_compression": "波动率压缩",
                     "washout": "缩量洗盘",
                     "false_break": "假跌破",
                     "reseal": "放量回封",
                     "leader": "龙头换手",
                     "hot_sector": "热点板块"}.get(key, key)
            status = "✅" if val else "❌"
            print(f"    {status} {label}: {val}")

        if res.get("vol_details"):
            vd = res["vol_details"]
            print(f"\n  压缩详情（squeeze_count={vd['squeeze_count']}）：")
            for k, v in vd.items():
                print(f"    {k}: {v}")

        print(f"\n  加分原因：")
        for r in res["reasons"]:
            print(f"    + {r}")

    elif args.scan:
        # ── 全市场扫描 ─────────────────────────────
        print("📋 全市场主力行为扫描中...")
        codes = gt.get_all_stock_codes()
        names = gt.load_stock_names()
        hot_secs = gt.get_top_sectors(n=15)
        sec_map  = gt.get_stock_sector_map()

        results = scan_market(codes, names,
                              top_n=args.top,
                              min_score=args.min_score,
                              hot_sectors=hot_secs,
                              stock_sector_map=sec_map)

        print(f"\n{'='*70}")
        print(f"  全市场主力行为扫描  |  共 {len(results)} 只达标（评分≥{args.min_score}）")
        print(f"{'='*70}")
        print(f"  {'代码':<8} {'名称':<10} {'评分':<5} {'收盘':<8} {'换手%':<7} 各维度")
        print(f"  {'-'*70}")
        for r in results[:args.top]:
            flags = []
            if r["vol_compression"]: flags.append("压")
            if r["washout"]:         flags.append("洗")
            if r["false_break"]:     flags.append("假")
            if r["reseal"]:          flags.append("封")
            if r["leader"]:          flags.append("换")
            if r["hot_sector"]:      flags.append("热")
            print(f"  {r['code']:<8} {r['name']:<10} "
                  f"{r['score']:<5} {r['close_today']:<8} "
                  f"{r['true_turnover_today']:<7} {'/'.join(flags) if flags else '—'}")
            for reason in r["reasons"]:
                print(f"      {reason}")

    else:
        parser.print_help()
