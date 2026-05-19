#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_force_scan.py — 盘后主力行为评分系统
============================================================
基于 gain_turnover.py 统一数据接口，对全市场A股进行主力行为扫描。

功能：
  1. 波动率压缩检测（ATR收缩 + 布林收口 + 振幅收缩 + 缩量，四合一）
  2. 主力行为识别：
     · 缩量洗盘（价格在MA20上方 + 近5日量/波幅同时收缩）
     · 假跌破（盘中跌破MA20×0.98后收回）
     · 放量回封（高开低走再封涨停 + 放量）
     · 龙头换手（高换手率 + 股价不跌）
  3. 连续N天信号叠加加分
  4. RPS 评分 + 板块热点加权
  5. 全市场扫描，输出 TXT / JSON / CSV

输出文件（与 screen_double.py 同目录结构）：
  force_scan_{date}.txt   — 人读表格
  force_scan_{date}.jsonl  — 每行一个JSON（code/name/...）
  force_scan_{date}.csv    — CSV排序表
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, date as date_type, timedelta
from pathlib import Path

import numpy as np

WORKSPACE = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(WORKSPACE))

from stock_trend import gain_turnover as gt

# ══════════════════════════════════════════════════════════════
# 常量
# ══════════════════════════════════════════════════════════════
OUTPUT_DIR = WORKSPACE / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 检测参数 ────────────────────────────────────────────────
L1_MIN_BARS      = 65    # 最低K线数量
CONT_DAYS        = 2      # 连续N天信号叠加
BB_WIDTH_MIN     = 3.0   # 布林收口阈值（%）
VOL_SHRINK       = 0.7    # 洗盘：5日均量 < 20日均量 × 此值
AMP_SHRINK       = 0.7    # 洗盘：5日振幅均值 < 20日振幅均值 × 此值
VOL_RESEAL       = 2.0    # 回封：今日量 > 20日均量 × 此倍数
LEADER_TURNOVER  = 20.0   # 换手：真实换手率 > 此值（%）触发龙头信号

# ── 评分权重 ────────────────────────────────────────────────
SCORE_COMPRESSION = 10     # 波动率压缩（每连续1天额外+3）
SCORE_WASHOUT     = 10
SCORE_FALSE_BREAK= 8
SCORE_RESEAL      = 15
SCORE_LEADER      = 10
SCORE_RPS         = 10
SCORE_HOT_SECTOR  = 5

# ══════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════
def rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    n = len(arr)
    out = np.full(n, np.nan, dtype=float)
    if n < window:
        return out
    out[window - 1:] = np.convolve(arr, np.ones(window) / window, mode="valid")
    return out


def _true_range(high: np.ndarray, low: np.ndarray,
                close: np.ndarray) -> np.ndarray:
    """TR = max(H-L, |H-PC|, |L-PC|)，等长数组（首个为 nan）。"""
    n = len(high)
    tr = np.full(n, np.nan)
    if n < 2:
        return tr
    pc = close[:-1]
    tr[1:] = np.maximum(high[1:] - low[1:],
                        np.maximum(np.abs(high[1:] - pc),
                                   np.abs(low[1:] - pc)))
    return tr


def _limit_up_pct(code: str) -> float:
    """根据代码返回涨停板幅度（10%/20%/30%）。"""
    pure = gt.normalize_symbol(code)
    if pure.startswith(("30", "301")):
        return 0.20
    if pure.startswith(("68", "430", "830", "87", "920")):
        return 0.20
    return 0.10


# ══════════════════════════════════════════════════════════════
# 主力行为检测（逐日，返回 bool 数组）
# ══════════════════════════════════════════════════════════════
def _washout_array(close: np.ndarray, high: np.ndarray,
                   low: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """返回每日是否缩量洗盘信号（布尔数组）。"""
    n = len(close)
    out = np.zeros(n, dtype=bool)
    if n < 21:
        return out

    ma20 = rolling_mean(close, 20)
    vol5 = rolling_mean(volume, 5)
    vol20 = rolling_mean(volume, 20)
    amp5  = rolling_mean((high - low) / np.where(close > 0, close, 1.0) * 100, 5)
    amp20 = rolling_mean((high - low) / np.where(close > 0, close, 1.0) * 100, 20)

    for i in range(20, n):
        if np.isnan(ma20[i]) or np.isnan(vol5[i]) or np.isnan(vol20[i]):
            continue
        if np.isnan(amp5[i]) or np.isnan(amp20[i]):
            continue
        # 价格在MA20上方
        if close[i] <= ma20[i]:
            continue
        out[i] = (vol5[i] < vol20[i] * VOL_SHRINK) and (amp5[i] < amp20[i] * AMP_SHRINK)

    return out


def _false_break_array(close: np.ndarray, low: np.ndarray) -> np.ndarray:
    """返回每日是否假跌破信号（盘中跌破MA20×0.98后收上）。"""
    n = len(close)
    out = np.zeros(n, dtype=bool)
    if n < 21:
        return out

    ma20 = rolling_mean(close, 20)
    for i in range(21, n):
        if np.isnan(ma20[i]):
            continue
        # 盘中跌破（最低 < MA20×0.98），收盘站稳（> MA20）
        out[i] = (low[i] < ma20[i] * 0.98) and (close[i] > ma20[i])

    return out


def _reseal_array(close: np.ndarray, high: np.ndarray,
                  low: np.ndarray, open_: np.ndarray,
                  volume: np.ndarray, code: str) -> np.ndarray:
    """返回每日是否放量回封信号。"""
    n = len(close)
    out = np.zeros(n, dtype=bool)
    if n < 21:
        return out

    limit_pct = _limit_up_pct(code)
    vol20 = rolling_mean(volume, 20)

    for i in range(21, n):
        prev_close = close[i - 1]
        limit_up = prev_close * (1.0 + limit_pct)

        if np.isnan(vol20[i]) or vol20[i] <= 0:
            continue

        v = volume[i]; h = high[i]; o = open_[i]; c = close[i]
        touched   = h >= limit_up * 0.995
        sealed    = c >= limit_up * 0.995
        opened_low = o < limit_up * 0.98    # 高开后低走（消化抛压）
        surge    = v > vol20[i] * VOL_RESEAL

        out[i] = touched and sealed and opened_low and surge

    return out


def _leader_array(true_turnover: np.ndarray, close: np.ndarray) -> np.ndarray:
    """返回每日是否龙头高换手信号。"""
    n = len(close)
    out = np.zeros(n, dtype=bool)
    if n < 2:
        return out

    for i in range(1, n):
        tt = true_turnover[i]
        if np.isnan(tt) or tt < LEADER_TURNOVER:
            continue
        out[i] = close[i] >= close[i - 1]   # 股价不跌

    return out


# ══════════════════════════════════════════════════════════════
# RPS 计算
# ══════════════════════════════════════════════════════════════
def compute_rps(close: np.ndarray, period: int = 20) -> float:
    """区间涨幅作为简化RPS（0~100）。"""
    n = len(close)
    if n <= period:
        return 50.0
    ret = (close[-1] / close[-period - 1] - 1) * 100.0
    return float(np.clip(ret, 0.0, 100.0))


# ══════════════════════════════════════════════════════════════
# 单票评分
# ══════════════════════════════════════════════════════════════
def score_stock(df: gt.PreparedData | pd.DataFrame,
               code: str,
               rps: float = 0.0,
               hot_sectors: dict = None,
               stock_sector_map: dict = None) -> dict | None:
    """
    盘后主力行为综合评分。

    参数：
        df               : PreparedData 或 DataFrame（包含至少 L1_MIN_BARS 根K线）
        code             : 股票代码（用于涨停幅度判断）
        rps              : 区间RPS（0=不加分）
        hot_sectors      : {sector_name: pct} 热板块字典
        stock_sector_map : {code: sector_name} 映射
    返回：
        信号字典（score/reasons/各维度详情），不足数据返回 None
    """
    # ── 提取数组 ─────────────────────────────────────────
    if isinstance(df, gt.PreparedData):
        close   = df.close
        high    = df.high
        low     = df.low
        open_   = df.open_
        volume  = df.volume
        true_turnover = df.true_turnover
        n       = len(close)
    else:
        close   = df["close"].values.astype(float)
        high    = df["high"].values.astype(float)
        low     = df["low"].values.astype(float)
        open_   = df["open"].values.astype(float)
        volume  = df["volume"].values.astype(float)
        true_turnover = df["true_turnover"].values.astype(float) \
            if "true_turnover" in df.columns else df["turnover"].values.astype(float)
        n       = len(close)

    if n < L1_MIN_BARS:
        return None

    # ── 各维度信号数组 ─────────────────────────────────
    washout_arr    = _washout_array(close, high, low, volume)
    false_break_arr = _false_break_array(close, low)
    reseal_arr     = _reseal_array(close, high, low, open_, volume, code)
    leader_arr     = _leader_array(true_turnover, close)

    # ── 波动率压缩历史数组 ────────────────────────────────
    # 调用 gain_turnover 低层函数，拿到完整历史数组重建 squeeze 序列
    _, atr_diff_arr, bb_width_arr, _, amplitude_arr, \
        amplitude_ma_arr, _, volume_ratio_arr, _ = \
        gt.compute_vol_compression(high, low, close, volume)

    # squeeze_arr: 每日是否触发压缩（4项中>=3项触发）
    squeeze_arr = np.zeros(n, dtype=bool)
    for i in range(n):
        cnt = 0
        if not np.isnan(atr_diff_arr[i])     and atr_diff_arr[i] < 0:     cnt += 1
        if not np.isnan(bb_width_arr[i])     and bb_width_arr[i] < BB_WIDTH_MIN: cnt += 1
        if not np.isnan(amplitude_arr[i])     and not np.isnan(amplitude_ma_arr[i]) \
           and amplitude_arr[i] < amplitude_ma_arr[i]:                         cnt += 1
        if not np.isnan(volume_ratio_arr[i])  and volume_ratio_arr[i] < VOL_SHRINK: cnt += 1
        squeeze_arr[i] = cnt >= 3

    # ── 连续N天信号判断 ─────────────────────────────────
    def _consecutive(arr: np.ndarray, days: int) -> bool:
        """arr末尾是否有连续days天全部为True。"""
        if len(arr) < days:
            return False
        return bool(np.all(arr[-days:]))

    def _consecutive_sum(arr: np.ndarray, days: int) -> int:
        """arr末尾连续days天中有多少天为True。"""
        if len(arr) < days:
            return int(np.sum(arr[-days:])) if len(arr) >= 1 else 0
        return int(np.sum(arr[-days:]))

    vol_signal = _consecutive(squeeze_arr, CONT_DAYS)
    cont_vol_days = _consecutive_sum(squeeze_arr, CONT_DAYS)

    # ── 最新日信号 ─────────────────────────────────────
    today_washout    = bool(washout_arr[-1])
    today_false_break = bool(false_break_arr[-1])
    today_reseal     = bool(reseal_arr[-1])
    today_leader     = bool(leader_arr[-1])
    cont_washout_days = _consecutive_sum(washout_arr, CONT_DAYS)

    # ── 今日压缩详情（用于显示）────────────────────────
    today_squeeze_cnt = int(np.sum(squeeze_arr[-3:])) if n >= 3 else 0
    vc = gt.vol_compression_scalar(high, low, close, volume)

    # ── 板块热点 ───────────────────────────────────────
    is_hot = False
    if hot_sectors and stock_sector_map:
        sector = stock_sector_map.get(code, "")
        is_hot = sector in hot_sectors

    # ── 评分汇总 ───────────────────────────────────────
    score = 0
    reasons = []

    if vol_signal:
        # 连续CONT_DAYS天压缩 → 满分；否则按触发天数部分加分
        extra = max(0, (cont_vol_days - CONT_DAYS)) * 3
        bonus = SCORE_COMPRESSION + extra
        bonus = min(bonus, SCORE_COMPRESSION + 6)
        score += bonus
        reasons.append(f"波动率压缩(连{CONT_DAYS}天{cont_vol_days}天触发)+{bonus:.0f}")
    else:
        reasons.append(f"波动率未连续压缩(近{CONT_DAYS}天仅{cont_vol_days}天)")

    if today_washout:
        score += SCORE_WASHOUT
        reasons.append(f"缩量洗盘+{SCORE_WASHOUT}")

    if today_false_break:
        score += SCORE_FALSE_BREAK
        reasons.append(f"假跌破MA20+{SCORE_FALSE_BREAK}")

    if today_reseal:
        score += SCORE_RESEAL
        reasons.append(f"放量回封涨停+{SCORE_RESEAL}")

    if today_leader:
        score += SCORE_LEADER
        reasons.append(f"龙头高换手+{SCORE_LEADER}")

    if rps > 0:
        score += SCORE_RPS
        reasons.append(f"RPS={rps:.0f}+{SCORE_RPS}")

    if is_hot:
        score += SCORE_HOT_SECTOR
        reasons.append(f"热点板块+{SCORE_HOT_SECTOR}")

    return {
        "score":            score,
        "reasons":         reasons,
        # 各维度详情
        "vol_compression":  vol_signal,
        "vol_days":         cont_vol_days,
        "today_squeeze_cnt": today_squeeze_cnt,
        "atr_shrink":       vc["atr_shrink"],
        "bb_squeeze":       vc["bb_squeeze"],
        "amp_shrink":       vc["amp_shrink"],
        "vol_squeeze":      vc["vol_squeeze"],
        "washout":          today_washout,
        "washout_days":     cont_washout_days,
        "false_break":      today_false_break,
        "reseal":           today_reseal,
        "leader":           today_leader,
        "hot_sector":       is_hot,
        "rps":              round(rps, 1),
        "close_today":      round(float(close[-1]), 2),
        "true_turnover":   round(float(true_turnover[-1]), 2)
                            if not np.isnan(true_turnover[-1]) else None,
    }


# ══════════════════════════════════════════════════════════════
# 全市场扫描
# ══════════════════════════════════════════════════════════════
def scan_market(top_n: int = 100,
               min_score: int = 15,
               end_date: str = None) -> list[dict]:
    """全市场主力行为扫描。"""
    print("📋 加载全市场数据...", flush=True)
    codes     = gt.get_all_stock_codes()
    names     = gt.load_stock_names()
    hot_secs  = gt.get_top_sectors(n=15)
    sec_map   = gt.get_stock_sector_map()

    print(f"  股票数量: {len(codes)}  热点板块: {len(hot_secs)}个  板块映射: {len(sec_map)}条",
          flush=True)

    results = []
    done   = 0
    t0     = datetime.now()

    for code in codes:
        done += 1
        if done % 500 == 0:
            elapsed = (datetime.now() - t0).total_seconds()
            eta = elapsed / done * (len(codes) - done)
            print(f"  进度: {done}/{len(codes)}  ({done/len(codes)*100:.0f}%) "
                  f"ETA={eta:.0f}s  已达标={len(results)}", end="\r", flush=True)

        try:
            df = gt.load_qfq_history(code, end_date=end_date)
            if df is None or len(df) < L1_MIN_BARS:
                continue

            rps = compute_rps(df["close"].values.astype(float))
            res = score_stock(df, code, rps=rps,
                             hot_sectors=hot_secs,
                             stock_sector_map=sec_map)
            if res is None or res["score"] < min_score:
                continue

            name = names.get(code,
                           names.get(gt.normalize_symbol(code), "未知"))
            results.append({"code": code, "name": name, **res})

        except Exception:
            continue

    elapsed = (datetime.now() - t0).total_seconds()
    print(f"\n✅ 扫描完成: {done}只  达标={len(results)}只  耗时={elapsed:.1f}s",
          flush=True)
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


# ══════════════════════════════════════════════════════════════
# 输出
# ══════════════════════════════════════════════════════════════
W = 120

def _pad(s: str, w: int, align: str = "<") -> str:
    """按显示宽度补空格（支持 CJK 2 宽度）。"""
    def _cjk(s2: str) -> int:
        return sum(2 if ord(c) > 0x1100 else 1 for c in s2)
    cw = _cjk(s)
    pad = w - cw
    if pad <= 0:
        return s
    return s + " " * pad if align == "<" else " " * pad + s


def _flag(v) -> str:
    return "✅" if v else "  "

def _fmt_v(v, decimals: int = 2) -> str:
    if v is None:
        return "N/A"
    return f"{v:.{decimals}f}"


def render_txt(results: list[dict], target_date: str,
               total: int, elapsed: float, min_score: int = 15) -> list[str]:
    """生成 TXT 多行文字（供显示+保存）。"""
    lines = []
    sep = "─" * W

    lines.append(sep)
    header = (
        f"  main_force_scan.py  主力行为评分系统"
        f"  日期={target_date}  达标={len(results)}只  耗时={elapsed:.1f}s"
    )
    lines.append(header)
    lines.append(sep)

    # 表头
    COLS = [
        ("代码",     8),
        ("名称",     9),
        ("评分",     5),
        ("收盘",     7),
        ("换手%",    7),
        ("RPS",      5),
        ("压缩",     4),
        ("洗盘",     4),
        ("假跌",     4),
        ("回封",     4),
        ("换手",     4),
        ("热点",     4),
        ("原因",    40),
    ]
    heads = "  ".join(_pad(t, w) for t, w in COLS)
    lines.append(heads)
    lines.append(sep)

    for r in results:
        name = r["name"][:8]
        flags = "".join([
            _flag(r["vol_compression"]),
            _flag(r["washout"]),
            _flag(r["false_break"]),
            _flag(r["reseal"]),
            _flag(r["leader"]),
            _flag(r["hot_sector"]),
        ])
        reason_str = "  ".join(r["reasons"][:3])
        row = (
            f"  {_pad(r['code'], 8)} "
            f"{_pad(name, 9)} "
            f"{_pad(str(r['score']), 5)} "
            f"{_pad(_fmt_v(r['close_today']), 7)} "
            f"{_pad(_fmt_v(r['true_turnover']) + '%', 7)} "
            f"{_pad(_fmt_v(r['rps'], 0), 5)} "
            f"{flags}  "
            f"{reason_str}"
        )
        lines.append(row)

    lines.append(sep)
    lines.append(f"  总计: {total} 只  |  达标: {len(results)} 只  "
                 f"|  最小评分门槛: {min_score}  |  Ctrl+C 退出")
    return lines


def save_results(results: list[dict], target_date: str,
                elapsed: float) -> None:
    """保存 TXT + JSONL + CSV。"""
    # TXT
    txt_path = OUTPUT_DIR / f"force_scan_{target_date}.txt"
    lines = render_txt(results, target_date, len(results), elapsed)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"💾 TXT 已保存: {txt_path}", flush=True)

    # JSONL
    json_path = OUTPUT_DIR / f"force_scan_{target_date}.jsonl"
    with open(json_path, "w", encoding="utf-8") as f:
        for r in results:
            obj = {
                "code":       r["code"],
                "name":       r["name"],
                "score":      r["score"],
                "close":      r["close_today"],
                "turnover":   r["true_turnover"],
                "rps":        r["rps"],
                "vol_compression": r["vol_compression"],
                "washout":    r["washout"],
                "false_break": r["false_break"],
                "reseal":     r["reseal"],
                "leader":     r["leader"],
                "hot_sector": r["hot_sector"],
                "reasons":    r["reasons"],
                "signal_date": target_date,
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"💾 JSONL 已保存: {json_path}", flush=True)

    # CSV
    csv_path = OUTPUT_DIR / f"force_scan_{target_date}.csv"
    import csv as csv_mod
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv_mod.writer(f)
        w.writerow(["代码", "名称", "评分", "收盘", "换手%", "RPS",
                   "波动率压缩", "缩量洗盘", "假跌破", "放量回封",
                   "龙头换手", "热点板块", "原因", "信号日期"])
        for r in results:
            w.writerow([
                r["code"], r["name"], r["score"],
                r["close_today"], r["true_turnover"], r["rps"],
                int(r["vol_compression"]), int(r["washout"]),
                int(r["false_break"]), int(r["reseal"]),
                int(r["leader"]), int(r["hot_sector"]),
                "|".join(r["reasons"]), target_date,
            ])
    print(f"💾 CSV 已保存: {csv_path}", flush=True)


# ══════════════════════════════════════════════════════════════
# 打印逻辑说明
# ══════════════════════════════════════════════════════════════
def print_logic():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║              main_force_scan.py  主力行为评分系统                   ║
╚══════════════════════════════════════════════════════════════════════╝

【第一层：波动率压缩检测】
  ATR 收缩    → ATR 1日差分 < 0
  布林收口    → 布林带宽 < 3.0%
  振幅收缩    → 当日振幅 < 20日振幅均线
  缩量        → 量比（当日量/20日均量）< 0.7
  · 4项中触发 ≥ 3 项视为当日压缩
  · 必须连续2天触发 → 才能加波动率压缩分（+10分）
  · 额外连续天数每天+3分（最多+16分）

【第二层：主力行为识别】
  缩量洗盘    → 价格>MA20 且 5日均量<0.7×20日均量 且 5日振幅<0.7×20日振幅
  假跌破      → 盘中最低<MA20×0.98，收盘站稳MA20
  放量回封    → 高开低走消化抛压后回封涨停（2.0倍量确认）
  龙头换手    → 真实换手率>20.0% 且 收盘不跌

【评分权重】
  波动率压缩  +10分    缩量洗盘    +10分
  假跌破      +8分     放量回封    +15分
  龙头换手    +10分    RPS         +10分
  热点板块    +5分

【输出文件】
  force_scan_{date}.txt      — 人读表格
  force_scan_{date}.jsonl   — 每行一个JSON
  force_scan_{date}.csv     — CSV排序表
""")


# ══════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="main_force_scan 盘后主力行为评分系统")
    parser.add_argument("--date", "-d", default=None,
                        help="截止日期 YYYY-MM-DD（默认昨天）")
    parser.add_argument("--top-n", "-n", type=int, default=100,
                        help="最多输出前N只（默认100）")
    parser.add_argument("--min-score", "-m", type=int, default=15,
                        help="最低评分门槛（默认15）")
    parser.add_argument("--code", "-c", default=None,
                        help="单只股票代码（不走全市场）")
    parser.add_argument("--logic", "-l", action="store_true",
                        help="显示评分逻辑说明")

    if "--help" in sys.argv or "-h" in sys.argv:
        print_logic()
        parser.print_help()
        print()
        sys.exit(0)

    args = parser.parse_args()

    if args.logic:
        print_logic()
        return

    # ── 确定日期 ────────────────────────────────────────
    if args.date:
        target_date = args.date
    else:
        today = datetime.now().date()
        # 跳过周末
        d = today - timedelta(days=1)
        while d.weekday() >= 5:
            d -= timedelta(days=1)
        target_date = d.strftime("%Y-%m-%d")

    # ── 单票模式 ─────────────────────────────────────────
    if args.code:
        df = gt.load_qfq_history(args.code, end_date=target_date)
        if df is None or len(df) < L1_MIN_BARS:
            print(f"❌ {args.code} 数据不足（需要≥{L1_MIN_BARS}根K线）", flush=True)
            sys.exit(1)

        hot_secs = gt.get_top_sectors(n=15)
        sec_map  = gt.get_stock_sector_map()
        rps      = compute_rps(df["close"].values.astype(float))
        res      = score_stock(df, args.code, rps=rps,
                              hot_sectors=hot_secs,
                              stock_sector_map=sec_map)
        name = gt.get_stock_name(args.code, gt.load_stock_names())

        if res is None:
            print(f"❌ {args.code} 评分失败", flush=True)
            sys.exit(1)

        print(f"\n{'='*80}", flush=True)
        print(f"  {name} ({args.code})  主力行为评分  |  日期={target_date}", flush=True)
        print(f"{'='*80}", flush=True)
        print(f"  总评分   : {res['score']} 分", flush=True)
        print(f"  今日收盘 : {res['close_today']}", flush=True)
        print(f"  真实换手 : {res['true_turnover']}%", flush=True)
        print(f"  RPS({20}日): {res['rps']}", flush=True)
        print()
        print(f"  {'维度':<14} {'状态':<6}  {'说明'}", flush=True)
        print(f"  {'─'*70}", flush=True)

        details = [
            ("波动率压缩", res["vol_compression"],
             f"连3天{res['vol_days']}天触发 "
             f"(ATR收缩={res['atr_shrink']} 布林收口={res['bb_squeeze']} "
             f"振幅收缩={res['amp_shrink']} 缩量={res['vol_squeeze']})"),
            ("缩量洗盘",   res["washout"],     "价格>MA20 + 近5日量/波幅双收缩"),
            ("假跌破",     res["false_break"],  "盘中<MA20×0.98收盘站稳MA20"),
            ("放量回封",   res["reseal"],      f"摸涨停+封涨停+高开低走+{VOL_RESEAL}倍放量"),
            ("龙头换手",   res["leader"],       f"真实换手率>{LEADER_TURNOVER}%+收盘不跌"),
            ("热点板块",   res["hot_sector"],   "所属板块在今日涨幅前15名"),
        ]
        for label, val, desc in details:
            flag = "✅" if val else "❌"
            print(f"  {flag} {label:<12} {'('+desc+')' if val else ''}", flush=True)

        print()
        print(f"  加分原因：", flush=True)
        for reason in res["reasons"]:
            print(f"    + {reason}", flush=True)
        return

    # ── 全市场扫描 ──────────────────────────────────────
    print(f"\n{'='*80}", flush=True)
    print(f"  main_force_scan.py  盘后主力行为评分  |  日期={target_date}", flush=True)
    print(f"{'='*80}", flush=True)

    t0 = datetime.now()
    results = scan_market(top_n=args.top_n, min_score=args.min_score,
                          end_date=target_date)
    elapsed = (datetime.now() - t0).total_seconds()

    # 控制台输出
    print(flush=True)
    lines = render_txt(results, target_date, len(results), elapsed)
    for line in lines:
        print(line, flush=True)

    # 保存文件
    save_results(results, target_date, elapsed)


if __name__ == "__main__":
    main()
