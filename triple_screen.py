#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三步量化选股系统（整合版）
==========================
Step 1: rps_strong_screen  →  RPS强势蓄势股（适合介入）
Step 2: trend_strong_screen →  趋势验证（均线健康）
Step 3: gain_turnover_screen → 精准入场点

策略设计：
  Step1: 综合RPS≥80，RSI 40~82（健康区间，非超买），20日涨幅≤40%
         → 选出"市场里强，但还没暴涨"的蓄势股
  Step2: trend 验证趋势，确认均线多头
  Step3: gain_turnover 信号窗口启动

输出：~/stock_reports/triple_screen_YYYY-MM-DD.txt
      （与 gain_turnover_screen.py 保持一致）

用法：
  python triple_screen.py                           # 默认全市场
  python triple_screen.py --date 2026-04-15        # 复盘
  python triple_screen.py --codes sz000678         # 指定股票
  python triple_screen.py --rps-composite 85       # 调RPS门槛
  python triple_screen.py --gain-days 3           # 调信号窗口
  python triple_screen.py --check-fundamental      # 加基本面检查
  python triple_screen.py --check-volume-surge     # 加放量检查
"""

import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd

WORKSPACE = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(WORKSPACE))

from stock_trend import gain_turnover as gt
from stock_trend import trend_strong_screen as tss
from stock_trend import rps_strong_screen as rps

# ── 默认参数 ─────────────────────────────────────────────
DEFAULT_RPS_COMPOSITE = 75.0   # Step1: RPS综合分门槛
DEFAULT_RSI_LOW = 50.0         # Step1: RSI下限（须在均线上方，下跌趋势排除）
DEFAULT_RSI_HIGH = 80.0        # Step1: RSI上限（>82才超买排除，75~82高位区扣分）
DEFAULT_RPS20_MIN = 75.0       # Step1: RPS20门槛（近期强势）
DEFAULT_MAX_RET20 = 40.0       # Step1: 20日涨幅上限（避开暴涨）
DEFAULT_MAX_RET5 = 30.0        # Step1: 近5日涨幅上限（近期过速上涨则排除）
DEFAULT_RET3_MIN = 3.0         # Step1: 近3日涨幅下限（剔除横盘，等于窗口加速确认）
DEFAULT_MIN_TURNOVER_STEP1 = 2.0  # Step1: 5日均换手率下限（%%，市值相对）
DEFAULT_TREND_TOP = 100       # Step2: trend 保留数量（0=全部）
DEFAULT_TREND_SCORE = 30.0    # Step2: 趋势评分门槛
DEFAULT_GAIN_DAYS = 3
DEFAULT_GAIN_MIN = 2.0
DEFAULT_GAIN_MAX = 10.0
DEFAULT_QUALITY_DAYS = 20
DEFAULT_WORKERS = 8


# ─────────────────────────────────────────────────────────
# Step 1: RPS 扫描 → 蓄势强势股
# ─────────────────────────────────────────────────────────
def step1_rps(
    codes: list | None,
    rps_composite: float,
    rps20_min: float,
    rsi_low: float,
    rsi_high: float,
    max_ret20: float,
    max_ret5: float,
    ret3_min: float,
    min_turnover: float,
    max_workers: int,
    target_date: datetime | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """返回 (筛选后的df, 全市场df)"""
    t0 = time.time()

    # 全市场扫描（始终用全市场算RPS排名，保证相对排名准确）
    all_codes = rps.get_all_stock_codes()
    print(f"\n📊 Step 1/3 — RPS 全市场扫描（{len(all_codes)} 只）")

    df_all = rps.scan_rps(all_codes, top_n=len(all_codes), max_workers=max_workers, target_date=target_date)

    # 若指定了 codes，则只保留指定范围
    if codes is not None:
        codes_lower = {c.lower() for c in codes}
        df_all = df_all[df_all["code"].str.lower().isin(codes_lower)]
        print(f"   限定范围: {len(codes)} 只（其余用于排名计算）")

    # 筛选策略：适合介入的蓄势股
    df = df_all[
        (df_all["composite"] >= rps_composite) &
        (df_all["ret20_rps"] >= rps20_min) &
        (df_all["rsi"] >= rsi_low) &
        (df_all["rsi"] <= rsi_high) &
        (df_all["ret20"] <= max_ret20) &
        (df_all["ret20"] >= -10) &
        (df_all["ret5"] <= max_ret5) &
        (df_all["ret20"] >= ret3_min) &
        (df_all["avg_turnover_5"] >= min_turnover)
    ].copy()

    df = df.sort_values("composite", ascending=False)

    print(f"   策略: RPS综合≥{rps_composite}, RPS20≥{rps20_min}, RSI[{rsi_low},{rsi_high}], "
          f"20日涨幅≤{max_ret20}%, 近5日≤{max_ret5}%, 3日≥{ret3_min}%, "
          f"5日换手≥{min_turnover}%")
    print(f"✅ Step1 完成: {len(df_all)} 只扫描 → {len(df)} 只蓄势强势股，用时 {time.time()-t0:.1f}s")
    for _, row in df.head(5).iterrows():
        print(f"   {row['code']} {row['name']:<8} 综合={row['composite']:.1f}  "
              f"RPS20={row['ret20_rps']:.1f}  RSI={row['rsi']:.1f}  20日={row['ret20']:+.2f}%")

    return df, df_all


# ─────────────────────────────────────────────────────────
# Step 2: trend_strong 趋势验证
# ─────────────────────────────────────────────────────────
def step2_trend(
    step1_df: pd.DataFrame,
    top_n: int,
    min_score: float,
    max_workers: int,
    target_date: datetime | None,
) -> tuple[pd.DataFrame, list]:
    t0 = time.time()
    codes = step1_df["code"].str.lower().tolist()

    no_limit = top_n <= 0
    limit_str = "全部" if no_limit else f"Top{top_n}"
    print(f"\n📊 Step 2/3 — trend_strong 趋势验证（{len(codes)} 只 → 保留 {limit_str}）")

    raw_results = tss.scan_market(
        codes=codes,
        top_n=top_n,
        score_threshold=min_score,
        max_workers=max_workers,
        target_date=target_date,
    )

    rows = []
    for item in raw_results:
        if not isinstance(item, tuple) or len(item) < 4:
            continue
        code = item[0]
        name = item[1] or ""
        score = float(item[2]) if item[2] is not None else 0
        factors = item[3] if isinstance(item[3], dict) else {}
        f_trend = factors.get("trend", {})
        f_mom = factors.get("momentum", {})
        f_vol = factors.get("volume", {})
        trend_score = (
            f_trend.get("above_score", 0) + f_trend.get("bull_score", 0) +
            f_trend.get("div_score", 0) + f_trend.get("slope_score", 0)
        )
        momentum_score = (
            f_mom.get("gain_20d_score", 0) + f_mom.get("gain_10d_score", 0) +
            f_mom.get("new_high_score", 0) + f_mom.get("recent_strong_bonus", 0)
        )
        vol_score = (
            f_vol.get("vr_score", 0) + f_vol.get("ar_score", 0) + f_vol.get("match_score", 0)
        )
        rows.append({
            "code": code, "name": name, "total_score": score,
            "trend": trend_score, "momentum": momentum_score, "vol": vol_score,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print(f"⚠️ Step2: trend 筛选后无股票")
        return df, raw_results

    df = df.sort_values("total_score", ascending=False)
    if not no_limit:
        df = df.head(top_n)
    print(f"✅ Step2 完成: {len(df)} 只趋势健康，用时 {time.time()-t0:.1f}s")
    for _, row in df.head(5).iterrows():
        print(f"   {row['code']} {row['name']:<8} 总分={row['total_score']:.1f}  "
              f"趋势={row['trend']:.1f} 动量={row['momentum']:.1f} 量价={row['vol']:.1f}")

    return df, raw_results


# ─────────────────────────────────────────────────────────
# Step 3: gain_turnover 入场点筛选（输出格式与 screen_market 一致）
# ─────────────────────────────────────────────────────────
def step3_gain(
    step2_df: pd.DataFrame,
    signal_days: int,
    min_gain: float,
    max_gain: float,
    quality_days: int,
    target_date: datetime | None,
    check_fundamental: bool,
    sector_bonus: bool,
    check_volume_surge: bool,
    max_workers: int,
) -> list:
    t0 = time.time()
    codes = step2_df["code"].str.lower().tolist()

    print(f"\n📊 Step 3/3 — gain_turnover 入场点筛选（{len(codes)} 只）")

    config = gt.StrategyConfig(
        signal_days=signal_days,
        min_gain=min_gain,
        max_gain=max_gain,
        quality_days=quality_days,
        check_fundamental=check_fundamental,
        sector_bonus=sector_bonus,
        check_volume_surge=check_volume_surge,
    )

    from stock_trend.gain_turnover_screen import screen_market
    results = screen_market(
        codes=codes,
        config=config,
        target_date=target_date,
        top_n=len(codes),
        max_workers=max_workers,
        refresh_cache=False,
    )

    print(f"✅ Step3 完成: {len(results)} 只入场候选，用时 {time.time()-t0:.1f}s")
    return results


# ─────────────────────────────────────────────────────────
# 连续命中计数
# ─────────────────────────────────────────────────────────
def load_consecutive_counts(target_date: datetime | None) -> tuple[dict[str, int], dict[str, int]]:
    """
    读取上一个交易日 triple_screen 输出，返回两个 dict：
      (code → 连号次数, code → RSI连档天数)
    如果今天出现在上交易日输出里 → 今天 = 上交易日次数 + 1
    RSI连档 = 上交易日连档 + 1（如果昨天的RSI在高位区>72）；否则 = 0
    自动处理周一（找上周五）的情况。
    """
    reports_dir = Path.home() / "stock_reports"
    today = target_date or datetime.now()
    today_str = today.strftime("%Y-%m-%d")

    candidates = []
    for p in reports_dir.glob("triple_screen_*.txt"):
        if p.name != f"triple_screen_{today_str}.txt":
            candidates.append(p)

    if not candidates:
        return {}, {}

    prev_file = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]

    import re
    code_pat = re.compile(r"^(sh|sz|bj)(\d{6})$")
    consec_pat = re.compile(r"连号[:：]?(\d+)")
    rsi_pat = re.compile(r"连档[:：]?(\d+)")

    consec_counts: dict[str, int] = {}
    rsi_high_counts: dict[str, int] = {}

    for line in prev_file.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        m = code_pat.match(parts[0].lower())
        if not m:
            continue
        code = f"{m.group(1)}{m.group(2)}"

        # 连号
        last = parts[-1]
        mc = consec_pat.search(last)
        if mc:
            consec_counts[code] = int(mc.group(1))
        else:
            try:
                consec_counts[code] = int(last)
            except ValueError:
                consec_counts[code] = 0

        # RSI 连档（在倒数第二列或特定列找 "连档:N"）
        rsi_high_counts[code] = 0
        for p in reversed(parts):
            mr = rsi_pat.search(p)
            if mr:
                rsi_high_counts[code] = int(mr.group(1))
                break

    return consec_counts, rsi_high_counts


# ─────────────────────────────────────────────────────────
# 打印最终结果 + 保存（与 gain_turnover_screen 格式完全一致）
# ─────────────────────────────────────────────────────────
from gain_turnover import _rpad, _lpad

def save_and_print(results: list, step1_all: pd.DataFrame, step2_df: pd.DataFrame,
                   output_path: Path | None, target_date: datetime | None):
    """打印并保存最终结果，格式与 gain_turnover_screen.py 的 format_signal_results 完全一致"""
    if not results:
        print("\n⚠️ 最终无交集股票（三步筛选均通过）")
        return

    # 合并 RPS / trend 数据
    rps_dict = {row["code"].lower(): row for _, row in step1_all.iterrows()}
    trend_dict = {row["code"].lower(): row for _, row in step2_df.iterrows()}

    # 读取昨天连号和RSI连档，今天 = 昨天 + 1；昨天没有的 = 0
    yesterday_consec, yesterday_rsi_high = load_consecutive_counts(target_date)

    date_str = target_date.strftime("%Y-%m-%d") if target_date else datetime.now().strftime("%Y-%m-%d")
    title = f"三步量化选股 {date_str}"

    lines = []
    lines.append("=" * 160)
    lines.append(f"📊 {title}（共 {len(results)} 只）")
    lines.append("=" * 160)

    # 列头（与 gain_turnover_screen 一致：文本左对齐，数字右对齐）
    col_spec = (
        f"{_rpad('代码',10)}\t{_rpad('名称',8)}\t{_rpad('日期',12)}"
        f"\t{_lpad('总分',6)}\t{_lpad('窗口涨幅',9)}"
        f"\t{_lpad('RPS综合',8)}\t{_lpad('趋势',6)}"
        f"\t{_lpad('5日换手%',10)}"
        f"\t{_lpad('RSI',6)}\t{_rpad('风险',8)}\t{_rpad('RSI动量',8)}\t{_rpad('量加速',8)}"
        f"\t{_lpad('偏离MA20',9)}"
        f"\t{_lpad('收盘',7)}\t{_lpad('扣分',8)}\t{_lpad('连号',5)}\t{_lpad('连档',5)}"
    )
    lines.append(col_spec)
    lines.append("-" * 160)

    # 按综合评分排序：gain×0.4 + RPS综合×0.3 + trend×0.3
    # 连号≥3天 → 趋势疲劳降权20%
    def composite_score(r):
        info = rps_dict.get(r.code.lower(), {})
        t_info = trend_dict.get(r.code.lower(), {})
        rps_c = info.get("composite", 0.0)
        trend_s = t_info.get("total_score", 0.0)
        consec_today = (yesterday_consec.get(r.code.lower(), 0) + 1
                        if r.code.lower() in yesterday_consec else 0)
        fatigue_penalty = 0.8 if consec_today >= 3 else 1.0
        return (r.score * 0.4 + rps_c * 0.3 + trend_s * 0.3) * fatigue_penalty

    results = sorted(results, key=composite_score, reverse=True)

    for r in results:
        code = r.code or ""
        name = r.name or ""
        signal_date = r.signal_date or ""

        # RPS 数据
        info = rps_dict.get(code.lower(), {})
        rps_c = info.get("composite", 0.0)

        # trend 数据
        t_info = trend_dict.get(code.lower(), {})
        trend_score = t_info.get("total_score", 0.0)

        # 扣分列
        if r.sector_bonus_applied > 0:
            penalty_str = f"+{int(r.sector_bonus_applied)}({r.sector_name})"
        elif r.fundamental_penalty:
            penalty_str = f"-{r.fundamental_penalty}"
        else:
            penalty_str = "-"
        extras = []
        if r.sector_bonus_applied > 0:
            extras.append(f"+{int(r.sector_bonus_applied)}({r.sector_name})")
        if r.limit_up_bonus > 0:
            extras.append(f"+{int(r.limit_up_bonus)}涨停")
        if extras:
            penalty_str = " ".join(extras)

        # 连号 = 昨天连号 + 1；昨天没有的 = 0
        consec_today = (yesterday_consec.get(code.lower(), 0) + 1
                        if code.lower() in yesterday_consec else 0)

        # RSI 连档：昨天也在高位区(>72) → +1；否则 → 0
        # RSI 高位区：> 72
        rsi_val = r.rsi14
        is_high_rsi = rsi_val > 72
        prev_rsi_high = yesterday_rsi_high.get(code.lower(), 0)
        rsi_high_today = (prev_rsi_high + 1 if is_high_rsi and code.lower() in yesterday_rsi_high else (1 if is_high_rsi else 0))

        # RSI 风险等级
        risk_tier = getattr(r, 'rsi_tier', '') or ''
        if not risk_tier:
            if rsi_val < 50:
                risk_tier = "🔵低位"
            elif rsi_val <= 65:
                risk_tier = "🟢健康"
            elif rsi_val <= 72:
                risk_tier = "🟡偏强"
            elif rsi_val <= 75:
                risk_tier = "🔴高位"
            elif rsi_val <= 78:
                risk_tier = "高位热"
            elif rsi_val <= 82:
                risk_tier = "高位热"
            else:
                risk_tier = "❌超买"

        row = (
            f"{_rpad(code,10)}\t{_rpad(name,8)}\t{_rpad(signal_date,12)}"
            f"\t{_lpad(f'{r.score:.1f}',6)}\t{_lpad(f'{r.total_gain_window:+.2f}%',9)}"
            f"\t{_lpad(f'{rps_c:.1f}',8)}\t{_lpad(f'{trend_score:.1f}',6)}"
            f"\t{_lpad(f'{r.avg_turnover_5:.2f}%',10)}"
            f"\t{_lpad(f'{r.rsi14:.1f}',6)}\t{_rpad(risk_tier,8)}"
            f"\t{_rpad(f'{getattr(r, 'rsi_momentum', 0):+.1f}',8)}"
            f"\t{_rpad(f'x{getattr(r, "volume_accel", 0):.2f}',8)}"
            f"\t{_lpad(f'{r.extension_pct:+.2f}%',9)}"
            f"\t{_lpad(f'{r.close:.2f}',7)}\t{_lpad(penalty_str,8)}\t{_lpad(str(consec_today),5)}\t{_lpad(str(rsi_high_today),5)}"
        )
        lines.append(row)

    lines.append("-" * 160)

    # 底部评分说明
    bonus_parts = []
    if any(r.sector_bonus_applied > 0 for r in results):
        bonus_parts.append("热门板块+8")
    if any(r.limit_up_bonus > 0 for r in results):
        bonus_parts.append("近10日涨停+10")
    bonus_note = (" + " + " + ".join(bonus_parts)) if bonus_parts else ""
    lines.append(f"评分: 稳定性20 + 信号强度10 + 趋势25 + 流动性15 + 量能15 + K线5 + RSI10 + RSI动量 + 量能加速度{bonus_note}")
    lines.append(f"RSI分层: 🟡偏强65~72扣5分 | 🔴高位72~75扣10分 | 高位热75~82扣15~25分")
    lines.append(f"RSI动量: >+5得5分 | >+8得10分 | <-5扣5分")
    lines.append(f"量加速: x≥1.2得2分 | x≥1.5得5分 | x≥2.0得8分")
    lines.append(f"连号≥3天 → 综合评分×0.8（趋势疲劳降权）")
    lines.append(f"综合评分 = gain×0.4 + RPS综合×0.3 + 趋势×0.3（用于最终排序）" )

    output_text = "\n".join(lines)
    print("\n" + output_text)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(output_text)
            f.write("\n")
        print(f"\n💾 结果已写入: {output_path.resolve()}")


# ─────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────
def main():
    import argparse

    parser = argparse.ArgumentParser(description="三步量化选股系统")
    parser.add_argument("--rps-composite", type=float, default=DEFAULT_RPS_COMPOSITE, help=f"RPS综合门槛（默认{DEFAULT_RPS_COMPOSITE}）")
    parser.add_argument("--rsi-low", type=float, default=DEFAULT_RSI_LOW, help=f"RSI下限（默认{DEFAULT_RSI_LOW}）")
    parser.add_argument("--rsi-high", type=float, default=DEFAULT_RSI_HIGH, help=f"RSI上限（默认{DEFAULT_RSI_HIGH}）")
    parser.add_argument("--rps20-min", type=float, default=DEFAULT_RPS20_MIN, help=f"RPS20门槛（默认{DEFAULT_RPS20_MIN}）")
    parser.add_argument("--max-ret20", type=float, default=DEFAULT_MAX_RET20, help=f"20日涨幅上限（默认{DEFAULT_MAX_RET20}）")
    parser.add_argument("--max-ret5", type=float, default=DEFAULT_MAX_RET5, help=f"近5日涨幅上限（默认{DEFAULT_MAX_RET5}）")
    parser.add_argument("--ret3-min", type=float, default=DEFAULT_RET3_MIN, help=f"近3日涨幅下限（默认{DEFAULT_RET3_MIN}）")
    parser.add_argument("--min-turnover-step1", type=float, default=DEFAULT_MIN_TURNOVER_STEP1, help=f"Step1 5日均换手率下限/%%（默认{DEFAULT_MIN_TURNOVER_STEP1}）")
    parser.add_argument("--trend-top", type=int, default=0, help="Step2 保留数量（默认0=全部）")
    parser.add_argument("--trend-score", type=float, default=30.0, help="Step2 趋势评分门槛（默认30.0）")
    parser.add_argument("--days", type=int, default=DEFAULT_GAIN_DAYS, help=f"信号窗口天数（默认{DEFAULT_GAIN_DAYS}）")
    parser.add_argument("--min-gain", type=float, default=DEFAULT_GAIN_MIN, help=f"最小日涨幅百分比（默认{DEFAULT_GAIN_MIN}）")
    parser.add_argument("--max-gain", type=float, default=DEFAULT_GAIN_MAX, help=f"最大日涨幅百分比（默认{DEFAULT_GAIN_MAX}）")
    parser.add_argument("--quality-days", type=int, default=DEFAULT_QUALITY_DAYS, help=f"质量窗口天数（默认{DEFAULT_QUALITY_DAYS}）")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help=f"并行线程数（默认{DEFAULT_WORKERS}）")
    parser.add_argument("--codes", nargs="+", default=None, help="指定股票代码（跳过全市场Step1）")
    parser.add_argument("--date", type=str, default=None, help="截止日期 YYYY-MM-DD（复盘用）")
    parser.add_argument("--check-fundamental", action="store_true", help="开启基本面检查（亏损扣分）")
    parser.add_argument("--sector-bonus", action="store_true", help="开启热门板块加分")
    parser.add_argument("--check-volume-surge", action="store_true", help="开启放量检查")
    args = parser.parse_args()

    target_date = None
    if args.date:
        target_date = datetime.strptime(args.date, "%Y-%m-%d")
        print(f"📅 复盘模式: {args.date}")

    total_t0 = time.time()
    print(f"\n{'#'*60}")
    print(f"# 三步量化选股系统")
    print(f"# Step1: RPS综合≥{args.rps_composite}, RSI[{args.rsi_low},{args.rsi_high}], RPS20≥{args.rps20_min}, "
          f"近5日≤{args.max_ret5}%, 3日≥{args.ret3_min}%, 5日换手≥{args.min_turnover_step1}%")
    print(f"# Step2: trend_strong 评分≥{args.trend_score}{', Top'+str(args.trend_top) if args.trend_top > 0 else ''}")
    print(f"# Step3: gain_turnover {args.days}天窗口[{args.min_gain},{args.max_gain}%]")
    if args.check_fundamental:
        print(f"#        + 基本面检查")
    if args.sector_bonus:
        print(f"#        + 板块加分")
    if args.check_volume_surge:
        print(f"#        + 放量检查")
    print(f"{'#'*60}")

    # Step 1
    step1_df, step1_all = step1_rps(
        codes=args.codes,
        rps_composite=args.rps_composite,
        rps20_min=args.rps20_min,
        rsi_low=args.rsi_low,
        rsi_high=args.rsi_high,
        max_ret20=args.max_ret20,
        max_ret5=args.max_ret5,
        ret3_min=args.ret3_min,
        min_turnover=args.min_turnover_step1,
        max_workers=args.workers,
        target_date=target_date,
    )

    if step1_df.empty:
        print("\n⚠️ Step1 无符合RPS策略的股票，退出")
        return

    # Step 2
    step2_df, _ = step2_trend(
        step1_df=step1_df,
        top_n=0 if args.trend_top == 0 else args.trend_top,
        min_score=args.trend_score,
        max_workers=args.workers,
        target_date=target_date,
    )

    if step2_df.empty:
        print("\n⚠️ Step2 trend筛选后无股票，退出")
        return

    # Step 3
    results = step3_gain(
        step2_df=step2_df,
        signal_days=args.days,
        min_gain=args.min_gain,
        max_gain=args.max_gain,
        quality_days=args.quality_days,
        target_date=target_date,
        check_fundamental=args.check_fundamental,
        sector_bonus=args.sector_bonus,
        check_volume_surge=args.check_volume_surge,
        max_workers=args.workers,
    )

    # 输出（路径与 gain_turnover_screen 保持一致）
    date_str = target_date.strftime("%Y-%m-%d") if target_date else datetime.now().strftime("%Y-%m-%d")
    output_path = Path.home() / "stock_reports" / f"triple_screen_{date_str}.txt"

    save_and_print(results, step1_all, step2_df, output_path, target_date)

    print(f"\n⏱️  总耗时: {time.time()-total_t0:.1f}s")


if __name__ == "__main__":
    main()
