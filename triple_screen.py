#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三步量化选股系统（整合版）
==========================
Step1: 综合RPS≥75，RSI 50~80，20日涨幅≤50%
Step2: trend 验证趋势，确认均线多头
Step3: gain_turnover 信号窗口启动（信号分仅含趋势+位置）

综合评分：gain×0.2 + RPS综×0.5 + 趋势×0.3

输出：~/stock_reports/triple_screen_YYYY-MM-DD.txt
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
from gain_turnover import _rpad, _lpad, normalize_prefixed
DEFAULT_RPS_COMPOSITE = 75.0   # Step1: RPS综合分门槛
DEFAULT_RSI_LOW = 50.0         # Step1: RSI下限（须在均线上方，下跌趋势排除）
DEFAULT_RSI_HIGH = 88.0        # Step1: RSI上限（>88超买过滤；82~88在Step2扣分）
DEFAULT_RPS20_MIN = 75.0       # Step1: RPS20门槛（近期强势）
DEFAULT_MAX_RET20 = 50.0       # Step1: 20日涨幅上限（避开暴涨）
DEFAULT_MAX_RET5 = 30.0        # Step1: 近5日涨幅上限（近期过速上涨则排除）
DEFAULT_RET3_MIN = 5.0         # Step1: 近3日涨幅下限（剔除横盘，等于窗口加速确认）
DEFAULT_MIN_TURNOVER_STEP1 = 2.0  # Step1: 5日均换手率下限（%%，市值相对）
DEFAULT_TREND_TOP = 100       # Step2: trend 保留数量（0=全部）
DEFAULT_TREND_SCORE = 30.0    # Step2: 趋势评分门槛
DEFAULT_GAIN_DAYS = 3
DEFAULT_GAIN_MIN = 2.0
DEFAULT_GAIN_MAX = 10.0
DEFAULT_QUALITY_DAYS = 20
DEFAULT_WORKERS = 8
DEFAULT_MARKET_STOP_LOSS = -5.0  # 市场21日涨幅低于此值则跳过


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

    # ── 排除当日无数据的股票 ──────────────────────────────
    if target_date is not None:
        target_str = target_date.strftime("%Y-%m-%d")
        before_count = len(df_all)
        # scan_rps 返回空 DataFrame（0只有效）时无 data_date 列，需防御
        if "data_date" not in df_all.columns:
            print(f"   ⚠️  无有效股票数据（{len(df_all)} 只），跳过")
            return pd.DataFrame(), pd.DataFrame()
        df_all = df_all[df_all["data_date"] == target_str]
        after_count = len(df_all)
        if before_count != after_count:
            print(f"   ⚠️  排除当日无数据股票: {before_count - after_count} 只（缓存最新日期 < {target_str}），剩余 {after_count} 只")

    # 若指定了 codes，则只保留指定范围（规范化前缀）
    if codes is not None:
        codes_normalized = [normalize_prefixed(c) for c in codes]
        codes_lower = {c.lower() for c in codes_normalized}
        df_all = df_all[df_all["code"].str.lower().isin(codes_lower)]
        print(f"   限定范围: {len(codes)} 只（其余用于排名计算）")

    # 筛选逻辑：与 rps_strong_screen.py 一致（仅 RPS 综合 + RPS20 门槛）
    df = df_all[
        (df_all["composite"] >= rps_composite) &
        (df_all["ret20_rps"] >= rps20_min)
    ].copy()

    df = df.sort_values("composite", ascending=False).head(300)

    print(f"   策略: RPS综合≥{rps_composite}, RPS20≥{rps20_min}（与rps_strong_screen逻辑一致）")
    print(f"✅ Step1 完成: {len(df_all)} 只扫描 → Top{len(df)} 用时 {time.time()-t0:.1f}s")
    for _, row in df.head(5).iterrows():
        print(f"   {row['code']} {row.get('name',''):<8} 综合={row['composite']:.1f}  "
              f"RPS20={row['ret20_rps']:.1f}")

    return df, df_all


# ─────────────────────────────────────────────────────────
# Step 2: trend_strong 趋势验证
# ─────────────────────────────────────────────────────────
def step2_trend(
    step1_df: pd.DataFrame,
    step1_all: pd.DataFrame,
    top_n: int,
    min_score: float,
    max_workers: int,
    target_date: datetime | None,
    show_rejected: bool = False,
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
            "trend": trend_score,   # 趋势维度仅展示
        })

    # 按 step1 顺序（综合分排序）保持不变，step2 只打分不排席
    df = pd.DataFrame(rows)
    if df.empty:
        print(f"⚠️ Step2: trend 筛选后无股票")
        return df, raw_results

    if not no_limit:
        df = df.head(top_n)

    # ── show_rejected 时输出完整评分表 ───────────────────────
    if show_rejected and not df.empty:
        # 合并 RPS 数据（从 step1_all）
        rps_cols = ["code", "ret20_rps", "ret60_rps", "ret120_rps", "composite"]
        rps_df = step1_all[rps_cols].copy() if all(c in step1_all.columns for c in rps_cols) else pd.DataFrame()
        if not rps_df.empty:
            rps_df["code"] = rps_df["code"].str.lower()
            df["code"] = df["code"].str.lower()
            df = df.merge(rps_df, on="code", how="left")

        # 保持 step1 顺序（按 composite 降序），code 在 step1_df 中的位置
        code_order = {c: i for i, c in enumerate(step1_df["code"].str.lower().tolist())}
        df["_order"] = df["code"].map(code_order)
        df = df.sort_values("_order").drop(columns=["_order"])

        # 列顺序：代码 | 名称 | RPS20 | RPS60 | RPS120 | 综合分 | 总分 | 趋势
        cols_show = ["code", "name", "ret20_rps", "ret60_rps", "ret120_rps",
                     "composite", "total_score", "trend"]
        show_df = df[[c for c in cols_show if c in df.columns]].copy()

        # 自动列宽对齐
        header = ["代码", "名称", "RPS20", "RPS60", "RPS120", "综合分", "总分", "趋势"]
        rows_out = []
        for _, r in show_df.iterrows():
            rows_out.append([
                r.get("code", ""),
                r.get("name", ""),
                f"{r.get('ret20_rps', 0):.1f}" if pd.notna(r.get('ret20_rps')) else "-",
                f"{r.get('ret60_rps', 0):.1f}" if pd.notna(r.get('ret60_rps')) else "-",
                f"{r.get('ret120_rps', 0):.1f}" if pd.notna(r.get('ret120_rps')) else "-",
                f"{r.get('composite', 0):.1f}" if pd.notna(r.get('composite')) else "-",
                f"{r.get('total_score', 0):.1f}",
                f"{r.get('trend', 0):.1f}",
            ])

        # 名称列（中文字符宽度问题：用全角空格补齐到偶数宽度）
        col_widths_raw = [max(len(str(row[i])) for row in [header] + rows_out) for i in range(len(header))]
        # 名称列(i==1)加倍宽度（中文2字节≈英文1字节）
        col_widths = [max(w * 2 if i == 1 else w, len(header[i])) for i, w in enumerate(col_widths_raw)]

        print(f"\n📋 Step2 趋势评分表（共 {len(rows_out)} 只）")
        # 不用分隔符，纯空格列对齐
        print("  ".join(h.ljust(col_widths[i]) for i, h in enumerate(header)))
        print("  ".join("─" * col_widths[i] for i in range(len(header))))
        for row in rows_out:
            print("  ".join(str(row[i]).ljust(col_widths[i]) for i in range(len(row))))
        print("  ".join("─" * col_widths[i] for i in range(len(header))) + "\n")

    print(f"✅ Step2 完成: {len(df)} 只趋势健康，用时 {time.time()-t0:.1f}s")
    for _, row in df.head(5).iterrows():
        print(f"   {row['code']} {row.get('name',''):<8} 总分={row['total_score']:.1f}  趋势={row['trend']:.1f}")

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
    max_workers: int = 8,
    min_turnover: float = 2.0,
    score_threshold: float = 40.0,
    show_rejected: bool = False,
) -> tuple[list, list]:
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
        min_turnover=min_turnover,
        score_threshold=score_threshold,
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

    print(f"✅ Step3 完成: {len(results)} 只入场候选，用时 {time.time()-t0:.1f}s\n")

    # ── 未入选股票诊断 ──────────────────────────────────
    rejected = []
    if show_rejected:
        passed_codes = {r.code.lower() for r in results}
        rejected_codes = [c for c in codes if c.lower() not in passed_codes]
        end_date = target_date.strftime("%Y-%m-%d") if target_date else None
        from stock_trend.gain_turnover import prepare_data
        from gain_turnover_screen import load_qfq_history
        for code in rejected_codes:
            c = gt.normalize_prefixed(code)
            df = load_qfq_history(c, end_date=end_date, adjust=config.adjust, refresh=False)
            if df is None or df.empty:
                rejected.append({"code": code, "name": "", "reasons": ["数据加载失败"]})
                continue
            if target_date is not None:
                df = df[df["date"] <= pd.Timestamp(target_date.date())].reset_index(drop=True)
            if df.empty:
                rejected.append({"code": code, "name": "", "reasons": ["无目标日期数据"]})
                continue
            prep = prepare_data(df)
            if prep is None:
                rejected.append({"code": code, "name": "", "reasons": ["数据不足"], "idx": -1})
                continue
            idx = len(prep.dates) - 1
            reasons = gt.diagnose_rejection(prep, idx, config)
            name = gt.get_stock_name(c, gt.load_stock_names()) if hasattr(gt, 'get_stock_name') else ""
            rejected.append({"code": code, "name": name, "reasons": reasons, "idx": idx})

        if rejected:
            print(f"📋 未入选股票 {len(rejected)} 只（Top50中 Step3 未通过）：")
            print(f"   {'代码':<12} {'名称':<8} {'idx':<5} 失败原因")
            print(f"   {'─'*80}")
            for item in sorted(rejected, key=lambda x: x.get('code','')):
                rsn_str = "; ".join(item["reasons"]) if item["reasons"] else "未知"
                print(f"   {item['code']:<12} {item.get('name',''):<8} {item.get('idx',''):<5} {rsn_str}")
            print()

    return results, rejected


# ─────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────
# 打印最终结果 + 保存
# ─────────────────────────────────────────────────────────
def save_and_print(results: list, step1_all: pd.DataFrame, step2_df: pd.DataFrame,
                   output_path: Path | None, target_date: datetime | None, section: str = "",
                   write_mode: str = "w"):
    """打印并保存最终结果，格式与 gain_turnover_screen.py 的 format_signal_results 完全一致"""
    if not results:
        print("\n⚠️ 最终无交集股票（三步筛选均通过）")
        return

    # 合并 RPS / trend 数据
    rps_dict = {row["code"].lower(): row for _, row in step1_all.iterrows()}
    trend_dict = {row["code"].lower(): row for _, row in step2_df.iterrows()}

    date_str = target_date.strftime("%Y-%m-%d") if target_date else datetime.now().strftime("%Y-%m-%d")
    section_tag = f"🚀 {section} " if section else "📊 三步量化选股 "
    title = f"{section_tag}{date_str}"

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
        f"\t{_lpad('RSI',6)}\t{_rpad('风险',8)}"
        f"\t{_lpad('偏离MA20',9)}"
        f"\t{_lpad('收盘',7)}\t{_lpad('加分',8)}"
    )
    lines.append(col_spec)
    lines.append("-" * 160)

    # 按综合评分排序：gain×0.2 + RPS综合×0.5 + trend×0.3
    def composite_score(r):
        info = rps_dict.get(r.code.lower(), {})
        t_info = trend_dict.get(r.code.lower(), {})
        rps_c = info.get("composite", 0.0)
        trend_s = t_info.get("total_score", 0.0)
        return r.score * 0.2 + rps_c * 0.5 + trend_s * 0.3

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

        # 加分列
        extras = []
        if r.sector_bonus_applied > 0:
            extras.append(f"+{int(r.sector_bonus_applied)}({r.sector_name})")
        if r.limit_up_bonus > 0:
            extras.append(f"+{int(r.limit_up_bonus)}涨停")
        bonus_str = " ".join(extras) if extras else "-"

        # RSI 风险等级
        rsi_val = r.rsi14
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
                risk_tier = "🔴高位热"
            elif rsi_val <= 82:
                risk_tier = "🔴强弩"
            else:
                risk_tier = "❌超买"

        row = (
            f"{_rpad(code,10)}\t{_rpad(name,8)}\t{_rpad(signal_date,12)}"
            f"\t{_lpad(f'{r.score:.1f}',6)}\t{_lpad(f'{r.total_gain_window:+.2f}%',9)}"
            f"\t{_lpad(f'{rps_c:.1f}',8)}\t{_lpad(f'{trend_score:.1f}',6)}"
            f"\t{_lpad(f'{r.avg_turnover_5:.2f}%',10)}"
            f"\t{_lpad(f'{r.rsi14:.1f}',6)}\t{_rpad(risk_tier,8)}"
            f"\t{_lpad(f'{r.extension_pct:+.2f}%',9)}"
            f"\t{_lpad(f'{r.close:.2f}',7)}\t{_lpad(bonus_str,8)}"
        )
        lines.append(row)

    lines.append("-" * 160)

    # 底部评分说明
    bonus_parts = []
    if any(r.sector_bonus_applied > 0 for r in results):
        bonus_parts.append("热门板块+8")
    if any(r.limit_up_bonus > 0 for r in results):
        bonus_parts.append("近10日涨停+3")
    bonus_note = (" + " + " + ".join(bonus_parts)) if bonus_parts else ""
    lines.append(f"评分: 稳定性20 + 信号强度10 + 趋势25 + 流动性15 + 量能15 + K线5 + RSI10{bonus_note}")
    lines.append(f"RSI分层(Step2扣分): 🟡>75~82扣2分 | 🔴82~88扣5分")
    lines.append(f"综合评分 = gain×0.2 + RPS综合×0.5 + 趋势×0.3（用于最终排序）" )

    output_text = "\n".join(lines)
    print("\n" + output_text)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mode = write_mode
        with open(output_path, mode, encoding="utf-8") as f:
            f.write(output_text)
            f.write("\n")
        if write_mode == "w":
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
    parser.add_argument("--min-turnover-step3", type=float, default=2.0, help="Step3 5日均换手率下限/%%（默认2.0）")
    parser.add_argument("--score-threshold-step3", type=float, default=40.0, help="Step3 评分门槛（默认40.0）")
    parser.add_argument("--market-stop-loss", type=float, default=DEFAULT_MARKET_STOP_LOSS, help=f"市场止损（%%，默认{DEFAULT_MARKET_STOP_LOSS}）")
    parser.add_argument("--show-rejected", action="store_true", help="输出 Step3 未入选股票的失败原因")
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
    print(f"{'#'*60}")

    # 市场止损检查
    from stock_trend.trend_strong_screen import get_market_gain, INDEX_CODES
    market = get_market_gain(INDEX_CODES, days=21, target_date=target_date)
    if market < args.market_stop_loss:
        print(f"❌ 市场21日涨幅{market:.2f}% < 止损线{args.market_stop_loss}%，停止选股")
        return
    print(f"📈 市场21日涨幅: {market:.2f}%")

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

    # Step 2（Step1 已取 Top50）
    step2_df, _ = step2_trend(
        step1_df=step1_df,
        step1_all=step1_all,
        top_n=0 if args.trend_top == 0 else args.trend_top,
        min_score=args.trend_score,
        max_workers=args.workers,
        target_date=target_date,
        show_rejected=args.show_rejected,
    )

    if step2_df.empty:
        print("\n⚠️ Step2 trend筛选后无股票，退出")
        return

    # Step 3
    results, rejected = step3_gain(
        step2_df=step2_df,
        signal_days=args.days,
        min_gain=args.min_gain,
        max_gain=args.max_gain,
        quality_days=args.quality_days,
        target_date=target_date,
        check_fundamental=args.check_fundamental,
        sector_bonus=args.sector_bonus,
        max_workers=args.workers,
        min_turnover=args.min_turnover_step3,
        score_threshold=args.score_threshold_step3,
        show_rejected=args.show_rejected,
    )

    # 输出（路径与 gain_turnover_screen 保持一致）
    date_str = target_date.strftime("%Y-%m-%d") if target_date else datetime.now().strftime("%Y-%m-%d")
    output_path = Path.home() / "stock_reports" / f"triple_screen_{date_str}.txt"

    # 分类：启动型 vs 趋势跟随型
    startup = [r for r in results if r.total_gain_window > 10 and r.avg_turnover_5 > 3]
    trend_follow = [r for r in results if r.total_gain_window <= 10 or r.avg_turnover_5 <= 3]

    save_and_print(startup, step1_all, step2_df, output_path, target_date, section="启动型", write_mode="w")
    save_and_print(trend_follow, step1_all, step2_df, output_path, target_date, section="趋势型", write_mode="a")

    print(f"\n⏱️  总耗时: {time.time()-total_t0:.1f}s")


if __name__ == "__main__":
    main()
