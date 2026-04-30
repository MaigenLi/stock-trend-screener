#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
复盘选股系统
============

用法：
    python screen.py --date 2026-04-22

输出：
    ~/stock_reports/review_screen_YYYY-MM-DD.txt
"""

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

WORKSPACE = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(WORKSPACE))

from stock_trend.review_screen.data_cache import load_qfq_history, preload_all_codes
from stock_trend.review_screen.indicators import compute_all, detect_volume_price_wave
from stock_trend.review_screen.filter_rules import FilterConfig, check_filters
from stock_trend.review_screen.scorer import score_stock, score_wave_quality, score_detail, classify_phase
from stock_trend.review_screen.utils import find_ascending_start

DEFAULT_WORKERS = 8

# ─────────────────────────────────────────
# 视觉对齐工具（中文=2字符宽度）
# ─────────────────────────────────────────
import unicodedata

def _vw(s):
    return sum(2 if unicodedata.east_asian_width(c) in ("W","F") else 1 for c in str(s))

def _pr(s, w):
    return str(s) + " " * max(0, w - _vw(s))

def _pl(s, w):
    return " " * max(0, w - _vw(s)) + str(s)

# 每列（标签，宽度，对齐）'>'=右，'<'=左
_COLS = [
    ("代码",     10, ">"),
    ("名称",      8, "<"),
    ("日期",     12, "<"),
    ("评分",      6, ">"),
    ("红柱",      5, ">"),
    ("收盘",      9, ">"),
    ("3日%",     7, ">"),
    ("换手%",    7, ">"),
    ("量比",      6, ">"),
    ("波量比",    7, ">"),
    ("MA5距%",   8, ">"),
    ("RSI",      6, ">"),
    ("MA20",     8, ">"),
    ("MA60",     8, ">"),
]

def _make_row(values):
    parts = []
    for (label, w, align), v in zip(_COLS, values):
        parts.append(_pl(v, w) if align == ">" else _pr(v, w))
    return " ".join(parts)

def _header_row():
    return _make_row([l for l, _, _ in _COLS])


# ─────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────

def normalize_code(code: str) -> str:
    """标准化股票代码（小写前缀，匹配缓存文件名）"""
    c = code.strip().upper()
    if not c.startswith(("SH", "SZ")):
        c = ("SH" if c.startswith("6") else "SZ") + c
    return c.lower()  # 小写，匹配缓存文件名


def load_stock_names() -> dict:
    """加载股票名称映射（复用 gain_turnover）"""
    from stock_trend.gain_turnover import load_stock_names as _load
    return _load()


def get_stock_name(code: str, names: dict) -> str:
    """获取股票名称（复用 gain_turnover）"""
    from stock_trend.gain_turnover import get_stock_name as _get
    return _get(code, names)


# ─────────────────────────────────────────
# 单股评估
# ─────────────────────────────────────────

def evaluate_stock(code: str, target_date: datetime | None, cfg: FilterConfig, names: dict,
                return_failed: bool = False) -> dict | None:
    """
    评估单只股票

    Args:
        code: 股票代码
        target_date: 信号日期
        cfg: 筛选配置
        names: 股票名称映射（由 scan_market 加载一次后传入）
        return_failed: 不过时也返回带原因的结果

    Returns:
        完整结果字典 或 None（不通过筛选且 return_failed=False）
    """
    c = normalize_code(code)
    end_str = target_date.strftime("%Y-%m-%d") if target_date else None

    df = load_qfq_history(c, end_date=end_str, refresh=False)
    if df is None or df.empty:
        if return_failed:
            return {"code": c, "name": get_stock_name(c, names), "failed": True, "reason": "无数据"}
        return None

    # 按日期截取
    if target_date is not None:
        df = df[df["date"] <= pd.Timestamp(target_date.date())].reset_index(drop=True)
    if len(df) < cfg.min_bars:
        if return_failed:
            return {"code": c, "name": get_stock_name(c, names), "failed": True, "reason": f"数据仅{len(df)}天<{cfg.min_bars}天"}
        return None

    # 计算指标（使用配置中的窗口参数）
    ind = compute_all(df, ma10_break_window=cfg.max_broke_ma10_days)
    if not ind:
        if return_failed:
            return {"code": c, "name": get_stock_name(c, names), "failed": True, "reason": "指标计算失败"}
        return None

    # 筛选（软扣分：非核心条件不达标不拒绝，扣减总分）
    passed, reason, soft_penalty = check_filters(ind, cfg)
    if not passed:
        if return_failed:
            return {"code": c, "name": get_stock_name(c, names), "failed": True, "reason": reason}
        return None

    # 评分（软扣分从总分中扣除）
    total_score = score_stock(ind) - soft_penalty
    ind["wave_quality_score"] = score_wave_quality(ind.get("waves", []))
    ind["code"] = c
    ind["name"] = get_stock_name(c, names)
    ind["score"] = max(total_score, 0)
    ind["_reason"] = reason

    # 阶段标签（基于筛选结果和质量评分综合判断）
    phase, phase_reason = classify_phase(ind, filter_passed=True, soft_penalty=soft_penalty)
    ind["phase"] = phase
    ind["phase_reason"] = phase_reason

    return ind


# ─────────────────────────────────────────
# 全市场扫描
# ─────────────────────────────────────────

# ─────────────────────────────────────────
# 市场环境检测
# ─────────────────────────────────────────

def _detect_market_mode(target_date: datetime | None) -> bool:
    """
    检测市场环境：大盘是否在熊市（HS300 < MA20）
    Returns: True = 熊市（需提高选股门槛），False = 正常/牛市
    """
    try:
        import akshare as ak
        symbol = "sh000300"  # 沪深300
        df_index = ak.stock_zh_index_daily(symbol=symbol)
        if df_index is None or df_index.empty:
            return False
        df_index["date"] = pd.to_datetime(df_index["date"])
        if target_date is not None:
            df_index = df_index[df_index["date"] <= pd.Timestamp(target_date.date())].reset_index(drop=True)
        if len(df_index) < 25:
            return False
        close = df_index["close"].values
        ma20 = float(pd.Series(close).rolling(20).mean().iloc[-1])
        latest_close = float(close[-1])
        is_bear = latest_close < ma20
        print(f"  📈 大盘环境: HS300={latest_close:.2f} MA20={ma20:.2f} → {'🐻 熊市' if is_bear else '🐂 正常'}")
        return is_bear
    except Exception:
        return False


def scan_market(
    codes: list,
    target_date: datetime | None,
    max_workers: int = DEFAULT_WORKERS,
    cfg: FilterConfig | None = None,
    return_failed: bool = False,
) -> list[dict]:
    """多线程扫描全市场"""
    if cfg is None:
        cfg = FilterConfig()
    results = []
    t0 = time.time()
    total = len(codes)

    # ── 市场环境检测：熊市时提高选股门槛 ─────────────────────────
    market_bear = _detect_market_mode(target_date)
    if market_bear:
        # 熊市：要求更高动能，min_gain5 从 3% 提到 5%
        original_min_gain5 = cfg.min_gain5
        cfg.min_gain5 = max(cfg.min_gain5, 5.0)
        print(f"  ⚠️ 熊市模式：min_gain5 {original_min_gain5}% → {cfg.min_gain5}%")

    # 名称映射只加载一次（避免每只股票重复读文件）
    names = load_stock_names()

    def work(code: str) -> dict | None:
        return evaluate_stock(code, target_date, cfg, names, return_failed=return_failed)

    done = [0]

    def log_progress(futures):
        for _ in as_completed(futures):
            done[0] += 1
            if done[0] % 500 == 0 or done[0] == total:
                elapsed = time.time() - t0
                speed = done[0] / elapsed if elapsed > 0 else 0
                eta = (total - done[0]) / speed if speed > 0 else 0
                print(f"  进度: {done[0]}/{total} ({done[0]*100//total}%)  速度:{speed:.0f}只/秒  ETA:{eta:.0f}秒", flush=True)

    print(f"📋 全市场扫描: {total} 只")
    print(f"🚀 开始筛选（workers={max_workers}）...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(work, c): c for c in codes}
        log_progress(futures)
        for future in as_completed(futures):
            r = future.result()
            if r is not None:
                results.append(r)

    if return_failed:
        failed = total - len([r for r in results if not r.get("failed", False)])
        print(f"✅ 扫描完成: {len(results)}/{total} 只（通过{total-failed}只，拒绝{failed}只），用时{time.time()-t0:.1f}秒")
    else:
        print(f"✅ 扫描完成: {len(results)}/{total} 只通过，用时 {time.time()-t0:.1f}秒")
    return results


# ─────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(description="复盘选股系统")
    parser.add_argument("--date", type=str, required=True, help="信号日期 YYYY-MM-DD")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help=f"并行线程数（默认{DEFAULT_WORKERS}）")
    parser.add_argument("--codes", nargs="+", default=None, help="指定股票代码列表（跳过全市场）")
    parser.add_argument("--output", type=str, default=None, help="输出文件路径")
    parser.add_argument("--waves", action="store_true", help="显示完整涨跌波段详情")
    parser.add_argument("--latest-wave-down", action="store_true", help="只选当前处于下跌波段的股票（蓄势找买点）")
    parser.add_argument("--reason", action="store_true", help="显示未通过股票的原因")
    parser.add_argument("--days", type=int, default=1, help="持续多少天（默认1天）")
    parser.add_argument("--buy", action="store_true", help="买入精选：RSI未过热 + 5日涨幅合理 + 止损空间<5%")
    args = parser.parse_args()

    # 解析日期
    try:
        target_date = datetime.strptime(args.date, "%Y-%m-%d")
    except ValueError:
        print(f"❌ 日期格式错误: {args.date}，应为 YYYY-MM-DD")
        sys.exit(1)

    # ─────────────────────────────────────────
    # 股票范围 + 筛选配置（多日和单日都要用，提前定义）
    # ─────────────────────────────────────────
    if args.codes:
        codes = args.codes
        print(f"\n📊 筛选（指定 {len(codes)} 只）")
    else:
        codes = preload_all_codes()
        print(f"\n📊 筛选（全市场 {len(codes)} 只）")

    cfg = FilterConfig(
        require_latest_wave_down=args.latest_wave_down,
    )

    date_str = args.date
    # ─────────────────────────────────────────
    # 多日持续模式
    # ─────────────────────────────────────────
    if args.days > 1:
        import akshare as ak
        trade_df = ak.tool_trade_date_hist_sina()
        all_trade_dates = pd.to_datetime(trade_df['trade_date']).dt.strftime('%Y-%m-%d').tolist()
        date_str = args.date
        if date_str not in all_trade_dates:
            print(f"❌ {date_str} 不是交易日，请使用正确的 YYYY-MM-DD 格式")
            sys.exit(1)
        start_idx = all_trade_dates.index(date_str)
        consecutive_dates = all_trade_dates[start_idx:start_idx + args.days]
        if len(consecutive_dates) < args.days:
            print(f"❌ 起始日期后不足 {args.days} 个交易日")
            sys.exit(1)
        print(f"\n📅 复盘选股: {args.date} × {args.days} 天持续筛选")
        print(f"📆 交易日: {' / '.join(consecutive_dates)}")

        # ── 每日扫描 ───────────────────────────────
        daily_passed = {}  # date_str -> list of passed results
        for day_str in consecutive_dates:
            day_dt = datetime.strptime(day_str, "%Y-%m-%d")
            day_results = scan_market(
                codes=codes,
                target_date=day_dt,
                max_workers=args.workers,
                cfg=cfg,
                return_failed=False,
            )
            day_passed = [r for r in day_results if not r.get("failed", False)]
            daily_passed[day_str] = day_passed
            print(f"  {day_str}: {len(day_passed)} 只通过")

        # ── 找交集：连续 N 天都通过的股票 ─────────────────
        if not daily_passed:
            print("\n⚠️  所有日期均无通过股票")
            sys.exit(0)
        # 交集：以第一天的代码为基准，筛出每天都有结果的
        base_set = {r['code']: r for r in daily_passed[consecutive_dates[0]]}
        for day_str in consecutive_dates[1:]:
            day_codes = {r['code'] for r in daily_passed[day_str]}
            base_set = {c: r for c, r in base_set.items() if c in day_codes}

        intersection_results = list(base_set.values())
        print(f"\n🎯 连续 {args.days} 天都通过: {len(intersection_results)} 只")
        intersection_results.sort(key=lambda x: x["score"], reverse=True)

        # 覆盖后续使用的变量
        passed_results = intersection_results
        date_str = f"{args.date}～{consecutive_dates[-1]}"
        target_date = datetime.strptime(consecutive_dates[-1], "%Y-%m-%d")

    print(f"\n📅 复盘选股: {date_str}")
    date_str_display = date_str

    # 输出路径
    output_path = Path(args.output) if args.output else Path.home() / "stock_reports" / f"review_screen_{date_str}.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 单日扫描（--days=1 或未指定时）
    if args.days <= 1:
        target_date = datetime.strptime(args.date, "%Y-%m-%d")
        results = scan_market(
            codes=codes,
            target_date=target_date,
            max_workers=args.workers,
            cfg=cfg,
            return_failed=args.reason,
        )
        passed_results = [r for r in results if not r.get("failed", False)]

    if not passed_results and not args.reason:
        print("\n⚠️  无符合筛选条件的股票")
        if not args.waves:
            sys.exit(0)

    # 按评分降序
    passed_results.sort(key=lambda x: x["score"], reverse=True)

    # ─────────────────────────────────────────
    # 波段详情格式化
    # ─────────────────────────────────────────
    def _load_df_for_waves(code: str, target_date: datetime):
        """重新加载数据用于波段分析"""
        from stock_trend.review_screen.data_cache import load_qfq_history

        c = normalize_code(code)
        end_str = target_date.strftime("%Y-%m-%d") if target_date else None
        df = load_qfq_history(c, end_date=end_str, refresh=False)
        if df is None or df.empty:
            return None, None
        df = df.sort_values("date").reset_index(drop=True)
        df = df[df["date"] <= pd.Timestamp(target_date.date())].reset_index(drop=True)
        return df, c

    def _format_wave_analysis(r: dict, df: pd.DataFrame, code: str) -> list[str]:
        """生成单只股票的完整波段分析文本"""
        close = df["close"].values
        volume = df["volume"].values
        high = df["high"].values
        low = df["low"].values
        lookback = min(60, len(close) - 1)
        result = detect_volume_price_wave(close, volume, lookback=lookback, high=high, low=low)
        waves = result.get("waves", [])
        if not waves:
            return []

        lines = []
        lines.append(f"\n{'─'*60}")
        lines.append(f"  {r['code']} {r['name']} 评分={r['score']}  红柱={r['red_days']}天  信号日={date_str}")
        lines.append(f"{'─'*60}")

        from scorer import score_wave_quality
        up_waves = [w for w in waves if w.direction == 'up']
        down_waves = [w for w in waves if w.direction == 'down']
        start_u_idx = find_ascending_start(up_waves)

        # ── 计算显示起点：丢弃 start_u_idx 个上涨及其间下跌 ────
        # 起点 = start_u_idx 对应上涨段在 waves 中的位置
        u1_wave = up_waves[start_u_idx]
        display_start = next(i for i, w in enumerate(waves) if w is u1_wave)

        display_waves = waves[display_start:]
        lines.append(f'  近{lookback}日涨跌波段（共{len(display_waves)}个）：')
        lines.append('')

        # ── 构建评分标注 ───────────────────────────────────
        scored_ups = up_waves[start_u_idx:]
        scored_downs = down_waves[start_u_idx + 1:]  # 跳过 d0、d2（第一个不评分），从 d4 开始评分对比

        # 波段索引映射（scored_index -> wave_index）
        up_to_wi = {}
        ui = 0
        for wi, w in enumerate(waves):
            if w.direction == 'up':
                if ui >= start_u_idx:
                    up_to_wi[ui - start_u_idx] = wi
                ui += 1

        # 下跌标注：d2 不评分只标注，d4/d6/... 从 scored_downs 中取
        down_to_wi = {}
        di = 0
        for wi, w in enumerate(waves):
            if w.direction == 'down':
                down_to_wi[di] = wi
                di += 1

        # 生成标注文本
        annotations = {}

        def _up_label(si):
            return f'u{si*2+1}'
        def _down_label(si):
            return f'd{si*2+2}'

        # ── 构建 display_start 之后波段的全局标注 ───────────────────────
        # display_idx=0→u1, display_idx=1→d2, display_idx=2→u3, ...
        display_up_count = 0   # u1/u3/u5 的枚举
        display_down_count = 0 # d2/d4/d6 的枚举
        scored_up_idx = 0      # 在 scored_ups 中的位置
        scored_down_idx = 0     # 在 scored_downs 中的位置

        for wi, w in enumerate(waves):
            if wi < display_start:
                continue
            if w.direction == 'up':
                lbl = _up_label(display_up_count)
                display_up_count += 1
                if scored_up_idx < len(scored_ups) and scored_ups[scored_up_idx] is w:
                    si = scored_up_idx
                    if si == 0:
                        annotations[wi] = f'{lbl}:'
                    elif si == 1:
                        prev = scored_ups[0]
                        annotations[wi] = f'{lbl}: {lbl}({w.wave_high:.2f}) > {_up_label(0)}({prev.wave_high:.2f}) → +2'
                    else:
                        prev = scored_ups[si-1]
                        max_prior = max(s.wave_high for s in scored_ups[:si])
                        if w.wave_high > prev.wave_high and w.wave_high > max_prior:
                            annotations[wi] = f'{lbl}: {lbl}({w.wave_high:.2f}) > {_up_label(si-1)}({prev.wave_high:.2f}) → +8 (创历史新高)'
                        elif w.wave_high > prev.wave_high:
                            annotations[wi] = f'{lbl}: {lbl}({w.wave_high:.2f}) > {_up_label(si-1)}({prev.wave_high:.2f}) → +1'
                        else:
                            annotations[wi] = f'{lbl}: {lbl}({w.wave_high:.2f}) < {_up_label(si-1)}({prev.wave_high:.2f}) → -3'
                    scored_up_idx += 1
                else:
                    annotations[wi] = f'{lbl}:'
            else:  # down
                lbl = _down_label(display_down_count)
                display_down_count += 1
                if scored_down_idx < len(scored_downs) and scored_downs[scored_down_idx] is w:
                    # 有评分的下跌波段（d4+）
                    # 找前一个同向波段（上一个下跌波段）用于对比
                    di_global = display_down_count - 1  # 当前下跌在显示序列中的 di
                    prev_wi = None
                    for pw_i in range(wi - 1, display_start - 1, -1):
                        if waves[pw_i].direction == 'down':
                            prev_wi = pw_i
                            break
                    if prev_wi is not None:
                        prev_w = waves[prev_wi]
                        prev_lbl = _down_label(display_down_count - 2)  # 前一个下跌的标签
                        if w.wave_low < prev_w.wave_low:
                            annotations[wi] = f'{lbl}: {lbl}({w.wave_low:.2f}) < {prev_lbl}({prev_w.wave_low:.2f}) → -1'
                        else:
                            annotations[wi] = f'{lbl}: {lbl}({w.wave_low:.2f}) >= {prev_lbl}({prev_w.wave_low:.2f}) → +0'
                    else:
                        annotations[wi] = f'{lbl}:'
                    scored_down_idx += 1
                else:
                    # d2 等不参与评分的下跌波段
                    annotations[wi] = f'{lbl}:'

        # ── 输出每行 ─────────────────────────────────────
        for wi, w in enumerate(waves):
            if wi < display_start:
                continue
            tag = annotations.get(wi, '')
            dir_icon = '↑' if w.direction == 'up' else '↓'
            start_date = str(df['date'].iloc[w.start])[:10]
            end_date = str(df['date'].iloc[w.end])[:10]
            lines.append(
                f'  {dir_icon} {start_date}～{end_date}  '
                f'({w.days:2d}天) '
                f'量={w.avg_volume:>10.0f}  '
                f'爆={w.volume_power:.1f}x  '
                f'涨跌={w.pct:+.2f}%  '
                f'{tag}'
            )

        lines.append('')
        lines.append(f'  📊 波段质量评分（共 {score_wave_quality(waves):+.1f} 分）')

        # ── 综合评分明细 ──────────────────────────────────
        ind_for_detail = compute_all(df, ma10_break_window=cfg.max_broke_ma10_days)
        if ind_for_detail:
            d = score_detail(ind_for_detail)
            wq = score_wave_quality(waves)
            lines.append('')
            lines.append(f'  📊 综合评分明细（总分={r["score"]:.1f}）：')
            lines.append(f'    DIF强度       {d["dif_score"]:>5.1f} / 25')
            lines.append(f'    红柱新鲜度   {d["red_score"]:>5.1f} / 20')
            lines.append(f'    量能质量     {d["turnover_score"]:>5.1f} /  8（换手）')
            lines.append(f'    量比         {d["volume_score"]:>5.1f} /  5')
            lines.append(f'    波段结构     {d["vol_structure_score"]:>5.1f} /  8')
            lines.append(f'    爆发力       {d["vol_burst_score"]:>5.1f} /  4')
            lines.append(f'    均线质量     {d["ma_score"]:>5.1f} / 15')
            lines.append(f'    回调支撑     {d["support_score"]:>5.1f} /  5')
            lines.append(f'    整理模式     {d["consolidation_score"]:>5.1f} / 10')
            lines.append(f'    波段质量     {wq:>+6.1f}')

        return lines

    # ─────────────────────────────────────────
    # 输出结果
    # ─────────────────────────────────────────
    print(f"\n{'='*120}")
    print(f"📊 复盘选股 {date_str}（通过 {len(passed_results)} 只）")
    print("=" * 120)

    header = _header_row()
    print(header)
    print("-" * 120)

    lines = [f"📊 复盘选股 {date_str}（通过 {len(passed_results)} 只）", "=" * 120, header, "-" * 120]

    for r in passed_results:
        ma5_dist = r.get('ma5_distance_pct', 0.0)
        wave_ratio = r.get('wave_up_vs_down_ratio', 0.0)
        sl_ref = r.get('stop_loss_ref')
        row = _make_row([
            r['code'], r['name'], date_str,
            f"{r['score']:.1f}", r['red_days'],
            f"{r['close']:.2f}",
            f"{r['gain3']:+.1f}%", f"{r['turnover_est']:.1f}%",
            f"{r['vol_ratio']:.2f}", f"{wave_ratio:.2f}",
            f"{ma5_dist:+.1f}%", f"{r['rsi']:.1f}",
            f"{r['ma20']:.2f}", f"{r['ma60']:.2f}",
        ])
        print(row)
        lines.append(row)

    print("=" * 120)
    lines.append("=" * 120)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n💾 结果已写入: {output_path}")

    # ─────────────────────────────────────────
    # 拒绝原因（当 --reason 参数时）
    # ─────────────────────────────────────────
    if args.reason:
        failed_results = [r for r in results if r.get("failed", False)]
        if failed_results:
            print(f"\n{'='*80}")
            print(f"❌ 未通过筛选的股票（{len(failed_results)} 只）")
            print(f"{'='*80}")
            # 只显示前 50 只
            for r in failed_results[:50]:
                print(f"  {r['code']} {r['name']:<8} ✗ {r['reason']}")
            if len(failed_results) > 50:
                print(f"  ... 还有 {len(failed_results) - 50} 只未显示")

    # ─────────────────────────────────────────
    # 波段详情（当 --waves 参数时）
    # ─────────────────────────────────────────
    if args.waves:
        print(f"\n\n{'='*80}")
        print(f"📈 完整涨跌波段详情（近60日，仅Top10）")
        print(f"{'='*80}")

        wave_items = passed_results[:10] if passed_results else []
        
        # --waves --code 模式：即使不通过也显示
        if args.codes and not wave_items:
            from stock_trend.review_screen.data_cache import load_qfq_history
            from stock_trend.review_screen.indicators import detect_volume_price_wave
            for code_str in args.codes:
                c = normalize_code(code_str)
                df, _ = _load_df_for_waves(code_str, target_date)
                if df is not None:
                    # 构造一个最小结果字典
                    close_arr = df['close'].values
                    # 找几条关键信息
                    name = load_stock_names().get(c or code_str, c or code_str)
                    fake_r = {
                        'code': c or code_str,
                        'name': name,
                        'score': 0.0,
                        'red_days': 0,
                    }
                    wave_lines = _format_wave_analysis(fake_r, df, c or code_str)
                    for wl in wave_lines:
                        print(wl)

        for r in wave_items:
            df, c = _load_df_for_waves(r['code'], target_date)
            if df is not None:
                wave_lines = _format_wave_analysis(r, df, c)
                for wl in wave_lines:
                    print(wl)

    # Top10摘要
    print(f"\n🏆 Top10（评分/3日涨幅/换手率/波段量比/止损参考）：")
    for i, r in enumerate(passed_results[:10], 1):
        sl = f"{r['stop_loss_ref']:.2f}" if r.get('stop_loss_ref') else "N/A"
        wave_ratio = r.get('wave_up_vs_down_ratio', 0.0)
        wave_quality = r.get('wave_quality_score', 0.0)
        wave_dir = r.get('wave_last_dir', 'N/A')
        ma5_d = r.get('ma5_distance_pct', 0.0)
        strong = r.get('up_stronger_than_down', False)
        main_trend = r.get('is_main_trend', False)
        second_break = r.get('is_second_break', False)
        structure_reason = r.get('structure_reason', '')
        strong_mark = '🔥' if strong else '⚠️'
        # 主升浪+二级启动挂额外标签
        extra = ''
        if main_trend:
            extra += '🏆'
        if second_break:
            extra += '🔄'
        print(f"  {i:2d}. {r['code']} {r['name']:<6} "
              f"评分{r['score']:>5.1f}  "
              f"3日{r['gain3']:>+6.2f}%  "
              f"换手{r['turnover_est']:.1f}%  "
              f"波量比{wave_ratio:.2f}({wave_dir})  "
              f"波评分{wave_quality:+.1f}  "
              f"MA5距{ma5_d:>+5.1f}%  "
              f"{strong_mark}{extra}  "
              f"{structure_reason}  "
              f"[{r.get('phase', '不明')}]  "
              f"止损{sl}")

    # ─────────────────────────────────────────
    # 买入精选（当 --buy 参数时）
    # ─────────────────────────────────────────
    if args.buy and passed_results:
        BUY_RSI_MAX = 72.0          # RSI 超过此值 = 过热，不选
        BUY_GAIN5_MAX = 15.0       # 5日涨幅超过此值 = 追高，不选
        BUY_STOP_LOSS_MAX = 5.0     # 止损空间超过此值（%），不选

        buy_candidates = []
        for r in passed_results:
            rsi = r.get('rsi', 50)
            gain5 = r.get('gain5', 0)
            close = r.get('close', 0)
            sl_ref = r.get('stop_loss_ref')

            # RSI 过热过滤
            if rsi >= BUY_RSI_MAX:
                continue
            # 5日涨幅过大过滤
            if gain5 >= BUY_GAIN5_MAX:
                continue
            # 止损空间计算
            if sl_ref is not None and close > 0:
                stop_pct = (close - sl_ref) / close * 100
                if stop_pct <= 0 or stop_pct > BUY_STOP_LOSS_MAX:
                    continue
            else:
                continue  # 无止损参考则跳过

            buy_candidates.append(r)

        buy_candidates.sort(key=lambda x: x['score'], reverse=True)

        print(f"\n{'='*120}")
        print(f"🟢 买入精选（RSI<{BUY_RSI_MAX:.0f} + 5日涨幅<{BUY_GAIN5_MAX:.0f}% + 止损空间<{BUY_STOP_LOSS_MAX:.0f}%，共 {len(buy_candidates)} 只）")
        print(f"{'='*120}")
        print(f"{'代码':<10} {'名称':<8} {'评分':>6} {'5日%':>7} {'RSI':>5} {'换手%':>6} {'止损位':>8} {'止损%':>6} {'阶段':<8}")
        print('-' * 120)

        lines_buy = [f"🟢 买入精选（RSI<{BUY_RSI_MAX:.0f} + 5日涨幅<{BUY_GAIN5_MAX:.0f}% + 止损空间<{BUY_STOP_LOSS_MAX:.0f}%，共 {len(buy_candidates)} 只）",
                     "=" * 120,
                     f"{'代码':<10} {'名称':<8} {'评分':>6} {'5日%':>7} {'RSI':>5} {'换手%':>6} {'止损位':>8} {'止损%':>6} {'阶段':<8}",
                     "-" * 120]

        for r in buy_candidates:
            sl_ref = r.get('stop_loss_ref')
            close = r.get('close', 0)
            stop_pct = (close - sl_ref) / close * 100 if sl_ref and close > 0 else 0
            sl_str = f"{sl_ref:.2f}" if sl_ref else "N/A"
            phase = r.get('phase', '不明')
            print(f"{r['code']:<10} {r['name']:<8} {r['score']:>6.1f} {r['gain5']:>+7.1f}% {r['rsi']:>5.1f} {r['turnover_est']:>6.1f}% {sl_str:>8} {stop_pct:>5.1f}%  {phase}")
            lines_buy.append(f"{r['code']:<10} {r['name']:<8} {r['score']:>6.1f} {r['gain5']:>+7.1f}% {r['rsi']:>5.1f} {r['turnover_est']:>6.1f}% {sl_str:>8} {stop_pct:>5.1f}%  {phase}")

        print('=' * 120)
        lines_buy.append('=' * 120)

        # 写入同一文件（追加）
        with open(output_path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines_buy) + "\n")

        print(f"\n💾 买入精选已追加写入: {output_path}")
