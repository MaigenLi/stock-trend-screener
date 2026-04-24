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
    """加载股票名称映射（兼容嵌套stocks结构）"""
    names_file = Path.home() / "stock_code" / "results" / "all_stock_names_final.json"
    if not names_file.exists():
        return {}
    import json
    try:
        data = json.loads(names_file.read_text(encoding="utf-8"))
        stocks = data.get("stocks", {}) if isinstance(data, dict) else {}
        names = {}
        for code, info in stocks.items():
            if not isinstance(info, dict):
                continue
            name = info.get("name", "未知")
            names[code.lower()] = name
            pure = info.get("code", "")
            if pure:
                names[pure.lower()] = name
        return names
    except Exception:
        return {}


def get_stock_name(code: str, names: dict) -> str:
    """获取股票名称"""
    return names.get(code, names.get(code[2:], ""))


# ─────────────────────────────────────────
# 单股评估
# ─────────────────────────────────────────

def evaluate_stock(code: str, target_date: datetime | None, cfg: FilterConfig, names: dict) -> dict | None:
    """
    评估单只股票

    Args:
        code: 股票代码
        target_date: 信号日期
        cfg: 筛选配置
        names: 股票名称映射（由 scan_market 加载一次后传入）

    Returns:
        完整结果字典 或 None（不通过筛选）
    """
    c = normalize_code(code)
    end_str = target_date.strftime("%Y-%m-%d") if target_date else None

    df = load_qfq_history(c, end_date=end_str, refresh=False)
    if df is None or df.empty:
        return None

    # 按日期截取
    if target_date is not None:
        df = df[df["date"] <= pd.Timestamp(target_date.date())].reset_index(drop=True)
    if len(df) < cfg.min_bars:
        return None

    # 计算指标（使用配置中的窗口参数）
    ind = compute_all(df, ma10_break_window=cfg.max_broke_ma10_days)
    if not ind:
        return None

    # 筛选
    passed, reason = check_filters(ind, cfg)
    if not passed:
        return None

    # 评分
    total_score = score_stock(ind)
    ind["wave_quality_score"] = score_wave_quality(ind.get("waves", []))
    ind["code"] = c
    ind["name"] = get_stock_name(c, names)
    ind["score"] = total_score
    ind["_reason"] = reason

    # 阶段标签
    phase, phase_reason = classify_phase(ind)
    ind["phase"] = phase
    ind["phase_reason"] = phase_reason

    return ind


# ─────────────────────────────────────────
# 全市场扫描
# ─────────────────────────────────────────

def scan_market(
    codes: list,
    target_date: datetime | None,
    max_workers: int = DEFAULT_WORKERS,
    cfg: FilterConfig | None = None,
) -> list[dict]:
    """多线程扫描全市场"""
    if cfg is None:
        cfg = FilterConfig()
    results = []
    t0 = time.time()
    total = len(codes)

    # 名称映射只加载一次（避免每只股票重复读文件）
    names = load_stock_names()

    def work(code: str) -> dict | None:
        return evaluate_stock(code, target_date, cfg, names)

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
    args = parser.parse_args()

    # 解析日期
    try:
        target_date = datetime.strptime(args.date, "%Y-%m-%d")
    except ValueError:
        print(f"❌ 日期格式错误: {args.date}，应为 YYYY-MM-DD")
        sys.exit(1)

    print(f"\n📅 复盘选股: {args.date}")
    date_str = args.date

    # 输出路径
    output_path = Path(args.output) if args.output else Path.home() / "stock_reports" / f"review_screen_{date_str}.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 股票范围
    if args.codes:
        codes = args.codes
        print(f"\n📊 筛选（指定 {len(codes)} 只）")
    else:
        codes = preload_all_codes()
        print(f"\n📊 筛选（全市场 {len(codes)} 只）")

    # 构建筛选配置
    cfg = FilterConfig(
        require_latest_wave_down=args.latest_wave_down,
    )

    # 扫描
    results = scan_market(
        codes=codes,
        target_date=target_date,
        max_workers=args.workers,
        cfg=cfg,
    )

    if not results:
        print("\n⚠️  无符合筛选条件的股票")
        sys.exit(0)

    # 按评分降序
    results.sort(key=lambda x: x["score"], reverse=True)

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

        for si, su in enumerate(scored_ups):
            lbl_curr = _up_label(si)
            wi = up_to_wi[si]
            if si == 0:
                annotations[wi] = f'{lbl_curr}:'
            elif si == 1:
                prev = scored_ups[0]
                annotations[wi] = f'{lbl_curr}: {lbl_curr}({su.wave_high:.2f}) > {_up_label(0)}({prev.wave_high:.2f}) → +2'
            else:
                prev = scored_ups[si-1]
                max_prior = max(s.wave_high for s in scored_ups[:si])
                if su.wave_high > prev.wave_high and su.wave_high > max_prior:
                    annotations[wi] = f'{lbl_curr}: {lbl_curr}({su.wave_high:.2f}) > {_up_label(si-1)}({prev.wave_high:.2f}) → +8 (创历史新高)'
                elif su.wave_high > prev.wave_high:
                    annotations[wi] = f'{lbl_curr}: {lbl_curr}({su.wave_high:.2f}) > {_up_label(si-1)}({prev.wave_high:.2f}) → +1'
                elif su.wave_high < prev.wave_high:
                    annotations[wi] = f'{lbl_curr}: {lbl_curr}({su.wave_high:.2f}) < {_up_label(si-1)}({prev.wave_high:.2f}) → -3'

        # d2 标注（不评分，只标）
        d2_idx = down_to_wi.get(start_u_idx)  # start_u_idx=1 对应 downs[1] = d2
        if d2_idx is not None:
            annotations[d2_idx] = 'd2:'

        # d4+ 评分（从 scored_downs 取）
        for si, sd in enumerate(scored_downs):
            lbl_curr = _down_label(si + 1)  # scored_downs[0] 对应 d4
            # 在 waves 中找到这个下跌段
            d_idx = start_u_idx + 1 + si
            wi = down_to_wi.get(d_idx)
            if wi is None:
                continue
            prev_idx = d_idx - 1
            prev_wi = down_to_wi.get(prev_idx)
            if prev_wi is None:
                continue
            prev_w = waves[prev_wi]
            prev_low = prev_w.wave_low
            if sd.wave_low < prev_low:
                annotations[wi] = f'{lbl_curr}: {lbl_curr}({sd.wave_low:.2f}) < {_down_label(si)}({prev_low:.2f}) → -1'
            else:
                annotations[wi] = f'{lbl_curr}: {lbl_curr}({sd.wave_low:.2f}) >= {_down_label(si)}({prev_low:.2f}) → +0'

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
    print(f"📊 复盘选股 {date_str}（共 {len(results)} 只）")
    print("=" * 120)

    header = _header_row()
    print(header)
    print("-" * 120)

    lines = [f"📊 复盘选股 {date_str}（共 {len(results)} 只）", "=" * 120, header, "-" * 120]

    for r in results:
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
    # 波段详情（当 --waves 参数时）
    # ─────────────────────────────────────────
    if args.waves:
        print(f"\n\n{'='*80}")
        print(f"📈 完整涨跌波段详情（近60日，仅Top10）")
        print(f"{'='*80}")

        for i, r in enumerate(results[:10]):
            df, c = _load_df_for_waves(r['code'], target_date)
            if df is not None:
                wave_lines = _format_wave_analysis(r, df, c)
                for wl in wave_lines:
                    print(wl)

    # Top10摘要
    print(f"\n🏆 Top10（评分/3日涨幅/换手率/波段量比/止损参考）：")
    for i, r in enumerate(results[:10], 1):
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
