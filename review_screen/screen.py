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
from stock_trend.review_screen.scorer import score_stock, score_wave_quality, score_detail

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

def evaluate_stock(code: str, target_date: datetime | None, cfg: FilterConfig) -> dict | None:
    """
    评估单只股票

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

    # 计算指标
    ind = compute_all(df)
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
    ind["name"] = get_stock_name(c, load_stock_names())
    ind["score"] = total_score
    ind["_reason"] = reason

    return ind


# ─────────────────────────────────────────
# 全市场扫描
# ─────────────────────────────────────────

def scan_market(
    codes: list,
    target_date: datetime | None,
    max_workers: int = DEFAULT_WORKERS,
) -> list[dict]:
    """多线程扫描全市场"""
    cfg = FilterConfig()
    results = []
    t0 = time.time()
    total = len(codes)

    def work(code: str) -> dict | None:
        return evaluate_stock(code, target_date, cfg)

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

    # 扫描
    results = scan_market(
        codes=codes,
        target_date=target_date,
        max_workers=args.workers,
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
        lookback = min(60, len(close) - 1)
        result = detect_volume_price_wave(close, volume, lookback=lookback)
        waves = result.get("waves", [])
        if not waves:
            return []

        lines = []
        lines.append(f"\n{'─'*60}")
        lines.append(f"  {r['code']} {r['name']} 评分={r['score']}  红柱={r['red_days']}天  信号日={date_str}")
        lines.append(f"{'─'*60}")

        # 近60日波段列表（从老到新）
        lines.append(f"  近{lookback}日涨跌波段（共{len(waves)}个）：")
        lines.append("")

        for j, w in enumerate(waves):
            start_date = str(df["date"].iloc[w["start_idx"]])[:10]
            end_date = str(df["date"].iloc[w["end_idx"]])[:10]
            dir_icon = "↑" if w["direction"] == "up" else "↓"
            price_str = f"{w['price_change']:+.2f}%"

            # 判断健康状态
            if w["direction"] == "up":
                # 找前一个跌段做对比
                if j > 0:
                    prev_w = waves[j - 1]
                    if prev_w["direction"] == "down":
                        vol_ratio = w["avg_volume"] / max(prev_w["avg_volume"], 1)
                        if vol_ratio > 1.5:
                            status = "✅ 放量上涨"
                        elif vol_ratio > 1.2:
                            status = "✅ 温和放量"
                        elif vol_ratio > 1.0:
                            status = "⚠️ 持平"
                        else:
                            status = "❌ 缩量"
                    else:
                        status = "✅"
                else:
                    status = "✅"
            else:
                # 跌段：找前一个涨段做对比
                if j > 0:
                    prev_w = waves[j - 1]
                    if prev_w["direction"] == "up":
                        vol_ratio = prev_w["avg_volume"] / max(w["avg_volume"], 1)
                        if vol_ratio > 1.5:
                            status = "✅ 缩量调整"
                        elif vol_ratio > 1.2:
                            status = "⚠️ 轻微缩量"
                        elif vol_ratio > 1.0:
                            status = "⚠️ 量能持平"
                        else:
                            status = "❌ 放量下跌"
                    else:
                        status = "⚠️"
                else:
                    status = "⚠️"

            lines.append(
                f"  {dir_icon} {start_date}～{end_date}  "
                f"({w['len']:2d}天) "
                f"均量={w['avg_volume']:>10.0f}  "
                f"涨跌={price_str:>8}  {status}"
            )

        # 关键比对结论
        up_waves = [w for w in waves if w["direction"] == "up"]
        down_waves = [w for w in waves if w["direction"] == "down"]

        lines.append("")

        # 波段质量评分详细分解
        wq_total = score_wave_quality(waves)
        wq_lines = []
        for i in range(1, len(up_waves)):
            curr_h = up_waves[i]["wave_high"]
            prev_h = up_waves[i - 1]["wave_high"]
            max_prior = max(up_waves[k]["wave_high"] for k in range(i))
            lbl_curr = f"u{2*i+1}"
            lbl_prev = f"u{2*i-1}"
            if curr_h > prev_h:
                if i == 1:
                    wq_lines.append(f"      {lbl_curr}({curr_h:.2f}) > {lbl_prev}({prev_h:.2f}) → +2")
                else:
                    max_prior = max(up_waves[k]["wave_high"] for k in range(i))
                    if curr_h > max_prior:
                        wq_lines.append(f"      {lbl_curr}({curr_h:.2f}) > {lbl_prev}({prev_h:.2f}) → +8 (创历史新高)")
                    else:
                        wq_lines.append(f"      {lbl_curr}({curr_h:.2f}) > {lbl_prev}({prev_h:.2f}) → +1")
            elif curr_h < prev_h:
                wq_lines.append(f"      {lbl_curr}({curr_h:.2f}) < {lbl_prev}({prev_h:.2f}) → -3")
            else:
                wq_lines.append(f"      {lbl_curr}({curr_h:.2f}) < {lbl_prev}({prev_h:.2f}) → 0")
        for i in range(1, len(down_waves)):
            curr_lo = down_waves[i]["wave_low"]
            prev_lo = down_waves[i - 1]["wave_low"]
            lbl_curr = f"d{2*i+2}"
            lbl_prev = f"d{2*i}"
            if curr_lo < prev_lo:
                wq_lines.append(f"      {lbl_curr}({curr_lo:.2f}) < {lbl_prev}({prev_lo:.2f}) → -1")
            else:
                wq_lines.append(f"      {lbl_curr}({curr_lo:.2f}) >= {lbl_prev}({prev_lo:.2f}) → +0")
        lines.append("")
        lines.append(f"  📊 波段质量评分（共 {wq_total:+.1f} 分）：")
        for wl in wq_lines:
            lines.append(wl)
        lines.append("")
        if up_waves and down_waves:
            last_up = up_waves[-1]
            last_down = down_waves[-1]
            ratio = last_up["avg_volume"] / max(last_down["avg_volume"], 1)
            all_up_avg = float(np.mean([w["avg_volume"] for w in up_waves]))
            all_down_avg = float(np.mean([w["avg_volume"] for w in down_waves]))
            all_ratio = all_up_avg / max(all_down_avg, 1)

            lines.append(f"  📊 波段量比分析：")
            lines.append(f"    最近涨段({last_up['len']}天)均量={last_up['avg_volume']:.0f} vs "
                        f"最近跌段({last_down['len']}天)均量={last_down['avg_volume']:.0f} → "
                        f"比值={ratio:.2f} {'✅' if ratio > 1.2 else '❌'}")
            lines.append(f"    全部涨段均量={all_up_avg:.0f} vs 全部跌段均量={all_down_avg:.0f} → "
                        f"比值={all_ratio:.2f} {'✅' if all_ratio > 1.0 else '❌'}")
            lines.append(f"    波段模式评分: {result['pattern_score']} {'✅' if result['pattern_score'] >= 0.4 else '❌'}")

        # ── 综合评分明细 ──────────────────────────────────
        # 重建完整指标以获取评分明细
        ind_for_detail = compute_all(df)
        if ind_for_detail:
            d = score_detail(ind_for_detail)
            wq = r.get("wave_quality_score", 0.0)
            lines.append("")
            lines.append(f"  📊 综合评分明细（总分={r['score']:.1f}）：")
            lines.append(f"    DIF强度       {d['dif_score']:>5.1f} / 25")
            lines.append(f"    红柱新鲜度   {d['red_score']:>5.1f} / 20")
            lines.append(f"    量能质量     {d['turnover_score']:>5.1f} /  8（换手）")
            lines.append(f"    量比         {d['volume_score']:>5.1f} /  5")
            lines.append(f"    波段结构     {d['vol_structure_score']:>5.1f} /  8")
            lines.append(f"    爆发力       {d['vol_burst_score']:>5.1f} /  4")
            lines.append(f"    均线质量     {d['ma_score']:>5.1f} / 15")
            lines.append(f"    回调支撑     {d['support_score']:>5.1f} /  5")
            lines.append(f"    整理模式     {d['consolidation_score']:>5.1f} / 10")
            lines.append(f"    波段质量     {wq:>+6.1f}")

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
        print(f"📈 完整涨跌波段详情（近60日）")
        print(f"{'='*80}")

        # 显示所有通过的结果（太多时限制前30只）
        show_count = min(len(results), 30)
        for i, r in enumerate(results[:show_count]):
            df, c = _load_df_for_waves(r['code'], target_date)
            if df is not None:
                wave_lines = _format_wave_analysis(r, df, c)
                for wl in wave_lines:
                    print(wl)

        if len(results) > show_count:
            print(f"\n...（还有{len(results) - show_count}只，显示上限30只）")

    # Top10摘要
    print(f"\n🏆 Top10（评分/3日涨幅/换手率/波段量比/止损参考）：")
    for i, r in enumerate(results[:10], 1):
        sl = f"{r['stop_loss_ref']:.2f}" if r.get('stop_loss_ref') else "N/A"
        wave_ratio = r.get('wave_up_vs_down_ratio', 0.0)
        wave_quality = r.get('wave_quality_score', 0.0)
        wave_dir = r.get('wave_last_dir', 'N/A')
        ma5_d = r.get('ma5_distance_pct', 0.0)
        print(f"  {i:2d}. {r['code']} {r['name']:<6} "
              f"评分{r['score']:>5.1f}  "
              f"3日{r['gain3']:>+6.2f}%  "
              f"换手{r['turnover_est']:.1f}%  "
              f"波量比{wave_ratio:.2f}({wave_dir})  "
              f"波评分{wave_quality:+.1f}  "
              f"MA5距{ma5_d:>+5.1f}%  "
              f"止损{sl}")
