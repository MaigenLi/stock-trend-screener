#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
合并选股脚本 —— real_screen + triple_screen 对比输出

逻辑：
  第一步：real_screen 逻辑 → 集合 A
  第二步：triple_screen 逻辑 → 集合 B
  第三步：A 写入输出文件
  第四步：A ∩ B 交集写入文件
  第五步：A \\ B（A独有）写入文件
  第六步：B \\ A（B独有）写入文件

输出格式与 triple_screen 完全一致（启动型 / 趋势型分类）
输出文件：~/stock_reports/triple_screen_{日期}.txt（覆盖原 triple_screen 输出）
"""

import sys, json
from pathlib import Path
from datetime import datetime, timedelta

WORKSPACE = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(WORKSPACE))

import rps_strong_screen as rps
import trend_strong_screen as tss
import gain_turnover as gt
from stock_trend.gain_turnover_screen import screen_market
from stock_trend.gain_turnover import _lpad, _rpad

# ─── 辅助函数 ──────────────────────────────────────────────

def get_prev_trading_day(target_date: datetime) -> str:
    """获取前一个交易日（跳过周末）"""
    for delta in [1, 2, 3]:
        cand = target_date - timedelta(days=delta)
        if cand.weekday() < 5:
            return cand.strftime("%Y-%m-%d")
    return (target_date - timedelta(days=1)).strftime("%Y-%m-%d")


def load_triple_state():
    """读取 triple_screen 的昨日 state"""
    p = Path.home() / "stock_reports" / "triple_screen_state.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def load_real_state():
    """读取 real_screen 的昨日 state"""
    p = Path.home() / "stock_reports" / "real_trading_state.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def save_real_state(state):
    p = Path.home() / "stock_reports" / "real_trading_state.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def get_trade_dates(start: str, end: str) -> list:
    """获取交易日列表"""
    df = tss.get_index_kline("sh000001")
    if df is None or df.empty:
        return []
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df[(df["date"] >= start) & (df["date"] <= end)]
    return sorted(df["date"].tolist())


import pandas as pd


# ─── Step1：RPS 扫描（real_screen 逻辑）────────────────────

def step1_real(all_codes, target_date, workers=8):
    """real_screen 的 Step1：加权评分取前60%"""
    df = rps.scan_rps(all_codes, top_n=len(all_codes),
                       max_workers=workers, target_date=target_date)
    scores = []
    W_RPS = 0.4; W_RET = 0.2; W_RSI = 0.2; W_TURNOVER = 0.2
    for _, row in df.iterrows():
        s = 0
        s += row.get("composite", 0) * W_RPS
        ret20 = row.get("ret20", 0)
        if 5 < ret20 < 40:
            s += 100 * W_RET
        rsi = row.get("rsi", 50)
        if 50 < rsi < 75:
            s += 100 * W_RSI
        turnover = row.get("avg_turnover_5", 0)
        if turnover > 2:
            s += 100 * W_TURNOVER
        scores.append(s)
    df = df.copy()
    df["score"] = scores
    top_codes = df[df["score"] > df["score"].quantile(0.6)]["code"].tolist()
    rps_map = {str(row["code"]).lower(): row.get("composite", 0)
               for _, row in df.iterrows()}
    return top_codes, rps_map, df


# ─── Step1：RPS 扫描（triple_screen 逻辑）────────────────

def step1_triple(all_codes, target_date, workers=8,
                 rps_composite=75.0, rsi_low=50.0, rsi_high=80.0,
                 rps20_min=75.0, max_ret20=40.0, max_ret5=30.0,
                 ret3_min=3.0, min_turnover=2.0):
    """triple_screen 的 Step1：硬指标过滤"""
    df = rps.scan_rps(all_codes, top_n=len(all_codes),
                       max_workers=workers, target_date=target_date)
    if "data_date" in df.columns:
        ts = target_date.strftime("%Y-%m-%d")
        df = df[df["data_date"] == ts]
    df_f = df[
        (df["composite"]       >= rps_composite) &
        (df["ret20_rps"]       >= rps20_min) &
        (df["rsi"]             >= rsi_low) &
        (df["rsi"]             <= rsi_high) &
        (df["ret20"]           <= max_ret20) &
        (df["ret20"]           >= -10) &
        (df["ret5"]            <= max_ret5) &
        (df["ret3"]            >= ret3_min) &
        (df["avg_turnover_5"]  >= min_turnover)
    ].copy()
    return df_f, df


# ─── Step2：趋势扫描（两者共用 tss.scan_market）───────────

def step2(codes, target_date, score_threshold=30, workers=8):
    """trend 扫描，返回 (codes_list, results_raw)"""
    raw = tss.scan_market(codes=codes, top_n=len(codes),
                           score_threshold=score_threshold,
                           max_workers=workers, target_date=target_date)
    valid = [r[0] for r in raw if isinstance(r, tuple) and len(r) >= 4]
    return valid, raw


# ─── Step3：gain_turnover（两者共用 gain_turnover_screen）─

def step3(codes, target_date, config, workers=8):
    """gain_turnover Step3"""
    return screen_market(codes=codes, config=config,
                         target_date=target_date, top_n=len(codes),
                         max_workers=workers, refresh_cache=False)


# ─── 风控（real_screen 版本）──────────────────────────────

def risk_control_real(results, prev_state, target_date):
    """real_screen 风控：连号≥3 / RSI连档≥3 过滤"""
    today_str = target_date.strftime("%Y-%m-%d")
    yesterday_str = get_prev_trading_day(target_date)

    final = []
    for r in results:
        code = r.code.lower()
        prev = prev_state.get(code, {"consec": 0, "rsi_high": 0, "last_date": ""})

        last_date = prev.get("last_date", "")
        is_new = (last_date != yesterday_str)

        is_high_rsi = r.rsi14 > 72
        prev_rsi = prev.get("rsi_high", 0)

        if is_new:
            consec = 1
            rsi_high = 1 if is_high_rsi else 0
        else:
            consec = prev.get("consec", 0) + 1
            rsi_high = (prev_rsi + 1) if (is_high_rsi and prev_rsi > 0) else (1 if is_high_rsi else 0)

        if consec >= 3:
            continue
        if rsi_high >= 3:
            continue

        prev_state[code] = {"consec": consec, "rsi_high": rsi_high, "last_date": today_str}
        final.append(r)

    return final, prev_state


# ─── 分类：启动型 / 趋势型 ─────────────────────────────────

def classify(results):
    startup     = [r for r in results if r.total_gain_window > 10 and r.avg_turnover_5 > 3]
    trend_follow = [r for r in results if not (r.total_gain_window > 10 and r.avg_turnover_5 > 3)]
    return startup, trend_follow


# ─── 输出格式化（与 triple_screen.save_and_print 一致）────

def make_lines(results, step1_df, step2_raw, yesterday_state,
               section_tag, date_str, inter_set=None):
    if inter_set is None:
        inter_set = set()
    """生成与 triple_screen 完全一致的表格行"""
    if not results:
        return []

    rps_dict  = {str(row["code"]).lower(): row for _, row in step1_df.iterrows()}
    trend_dict = {str(r[0]).lower(): r for r in step2_raw if isinstance(r, tuple)}

    # 兼容两种 state 格式：
    #   triple_screen: {date: {"consec": {code: val}, "rsi_high": {code: val}}}
    #   real_screen:  {code: {"consec": val, "rsi_high": val, "last_date": "..."}}
    yesterday_consec_all = {}
    yesterday_rsi_high_all = {}
    sample_val = None
    for d, v in yesterday_state.items():
        if isinstance(v, dict):
            sample_val = v
            break
    if sample_val is not None:
        if "consec" in sample_val and isinstance(sample_val["consec"], dict):
            # triple_screen 格式：{"consec": {code: val}, "rsi_high": {code: val}}
            yesterday_consec_all = sample_val.get("consec", {})
            yesterday_rsi_high_all = sample_val.get("rsi_high", {})
        else:
            # real_screen 格式：{code: {consec: val, rsi_high: val}}
            for code, v in yesterday_state.items():
                if isinstance(v, dict):
                    yesterday_consec_all[code] = v.get("consec", 0)
                    yesterday_rsi_high_all[code] = v.get("rsi_high", 0)

    def composite_score(r):
        info  = rps_dict.get(r.code.lower(), {})
        t_inf = trend_dict.get(r.code.lower(), (None,)*5)
        rps_c   = info.get("composite", 0.0) if isinstance(info, dict) else 0.0
        trend_s = float(t_inf[2]) if t_inf[2] is not None else 0.0
        prev_c  = yesterday_consec_all.get(r.code.lower(), 0)
        consec_today = prev_c + 1 if prev_c > 0 else 0
        penalty = 0.8 if consec_today >= 3 else 1.0
        return (r.score * 0.4 + rps_c * 0.3 + trend_s * 0.3) * penalty

    results = sorted(results, key=composite_score, reverse=True)

    lines = []
    lines.append("=" * 160)
    lines.append(f"📊 {section_tag}{date_str}（共 {len(results)} 只）")
    lines.append("=" * 160)

    col_spec = (
        f"{_rpad('代码',10)}\t{_rpad('名称',8)}\t{_rpad('日期',12)}"
        f"\t{_lpad('总分',6)}\t{_lpad('窗口涨幅',9)}"
        f"\t{_lpad('RPS综合',8)}\t{_lpad('趋势',6)}"
        f"\t{_lpad('5日换手%',10)}"
        f"\t{_lpad('RSI',6)}\t{_rpad('风险',8)}"
        f"\t{_lpad('偏离MA20',9)}"
        f"\t{_lpad('收盘',7)}\t{_lpad('扣分',8)}\t{_lpad('连号',5)}\t{_lpad('连档',5)}"
    )
    lines.append(col_spec)
    lines.append("-" * 160)

    for r in results:
        code = r.code or ""
        name = r.name or ""
        signal_date = getattr(r, 'signal_date', '') or ''

        info  = rps_dict.get(code.lower(), {})
        t_inf = trend_dict.get(code.lower(), (None,)*5)
        rps_c   = info.get("composite", 0.0) if isinstance(info, dict) else 0.0
        # 优先用 Step1 注入的 RPS综合分（更准确反映全市场排名）
        rps_c = getattr(r, 'rps_composite', rps_c)
        trend_s = float(t_inf[2]) if t_inf[2] is not None else 0.0

        # 交集标记（🟢）
        is_inter = (code.lower() in inter_set)

        sector_bonus = getattr(r, 'sector_bonus_applied', 0) or 0
        limit_up_bonus = getattr(r, 'limit_up_bonus', 0) or 0
        if is_inter:
            penalty_str = "🟢交集"
        elif sector_bonus > 0:
            penalty_str = f"+{int(sector_bonus)}({getattr(r,'sector_name','')})"
        elif limit_up_bonus > 0:
            penalty_str = f"+{int(limit_up_bonus)}涨停"
        else:
            penalty_str = "-"

        prev_c = yesterday_consec_all.get(code.lower(), 0)
        consec_today = prev_c + 1 if prev_c > 0 else 0
        prev_rsi = yesterday_rsi_high_all.get(code.lower(), 0)
        rsi_val = r.rsi14
        is_high_rsi = rsi_val > 72
        rsi_high_today = (prev_rsi + 1) if (is_high_rsi and prev_rsi > 0) else (1 if is_high_rsi else 0)

        if rsi_val < 50:
            risk_tier = "🔵低位"
        elif rsi_val <= 65:
            risk_tier = "🟢健康"
        elif rsi_val <= 72:
            risk_tier = "🟡偏强"
        elif rsi_val <= 75:
            risk_tier = "🔴高位"
        elif rsi_val <= 82:
            risk_tier = "高位热"
        else:
            risk_tier = "❌超买"

        ext_pct = getattr(r, 'extension_pct', None) or 0.0

        row = (
            f"{_rpad(code,10)}\t{_rpad(name,8)}\t{_rpad(signal_date,12)}"
            f"\t{_lpad(f'{r.score:.1f}',6)}\t{_lpad(f'{r.total_gain_window:+.2f}%',9)}"
            f"\t{_lpad(f'{rps_c:.1f}',8)}\t{_lpad(f'{trend_s:.1f}',6)}"
            f"\t{_lpad(f'{r.avg_turnover_5:.2f}%',10)}"
            f"\t{_lpad(f'{rsi_val:.1f}',6)}\t{_rpad(risk_tier,8)}"
            f"\t{_lpad(f'{ext_pct:+.2f}%',9)}"
            f"\t{_lpad(f'{r.close:.2f}',7)}\t{_lpad(penalty_str,8)}\t{_lpad(str(consec_today),5)}\t{_lpad(str(rsi_high_today),5)}"
        )
        lines.append(row)

    lines.append("-" * 160)
    bonus_note = ""
    if any(getattr(r, 'sector_bonus_applied', 0) > 0 for r in results):
        bonus_note += " + 热门板块+8"
    if any(getattr(r, 'limit_up_bonus', 0) > 0 for r in results):
        bonus_note += " + 近10日涨停+3"
    inter_note = " 🟢=A∩B交集" if inter_set else ""
    lines.append(f"评分: 稳定性20 + 信号强度10 + 趋势25 + 流动性15 + 量能15 + K线5 + RSI10{bonus_note}{inter_note}")
    lines.append("RSI分层: 🟡偏强65~72扣5分 | 🔴高位72~75扣10分 | 高位热75~82扣15~25分")
    lines.append("连号≥3天 → 综合评分×0.8（趋势疲劳降权）")
    lines.append("综合评分 = gain×0.4 + RPS综合×0.3 + 趋势×0.3（用于最终排序）")

    return lines


# ─── 主流程 ───────────────────────────────────────────────

def combined_screen(target_date=None, workers=8):
    if target_date is None:
        target_date = datetime.now()
    date_str = target_date.strftime("%Y-%m-%d")
    print(f"\n{'='*60}")
    print(f"📊 合并选股  {date_str}")
    print(f"{'='*60}")

    # ── 准备：读取昨日 state（用于连号/连档展示）────────────
    triple_state = load_triple_state()
    real_prev_state = load_real_state()

    yesterday_triple = {}
    yesterday_rsi_triple = {}
    for d, v in triple_state.items():
        if isinstance(v, dict):
            yesterday_triple = v.get("consec", {})
            yesterday_rsi_triple = v.get("rsi_high", {})
            break

    # ── 获取股票列表 ──────────────────────────────────────
    all_codes = rps.get_all_stock_codes()
    print(f"\n📋 Step1 RPS 扫描（{len(all_codes)} 只）...")

    # ── Step1 real（加权评分前60%）───────────────────────
    real_top, real_rps_map, real_step1_df = step1_real(all_codes, target_date, workers)
    print(f"   real_screen Step1 → {len(real_top)} 只")

    # ── Step1 triple（硬过滤）────────────────────────────
    triple_step1_df, triple_all_df = step1_triple(all_codes, target_date, workers)
    triple_top = triple_step1_df["code"].str.lower().tolist()
    print(f"   triple_screen Step1 → {len(triple_top)} 只")

    # ── Step2 trend ─────────────────────────────────────
    print(f"\n📋 Step2 趋势扫描 ...")
    real_trend_codes, real_trend_raw = step2(real_top, target_date,
                                              score_threshold=40, workers=workers)
    triple_trend_codes, triple_trend_raw = step2(triple_top, target_date,
                                                  score_threshold=30, workers=workers)
    print(f"   real_screen Step2(≥40) → {len(real_trend_codes)} 只")
    print(f"   triple_screen Step2(≥30) → {len(triple_trend_codes)} 只")

    # ── Step3 gain_turnover ─────────────────────────────
    print(f"\n📋 Step3 gain_turnover 筛选 ...")
    config = gt.StrategyConfig(
        signal_days=3, min_gain=2.0, max_gain=10.0, quality_days=20,
        check_fundamental=False, sector_bonus=False,
        check_volume_surge=True, min_turnover=2.0, score_threshold=40.0,
    )

    real_results = step3(real_trend_codes, target_date, config, workers)
    triple_results = step3(triple_trend_codes, target_date, config, workers)
    print(f"   real_screen Step3 → {len(real_results)} 只")
    print(f"   triple_screen Step3 → {len(triple_results)} 只")

    # 保存原始昨日 state（用于显示 A∩B / A\B 的连号/连档）
    real_state_for_display = dict(real_prev_state)

    # ── 风控（real_screen 版本）──────────────────────────
    print(f"\n📋 风控过滤（real_screen 规则）...")
    real_results, _ = risk_control_real(real_results, real_prev_state, target_date)
    save_real_state(real_prev_state)  # 覆盖为今日 state（供下次运行使用）
    print(f"   real_screen 风控后 → {len(real_results)} 只")

    # ── 补充字段（用于显示）────────────────────────────────
    # real_results: rps_composite / trend_score 在 step3_gain 中已由 screen_market 设置
    # triple_results: 需要手动补充（用于 B\A 组的显示）
    triple_rps_map  = {str(row["code"]).lower(): float(row["composite"])
                       for _, row in triple_step1_df.iterrows()}
    triple_trend_map = {str(r[0]).lower(): float(r[2]) if r[2] is not None else 0.0
                        for r in triple_trend_raw if isinstance(r, tuple)}
    for r in triple_results:
        code = r.code.lower()
        r.rps_composite = triple_rps_map.get(code, 0.0)
        r.trend_score  = triple_trend_map.get(code, 0.0)

    # ── 保存 triple_screen state（连号/连档追踪）─────────────
    # 读取昨日 triple_state，计算今日连号/连档，写回
    yesterday_triple = {}
    yesterday_rsi_triple = {}
    for d, v in triple_state.items():
        if isinstance(v, dict):
            yesterday_triple = v.get("consec", {})
            yesterday_rsi_triple = v.get("rsi_high", {})
            break

    today_triple = {}
    today_rsi_triple = {}
    today_str = target_date.strftime("%Y-%m-%d")
    yesterday_str = get_prev_trading_day(target_date)

    for r in triple_results:
        code = r.code.lower()
        prev_c  = yesterday_triple.get(code, 0)
        prev_r  = yesterday_rsi_triple.get(code, 0)
        consec  = prev_c + 1 if prev_c > 0 else 0
        is_high = r.rsi14 > 72
        rsi_h   = (prev_r + 1) if (is_high and prev_r > 0) else (1 if is_high else 0)
        today_triple[code]  = consec
        today_rsi_triple[code] = rsi_h

    new_triple_state = {
        today_str: {"consec": today_triple, "rsi_high": today_rsi_triple}
    }
    triple_state_path = Path.home() / "stock_reports" / "triple_screen_state.json"
    triple_state_path.write_text(json.dumps(new_triple_state, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── 集合运算 ─────────────────────────────────────────
    set_A = {r.code.lower() for r in real_results}
    set_B = {r.code.lower() for r in triple_results}
    inter = set_A & set_B
    only_A = set_A - set_B
    only_B = set_B - set_A

    print(f"\n📊 集合对比：")
    print(f"   A(real_screen)     = {len(set_A)} 只")
    print(f"   B(triple_screen)   = {len(set_B)} 只")
    print(f"   A ∩ B（交集）      = {len(inter)} 只")
    print(f"   A \\ B（A独有）     = {len(only_A)} 只")
    print(f"   B \\ A（B独有）     = {len(only_B)} 只")

    # ── 输出前：给 real_results 注入 RPS综合分 ───────────────
    for r in real_results:
        r.rps_composite = real_rps_map.get(r.code.lower(), 0.0)

    # ── 输出文件（简化版）─────────────────────────────────────
    output = Path.home() / "stock_reports" / f"triple_screen_{date_str}.txt"
    output.parent.mkdir(parents=True, exist_ok=True)

    def write_section(tag, results_list, df_src, trend_src, state_src,
                     inter_set, mode):
        """写一个分组。inter_set 非空时对交集股票加 🟢 标记。"""
        if not results_list:
            return mode
        startup, trend_f = classify(results_list)
        for sub_tag, grp in [(tag + " 启动型", startup),
                               (tag + " 趋势型", trend_f)]:
            if not grp:
                continue
            lines = make_lines(grp, df_src, trend_src, state_src,
                               sub_tag, date_str, inter_set=inter_set)
            with open(output, mode, encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            mode = "a"
        return mode

    triple_state_new = {
        today_str: {"consec": today_triple, "rsi_high": today_rsi_triple}
    }

    mode = "w"
    # 步骤3+4：集合A（real_screen格式），其中 A∩B 用 🟢 标记
    mode = write_section("🚀 Real Screen 集合A",
                         real_results,
                         real_step1_df, real_trend_raw,
                         real_state_for_display,
                         inter_set=inter, mode=mode)
    # 步骤5：B\A 差集（triple_screen格式）
    mode = write_section("🔴 B\\A 差集",
                         [r for r in triple_results if r.code.lower() in only_B],
                         triple_all_df, triple_trend_raw,
                         triple_state_new,
                         inter_set=set(), mode=mode)

    print(f"\n💾 已保存: {output}")
    return real_results, triple_results


# ─── 入口 ─────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="合并选股脚本（real_screen + triple_screen）")
    parser.add_argument("--date", type=str, help="日期 YYYY-MM-DD（默认今天）")
    parser.add_argument("--workers", type=int, default=8, help="并行线程数")
    args = parser.parse_args()

    target = datetime.strptime(args.date, "%Y-%m-%d") if args.date else datetime.now()
    combined_screen(target, workers=args.workers)
