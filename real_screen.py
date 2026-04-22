#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path

WORKSPACE = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(WORKSPACE))

import argparse
from datetime import datetime
import json

import pandas as pd

from stock_trend import gain_turnover as gt
from stock_trend import trend_strong_screen as tss
from stock_trend import rps_strong_screen as rps
from stock_trend.gain_turnover_screen import screen_market
from stock_trend.gain_turnover import _lpad, _rpad


# =========================
# 配置
# =========================

class Config:
    MARKET_STOP_LOSS = -5

    W_RPS = 0.4
    W_RET = 0.2
    W_RSI = 0.2
    W_TURNOVER = 0.2

    MIN_TREND_SCORE = 40

    SIGNAL_DAYS = 3
    MIN_GAIN = 2
    MAX_GAIN = 10    # 与 triple_screen 一致，精准入场

    MAX_POSITIONS = 50


# =========================
# 市场环境
# =========================

def get_market_strength(target_date=None):
    from stock_trend.trend_strong_screen import get_market_gain, INDEX_CODES

    return get_market_gain(
        INDEX_CODES,
        days=21,
        target_date=target_date
    )


# =========================
# 评分
# =========================

def score_stocks(df):
    scores = []

    for _, row in df.iterrows():
        score = 0

        score += row.get("composite", 0) * Config.W_RPS

        ret20 = row.get("ret20", 0)
        if 5 < ret20 < 40:
            score += 100 * Config.W_RET

        rsi = row.get("rsi", 50)
        if 50 < rsi < 75:
            score += 100 * Config.W_RSI

        turnover = row.get("avg_turnover_5", 0)
        if turnover > 2:
            score += 100 * Config.W_TURNOVER

        scores.append(score)

    df["score"] = scores
    return df.sort_values("score", ascending=False)


# =========================
# 趋势过滤
# =========================

def filter_trend(codes, target_date=None):
    actual_threshold = Config.MIN_TREND_SCORE
    raw = tss.scan_market(
        codes=codes,
        top_n=len(codes),
        score_threshold=actual_threshold,
        target_date=target_date
    )
    print(f"[DEBUG filter_trend] codes={len(codes)} thr={actual_threshold} returned={len(raw)}")

    valid = []
    for r in raw:
        if isinstance(r, tuple):
            valid.append(r[0])

    return valid, raw  # 返回 codes 列表 + 原始结果（用于建立 trend 评分映射）


# =========================
# 信号过滤
# =========================

def signal_filter(codes, target_date=None):
    strategy_config = gt.StrategyConfig(
        signal_days=Config.SIGNAL_DAYS,
        min_gain=Config.MIN_GAIN,
        max_gain=Config.MAX_GAIN,
        min_turnover=2.0,
        min_amount=1e8,
    )

    return screen_market(
        codes=codes,
        config=strategy_config,
        target_date=target_date,
        top_n=len(codes),
        refresh_cache=False
    )


# =========================
# =========================
# 分类
# =========================

def classify(results):
    startup = []
    trend_follow = []

    for r in results:
        if r.total_gain_window > 10 and r.avg_turnover_5 > 3:
            startup.append(r)
        else:
            trend_follow.append(r)

    return startup, trend_follow


# =========================
# 主流程
# =========================

def run(target_date=None, custom_config=None, codes=None):
    print("=== 实盘选股 ===")

    config = Config()

    if custom_config:
        for k, v in custom_config.items():
            setattr(config, k, v)

    if target_date:
        print(f"📅 日期: {target_date.strftime('%Y-%m-%d')}")

    # 市场判断
    market = get_market_strength(target_date)
    print(f"市场20日涨幅: {market:.2f}%")

    if market < config.MARKET_STOP_LOSS:
        print("❌ 市场太差，停止")
        return []

    # =========================
    # Step1
    # =========================
    # 指定代码模式：只扫描这些股票
    if codes is not None:
        all_codes = [gt.normalize_prefixed(c) for c in codes]
        print(f"🔍 指定范围: {len(all_codes)} 只")
    else:
        all_codes = rps.get_all_stock_codes()

    # 腾讯实时批量预取（盘中，且目标日期是今天才需要）
    import pandas as pd
    today = pd.Timestamp.today().normalize()
    is_today = (target_date is not None and pd.Timestamp(target_date).normalize() == today)
    if is_today:
        from stock_trend.gain_turnover import _prefetch_tencent_realtime as prefetch_tencent_realtime
        prefetch_tencent_realtime(all_codes)

    # 盘中实时：scan_rps 不带 target_date（避免 concat 开销），腾讯数据已在缓存
    # 复盘模式：正常传 target_date
    df = rps.scan_rps(
        all_codes,
        top_n=len(all_codes),
        target_date=None if is_today else target_date
    )

    df = score_stocks(df)
    if df.empty:
        print("⚠️  Step1 扫描无有效股票")
        return
    # 指定代码模式：直接取指定股票；全市场模式：取 top 60% quantile
    if codes is not None:
        top_codes = df["code"].tolist()
    else:
        q = df["score"].quantile(0.6)
        top_codes = df[df["score"] > q]["code"].tolist()
    print(f"[DEBUG] top_codes from Step1: {len(top_codes)}")

    print(f"Step1 → {len(top_codes)}")

    # 指定代码模式：过滤到指定代码
    if codes is not None:
        codes_lower = {c.lower() for c in all_codes}
        df = df[df["code"].str.lower().isin(codes_lower)]

    # 建立 RPS 综合分映射（用于最终排序）
    rps_score_map = {
        str(row["code"]).lower(): row.get("composite", 0)
        for _, row in df.iterrows()
        if pd.notna(row["code"])
    }

    # =========================
    # Step2
    # =========================
    trend_codes, trend_results = filter_trend(top_codes, target_date)
    print(f"[DEBUG] filter_trend({len(top_codes)} codes, target={target_date}) returned {len(trend_codes)} codes")
    if trend_codes:
        print(f"[DEBUG] first 5 trend_codes: {trend_codes[:5]}")

    # DEBUG: 打印top_codes的类型和前5个
    print(f"[DEBUG] top_codes sample: {top_codes[:5]} type={type(top_codes[0])}")
    # 直接验证top_codes[0]的评分
    first_code = top_codes[0]
    first_result = tss.evaluate_stock(first_code, min_volume=5e7, exclude_st=True,
                                       market_gain=market, names_cache=None, target_date=target_date)
    score_display = first_result["score"] if first_result else None
    print("[DEBUG] first_code=%s score=%s" % (first_code, score_display))

    # 建立 trend 评分映射（用于最终排序）
    trend_score_map = {
        str(r[0]).lower(): float(r[2]) if r[2] is not None else 0
        for r in trend_results
        if isinstance(r, tuple)
    }

    print(f"Step2 → {len(trend_codes)}")

    if not trend_codes:
        return []

    # =========================
    # Step3
    # =========================
    results = signal_filter(trend_codes, target_date)
    print(f"Step3 → {len(results)}")

    if not results:
        return []

    # 回填 RPS 综合分（用于最终排序）
    for r in results:
        r.rps_composite = rps_score_map.get(r.code.lower(), 0)
        r.trend_score = trend_score_map.get(r.code.lower(), 0)

    # =========================
    # 排序
    # =========================
    def final_score(r):
        # 与 triple_screen 一致：gain×0.5 + RPS综×0.1 + trend×0.5
        return (
            r.score * 0.5 +
            getattr(r, 'rps_composite', 0) * 0.1 +
            getattr(r, 'trend_score', 0) * 0.5
        )

    final = sorted(results, key=final_score, reverse=True)
    final = final[:config.MAX_POSITIONS]

    # =========================
    # 输出（格式与 triple_screen 完全一致）
    # =========================
    date_str = target_date.strftime("%Y-%m-%d") if target_date else datetime.now().strftime("%Y-%m-%d")
    output = Path.home() / "stock_reports" / f"real_screen_{date_str}.txt"
    output.parent.mkdir(parents=True, exist_ok=True)

    # 建立 RPS / trend 字典
    rps_dict  = {str(row["code"]).lower(): row for _, row in df.iterrows()}
    trend_dict = {str(r[0]).lower(): r for r in trend_results if isinstance(r, tuple)}

    # 合并 RPS / trend 数据到 result 对象（补充字段）
    for r in final:
        code = r.code.lower()
        r.rps_composite = rps_score_map.get(code, 0.0)
        t_info = trend_dict.get(code, (None, None, None, None, None))
        r.trend_score = float(t_info[2]) if isinstance(t_info, tuple) and t_info[2] is not None else 0.0

    lines = []
    lines.append("=" * 160)
    lines.append(f"📊 实盘选股 {date_str}（共 {len(final)} 只）")
    lines.append("=" * 160)

    # 列头（与 triple_screen 一致）
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

    for r in final:
        code = r.code or ""
        name = r.name or ""
        signal_date = getattr(r, 'signal_date', '') or ''

        rps_c = getattr(r, 'rps_composite', 0.0)
        trend_s = getattr(r, 'trend_score', 0.0)

        # 加分列
        extras = []
        if sector_bonus > 0:
            extras.append(f"+{int(sector_bonus)}({getattr(r, 'sector_name', '')})")
        if limit_up_bonus > 0:
            extras.append(f"+{int(limit_up_bonus)}涨停")
        bonus_str = " ".join(extras) if extras else "-"

        rsi_val = r.rsi14

        # RSI 风险分层
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

        # 偏离 MA20（gain_turnover 有 extension_pct，尝试获取；无则估算）
        ext_pct = getattr(r, 'extension_pct', None)
        if ext_pct is None:
            ext_pct = 0.0

        row = (
            f"{_rpad(code,10)}\t{_rpad(name,8)}\t{_rpad(signal_date,12)}"
            f"\t{_lpad(f'{r.score:.1f}',6)}\t{_lpad(f'{r.total_gain_window:+.2f}%',9)}"
            f"\t{_lpad(f'{rps_c:.1f}',8)}\t{_lpad(f'{trend_s:.1f}',6)}"
            f"\t{_lpad(f'{r.avg_turnover_5:.2f}%',10)}"
            f"\t{_lpad(f'{rsi_val:.1f}',6)}\t{_rpad(risk_tier,8)}"
            f"\t{_lpad(f'{ext_pct:+.2f}%',9)}"
            f"\t{_lpad(f'{r.close:.2f}',7)}\t{_lpad(bonus_str,8)}"
        )
        lines.append(row)

    lines.append("-" * 160)
    lines.append("评分: 稳定性20 + 信号强度10 + 趋势25 + 流动性15 + 量能15 + K线5 + RSI10")
    lines.append("RSI分层: 🟡偏强65~72扣5分 | 🔴高位72~75扣10分 | 高位热75~82扣15~25分")
    lines.append("综合评分 = gain×0.4 + RPS综合×0.3 + 趋势×0.3（用于最终排序）")

    output_text = "\n".join(lines)
    print("\n" + output_text)

    with open(output, "w", encoding="utf-8") as f:
        f.write(output_text)
        f.write("\n")
    print(f"\n💾 已保存: {output}")
    return final


# =========================
# CLI
# =========================

def main():
    parser = argparse.ArgumentParser(description="实盘选股系统")
    parser.add_argument("--date", type=str, help="YYYY-MM-DD")
    parser.add_argument("--codes", nargs="+", default=None, help="指定股票代码（单独分析这些股票）")
    parser.add_argument("--trend-score", type=float, default=Config.MIN_TREND_SCORE, help=f"Step2 趋势评分门槛（默认{Config.MIN_TREND_SCORE}）")
    parser.add_argument("--days", type=int, default=Config.SIGNAL_DAYS, help=f"信号窗口天数（默认{Config.SIGNAL_DAYS}）")
    parser.add_argument("--min-gain", type=float, default=Config.MIN_GAIN, help=f"最小日涨幅%%（默认{Config.MIN_GAIN}）")
    parser.add_argument("--max-gain", type=float, default=Config.MAX_GAIN, help=f"最大日涨幅%%（默认{Config.MAX_GAIN}）")

    args = parser.parse_args()

    target_date = datetime.strptime(args.date, "%Y-%m-%d") if args.date else None

    # 将 CLI 参数覆盖 Config
    custom_config = {
        "MIN_TREND_SCORE": args.trend_score,
        "SIGNAL_DAYS": args.days,
        "MIN_GAIN": args.min_gain,
        "MAX_GAIN": args.max_gain,
    }
    run(target_date, custom_config, codes=args.codes)


if __name__ == "__main__":
    main()
