#!/usr/bin/env python3
# -*- coding: utf-8 -*-

r"""
读取 combined_screen 输出文件，分析 B\A 差集股票为何未被 real_screen 选中。

对每只 B\A 股票，按 real_screen 的筛选流程逐步验证：
  Step1（加权评分 top 60%）→ Step2（趋势≥40）→ Step3（gain_turnover）→ 风控

输出：~/stock_reports/ba_analysis_{日期}.txt
"""

import sys, re, argparse
from pathlib import Path
from datetime import datetime

WORKSPACE = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(WORKSPACE))

import rps_strong_screen as rps
import trend_strong_screen as tss
import gain_turnover as gt
from stock_trend.gain_turnover_screen import screen_market


# ─── 读取 B\A 股票代码 ──────────────────────────────────────

def extract_codes(filepath):
    codes = set()
    with open(filepath) as f:
        for line in f:
            found = re.findall(r'(s[zh][0-9]{6})', line)
            codes.update(found)
    return codes


def get_ba_codes(date_str):
    """从 combined_screen 输出文件中提取 B\\A 差集代码"""
    path = Path.home() / "stock_reports" / f"triple_screen_{date_str}.txt"
    if not path.exists():
        print(f"❌ 文件不存在: {path}")
        return None, None

    # 找 B\A 差集区块（启动型 + 趋势型）
    # combined_screen 输出格式：标题行 → ===行 → 列头 → ---行 → 数据行 → ---行 → (下一个标题)
    # 跳过第一个 --- 分隔线之后的所有代码，直到遇到下一个 "=" 开头的分组标题
    codes = set()
    in_ba = False
    line_count = 0
    with open(path) as f:
        for line in f:
            line_count += 1
            if "🔴 B" in line and "差集" in line:
                in_ba = True
                line_count = 0
                continue
            if in_ba:
                # 第二个分组标题（等号行出现在第二行之后）
                if line.startswith("=") and line_count > 2:
                    break
                found = re.findall(r'(s[zh][0-9]{6})', line)
                codes.update(found)
    return codes, path


# ─── Step1：real_screen 加权评分 ─────────────────────────

def check_step1_real(codes, target_date, workers=8):
    """
    模拟 real_screen Step1：RPS加权评分取前60%
    返回：(通过codes, df_with_scores)
    """
    df = rps.scan_rps(
        rps.get_all_stock_codes(),
        top_n=len(rps.get_all_stock_codes()),
        max_workers=workers, target_date=target_date
    )

    scores = []
    W_RPS = 0.4; W_RET = 0.2; W_RSI = 0.2; W_TURNOVER = 0.2
    for _, row in df.iterrows():
        s = row.get("composite", 0) * W_RPS
        ret20 = row.get("ret20", 0)
        if 5 < ret20 < 40:
            s += 100 * W_RET
        rsi_val = row.get("rsi", 50)
        if 50 < rsi_val < 75:
            s += 100 * W_RSI
        turnover = row.get("avg_turnover_5", 0)
        if turnover > 2:
            s += 100 * W_TURNOVER
        scores.append(s)

    df = df.copy()
    df["real_score"] = scores
    threshold = df["real_score"].quantile(0.6)

    target_df = df[df["code"].str.lower().isin(codes)].copy()
    passed = target_df[target_df["real_score"] > threshold]["code"].str.lower().tolist()

    return passed, df, threshold


# ─── Step2：real_screen 趋势评分 ────────────────────────

def check_step2_real(step1_codes, target_date, workers=8):
    """
    模拟 real_screen Step2：趋势评分≥40
    返回：通过Step2的codes
    """
    raw = tss.scan_market(
        codes=step1_codes,
        top_n=len(step1_codes),
        score_threshold=40,   # real_screen 用 40
        max_workers=workers, target_date=target_date
    )
    passed = [str(r[0]).lower() for r in raw if isinstance(r, tuple) and len(r) >= 4]
    trend_map = {str(r[0]).lower(): r for r in raw if isinstance(r, tuple)}
    return passed, trend_map


# ─── Step3：gain_turnover 筛选 ──────────────────────────

def check_step3_real(step2_codes, target_date, workers=8):
    """
    模拟 real_screen Step3：gain_turnover
    返回：通过Step3的results
    """
    config = gt.StrategyConfig(
        signal_days=3, min_gain=2.0, max_gain=10.0, quality_days=20,
        check_fundamental=False, sector_bonus=False,
        check_volume_surge=True, min_turnover=2.0, score_threshold=40.0,
    )
    results = screen_market(
        codes=step2_codes, config=config,
        target_date=target_date, top_n=len(step2_codes),
        max_workers=workers, refresh_cache=False
    )
    return results


# ─── 主分析 ──────────────────────────────────────────────

def analyze_ba(date_str, workers=8):
    target = datetime.strptime(date_str, "%Y-%m-%d")

    print(f"\n{'='*60}")
    print(f"📊 B\\A 差集分析  {date_str}")
    print(f"{'='*60}")

    ba_codes, src_path = get_ba_codes(date_str)
    if ba_codes is None:
        return
    ba_codes = sorted(ba_codes)
    print(f"\nB\\A 差集共 {len(ba_codes)} 只股票待分析...")

    # Step1
    print(f"\n📋 Step1：real_screen 加权评分（top 60%）...")
    passed_s1, df_scores, threshold = check_step1_real(ba_codes, target, workers)
    print(f"   门槛: {threshold:.2f}  分")
    print(f"   通过: {len(passed_s1)} / {len(ba_codes)}")

    # Step2
    print(f"\n📋 Step2：trend 评分≥40...")
    passed_s2, trend_map = check_step2_real(
        [c for c in ba_codes if c in passed_s1], target, workers
    )
    print(f"   通过: {len(passed_s2)} / {len([c for c in ba_codes if c in passed_s1])}")

    # Step3
    print(f"\n📋 Step3：gain_turnover 筛选...")
    results_s3 = check_step3_real(passed_s2, target, workers)
    passed_s3 = {r.code.lower() for r in results_s3}
    print(f"   通过: {len(passed_s3)} / {len(passed_s2)}")

    # ── 逐只分析 ──────────────────────────────────────────
    out_path = Path.home() / "stock_reports" / f"ba_analysis_{date_str}.txt"

    lines = []
    lines.append("=" * 100)
    lines.append(f"📊 B\\A 差集分析报告  {date_str}")
    lines.append(f"来源文件: {src_path}")
    lines.append("=" * 100)
    lines.append("")

    for code in ba_codes:
        reasons = []

        # Step1 检查
        if code not in passed_s1:
            row = df_scores[df_scores["code"].str.lower() == code]
            if not row.empty:
                r = row.iloc[0]
                real_sc = r["real_score"]
                rps_c = r.get("composite", 0)
                ret20 = r.get("ret20", 0)
                rsi_v = r.get("rsi", 0)
                turn = r.get("avg_turnover_5", 0)
                reasons.append(
                    f"  Step1❌ 加权分数={real_sc:.1f} ≤ 门槛{threshold:.1f}  "
                    f"(RPS综={rps_c:.1f} ret20={ret20:.1f}% RSI={rsi_v:.1f} 换手={turn:.2f}%)"
                )
            else:
                reasons.append("  Step1❌ 数据不足（可能被 RPS 扫描过滤）")
        else:
            reasons.append("  Step1✅")

        # Step2 检查
        if code in passed_s1 and code not in passed_s2:
            t_info = trend_map.get(code)
            if t_info:
                ts = float(t_info[2]) if t_info[2] else 0
                reasons.append(f"  Step2❌ 趋势分={ts:.1f} < 40")
            else:
                reasons.append("  Step2❌ 未通过 trend 筛选（score<40）")
        elif code in passed_s1:
            reasons.append("  Step2✅")

        # Step3 检查
        if code in passed_s2 and code not in passed_s3:
            reasons.append("  Step3❌ 未通过 gain_turnover 筛选（质量窗口/量能/评分门槛）")
        elif code in passed_s2:
            reasons.append("  Step3✅")

        # 汇总
        failed_at = "❌ Step1" if "Step1❌" in reasons[0] else \
                    "❌ Step2" if "Step2❌" in reasons[1] else \
                    "❌ Step3" if "Step3❌" in reasons[2] else \
                    "✅ 全通过（但被风控过滤）"

        lines.append(f"{code.upper()}  →  {failed_at}")
        lines.extend(reasons)
        lines.append("")

    # 统计
    s1_fail = sum(1 for c in ba_codes if c not in passed_s1)
    s2_fail = sum(1 for c in ba_codes if c in passed_s1 and c not in passed_s2)
    s3_fail = sum(1 for c in ba_codes if c in passed_s2 and c not in passed_s3)

    lines.append("-" * 100)
    lines.append(f"📊 统计：共 {len(ba_codes)} 只")
    lines.append(f"   Step1 过滤（RPS加权评分 top 60% 未进）: {s1_fail} 只")
    lines.append(f"   Step2 过滤（趋势评分 < 40）            : {s2_fail} 只")
    lines.append(f"   Step3 过滤（gain_turnover 未通过）    : {s3_fail} 只")
    lines.append(f"   全通过但风控过滤                      : "
                 f"{len(ba_codes)-s1_fail-s2_fail-s3_fail} 只")
    lines.append("")
    lines.append("说明：real_screen 的风控（连号≥3 / RSI连档≥3）需要历史 state 数据，")
    lines.append("      无法仅从当日截面数据还原。")

    output = "\n".join(lines)
    print("\n" + output)
    out_path.write_text(output, encoding="utf-8")
    print(f"\n💾 已保存: {out_path}")


# ─── 入口 ───────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="分析 B\\A 差集股票被过滤原因")
    parser.add_argument("--date", type=str, required=True, help="日期 YYYY-MM-DD")
    parser.add_argument("--workers", type=int, default=8, help="并行线程数")
    args = parser.parse_args()
    analyze_ba(args.date, workers=args.workers)
