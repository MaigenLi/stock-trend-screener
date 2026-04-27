#!/usr/bin/env python3
"""
回测胜率报告生成 + 邮件发送
==============================

自动运行：
1. compute_indicators.py（如需）
2. backtest_engine.py（训练集 + 验证集）
3. 生成 Markdown 报告
4. 转换为 PDF 并发送邮件

用法：
    python backtest_report.py --train-start 2025-01-01 --train-end 2025-09-30 \
                              --val-start 2025-10-01 --val-end 2026-04-23 \
                              --hold 5 --interval 4
"""

import sys
import json
import argparse
import smtplib
import subprocess
from pathlib import Path
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication

WORKSPACE = Path(__file__).parent.parent.parent.resolve()
REPORT_DIR = Path.home() / "stock_reports"
REPORT_DIR.mkdir(exist_ok=True)

# ── 邮件配置（从 .env 读取）───────────────────────────────
_env = {}
env_path = Path.home() / ".openclaw" / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            _env[k.strip()] = v.strip()

SMTP_HOST = "smtp.qq.com"
SMTP_PORT = 587
EMAIL_FROM = _env.get("QQ_EMAIL", "920662304@qq.com")
EMAIL_PASSWORD = _env.get("QQ_PASS", "")
EMAIL_TO = _env.get("QQ_EMAIL", "920662304@qq.com")


def send_email_report(
    report_path: Path,
    subject: str,
    body_html: str,
    to_email: str = None,
) -> bool:
    """发送邮件报告"""
    if not EMAIL_PASSWORD:
        print("⚠️ 未在 ~/.openclaw/.env 中找到 QQ_PASS，跳过发送")
        return False

    to = to_email or EMAIL_TO

    msg = MIMEMultipart()
    msg["From"] = EMAIL_FROM
    msg["To"] = to
    msg["Subject"] = subject
    msg.attach(MIMEText(body_html, "html", "utf-8"))

    if report_path and report_path.exists():
        with open(report_path, "rb") as f:
            part = MIMEApplication(f.read(), Name=report_path.name)
            part["Content-Disposition"] = f'attachment; filename="{report_path.name}"'
            msg.attach(part)

    print(f"📧 发送邮件到 {to}...")
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.login(EMAIL_FROM, EMAIL_PASSWORD)
            server.sendmail(EMAIL_FROM, [to], msg.as_string())
        print(f"✅ 邮件已发送")
        return True
    except Exception as e:
        print(f"❌ 邮件发送失败: {e}")
        return False


def build_html_report(train_stats, val_stats, combined_stats, params: dict) -> tuple[str, Path]:
    """生成 HTML 报告"""
    today = datetime.now().strftime("%Y-%m-%d")

    # Markdown → HTML body
    md_content = f"""# 📊 screen_trend_filter 胜率回测报告

**生成时间**: {today}  
**回测区间**: 训练集 {params['train_start']} → {params['train_end']} | 验证集 {params['val_start']} → {params['val_end']}  
**持仓期**: T+{params['hold']}（持有 {params['hold']} 天）  
**信号间隔**: 每 {params['interval']} 个交易日  

---

## 🔬 验证集（真实胜率）

> 以下是模型从未见过的数据区间的表现

| 指标 | 数值 |
|------|------|
| **总信号数** | {val_stats['total_signals']} |
| **胜率** | **{val_stats['win_rate']}%** |
| **平均收益** | {val_stats['avg_pnl']:+.3f}% |
| **中位数收益** | {val_stats['median_pnl']:+.3f}% |
| **最大盈利** | {val_stats['max_pnl']:+.2f}% |
| **最大亏损** | {val_stats['min_pnl']:+.2f}% |
| **标准差** | {val_stats['std_pnl']} |
| **盈亏比** | {val_stats['profit_factor']} |

### 分持仓天数

| 持仓期 | 笔数 | 胜率 | 平均收益 |
|--------|------|------|----------|
"""

    for k, v in val_stats.get("by_hold_days", {}).items():
        md_content += f"| {k} | {v['count']} | {v['win_rate']}% | {v['avg_pnl']:+.3f}% |\n"

    md_content += f"""
### 分信号强度

| 信号类型 | 笔数 | 胜率 | 平均收益 |
|----------|------|------|----------|
| 🟢强信号(≥70分) | {val_stats['strong_signals']['count']} | {val_stats['strong_signals']['win_rate']}% | {val_stats['strong_signals']['avg_pnl']:+.3f}% |
| 🔵中信号(60-70分) | {val_stats['medium_signals']['count']} | {val_stats['medium_signals']['win_rate']}% | {val_stats['medium_signals']['avg_pnl']:+.3f}% |
| 🟡弱信号(<60分) | {val_stats['weak_signals']['count']} | {val_stats['weak_signals']['win_rate']}% | {val_stats['weak_signals']['avg_pnl']:+.3f}% |

### 止损统计

- 止损触发: {val_stats['stopped_count']} 次，平均收益 {val_stats['stopped_avg_pnl']:+.3f}%
- 非止损: {val_stats['not_stopped_count']} 次，平均收益 {val_stats['not_stopped_avg_pnl']:+.3f}%

---

## 📈 训练集（过拟合参考）

| 指标 | 数值 |
|------|------|
| **总信号数** | {train_stats['total_signals']} |
| **胜率** | {train_stats['win_rate']}% |
| **平均收益** | {train_stats['avg_pnl']:+.3f}% |

### 分信号强度

| 信号类型 | 笔数 | 胜率 | 平均收益 |
|----------|------|------|----------|
| 🟢强信号 | {train_stats['strong_signals']['count']} | {train_stats['strong_signals']['win_rate']}% | {train_stats['strong_signals']['avg_pnl']:+.3f}% |
| 🔵中信号 | {train_stats['medium_signals']['count']} | {train_stats['medium_signals']['win_rate']}% | {train_stats['medium_signals']['avg_pnl']:+.3f}% |
| 🟡弱信号 | {train_stats['weak_signals']['count']} | {train_stats['weak_signals']['win_rate']}% | {train_stats['weak_signals']['avg_pnl']:+.3f}% |

---

## 🏆 综合结论

"""

    # 自动生成结论
    val_wr = val_stats['win_rate']
    val_avg = val_stats['avg_pnl']
    val_combined = combined_stats

    if val_wr >= 55 and val_avg > 0:
        verdict = "✅ **策略有效** — 验证集胜率显著高于随机（50%），具备实盘参考价值"
    elif val_wr >= 50 and val_avg >= 0:
        verdict = "⚠️ **策略中性** — 胜率略高于随机，但收益有限，需优化参数"
    else:
        verdict = "❌ **策略失效** — 验证集表现不佳，建议重新设计筛选逻辑"

    conclusion = f"""
- **验证集胜率**: {val_wr}%（{val_stats['total_signals']} 笔信号）
- **验证集平均收益**: {val_avg:+.3f}%
- **强信号胜率**: {val_stats['strong_signals']['win_rate']}%（{val_stats['strong_signals']['count']} 笔）
- **盈亏比**: {val_stats['profit_factor']}

{verdict}

> ⚠️ 回测结果仅供参考，实盘表现受流动性、滑点、市场环境等多因素影响。
> 模拟验证日志路径: `~/stock_reports/simulation_log.json`

---
*本报告由 screen_trend_filter v3 自动生成*
"""

    md_content += conclusion

    # 转换为 HTML
    try:
        import markdown
        html_body = markdown.markdown(
            md_content,
            extensions=["tables", "fenced_code"],
            output_format="html",
        )
    except ImportError:
        # 不用 markdown 库，手动转简单 HTML
        html_body = f"<pre>{md_content}</pre>"

    html_full = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<style>
body {{ font-family: -apple-system, Arial, sans-serif; margin: 40px; color: #333; }}
h1 {{ color: #1a1a2e; border-bottom: 2px solid #0066cc; padding-bottom: 10px; }}
h2 {{ color: #0066cc; margin-top: 30px; }}
table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
th, td {{ border: 1px solid #ddd; padding: 10px; text-align: center; }}
th {{ background: #0066cc; color: white; }}
tr:nth-child(even) {{ background: #f5f5f5; }}
strong {{ color: #0066cc; }}
code {{ background: #f0f0f0; padding: 2px 6px; border-radius: 3px; }}
hr {{ border: none; border-top: 1px solid #ddd; margin: 30px 0; }}
</style>
</head><body>
{html_body}
</body></html>"""

    # 写 HTML 文件
    html_path = REPORT_DIR / f"backtest_report_{today}.html"
    html_path.write_text(html_full, encoding="utf-8")

    return html_full, html_path


def run_pipeline(args):
    """运行完整流水线"""
    today = datetime.now().strftime("%Y-%m-%d")
    train_out = REPORT_DIR / "train_results.json"
    val_out = REPORT_DIR / "val_results.json"

    sys.path.insert(0, str(WORKSPACE))
    from stock_trend.review_screen.backtest_engine import (
        run_full_backtest, analyze_results, preload_indicators,
    )

    # ── 1. 指标预计算检查 ───────────────────────────────
    indicator_count = len(list((WORKSPACE / ".cache" / "indicators").glob("*_indicators.json")))
    print(f"📊 当前已计算指标: {indicator_count} 只")

    if indicator_count < 4000:
        print("⚠️  指标数量不足（<4000），请先运行 compute_indicators.py")
        print(f"   python stock_trend/review_screen/compute_indicators.py --workers 8")
        return None

    # ── 2. 运行回测 ─────────────────────────────────────
    print("\n🚀 开始回测...")

    print(f"\n📂 训练集: {args.train_start} → {args.train_end}")
    train_results = run_full_backtest(
        args.train_start, args.train_end,
        hold_days=args.hold, skip_interval=args.interval,
    )
    train_stats = analyze_results(train_results)

    print(f"\n📂 验证集: {args.val_start} → {args.val_end}")
    val_results = run_full_backtest(
        args.val_start, args.val_end,
        hold_days=args.hold, skip_interval=args.interval,
    )
    val_stats = analyze_results(val_results)

    # 合并
    combined_results = train_results + val_results
    combined_stats = analyze_results(combined_results)

    # 保存
    with open(train_out, "w") as f:
        json.dump(train_results, f, ensure_ascii=False)
    with open(val_out, "w") as f:
        json.dump(val_results, f, ensure_ascii=False)

    print(f"\n💾 结果已保存: {train_out} ({len(train_results)}笔)  {val_out} ({len(val_results)}笔)")

    # ── 3. 生成报告 ─────────────────────────────────────
    params = {
        "train_start": args.train_start,
        "train_end": args.train_end,
        "val_start": args.val_start,
        "val_end": args.val_end,
        "hold": args.hold,
        "interval": args.interval,
    }

    body_html, html_path = build_html_report(train_stats, val_stats, combined_stats, params)

    # ── 4. 发送邮件 ─────────────────────────────────────
    subject = f"📊 screen_trend_filter 胜率回测报告 {today}"

    if args.no_email:
        print(f"\n⏭️ 已跳过邮件发送（--no-email）")
        print(f"📄 报告: {html_path}")
    else:
        success = send_email_report(
            report_path=html_path,
            subject=subject,
            body_html=body_html,
        )
        if success:
            print(f"\n✅ 报告已发送至 {EMAIL_TO}")
        else:
            print(f"\n⚠️ 邮件发送失败，报告保存在: {html_path}")

    return {
        "train_stats": train_stats,
        "val_stats": val_stats,
        "combined_stats": combined_stats,
        "html_path": html_path,
    }


def main():
    parser = argparse.ArgumentParser(description="回测胜率报告生成 + 邮件发送")
    parser.add_argument("--train-start", type=str, default="2025-01-01")
    parser.add_argument("--train-end", type=str, default="2025-09-30")
    parser.add_argument("--val-start", type=str, default="2025-10-01")
    parser.add_argument("--val-end", type=str, default="2026-04-23")
    parser.add_argument("--hold", type=int, default=5, help="持有天数")
    parser.add_argument("--interval", type=int, default=4, help="信号间隔")
    parser.add_argument("--no-email", action="store_true", help="不发送邮件")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"📊 screen_trend_filter 胜率回测报告")
    print(f"{'='*60}")

    result = run_pipeline(args)

    if result:
        print(f"\n{'='*60}")
        print(f"📋 验证集关键结论")
        print(f"{'='*60}")
        vs = result["val_stats"]
        print(f"  胜率:     {vs['win_rate']}%")
        print(f"  平均收益: {vs['avg_pnl']:+.3f}%")
        print(f"  盈亏比:   {vs['profit_factor']}")
        print(f"  🟢强信号胜率: {vs['strong_signals']['win_rate']}% ({vs['strong_signals']['count']}笔)")


if __name__ == "__main__":
    main()
