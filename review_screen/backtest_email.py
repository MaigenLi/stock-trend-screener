#!/usr/bin/env python3
"""
回测胜率报告邮件发送
====================

读取 train_top3.json 和 val_top3.json，生成HTML报告并发送邮件
"""

import json
import smtplib
import sys
from pathlib import Path
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication

WORKSPACE = Path(__file__).parent.parent.parent.resolve()
REPORT_DIR = Path.home() / "stock_reports"

# ── 邮件配置 ──────────────────────────────────────────────
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


def load_results(path: Path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return []


def analyze(results):
    """统计分析"""
    if not results:
        return {}
    import numpy as np
    pnls = [r["pnl_pct"] for r in results]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    from collections import defaultdict
    by_hold = defaultdict(list)
    for r in results:
        by_hold[r["hold_days_actual"]].append(r["pnl_pct"])

    stopped = [r["pnl_pct"] for r in results if r["stopped"]]
    not_stopped = [r["pnl_pct"] for r in results if not r["stopped"]]
    strong = [r for r in results if r["signal_score"] >= 70]
    medium = [r for r in results if 60 <= r["signal_score"] < 70]
    weak = [r for r in results if r["signal_score"] < 60]

    return {
        "total": len(results),
        "win_rate": round(len(wins) / len(results) * 100, 1),
        "avg_pnl": round(float(np.mean(pnls)), 3),
        "median_pnl": round(float(np.median(pnls)), 3),
        "std_pnl": round(float(np.std(pnls)), 2),
        "max_pnl": round(max(pnls), 2),
        "min_pnl": round(min(pnls), 2),
        "profit_factor": round(abs(np.sum(wins) / np.sum(losses)), 2) if losses else 99.99,
        "by_hold": {k: {"count": len(v),
                        "win_rate": round(len([x for x in v if x > 0]) / len(v) * 100, 1),
                        "avg": round(float(np.mean(v)), 3)}
                    for k, v in sorted(by_hold.items())},
        "stopped_n": len(stopped),
        "stopped_avg": round(float(np.mean(stopped)), 3) if stopped else 0,
        "not_stopped_n": len(not_stopped),
        "not_stopped_avg": round(float(np.mean(not_stopped)), 3) if not_stopped else 0,
        "strong_n": len(strong),
        "strong_wr": round(len([x for x in strong if x["pnl_pct"] > 0]) / len(strong) * 100, 1) if strong else 0,
        "strong_avg": round(float(np.mean([x["pnl_pct"] for x in strong])), 3) if strong else 0,
        "medium_n": len(medium),
        "medium_wr": round(len([x for x in medium if x["pnl_pct"] > 0]) / len(medium) * 100, 1) if medium else 0,
        "medium_avg": round(float(np.mean([x["pnl_pct"] for x in medium])), 3) if medium else 0,
        "weak_n": len(weak),
    }


def send_email(html_path: Path, subject: str, body: str) -> bool:
    if not EMAIL_PASSWORD:
        print("⚠️ 未配置邮件密码，跳过发送")
        return False
    msg = MIMEMultipart()
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "html", "utf-8"))
    if html_path and html_path.exists():
        with open(html_path, "rb") as f:
            part = MIMEApplication(f.read(), Name=html_path.name)
            part["Content-Disposition"] = f'attachment; filename="{html_path.name}"'
            msg.attach(part)
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.login(EMAIL_FROM, EMAIL_PASSWORD)
            server.sendmail(EMAIL_FROM, [EMAIL_TO], msg.as_string())
        print(f"✅ 邮件已发送至 {EMAIL_TO}")
        return True
    except Exception as e:
        print(f"❌ 发送失败: {e}")
        return False


def main():
    today = datetime.now().strftime("%Y-%m-%d")

    train = load_results(REPORT_DIR / "train_top3.json")
    val = load_results(REPORT_DIR / "val_top3.json")

    if not val:
        print("⚠️ 无验证集结果")
        sys.exit(1)

    ts = analyze(train)
    vs = analyze(val)

    # ── 生成 HTML 报告 ──────────────────────────────────
    val_wr = vs.get("win_rate", 0)
    val_avg = vs.get("avg_pnl", 0)
    val_pf = vs.get("profit_factor", 0)
    val_signals = vs.get("total", 0)

    if val_wr >= 50 and val_avg > 0.5:
        verdict_color = "#00c853"
        verdict_icon = "✅"
        verdict_text = "策略有效"
        verdict_detail = "验证集胜率显著高于50%，平均收益为正，具备实盘参考价值"
    elif val_wr >= 45 and val_avg >= 0:
        verdict_color = "#ff9800"
        verdict_icon = "⚠️"
        verdict_text = "策略中性"
        verdict_detail = "胜率低于50%但盈亏比>1，说明赚钱的股票涨幅大于亏钱的跌幅"
    elif val_pf > 1:
        verdict_color = "#ff9800"
        verdict_icon = "⚠️"
        verdict_text = "策略偏弱"
        verdict_detail = f"胜率{val_wr}%<50%，但盈亏比={val_pf}>1，仍有微弱正期望"
    else:
        verdict_color = "#f44336"
        verdict_icon = "❌"
        verdict_text = "策略失效"
        verdict_detail = "胜率和盈亏比均不理想，建议重新设计筛选逻辑"

    html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>screen_trend_filter 胜率回测报告</title>
<style>
  body {{ font-family: -apple-system, Arial, sans-serif; margin: 40px; color: #222; background: #fafafa; }}
  .card {{ background: white; border-radius: 12px; padding: 24px; margin: 16px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
  h1 {{ color: #1a237e; font-size: 24px; margin-bottom: 4px; }}
  h2 {{ color: #1565c0; font-size: 18px; margin-top: 24px; border-bottom: 2px solid #e3f2fd; padding-bottom: 8px; }}
  h3 {{ color: #333; font-size: 15px; margin-top: 16px; }}
  .subtitle {{ color: #888; font-size: 13px; margin-bottom: 20px; }}
  .tag {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; }}
  .tag-green {{ background: #e8f5e9; color: #2e7d32; }}
  .tag-red {{ background: #ffebee; color: #c62828; }}
  .tag-orange {{ background: #fff3e0; color: #e65100; }}
  table {{ border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 14px; }}
  th, td {{ border: 1px solid #e0e0e0; padding: 10px 12px; text-align: center; }}
  th {{ background: #1565c0; color: white; }}
  tr:nth-child(even) {{ background: #f5f5f5; }}
  .highlight {{ background: #fff8e1 !important; }}
  .big-num {{ font-size: 32px; font-weight: bold; color: #1565c0; }}
  .big-num-green {{ color: #2e7d32; }}
  .big-num-red {{ color: #c62828; }}
  .verdict {{ background: {verdict_color}18; border-left: 4px solid {verdict_color}; padding: 16px; border-radius: 8px; margin: 20px 0; }}
  .verdict-icon {{ font-size: 24px; margin-right: 8px; }}
  .verdict-title {{ font-size: 18px; font-weight: bold; color: {verdict_color}; }}
  .verdict-detail {{ color: #555; margin-top: 6px; font-size: 14px; }}
  .warn {{ background: #fff3cd; border: 1px solid #ffc107; border-radius: 6px; padding: 12px; margin: 16px 0; font-size: 13px; color: #856404; }}
  .footer {{ color: #aaa; font-size: 11px; margin-top: 30px; text-align: center; }}
  .metric-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 16px 0; }}
  .metric-box {{ background: #f8f9fa; border-radius: 8px; padding: 16px; text-align: center; }}
  .metric-label {{ font-size: 11px; color: #888; text-transform: uppercase; }}
  .metric-value {{ font-size: 22px; font-weight: bold; margin-top: 4px; }}
</style>
</head><body>

<div class="card">
  <h1>📊 screen_trend_filter TOP3 胜率回测报告</h1>
  <p class="subtitle">生成时间: {today} | 数据区间: 2024-01-02 ~ 2026-04-24 | 持仓: T+5 | 信号间隔: 每4个交易日</p>
</div>

<div class="verdict">
  <span class="verdict-icon">{verdict_icon}</span>
  <span class="verdict-title">{verdict_text}</span>
  <div class="verdict-detail">{verdict_detail}</div>
</div>

<div class="card">
  <h2>🔬 验证集核心结论（样本外）</h2>
  <div class="metric-grid">
    <div class="metric-box">
      <div class="metric-label">胜率</div>
      <div class="metric-value {'big-num-green' if val_wr >= 50 else 'big-num-red'}">{val_wr}%</div>
    </div>
    <div class="metric-box">
      <div class="metric-label">平均收益</div>
      <div class="metric-value {'big-num-green' if val_avg > 0 else 'big-num-red'}">{val_avg:+.3f}%</div>
    </div>
    <div class="metric-box">
      <div class="metric-label">盈亏比</div>
      <div class="metric-value {'big-num-green' if val_pf > 1.2 else ''}">{val_pf}</div>
    </div>
    <div class="metric-box">
      <div class="metric-label">总信号</div>
      <div class="metric-value">{val_signals}</div>
    </div>
  </div>

  <table>
    <tr><th>指标</th><th>验证集</th><th>训练集</th><th>对比</th></tr>
    <tr><td>胜率</td><td><b>{val_wr}%</b></td><td>{ts.get('win_rate',0)}%</td><td>{val_wr - ts.get('win_rate',0):+.1f}%</td></tr>
    <tr><td>平均收益</td><td><b>{val_avg:+.3f}%</b></td><td>{ts.get('avg_pnl',0):+.3f}%</td><td>{val_avg - ts.get('avg_pnl',0):+.3f}%</td></tr>
    <tr><td>中位数收益</td><td><b>{vs.get('median_pnl',0):+.3f}%</b></td><td>{ts.get('median_pnl',0):+.3f}%</td><td>{vs.get('median_pnl',0) - ts.get('median_pnl',0):+.3f}%</td></tr>
    <tr><td>盈亏比</td><td><b>{val_pf}</b></td><td>{ts.get('profit_factor',0)}</td><td>{val_pf - ts.get('profit_factor',0):+.2f}</td></tr>
    <tr><td>最大盈利</td><td><b>{vs.get('max_pnl',0):+.2f}%</b></td><td>{ts.get('max_pnl',0):+.2f}%</td><td>-</td></tr>
    <tr><td>最大亏损</td><td><b>{vs.get('min_pnl',0):+.2f}%</b></td><td>{ts.get('min_pnl',0):+.2f}%</td><td>-</td></tr>
    <tr><td>标准差</td><td><b>{vs.get('std_pnl',0)}</b></td><td>{ts.get('std_pnl',0)}</td><td>-</td></tr>
  </table>
</div>

<div class="card">
  <h2>📅 分持仓天数（验证集）</h2>
  <table>
    <tr><th>持仓期</th><th>笔数</th><th>胜率</th><th>平均收益</th></tr>
"""
    for k, v in vs.get("by_hold", {}).items():
        html += f"    <tr><td>{k}</td><td>{v['count']}</td><td>{v['win_rate']}%</td><td>{v['avg']:+.3f}%</td></tr>\n"

    html += f"""
  </table>
  <div class="warn">
    💡 注意：T+1～T+4 样本极少（合计{vs.get('total',0) - vs['by_hold'].get('T+5',{}).get('count',0)}笔），
    统计意义有限。T+5 持有到期是最主要场景。
  </div>
</div>

<div class="card">
  <h2>🏆 分信号强度（验证集）</h2>
  <table>
    <tr><th>类型</th><th>评分区间</th><th>笔数</th><th>胜率</th><th>平均收益</th></tr>
    <tr class="highlight"><td>🟢强信号</td><td>≥70分</td><td>{vs['strong_n']}</td><td>{vs['strong_wr']}%</td><td>{vs['strong_avg']:+.3f}%</td></tr>
    <tr><td>🔵中信号</td><td>60-70分</td><td>{vs['medium_n']}</td><td>{vs['medium_wr']}%</td><td>{vs['medium_avg']:+.3f}%</td></tr>
    <tr><td>🟡弱信号</td><td>&lt;60分</td><td>{vs['weak_n']}</td><td>-</td><td>-</td></tr>
  </table>
</div>

<div class="card">
  <h2>🛡️ 止损分析（验证集）</h2>
  <table>
    <tr><th>类型</th><th>次数</th><th>平均收益</th><th>说明</th></tr>
    <tr><td>止损触发（8%）</td><td>{vs['stopped_n']}</td><td class="{'big-num-red' if vs['stopped_avg'] < 0 else 'big-num-green'}">{vs['stopped_avg']:+.3f}%</td><td>收盘价触及止损线</td></tr>
    <tr><td>持有到期</td><td>{vs['not_stopped_n']}</td><td class="{'big-num-green' if vs['not_stopped_avg'] > 0 else 'big-num-red'}">{vs['not_stopped_avg']:+.3f}%</td><td>持有5天后收盘卖出</td></tr>
  </table>
</div>

<div class="card">
  <h2>⚠️ 重要风险提示</h2>
  <ul style="font-size:14px; color: #555; line-height: 1.8;">
    <li>验证集胜率 <b>{val_wr}% {'高于' if val_wr > 50 else '低于'} 50%</b>，意味着超过半数的信号以亏损告终</li>
    <li>盈亏比 = {val_pf}（{'盈利能覆盖亏损，有正期望' if val_pf > 1 else '盈亏基本持平或亏损'}）</li>
    <li>中位数收益 <b>{vs.get('median_pnl',0):+.3f}%</b>（表明大多数信号实际是亏损的，靠少数大涨赚钱）</li>
    <li>T+5 止损设置为 8%，但实际触发时平均损失 {-vs['stopped_avg']:.1f}%，执行会有滑点</li>
    <li>所有结果均为历史回测，不预示未来表现</li>
  </ul>
</div>

<div class="card">
  <h2>💡 改进建议</h2>
  <ol style="font-size:14px; color: #444; line-height: 1.9;">
    <li><b>加入 RSI 过滤</b>：当前 RSI>80 才过滤，可提高到 RSI>75，减少过热追高</li>
    <li><b>加入"连续涨幅"过滤</b>：近3日累计涨幅>15%才过滤，避免在高位追入</li>
    <li><b>缩小持仓数量</b>：同时持仓从10只减到5只，集中火力</li>
    <li><b>加入大盘过滤</b>：沪深300当日下跌>1%时，减少新买入信号</li>
    <li><b>修改止损</b>：从8%改为5%，减少单次最大亏损（盈亏比会改善）</li>
    <li><b>分层持仓</b>：强信号75%仓位，中信号25%仓位</li>
  </ol>
</div>

<div class="footer">
  本报告由 screen_trend_filter v3 自动生成 | 数据来源：.cache/indicators_merged.json | 报告路径：~/stock_reports/
</div>

</body></html>"""

    # 写 HTML 文件
    html_path = REPORT_DIR / f"backtest_report_top3_{today}.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"📄 报告已生成: {html_path}")

    # 发送邮件
    subject = f"📊 screen_trend_filter TOP3 胜率回测报告 {today}（验证集 {val_wr}%胜率）"
    send_email(html_path, subject, html)


if __name__ == "__main__":
    main()
