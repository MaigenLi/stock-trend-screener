#!/usr/bin/env python3
"""
monitor_backtest.py — 监控 batch_backtest 进程完成后自动分析并发邮件
========================================================================
用法:
  python monitor_backtest.py --pid 3731
  python monitor_backtest.py --pid 3731 --check-only  # 只检查，不发邮件

Cron 定时检查（每2分钟）:
  */2 * * * * cd ~/.openclaw/workspace/stock_trend && ~/.venv/bin/python monitor_backtest.py --pid 3731
"""

import sys, os, json, time, argparse, smtplib
from pathlib import Path
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

WORKSPACE = Path(__file__).resolve().parent
OUTPUT_DIR = WORKSPACE / "output"
SUMMARY_FILE = OUTPUT_DIR / "batch_backtest_summary.json"
LOCK_FILE = WORKSPACE / ".monitor_backtest.lock"

# ── 邮件配置（从 ~/.openclaw/.env 读取）───────────────
SMTP_HOST = "smtp.qq.com"
SMTP_PORT = 587

def _load_env():
    env_path = Path.home() / ".openclaw" / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                if k in ("QQ_EMAIL", "QQ_PASS"):
                    yield k, v.strip()

_env = dict(_load_env())
EMAIL_FROM = _env.get("QQ_EMAIL", "920662304@qq.com")
EMAIL_PASSWORD = _env.get("QQ_PASS", "")
EMAIL_TO = _env.get("QQ_EMAIL", "920662304@qq.com")


def process_alive(pid):
    """检查进程是否存活"""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def format_report(summary_data):
    """格式化为 HTML 邮件报告"""
    stats = summary_data["stats"]
    args_info = summary_data["args"]
    all_results = summary_data.get("all_results", [])
    errors = summary_data.get("errors", [])

    # 从所有结果中汇总
    all_trades = []
    for entry in all_results:
        for r in entry.get("results", []):
            all_trades.append(r)

    n = len(all_trades)
    if n == 0:
        return "<p>❌ 回测无交易数据</p>"

    win = [r for r in all_trades if r["ret"] > 0]
    loss = [r for r in all_trades if r["ret"] <= 0]
    gt20 = [r for r in all_trades if r["ret"] > 20]
    gt10 = [r for r in all_trades if r["ret"] > 10]
    sl8 = [r for r in all_trades if r.get("exit_reason") == "sl_8"]
    sl_ma10 = [r for r in all_trades if r.get("exit_reason") == "sl_ma10"]
    hold = [r for r in all_trades if r.get("exit_reason", "hold") == "hold"]

    rets = [r["ret"] for r in all_trades]
    avg_ret = sum(rets) / n
    gains_sum = sum(r for r in rets if r > 0)
    losses_sum = abs(sum(r for r in rets if r < 0))
    pf = gains_sum / losses_sum if losses_sum else float("inf")

    # 日期范围
    dates = sorted(set(e["date"] for e in all_results))
    date_range = f"{dates[0]} ~ {dates[-1]}" if dates else "N/A"

    # 最佳/最差
    best = max(all_trades, key=lambda x: x["ret"])
    worst = min(all_trades, key=lambda x: x["ret"])

    # 月度统计
    monthly = {}
    for r in all_trades:
        m = r.get("signal_date", "")[:7]
        if m not in monthly:
            monthly[m] = {"count": 0, "rets": [], "wins": 0}
        monthly[m]["count"] += 1
        monthly[m]["rets"].append(r["ret"])
        if r["ret"] > 0:
            monthly[m]["wins"] += 1

    monthly_rows = ""
    for m in sorted(monthly.keys()):
        d = monthly[m]
        avg = sum(d["rets"]) / len(d["rets"])
        wr = d["wins"] / d["count"] * 100
        monthly_rows += f"<tr><td>{m}</td><td align='right'>{d['count']}</td>"
        monthly_rows += f"<td align='right'>{wr:.0f}%</td>"
        monthly_rows += f"<td align='right' style='color:{'green' if avg>0 else 'red'}'>{avg:+.1f}%</td></tr>\n"

    # 止损统计
    stop_section = ""
    if sl8 or sl_ma10:
        stop_section = f"""
    <h3>🛡️ 止损统计</h3>
    <table border="1" cellpadding="4" cellspacing="0" style="border-collapse:collapse">
      <tr><th>类型</th><th>次数</th><th>占比</th><th>平均退出收益</th></tr>
      <tr><td>-8% 硬止损</td><td align='right'>{len(sl8)}</td>
          <td align='right'>{len(sl8)/n*100:.1f}%</td>
          <td align='right' style='color:red'>{sum(r['ret'] for r in sl8)/len(sl8):+.1f}%</td></tr>
      <tr><td>MA10 跌破</td><td align='right'>{len(sl_ma10)}</td>
          <td align='right'>{len(sl_ma10)/n*100:.1f}%</td>
          <td align='right' style='color:red'>{sum(r['ret'] for r in sl_ma10)/len(sl_ma10):+.1f}%</td></tr>
      <tr><td>正常持有</td><td align='right'>{len(hold)}</td>
          <td align='right'>{len(hold)/n*100:.1f}%</td>
          <td align='right'>{sum(r['ret'] for r in hold)/len(hold):+.1f}%</td></tr>
    </table>
    """

    return f"""
    <html><body style="font-family: 'Microsoft YaHei', Arial, sans-serif; max-width:700px">
    <h2>📊 screen_double 批量回测报告</h2>

    <h3>📋 回测参数</h3>
    <p>回测区间: {date_range}（{len(dates)} 个信号日）<br/>
    参数: mode={args_info.get('mode','?')}, gain20≥{args_info.get('gain20','?')}%, 
    换手≥{args_info.get('turnover','?')}%, top_n={args_info.get('top_n','?')}<br/>
    hold={args_info.get('hold_days',10)} 天, 止损={'开启' if args_info.get('stop_loss') else '关闭'}</p>

    <h3>📈 核心指标</h3>
    <table border="1" cellpadding="4" cellspacing="0" style="border-collapse:collapse">
      <tr><td><b>总交易</b></td><td align='right'>{n}</td><td><b>胜率</b></td>
          <td align='right' style='color:{'green' if stats.get('win_rate',0)>50 else 'red'}'>{stats.get('win_rate',0):.1f}%</td></tr>
      <tr><td><b>平均收益</b></td><td align='right' style='color:{'green' if avg_ret>0 else 'red'}'>{avg_ret:+.2f}%</td>
          <td><b>中位数</b></td><td align='right'>{stats.get('median_return',0):+.2f}%</td></tr>
      <tr><td><b>盈亏比</b></td><td align='right'>{pf:.2f}</td>
          <td><b>标准差</b></td><td align='right'>{stats.get('std_return',0):.2f}%</td></tr>
      <tr><td><b>最大盈利</b></td><td align='right' style='color:green'>{best['ret']:+.2f}% ({best['code']})</td>
          <td><b>最大亏损</b></td><td align='right' style='color:red'>{worst['ret']:+.2f}% ({worst['code']})</td></tr>
      <tr><td><b>&gt;20%</b></td><td align='right'>{len(gt20)} ({len(gt20)/n*100:.1f}%)</td>
          <td><b>&gt;10%</b></td><td align='right'>{len(gt10)} ({len(gt10)/n*100:.1f}%)</td></tr>
    </table>

    {stop_section}

    <h3>📅 月度表现</h3>
    <table border="1" cellpadding="3" cellspacing="0" style="border-collapse:collapse">
      <tr><th>月份</th><th>交易数</th><th>胜率</th><th>平均收益</th></tr>
      {monthly_rows}
    </table>

    <p style="color:#888; font-size:12px; margin-top:20px">
    生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
    数据异常: {len(errors)} 天</p>
    </body></html>
    """


def send_email(html_body, subject_extra=""):
    """发送 HTML 邮件"""
    if not EMAIL_PASSWORD:
        print("❌ 未配置邮箱密码，跳过发送")
        return False

    subject = f"📊 批量回测报告{subject_extra} - {datetime.now().strftime('%m-%d %H:%M')}"
    msg = MIMEMultipart()
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO
    msg["Subject"] = subject
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.login(EMAIL_FROM, EMAIL_PASSWORD)
            server.sendmail(EMAIL_FROM, [EMAIL_TO], msg.as_string())
        print(f"✅ 邮件已发送到 {EMAIL_TO}")
        return True
    except Exception as e:
        print(f"❌ 邮件发送失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="监控 batch_backtest 完成后分析并发邮件")
    parser.add_argument("--pid", type=int, required=True, help="batch_backtest 进程 PID")
    parser.add_argument("--check-only", action="store_true", help="只检查不发送")
    parser.add_argument("--force", action="store_true", help="强制执行（跳过进程检查）")
    args = parser.parse_args()

    # ── 防重复：已有锁文件跳过 ─────────────────────
    if LOCK_FILE.exists():
        print(f"⚠️ 锁文件已存在（可能已发送过报告），跳过")
        return

    # ── 检查进程 ──────────────────────────────────
    if not args.force and process_alive(args.pid):
        print(f"⏳ 进程 {args.pid} 仍在运行，等待...")
        return

    print(f"✅ 进程 {args.pid} 已结束，开始分析回测结果...")

    # ── 读取结果 ──────────────────────────────────
    if not SUMMARY_FILE.exists():
        print(f"❌ 结果文件不存在: {SUMMARY_FILE}")
        return

    try:
        with open(SUMMARY_FILE) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"❌ 读取结果文件失败: {e}")
        return

    # ── 生成报告 ──────────────────────────────────
    html = format_report(data)

    if args.check_only:
        print("📝 --check-only 模式，仅生成报告（不发邮件）")
        report_path = OUTPUT_DIR / "backtest_report.html"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"💾 报告已保存: {report_path}")
        return

    # ── 发送邮件 + 设锁 ────────────────────────────
    success = send_email(html)
    if success:
        LOCK_FILE.write_text(datetime.now().isoformat())
        print(f"🔒 锁文件已创建: {LOCK_FILE}")
    else:
        print("⚠️ 邮件发送失败，未创建锁文件（下次还会重试）")


if __name__ == "__main__":
    main()
