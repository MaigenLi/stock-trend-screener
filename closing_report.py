#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
收盘报告生成器
================================
每个交易日 17:00 自动运行：
  1. 获取大盘指数 + 板块数据
  2. 读取今日选股结果（gain_turnover_screen）
  3. 读取昨日信号验证结果（signal_validator）
  4. 生成 PDF 报告
  5. 发送邮件到 ~/.openclaw/.env 中配置的 QQ 邮箱

用法：
  python closing_report.py                    # 直接运行（生成+发送）
  python closing_report.py --no-email        # 仅生成 PDF，不发送
"""

from __future__ import annotations

import argparse
import smtplib
import subprocess
import sys
import time
import signal
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from pathlib import Path
from datetime import datetime

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

# ── 路径配置 ───────────────────────────────────────────
WORKSPACE = Path(__file__).parent.parent.resolve()
REPORTS_DIR = Path.home() / "stock_reports"
FONT_PATH = "/mnt/c/Windows/Fonts/simhei.ttf"
TRACKER_CSV = WORKSPACE / "stock_trend" / "feedback_tracker.csv"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ── PDF 字体 ─────────────────────────────────────────────
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.colors import HexColor, black, white
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

try:
    pdfmetrics.registerFont(TTFont("SimHei", FONT_PATH))
    pdfmetrics.registerFont(TTFont("SimHei-Bold", FONT_PATH))
    pdfmetrics.registerFontFamily("SimHei", "SimHei", "SimHei-Bold", "SimHei")
    BASE_FONT = "SimHei"
except Exception as e:
    print(f"[closing_report] 字体加载失败: {e}，使用 Helvetica")
    BASE_FONT = "Helvetica"

PAGE_W, PAGE_H = A4
MARGIN = 1.5 * cm
TEXT_W = PAGE_W - 2 * MARGIN

C_H1     = HexColor("#0f3460")   # 封面/H1/H2 背景（同 md2pdf）
C_BLUE   = HexColor("#0f3460")
C_H3     = HexColor("#1a237e")   # H3 背景
C_SECTION= HexColor("#0f3460")   # 区块标题背景（对齐 md2pdf H2）
C_LIGHT  = HexColor("#f5f5f5")
C_BORDER = HexColor("#cccccc")
C_DARK   = HexColor("#1a1a2e")
C_WARN   = HexColor("#fff8e1")
C_CODE   = HexColor("#1e1e1e")   # 代码块背景

def S(size, leading=None, color=black, align=TA_LEFT, bold=False):
    fn = "SimHei-Bold" if bold else "SimHei"
    return ParagraphStyle(
        "s", fontName=fn, fontSize=size,
        leading=leading or size * 1.45,   # 对齐 md2pdf: 1.45× 行高
        textColor=color, alignment=align
    )

TITLE_STYLE = S(22, bold=True, color=white, align=TA_CENTER)  # md2pdf封面: 22pt
SUB_STYLE   = S(10, color=HexColor("#cccccc"), align=TA_CENTER)
SEC_STYLE   = S(11, bold=True, color=white)    # md2pdf H2: 11pt bold
BODY_STYLE  = S(9, color=black)                # md2pdf 正文: 9pt
WARN_STYLE  = S(8.5, color=HexColor("#555555"))
TAG_STYLE   = S(8.5, color=HexColor("#222222"))
HDR_STYLE   = S(8, bold=True, color=white)     # md2pdf 表头: 8pt bold
CELL_STYLE  = S(8, color=black)                # md2pdf 表格: 8pt
SMALL_STYLE = S(8, color=HexColor("#555555"), align=TA_CENTER)
INDEX_STYLE = S(10, color=C_DARK, bold=True)
GREEN_STYLE= S(10, color=HexColor("#2e7d32"), bold=True)
RED_STYLE  = S(10, color=HexColor("#c62828"), bold=True)


def title_block(title: str, subtitle: str) -> Table:
    inner = [Paragraph(title, TITLE_STYLE), Paragraph(subtitle, SUB_STYLE)]
    tbl = Table([[inner]], colWidths=[TEXT_W])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), C_H1),   # md2pdf封面: #0f3460
        ("TOPPADDING",   (0,0), (-1,-1), 22),
        ("BOTTOMPADDING",(0,0), (-1,-1), 22),
        ("LEFTPADDING",  (0,0), (-1,-1), 16),
        ("RIGHTPADDING", (0,0), (-1,-1), 16),
    ]))
    return tbl


def section_block(text: str) -> Table:
    tbl = Table([[Paragraph(text, SEC_STYLE)]], colWidths=[TEXT_W])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0,0), (-1,-1), C_SECTION),
        ("TOPPADDING",   (0,0), (-1,-1), 6),
        ("BOTTOMPADDING",(0,0), (-1,-1), 6),
        ("LEFTPADDING",  (0,0), (-1,-1), 10),
        ("RIGHTPADDING", (0,0), (-1,-1), 10),
    ]))
    return tbl


def note_block(text: str) -> Table:
    tbl = Table([[Paragraph(text, WARN_STYLE)]], colWidths=[TEXT_W])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0,0), (-1,-1), C_WARN),
        ("BOX",          (0,0), (-1,-1), 1, HexColor("#ffe082")),
        ("TOPPADDING",   (0,0), (-1,-1), 7),
        ("BOTTOMPADDING",(0,0), (-1,-1), 7),
        ("LEFTPADDING",  (0,0), (-1,-1), 10),
        ("RIGHTPADDING", (0,0), (-1,-1), 10),
    ]))
    return tbl


def make_table(headers: list, rows: list, col_widths: list) -> Table:
    data = [[Paragraph(h, HDR_STYLE) for h in headers]] + \
           [[Paragraph(str(c), CELL_STYLE) for c in row] for row in rows]
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,0),  C_BLUE),
        ("TEXTCOLOR",     (0,0), (-1,0),  white),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [white, C_LIGHT]),
        ("GRID",          (0,0), (-1,-1), 0.4, C_BORDER),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING",   (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0), (-1,-1), 4),
        ("LEFTPADDING",  (0,0), (-1,-1), 5),
        ("RIGHTPADDING", (0,0), (-1,-1), 5),
    ]))
    return t


# ── 步骤0：获取大盘 + 板块数据 ─────────────────────────
def get_index_data() -> list[dict]:
    """从腾讯财经获取四大指数实时数据（稳定）。"""
    import urllib.request
    indices = [
        ("sh000001", "上证指数"),
        ("sz399001", "深证成指"),
        ("sz399006", "创业板指"),
        ("sh000300", "沪深300"),
    ]
    results = []
    codes_str = ",".join(c for c, _ in indices)
    url = f"https://qt.gtimg.cn/q={codes_str}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            txt = resp.read().decode("gbk", errors="replace")
        for line in txt.splitlines():
            if "=" not in line:
                continue
            body = line.split("=", 1)[1].strip('"; \n')
            fields = body.split("~")
            if len(fields) < 10:
                continue
            try:
                name = fields[1]
                price = float(fields[3])
                prev_close = float(fields[4])
                pct = (price - prev_close) / prev_close * 100
                # fields[37] = 成交额（万元），fields[36] = 成交量（手）
                amount = float(fields[37]) if fields[37].isdigit() else 0
                results.append({
                    "name": name,
                    "price": price,
                    "pct": pct,
                    "amount": amount / 10000,  # 万元 → 亿元
                })
            except (ValueError, IndexError):
                continue
    except Exception as e:
        print(f"⚠️ 指数数据获取失败: {e}")
    return results


def get_sector_data() -> tuple[list, list]:
    """
    获取板块数据：返回 (领涨板块列表, 领跌板块列表)，每项含 name/pct。
   优先使用新浪行业板块 API（GBK 编码，175个板块），
   失败时尝试 AkShare EastMoney（已被封）。
    """
    # ── 方案1：新浪行业板块（稳定）──────────────────────
    try:
        import re, urllib.request
        url = "https://vip.stock.finance.sina.com.cn/q/view/newFLJK.php?param=class"
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://finance.sina.com.cn"
        })
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read()
        txt = raw.decode("gbk", errors="replace")
        # 解析 JS 对象: "gn_xxx":"gn_xxx,名称,count,avg,pct,..."
        pattern = re.compile(r'"(gn_\w+)":"([^"]+)"')
        matches = pattern.findall(txt)
        boards = []
        for key, value in matches:
            parts = value.split(",")
            if len(parts) < 5:
                continue
            try:
                name = parts[1]
                pct = float(parts[4]) if parts[4].replace(".", "").replace("-", "").isdigit() else 0
                boards.append({"name": name, "pct": pct})
            except (ValueError, IndexError):
                continue
        boards.sort(key=lambda x: x["pct"], reverse=True)
        top = boards[:10]
        bottom = boards[-10:][::-1]
        print(f"📊 新浪行业板块: {len(boards)} 个, 领涨 {top[0]['name'] if top else '-'} {top[0]['pct'] if top else 0:+.2f}%")
        return top, bottom
    except Exception as e:
        print(f"⚠️ 新浪板块获取失败: {type(e).__name__}: {str(e)[:60]}")

    # ── 方案2：AkShare EastMoney（备用）─────────────────
    try:
        import akshare as ak
        board = ak.stock_board_industry_name_em()
        board = board.sort_values("涨跌幅", ascending=False)
        top = []
        for _, row in board.head(10).iterrows():
            try:
                top.append({"name": str(row["名称"]), "pct": float(row["涨跌幅"])})
            except (ValueError, KeyError):
                continue
        bottom = []
        for _, row in board.tail(10).iloc[::-1].iterrows():
            try:
                bottom.append({"name": str(row["名称"]), "pct": float(row["涨跌幅"])})
            except (ValueError, KeyError):
                continue
        return top, bottom
    except Exception as e:
        print(f"⚠️ AkShare 板块获取失败: {type(e).__name__}: {str(e)[:60]}")
        return [], []


# ── 步骤1：运行 gain_turnover_screen ─────────────────
def run_screen(top_n: int = 50) -> tuple[list, str]:
    print("📊 运行选股...")
    result = subprocess.run(
        [sys.executable, "gain_turnover_screen.py",
         "--top-n", str(top_n), "--output", "/dev/stdout"],
        capture_output=True, text=True,
        cwd=str(WORKSPACE / "stock_trend"),
        timeout=300,
    )
    raw = result.stdout + result.stderr
    if result.returncode != 0:
        print(f"⚠️ 选股脚本异常: {result.stderr[-200:]}")
    rows = parse_screen_table(raw)
    return rows, raw


def parse_screen_table(raw: str) -> list[dict]:
    import re
    rows = []
    code_pat = re.compile(r"^(sh|sz|bj)(\d{6})\s+(\S+)")
    for line in raw.splitlines():
        line = line.strip()
        m = code_pat.match(line.lower())
        if not m:
            continue
        parts = line.split("\t")
        if len(parts) < 10:
            continue
        try:
            rows.append({
                "code": f"{m.group(1)}{m.group(2)}",
                "name": parts[1].strip(),
                "score": float(parts[3].strip()) if parts[3].strip().replace(".", "").isdigit() else 0,
                "gain": parts[4].strip(),
                "amount": parts[5].strip() if len(parts) > 5 else "-",
                "turnover": parts[6].strip() if len(parts) > 6 else "-",
                "rsi": parts[7].strip() if len(parts) > 7 else "-",
                "ext": parts[8].strip() if len(parts) > 8 else "-",
                "close": parts[9].strip() if len(parts) > 9 else "-",
            })
        except (ValueError, IndexError):
            continue
    return rows


# ── 步骤2：读取 signal_validation 报告 ───────────────
def get_validation_results() -> tuple[str, str, list]:
    today = datetime.now()
    from datetime import timedelta
    yesterday = (today - timedelta(days=1)).strftime("%Y-%m-%d")
    val_file = REPORTS_DIR / f"signal_validation_{yesterday}.txt"
    if not val_file.exists():
        today_str = today.strftime("%Y-%m-%d")
        val_file = REPORTS_DIR / f"signal_validation_{today_str}.txt"
    if not val_file.exists():
        return "（昨日验证报告未找到）", "", []
    text = val_file.read_text(encoding="utf-8")
    rows = []
    import re
    code_pat = re.compile(r"^(sh|sz|bj)\d+\s+")
    tag_pat = re.compile(r"(🟢|🔵|🟡|🔴)")
    for line in text.splitlines():
        if code_pat.match(line.strip()):
            parts = line.split("\t")
            if len(parts) >= 14:
                tag = tag_pat.search(line)
                full_tag = parts[13].strip() if len(parts) > 13 else (tag.group(1) if tag else "🔴")
                rows.append({
                    "code": parts[0].strip(),
                    "name": parts[1].strip(),
                    "signal_date": parts[2].strip(),
                    "signal_price": parts[3].strip(),
                    "open": parts[4].strip(),
                    "close": parts[5].strip(),
                    "ret_actual": parts[6].strip(),
                    "ret_signal": parts[7].strip(),
                    "ret_high": parts[8].strip(),
                    "hit_3": parts[9].strip(),
                    "hit_5": parts[10].strip(),
                    "stop_loss": parts[11].strip(),
                    "score": parts[12].strip(),
                    "tag": full_tag if full_tag else (tag.group(1) if tag else "🔴"),
                })
    return text[:2000], text, rows


# ── 步骤3：读取反馈数据库统计 ─────────────────────────
def get_tracker_stats() -> str:
    if not TRACKER_CSV.exists():
        return "（反馈数据库为空）"
    try:
        import csv
        rows = list(csv.DictReader(open(TRACKER_CSV, encoding="utf-8")))
        if not rows:
            return "（反馈数据库为空）"
        total = len(rows)
        scores = [float(r["quality_score"]) for r in rows if r.get("quality_score")]
        rets = [float(r["ret_actual"]) for r in rows if r.get("ret_actual")]
        win = sum(1 for r in rets if r > 0)
        avg = sum(rets) / len(rets) if rets else 0
        avg_score = sum(scores) / len(scores) if scores else 0
        return (
            f"累计信号 {total} 只 | "
            f"胜率 {win}/{len(rets)}={win*100//len(rets) if rets else 0}% | "
            f"均收益 {avg:+.2f}% | 均评分 {avg_score:.1f}"
        )
    except Exception as e:
        return f"（读取统计失败: {e}）"


# ── 步骤4：构建 PDF ───────────────────────────────────
def build_pdf(today_str: str, screen_rows: list, val_full: str,
              val_rows: list, tracker_stats: str, index_data: list,
              top_sectors: list, bottom_sectors: list) -> Path:
    output_path = REPORTS_DIR / f"closing_report_{today_str}.pdf"
    story = []

    # 标题
    story.append(title_block(
        f"A 股收盘报告 {today_str}",
        f"生成时间: {datetime.now().strftime('%H:%M')} | gain_turnover 自我进化策略"
    ))
    story.append(Spacer(1, 0.3*cm))

    # ── 大盘指数 ──────────────────────────────────────────
    story.append(section_block("📊 今日大盘指数"))
    story.append(Spacer(1, 0.2*cm))
    if index_data:
        idx_rows = []
        for idx in index_data:
            pct = idx["pct"]
            sign = "+" if pct >= 0 else ""
            color_hex = "c62828" if pct >= 0 else "2e7d32"
            style_note = ParagraphStyle(
                "ip", fontName="SimHei", fontSize=10,
                textColor=HexColor(f"#{color_hex}"), alignment=TA_LEFT
            )
            idx_rows.append([
                Paragraph(idx["name"], style_note),
                Paragraph(f"{idx['price']:,.2f}", style_note),
                Paragraph(f"{sign}{pct:+.2f}%", style_note),
                Paragraph(f"{idx['amount']:.0f}亿", style_note),
            ])
        idx_tbl = Table(idx_rows, colWidths=[3*cm, 3.5*cm, 3*cm, 3*cm])
        idx_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,0),  HexColor("#0f3460")),
            ("TEXTCOLOR",    (0,0), (-1,0),  white),
            ("ROWBACKGROUNDS",(0,1), (-1,-1), [white, C_LIGHT]),
            ("GRID",         (0,0), (-1,-1), 0.4, C_BORDER),
            ("VALIGN",       (0,0), (-1,-1), "MIDDLE"),
            ("TOPPADDING",   (0,0), (-1,-1), 5),
            ("BOTTOMPADDING",(0,0), (-1,-1), 5),
            ("LEFTPADDING",  (0,0), (-1,-1), 6),
            ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ]))
        story.append(idx_tbl)
    else:
        story.append(Paragraph("（指数数据获取失败）", BODY_STYLE))
    story.append(Spacer(1, 0.3*cm))

    # ── 板块涨跌 ─────────────────────────────────────────
    story.append(section_block("🔥 行业板块涨跌 TOP"))
    story.append(Spacer(1, 0.2*cm))
    if top_sectors or bottom_sectors:
        sector_rows = []
        # 领涨
        for s in top_sectors[:8]:
            pct = s["pct"]
            sign = "+" if pct >= 0 else ""
            color = "c62828" if pct >= 0 else "2e7d32"
            p = ParagraphStyle("sp", fontName="SimHei", fontSize=9,
                               textColor=HexColor(f"#{color}"), alignment=TA_LEFT)
            sector_rows.append([
                Paragraph("📈", TAG_STYLE),
                Paragraph(s["name"], p),
                Paragraph(f"{sign}{pct:.2f}%", p),
            ])
        # 领跌
        for s in bottom_sectors[:8]:
            pct = s["pct"]
            sign = "+" if pct >= 0 else ""
            color = "c62828" if pct >= 0 else "2e7d32"
            p = ParagraphStyle("sp", fontName="SimHei", fontSize=9,
                               textColor=HexColor(f"#{color}"), alignment=TA_LEFT)
            sector_rows.append([
                Paragraph("📉", TAG_STYLE),
                Paragraph(s["name"], p),
                Paragraph(f"{sign}{pct:.2f}%", p),
            ])
        if sector_rows:
            sec_tbl = Table(sector_rows, colWidths=[1.2*cm, 4.5*cm, 2.5*cm])
            sec_tbl.setStyle(TableStyle([
                ("ROWBACKGROUNDS",(0,0), (-1,-1), [white, C_LIGHT]),
                ("GRID",          (0,0), (-1,-1), 0.3, C_BORDER),
                ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
                ("TOPPADDING",   (0,0), (-1,-1), 3),
                ("BOTTOMPADDING",(0,0), (-1,-1), 3),
                ("LEFTPADDING",  (0,0), (-1,-1), 5),
                ("RIGHTPADDING", (0,0), (-1,-1), 5),
            ]))
            story.append(sec_tbl)
    else:
        story.append(Paragraph("（板块数据获取超时，请检查 AkShare 接口）", BODY_STYLE))
    story.append(Spacer(1, 0.3*cm))

    # ── 策略数据库累计统计 ────────────────────────────────
    story.append(section_block("🧠 策略信号数据库累计统计"))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(tracker_stats, BODY_STYLE))
    story.append(Spacer(1, 0.4*cm))

    # ── 今日选股结果 ──────────────────────────────────────
    story.append(section_block(f"📈 今日选股结果（共 {len(screen_rows)} 只）"))
    story.append(Spacer(1, 0.2*cm))
    if screen_rows:
        cols = ["代码", "名称", "总分", "窗口涨幅", "20日额", "5日换手", "RSI", "偏离MA20", "收盘"]
        cw   = [2.5*cm, 2.5*cm, 1.8*cm, 2.5*cm, 2.2*cm, 2.0*cm, 1.5*cm, 2.2*cm, 2.0*cm]
        data_rows = [[
            r["code"], r["name"], f"{r['score']:.1f}",
            r["gain"], r["amount"], r["turnover"],
            r["rsi"], r["ext"], r["close"]
        ] for r in screen_rows[:50]]
        story.append(make_table(cols, data_rows, cw))
        if len(screen_rows) > 50:
            story.append(Paragraph(f"（仅显示前50只，共 {len(screen_rows)} 只）", SMALL_STYLE))
    else:
        story.append(note_block("⚠️ 今日选股结果为空，请检查脚本执行情况"))
    story.append(Spacer(1, 0.4*cm))

    # ── 昨日信号验证（表格样式，原文件解析）──────────────
    story.append(section_block("🔍 昨日信号验证（T+1 日收益）"))
    story.append(Spacer(1, 0.15*cm))
    if val_rows:
        # 从 val_full 提取统计行（第4行）
        stat_line = ""
        if val_full:
            for line in val_full.splitlines():
                if "均分=" in line or "上涨=" in line:
                    stat_line = line.strip()
                    break

        # 图例+统计行
        leg_para = ParagraphStyle("leg_para", fontName="SimHei", fontSize=9,
                                  textColor=HexColor("#ffd600"), bold=True, leading=13)
        leg_tbl = Table([[Paragraph(f"验证样本 {len(val_rows)} 只  {stat_line}", leg_para)]],
                        colWidths=[TEXT_W])
        leg_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), HexColor("#1a237e")),
            ("BOX", (0,0), (-1,-1), 0.5, HexColor("#ffd600")),
            ("TOPPADDING", (0,0), (-1,-1), 7),
            ("BOTTOMPADDING", (0,0), (-1,-1), 7),
            ("LEFTPADDING", (0,0), (-1,-1), 10),
            ("RIGHTPADDING", (0,0), (-1,-1), 10),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ]))
        story.append(leg_tbl)
        story.append(Spacer(1, 0.1*cm))

        def tagParagraph(tag_text):
            """彩色评价单元格"""
            if "🟢" in tag_text:
                bg, c = HexColor("#e8f5e9"), HexColor("#1b5e20")
            elif "🔵" in tag_text:
                bg, c = HexColor("#e3f2fd"), HexColor("#0d47a1")
            elif "🟡" in tag_text:
                bg, c = HexColor("#fff8e1"), HexColor("#e65100")
            else:
                bg, c = HexColor("#ffebee"), HexColor("#b71c1c")
            p = ParagraphStyle("ct", fontName="SimHei", fontSize=8,
                               textColor=c, bold=True, alignment=TA_CENTER)
            cell = Table([[Paragraph(tag_text, p)]], colWidths=[cw2[-1]])
            cell.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,-1), bg),
                ("BOX", (0,0), (-1,-1), 0.5, c),
                ("TOPPADDING", (0,0), (-1,-1), 0),
                ("BOTTOMPADDING", (0,0), (-1,-1), 0),
                ("LEFTPADDING", (0,0), (-1,-1), 0),
                ("RIGHTPADDING", (0,0), (-1,-1), 0),
                ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ]))
            return cell

        cols2 = ["代码", "名称", "今收", "真实收益", "参收益", "高涨", "评分", "评价"]
        cw2   = [2.0*cm, 2.2*cm, 2.5*cm, 2.2*cm, 2.2*cm, 2.0*cm, 1.7*cm, 2.2*cm]
        hdr_cells = [Paragraph(h, HDR_STYLE) for h in cols2]
        data_rows2 = []
        for r in val_rows:
            row = [
                Paragraph(r["code"], CELL_STYLE),
                Paragraph(r["name"], CELL_STYLE),
                Paragraph(r["close"], CELL_STYLE),
                Paragraph(r["ret_actual"], CELL_STYLE),
                Paragraph(r["ret_signal"], CELL_STYLE),
                Paragraph(r["ret_high"], CELL_STYLE),
                Paragraph(r["score"], CELL_STYLE),
                tagParagraph(r["tag"]),
            ]
            data_rows2.append(row)

        val_tbl = Table([hdr_cells] + data_rows2, colWidths=cw2, repeatRows=1)
        val_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,0),  C_BLUE),
            ("TEXTCOLOR",     (0,0), (-1,0),  white),
            ("ROWBACKGROUNDS",(0,1), (-1,-1), [white, C_LIGHT]),
            ("GRID",          (0,0), (-1,-1), 0.4, C_BORDER),
            ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
            ("TOPPADDING",   (0,0), (-1,-1), 3),
            ("BOTTOMPADDING",(0,0), (-1,-1), 3),
            ("LEFTPADDING",  (0,0), (-1,-1), 4),
            ("RIGHTPADDING", (0,0), (-1,-1), 4),
        ]))
        story.append(val_tbl)
    else:
        story.append(Paragraph("（验证报告为空）", BODY_STYLE))
    story.append(Spacer(1, 0.3*cm))

    # 评分说明
    story.append(note_block(
        "评分标准（基于 T+1 真实收益）：🟢优秀≥85  🔵良好≥70  🟡及格≥55  🔴失效<55  |  "
        "真实收益 = T+1开盘买入 → T+1收盘卖出（持有1天）"
    ))

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=MARGIN,
    )
    doc.build(story)
    print(f"✅ PDF 已生成: {output_path}")
    return output_path


# ── 步骤5：发送邮件 ───────────────────────────────────
def send_email(pdf_path: Path, today_str: str):
    if not EMAIL_PASSWORD:
        print("⚠️ 未在 ~/.openclaw/.env 中找到 QQ_PASS，跳过发送")
        return
    msg = MIMEMultipart()
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO
    msg["Subject"] = f"A股收盘报告 {today_str}"
    body = f"""
    <html><body>
    <h2>A股收盘报告 {today_str}</h2>
    <p>附件为今日收盘报告 PDF，请查收。</p>
    <hr/>
    <p style="color:#888;font-size:12px">
    本报告由 gain_turnover 自我进化策略系统自动生成。<br/>
    评分标准：🟢优秀≥85  🔵良好≥70  🟡及格≥55  🔴失效&lt;55<br/>
    真实收益 = T+1开盘买入 → T+1收盘卖出（持有1天）
    </p>
    </body></html>
    """
    msg.attach(MIMEText(body, "html", "utf-8"))
    if pdf_path.exists():
        with open(pdf_path, "rb") as f:
            part = MIMEApplication(f.read(), Name=pdf_path.name)
            part["Content-Disposition"] = f'attachment; filename="{pdf_path.name}"'
            msg.attach(part)
    print(f"📧 发送邮件到 {EMAIL_TO}...")
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.login(EMAIL_FROM, EMAIL_PASSWORD)
            server.sendmail(EMAIL_FROM, [EMAIL_TO], msg.as_string())
        print(f"✅ 邮件已发送")
    except Exception as e:
        print(f"❌ 邮件发送失败: {e}")


# ── 主流程 ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="收盘报告生成器")
    parser.add_argument("--top-n", type=int, default=50, help="选股数量，默认50")
    parser.add_argument("--no-email", action="store_true", help="仅生成 PDF，不发送邮件")
    parser.add_argument("--screen-only", action="store_true", help="仅运行选股，不生成报告")
    args = parser.parse_args()

    today_str = datetime.now().strftime("%Y-%m-%d")
    print(f"\n{'='*50}")
    print(f"📋 收盘报告生成器 {today_str}")
    print(f"{'='*50}\n")

    # 1. 获取大盘指数（腾讯财经，稳定）
    print("📊 获取大盘指数...")
    index_data = get_index_data()
    for idx in index_data:
        sign = "+" if idx["pct"] >= 0 else ""
        print(f"  {idx['name']}: {idx['price']:,.2f} {sign}{idx['pct']:.2f}%")

    # 2. 获取板块数据（AkShare，带超时）
    print("📊 获取板块数据...")
    top_sectors, bottom_sectors = get_sector_data()
    if top_sectors:
        print(f"  领涨: {', '.join(s['name'] for s in top_sectors[:5])}")
    if bottom_sectors:
        print(f"  领跌: {', '.join(s['name'] for s in bottom_sectors[:5])}")

    # 3. 获取策略统计数据
    tracker_stats = get_tracker_stats()
    print(f"  策略统计: {tracker_stats}")

    # 4. 获取验证结果
    _, val_full, val_rows = get_validation_results()

    if args.screen_only:
        print("✅ [仅选股模式] 完成")
        return

    # 5. 读取今日选股结果
    today_screen = REPORTS_DIR / f"daily_screen_{today_str}.txt"
    if today_screen.exists():
        raw = today_screen.read_text(encoding="utf-8")
        screen_rows = parse_screen_table(raw)
        print(f"📊 从 {today_screen.name} 读取到 {len(screen_rows)} 只")
    else:
        print("⚠️ 今日选股文件不存在，直接运行...")
        screen_rows, _ = run_screen(args.top_n)

    # 6. 生成 PDF
    pdf_path = build_pdf(
        today_str, screen_rows, val_full, val_rows, tracker_stats,
        index_data, top_sectors, bottom_sectors
    )

    # 7. 发送邮件
    if not args.no_email:
        send_email(pdf_path, today_str)
    else:
        print("⏭️ 已跳过邮件发送（--no-email）")

    print(f"\n✅ 收盘报告完成: {pdf_path}\n")


if __name__ == "__main__":
    main()
