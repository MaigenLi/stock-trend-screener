#!/usr/bin/env python3
"""
mootdx 实时行情监控脚本
功能：读取 JSON/JSONL 股票列表，按指定日期获取历史分时，逐分钟对比实时成交判断是否放量
放量定义：今日同分钟累计成交量 > 昨日同分钟累计成交量 × 阈值

界面：终端原地刷新，不翻屏
用法：
    python3 mootdx_volume_monitor.py \
        --date 2026-04-30    # 默认为今天，自动找最近交易日作基准
"""

import json, argparse, sys, time, re, os
from pathlib import Path
from datetime import datetime, date as date_type

# --------------------------------------------------------------------------- #
# 常量
# --------------------------------------------------------------------------- #
MARKET_BY_CODE = {
    1: lambda c: c.startswith(("6", "9")),   # 上海
    0: lambda c: c and c[0] in ("0", "1", "2", "3"),  # 深圳
}
TOTAL_MINUTES = 240
CLEAR_SCREEN  = "\033[2J\033[H"
CLEAR_LINE    = "\033[2K"
_ansi_pat     = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]|\x1b\[K")


# --------------------------------------------------------------------------- #
# 工具函数
# --------------------------------------------------------------------------- #
def parse_args():
    parser = argparse.ArgumentParser(
        description="mootdx 实时行情监控 — 分钟精度放量分析（原地刷新）"
    )
    parser.add_argument("--file", "-f",
                        default="output/watchlist.EBK",
                        help="输入文件路径（支持 .json / .jsonl / .EBK），默认 output/watchlist.EBK")
    parser.add_argument("--date", "-d",
                        default=datetime.now().strftime("%Y-%m-%d"),
                        help="指定日期（默认今天），基准分时数据自动取该日期往前最近的240分钟交易日")
    parser.add_argument("--interval", "-i", type=int, default=5,
                        help="轮询间隔（秒），默认5秒")
    parser.add_argument("--vol-ratio", "-r", type=float, default=1.3,
                        help="放量判定阈值，默认1.3倍")
    parser.add_argument("--count", "-c", type=int, default=0,
                        help="查询次数，0=无限循环（默认）")
    group = parser.add_argument_group("消息预警")
    group.add_argument("--news", "-n", action="store_true",
                        help="同时抓取个股重大公告/提示（港澳资讯最新提示，踩坑专用）")
    group.add_argument("--tavily", "-t", action="store_true",
                        help="开启 Tavily 网络新闻搜索（需 TAVILY_API_KEY 环境变量）")
    parser.add_argument("--output", "-o",
                        help="输出结果 JSON 文件路径")
    return parser.parse_args()


def auto_market(code: str) -> int:
    for market, pred in MARKET_BY_CODE.items():
        if pred(code):
            return market
    return 1


def load_stocks(path: str) -> list[dict]:
    """
    支持格式：
    - .json  / .jsonl → JSON 数组或逐行 JSON
    - .EBK / .ebk     → 通达信自选股导出（二进制 ASCII）
      格式：\r\n + [0|1] + XXXXXX + \r\n  (市场 0=深 1=沪 + 6位代码)
    """
    p = Path(path)
    if not p.is_absolute():
        p = Path.cwd() / path

    # ── EBK 格式 ──
    if p.suffix.lower() in (".ebk",):
        raw_bytes = p.read_bytes()
        text = raw_bytes.decode("ascii", errors="ignore")
        codes = [
            t[1:] for t in text.split("\r\n")
            if len(t) == 7 and t[0] in "01"
        ]
        codes = list(dict.fromkeys(codes))
        # 从名称库查找股票名
        names_cache = {}
        try:
            # gain_turnover 在 stock_trend 目录下
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
            from gain_turnover import load_stock_names as _load_names
            names_cache = _load_names()
        except Exception:
            pass
        return [{"code": c, "name": names_cache.get(c, "")} for c in codes]

    # ── JSON / JSONL ──
    raw = p.read_text().strip()

    if raw.startswith("["):
        data = json.loads(raw)
        stocks = data if isinstance(data, list) else [data]
    else:
        stocks = []
        for line in raw.split("\n"):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, list):
                stocks.extend(obj)
            elif isinstance(obj, dict):
                found = False
                for key in ["stocks", "data", "results", "items"]:
                    if key in obj and isinstance(obj[key], list):
                        stocks.extend(obj[key])
                        found = True
                        break
                if not found:
                    stocks.append(obj)

    return [{"code": str(s.get("code") or s.get("symbol") or "").strip(),
             "name": s.get("name") or ""}
            for s in stocks]


def get_client():
    from mootdx.quotes import Quotes
    return Quotes.factory(market="std")


def get_today_minute_bars(client, symbol: str, market: int) -> list:
    try:
        return client.client.get_minute_time_data(market, symbol)
    except Exception:
        return []


def get_yesterday_minute_bars(client, symbol: str, market: int,
                              yesterday_date: date_type) -> list:
    try:
        date_int = (yesterday_date.year * 10000
                    + yesterday_date.month * 100
                    + yesterday_date.day)
        return client.client.get_history_minute_time_data(market, symbol, date_int)
    except Exception:
        return []


def get_realtime(client, symbol: str, market: int) -> dict | None:
    try:
        result = client.client.get_security_quotes([(market, symbol)])
        return dict(result[0]) if result else None
    except Exception:
        return None


def find_last_trading_day(reference_date: date_type) -> date_type:
    """
    从 reference_date 往前找最近一个有240分钟分时数据的交易日
    跳过周末，一路往前试最多14天
    用 get_history_minute_time_data（精确取指定日期的分时）
    """
    from datetime import timedelta
    day = reference_date
    for _ in range(14):
        if day.weekday() >= 5:          # 跳过周末
            day -= timedelta(days=1)
            continue
        date_int = day.year * 10000 + day.month * 100 + day.day
        bars = get_history_minute_bar(date_int, "000001", 0)  # 用平安测试该日期
        if len(bars) >= 230:             # 230根以上算有效交易日
            return day
        day -= timedelta(days=1)
    return reference_date               # 兜底返回原日期


def get_history_minute_bar(date_int: int, symbol: str, market: int) -> list:
    """拉指定日期的历史分时，失败返回空列表"""
    try:
        client = get_client()
        return client.client.get_history_minute_time_data(market, symbol, date_int)
    except Exception:
        return []


# --------------------------------------------------------------------------- #
# 消息预警（踩坑检测）
# --------------------------------------------------------------------------- #

def get_company_news(client, market: int, code: str) -> str:
    """
    通过 mootdx 港澳资讯接口获取个股最新提示（重大公告/风险提示）
    client: TdxHq_API 实例（直接有 get_company_info_category 方法）
    返回摘要文本，失败返回空字符串
    """
    try:
        cats = client.get_company_info_category(market, code)
        if not cats:
            return ""
        # 找"最新提示"类别（通常第一个就是）
        target = next((c for c in cats if "最新提示" in c.get("name", "")), cats[0])
        content = client.get_company_info_content(
            market, code,
            target["filename"],
            target["start"],
            target["length"]
        )
        if not content:
            return ""
        # 提取前 300 字关键内容（去格式符号）
        lines = [l.strip() for l in content.split("\n") if l.strip() and not l.strip().startswith("☆")]
        # 取前 8 行非空行
        excerpt = " ".join(lines[:8])
        return excerpt[:300] if excerpt else ""
    except Exception:
        return ""


def _get_tavily_key() -> str:
    """从 openclaw.json 读取 Tavily API key"""
    try:
        p = Path.home() / ".openclaw" / "openclaw.json"
        with open(p) as f:
            d = json.load(f)
        return (d["plugins"]["entries"]["tavily"]["config"]["webSearch"]["apiKey"] or "")
    except Exception:
        return ""


def search_tavily_news(code: str, name: str, key: str = "") -> str:
    """
    用 Tavily API 搜索个股最新网络新闻/公告
    key 优先用传入值，传入空则自动从 openclaw.json 读取
    返回 AI 摘要字符串，失败返回空字符串
    """
    tavily_key = key.strip() or _get_tavily_key()
    if not tavily_key:
        return ""
    try:
        import urllib.request
        query = f"{name} ({code}) 股票 最新公告 重大消息"
        payload = json.dumps({
            "api_key": tavily_key,
            "query": query,
            "search_depth": "basic",
            "topic": "finance",
            "max_results": 3,
            "include_answer": True,
        }).encode("utf-8")
        req = urllib.request.Request(
            "https://api.tavily.com/search",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        answer = result.get("answer", "").strip()
        return answer[:300] if answer else ""
    except Exception:
        return ""


def current_minute_index(cur_time: datetime) -> int:
    h, m, s = cur_time.hour, cur_time.minute, cur_time.second
    total = h * 3600 + m * 60 + s
    AM_START = 9  * 3600 + 30 * 60
    AM_END   = 11 * 3600 + 29 * 60
    PM_START = 13 * 3600
    PM_END   = 15 * 3600 - 1

    if total < AM_START:
        return -1
    if total <= AM_END:
        return (total - AM_START) // 60
    if total < PM_START:
        return -1
    if total <= PM_END:
        return (total - PM_START) // 60 + 120
    return TOTAL_MINUTES - 1


def cumsum_at(bars: list, minute_idx: int) -> int:
    if minute_idx < 0 or not bars:
        return 0
    limit = min(minute_idx + 1, len(bars))
    return sum(int(bars[i].get("vol") or 0) for i in range(limit))


def total_vol(bars: list) -> int:
    return sum(int(b.get("vol") or 0) for b in bars)


def minute_to_str(idx: int) -> str:
    if idx < 0:
        return "--:--"
    secs = (34200 + idx * 60) if idx < 120 else (46800 + (idx - 120) * 60)
    return f"{secs // 3600:02d}:{(secs % 3600) // 60:02d}"


def cjk_width(s: str) -> int:
    """计算字符串在终端的显示宽度（CJK 占 2，ASCII 占 1）"""
    w = 0
    for ch in s:
        cp = ord(ch)
        if cp > 0x1100 and (
            cp <= 0x115f or cp == 0x2329 or cp == 0x232a
            or (0x2e80 <= cp <= 0xa4cf and cp != 0x303f)
            or (0xac00 <= cp <= 0xd7a3)
            or (0xf900 <= cp <= 0xfaff)
            or (0xfe10 <= cp <= 0xfe19)
            or (0xfe30 <= cp <= 0xfe6f)
            or (0xff00 <= cp <= 0xff60)
            or (0xffe0 <= cp <= 0xffe6)
            or (0x20000 <= cp <= 0x2fffd)
            or (0x30000 <= cp <= 0x3fffd)
        ):
            w += 2
        else:
            w += 1
    return w


def pad_to_width(s: str, target: int, align: str = "<") -> str:
    """按终端显示宽度补齐（<'左对齐 '>'右对齐）"""
    sw = cjk_width(s)
    pad = target - sw
    if pad <= 0:
        return s
    if align == ">":
        return " " * pad + s
    return s + " " * pad


def is_tty() -> bool:
    return sys.stdout.isatty()


def strip_ansi(s: str) -> str:
    return _ansi_pat.sub("", s)


# --------------------------------------------------------------------------- #
# 渲染引擎（原地刷新）
# --------------------------------------------------------------------------- #
#  列定义：代码 | 名称 | 昨收盘 | 今开盘 | 成交价 |     成交量     |     量比     | 放量标识
#  左对齐      左对齐    右对齐   右对齐   右对齐       右对齐           右对齐      左对齐

W_CODE  = 6
W_NAME  = 8
W_LAST  = 7
W_OPEN  = 7
W_PRICE = 8
W_VOL   = 12
W_RATIO = 8
W_FLAG  = 10

# 列间距（额外空格）
GAP_CODE_NAME = 2
GAP_NAME_LAST = 2
GAP_LAST_OPEN = 2
GAP_OPEN_PRICE = 2
GAP_PRICE_VOL = 3   # 成交价到成交量：加大
GAP_VOL_RATIO  = 4   # 成交量到量比：加大
GAP_RATIO_FLAG = 4   # 量比到放量标识：加大

# 终端显示总宽度（CJK 双重宽度感知）
SEP_TOTAL = (
    cjk_width("代码") + GAP_CODE_NAME
    + W_NAME + GAP_NAME_LAST  # 名称列以 display width 补齐到 W_NAME
    + cjk_width("昨收盘") + GAP_LAST_OPEN
    + cjk_width("今开盘") + GAP_OPEN_PRICE
    + cjk_width("成交价") + GAP_PRICE_VOL
    + W_VOL + GAP_VOL_RATIO
    + W_RATIO + GAP_RATIO_FLAG
    + cjk_width("放量标识") + 4  # ✅ + 空格 + verdic段
)

def sep_line() -> str:
    return f"{CLEAR_LINE}{'─' * SEP_TOTAL}\n"


def hdr_line() -> str:
    return (
        f"{CLEAR_LINE}"
        f"{pad_to_width('代码', W_CODE)}{' ' * GAP_CODE_NAME}"
        f"{pad_to_width('名称', W_NAME)}{' ' * GAP_NAME_LAST}"
        f"{pad_to_width('昨收盘', W_LAST, '>')}{' ' * GAP_LAST_OPEN}"
        f"{pad_to_width('今开盘', W_OPEN, '>')}{' ' * GAP_OPEN_PRICE}"
        f"{pad_to_width('成交价', W_PRICE, '>')}{' ' * GAP_PRICE_VOL}"
        f"{pad_to_width('成交量', W_VOL, '>')}{' ' * GAP_VOL_RATIO}"
        f"{pad_to_width('量比', W_RATIO, '>')}{' ' * GAP_RATIO_FLAG}"
        f"{pad_to_width('放量标识', W_FLAG)}"
        "\n"
    )


def data_line(code: str, name: str,
              last_close: float, today_open: float,
              price, vol: int,
              ratio: float, is_breakout: bool, verdict: str,
              err: str = "") -> str:
    flag = "✅" if is_breakout else "  "

    if err:
        return (
            f"{CLEAR_LINE}"
            f"{pad_to_width(code, W_CODE)}{' ' * GAP_CODE_NAME}"
            f"{pad_to_width(name, W_NAME)}  "
            f"[错误] {err}\n"
        )

    price_str = f"{price:>{W_PRICE}.2f}" if price else f"{'N/A':>{W_PRICE}}"
    vol_str   = f"{int(vol):>{W_VOL},}" if vol else f"{'N/A':>{W_VOL}}"
    ratio_str = f"{ratio * 100:>{W_RATIO - 1}.1f}%" if ratio else f"{'N/A':>{W_RATIO}}"
    chg_str   = ""
    if price and last_close:
        pct = (price - last_close) / last_close * 100
        chg_str = f" ({pct:+.1f}%)"

    return (
        f"{CLEAR_LINE}"
        f"{pad_to_width(code, W_CODE)}{' ' * GAP_CODE_NAME}"
        f"{pad_to_width(name, W_NAME)}{' ' * GAP_NAME_LAST}"
        f"{last_close:>{W_LAST}.2f}{' ' * GAP_LAST_OPEN}"
        f"{today_open:>{W_OPEN}.2f}{' ' * GAP_OPEN_PRICE}"
        f"{price_str}{' ' * GAP_PRICE_VOL}"
        f"{vol_str}{' ' * GAP_VOL_RATIO}"
        f"{ratio_str}{' ' * GAP_RATIO_FLAG}"
        f"{flag} {verdict}{chg_str}"
        "\n"
    )


def summary_line(total: int, breakouts: int, err_cnt: int) -> str:
    ok = total - err_cnt
    s = (
        f"{CLEAR_LINE}"
        f"合计：{total} 只  |  "
        f"放量：{breakouts}  |  "
        f"未放量：{ok - breakouts}"
        f"{'  |  错误：' + str(err_cnt) if err_cnt else ''}"
    )
    pad = SEP_TOTAL - len(strip_ansi(s))
    return f"{s}{' ' * max(0, pad)}\n"


def render(records: list, cur_time: datetime, cur_idx: int) -> None:
    now_str    = cur_time.strftime("%H:%M:%S")
    minute_lbl = minute_to_str(cur_idx)

    # 第一行：更新时间
    top = (
        f"{CLEAR_LINE}"
        f" 最新更新时间：{now_str}  |  "
        f"当前分钟：{minute_lbl}（{cur_idx}/{TOTAL_MINUTES}）"
    )
    pad = SEP_TOTAL - len(strip_ansi(top))
    top += " " * max(0, pad) + "\n"

    # 分隔线
    sep = sep_line()

    # 表头
    header = hdr_line()

    # 数据行
    data = "".join(
        data_line(
            r["code"], r["name"],
            r.get("current", {}).get("last_close", 0) or 0,
            r.get("current", {}).get("open", 0) or 0,
            r.get("current", {}).get("price"),
            r.get("current", {}).get("vol") or r.get("current", {}).get("today_cumsum_vol", 0) or 0,
            r.get("volume_analysis", {}).get("vol_ratio", 0.0),
            r.get("volume_analysis", {}).get("is_breakout", False),
            r.get("volume_analysis", {}).get("verdict", "---"),
            r.get("error", ""),
        )
        for r in records
    )

    # 汇总行
    total    = len(records)
    breakouts = sum(1 for r in records
                    if r.get("volume_analysis", {}).get("is_breakout"))
    err_cnt  = sum(1 for r in records if r.get("error"))
    footer   = summary_line(total, breakouts, err_cnt)

    output = top + sep + header + sep + data + sep + footer

    if is_tty():
        sys.stdout.write(CLEAR_SCREEN)
        sys.stdout.write(output)
    else:
        # 非 TTY：strip ANSI，用换行 + 分隔线输出
        sys.stdout.write("\n" + "=" * 60 + "\n" + strip_ansi(output))
    sys.stdout.flush()


# --------------------------------------------------------------------------- #
# 主逻辑
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    today = datetime.strptime(args.date.strip(), "%Y-%m-%d").date()

    # 自动检测最近交易日
    last_trading = find_last_trading_day(today)
    if last_trading < today:
        print(f"📅 指定日期 {today} → 基准分时取最近交易日 {last_trading}",
              flush=True)
    else:
        print(f"📅 基准分时日期：{last_trading}", flush=True)

    stocks = load_stocks(args.file)
    if not stocks:
        print("错误：文件为空或格式无法识别", file=sys.stderr)
        sys.exit(1)

    client = get_client()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 已连接 mootdx 服务器",
          flush=True)
    if not is_tty():
        print("（注意：非交互式终端，输出将有 ANSI 转义序列）",
              flush=True)

    # 预加载基准分时
    print(f"正在加载 {len(stocks)} 只股票的分时数据（基准日 {last_trading}）...",
          flush=True)
    yesterday_bars_map = {}

    for stock in stocks:
        code = stock["code"]
        mkt  = auto_market(code)
        bars = get_yesterday_minute_bars(client, code, mkt, last_trading)
        yesterday_bars_map[code] = bars
        print(f"  ✅ {code} {stock['name']}: "
              f"基准总成交量={total_vol(bars):,} ({len(bars)} 分钟棒)",
              flush=True)

    print(flush=True)

    count = 0
    try:
        while True:
            cycle_start = time.time()
            count += 1
            cur_time = datetime.now()
            cur_idx  = current_minute_index(cur_time)
            cur_idx_capped = (max(0, min(cur_idx, TOTAL_MINUTES - 1))
                              if cur_idx >= 0 else 0)

            records  = []
            breakouts = []

            # 批量获取实时行情（一次 API 调用，大幅提速）
            quotes_map = {}
            try:
                batch_codes = [(auto_market(s["code"]), s["code"]) for s in stocks]
                batch_result = client.client.get_security_quotes(batch_codes)
                for q in (batch_result or []):
                    quotes_map[q["code"]] = q
            except Exception:
                pass

            # 午休标识
            is_lunch_break = (cur_idx < 0
                              and cur_time.hour >= 11
                              and cur_time.hour < 13)

            for stock in stocks:
                code = stock["code"]
                name = stock["name"]
                yes_bars = yesterday_bars_map.get(code, [])
                mkt = auto_market(code)

                quote = quotes_map.get(code)

                # ── 消息预警（踩坑检测）────────
                news_excerpt = ""
                tavily_answer = ""
                if args.news:
                    news_excerpt = get_company_news(client.client, mkt, code)
                if args.tavily:
                    tavily_answer = search_tavily_news(code, name)

                # 今日累计成交量：优先用实时行情（准确），分时数据仅做后备
                today_cum = cur_cum = 0
                quote_vol = int(quote.get("vol") or 0) if quote else 0
                if quote_vol > 0:
                    today_cum = cur_cum = quote_vol
                else:
                    today_bars = get_today_minute_bars(client, code, mkt)
                    if cur_idx >= 0:
                        cur_cum = cumsum_at(today_bars, cur_idx)
                    else:
                        cur_cum = total_vol(today_bars)
                    today_cum = cur_cum

                if cur_idx >= 0 and yes_bars:
                    yes_cumsum = cumsum_at(yes_bars, cur_idx_capped)
                    threshold  = yes_cumsum * args.vol_ratio
                    ratio_val  = (cur_cum / yes_cumsum) if yes_cumsum > 0 else 0.0
                    is_breakout = cur_cum > threshold
                    verdict = "放量" if is_breakout else "未放量"
                elif is_lunch_break:
                    yes_cumsum = 0
                    threshold  = 0
                    ratio_val  = 0.0
                    is_breakout = False
                    verdict = "休市"
                else:
                    yes_cumsum = 0
                    threshold  = 0
                    ratio_val  = 0.0
                    is_breakout = False
                    verdict = ("盘前/盘后" if cur_idx < 0 else "无昨日"
                               if not yes_bars else "---")

                records.append({
                    "code": code,
                    "name": name,
                    "time": cur_time.strftime("%H:%M:%S"),
                    "target_date": args.date,
                    "cur_minute_idx": cur_idx,
                    "current": {
                        "price":          quote.get("price") if quote else None,
                        "open":           quote.get("open") if quote else None,
                        "last_close":     quote.get("last_close") if quote else None,
                        "vol":            cur_cum,
                        "today_cumsum_vol": cur_cum,
                        "amount":         quote.get("amount") if quote else None,
                    },
                    "volume_analysis": {
                        "cur_cum":          cur_cum,
                        "yesterday_cumsum": yes_cumsum,
                        "yesterday_total":  total_vol(yes_bars),
                        "cur_minute_idx":   cur_idx,
                        "threshold":        round(threshold, 0),
                        "vol_ratio":        round(ratio_val, 2),
                        "is_breakout":      is_breakout,
                        "verdict":          verdict,
                    },
                    "error": None if quote else "获取实时数据失败",
                    "news_excerpt": news_excerpt,
                    "tavily_answer": tavily_answer,
                })

                if is_breakout:
                    breakouts.append(records[-1])

            render(records, cur_time, cur_idx)

            if args.output:
                out_path = Path(args.output)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "query_time":          cur_time.isoformat(),
                        "target_date":         args.date,
                        "cur_minute_idx":      cur_idx,
                        "minute_label":        minute_to_str(cur_idx),
                        "interval":            args.interval,
                        "vol_ratio_threshold": args.vol_ratio,
                        "total":              len(records),
                        "breakout_count":      len(breakouts),
                        "records":             records,
                    }, f, ensure_ascii=False, indent=2)

            if args.count > 0 and count >= args.count:
                print(f"\n已达成指定查询次数 ({args.count})，退出。\n",
                      flush=True)
                break

            # 精确间隔：从本周期开始时间算起
            elapsed = time.time() - cycle_start
            if elapsed < args.interval:
                time.sleep(args.interval - elapsed)

    except KeyboardInterrupt:
        print(f"\n监控已停止（Ctrl+C）。共查询 {count} 次。\n",
              flush=True)


if __name__ == "__main__":
    main()