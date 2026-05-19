#!/usr/bin/env python3
"""
mootdx_signal_monitor.py — 股票实时买卖信号监控
==============================================================
功能：
  - 读取股票列表（.EBK / .json / .jsonl）
  - 利用 gain_turnover 离线历史数据计算均线/RSI/MACD
  - 利用 mootdx 实时行情计算当日涨跌幅/量比/现价
  - 综合多指标给出「买入/观望/卖出/关注」信号
  - 通达信 .blk 文件格式输出信号股列表

指标说明：
  趋势（Trend Score，0~100）：
    0~25  下降排列（ bearish = SELL）
    25~45 混合排列（ mixed = HOLD/WATCH）
    45~70 多头初现（ bullish = HOLD/WATCH）
    70~100 多头确认（ strong_bull = BUY）
  RSI：>70 超买，<30 超卖
  MACD柱：DIF>DEA 多头（看多），DIF<DEA 空头（看空）
  量比：>1.5 放量，>2.5 明显放量
  当日涨幅：>5% 强势，<-3% 弱势/出货嫌疑

买卖信号规则（综合评分 0~100）：
  买入信号 score >= 65（趋势>=45 + RSI<70 + 多头）
  卖出信号 score <= 30（趋势<=35 + RSI>65 + 空头）
  观望   score 31~64
  关注   信号日放量且价格突破关键位

用法：
  python mootdx_signal_monitor.py                          # 循环监控（默认60秒）
  python mootdx_signal_monitor.py --count 3               # 只查3次退出
  python mootdx_signal_monitor.py -f mylist.jsonl         # 自定义列表
  python mootdx_signal_monitor.py --top 30 --blk           # 输出前30名到.blk
"""

import sys, time, json, argparse, logging
from pathlib import Path
from datetime import datetime, date
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ── 路径设置 ────────────────────────────────────────────────
WORKSPACE = Path(__file__).resolve().parent.parent  # → stock_trend/
sys.path.insert(0, str(WORKSPACE))  # stock_trend/ 加入 import 路径

from gain_turnover import (
    load_qfq_history, load_stock_names, normalize_symbol,
    compute_rsi_scalar, rolling_mean,
)
from mootdx.quotes import Quotes as MooQuotes


# ═══════════════════════════════════════════════════════════════
# 常量
# ═══════════════════════════════════════════════════════════════
CLEAR_SCREEN = "\033[2J\033[H"
CLEAR_LINE   = "\033[2K"
SEP_TOTAL   = 140

# RSI 阈值
RSI_BUY_ZONE   = 35   # RSI < 35 超卖
RSI_SELL_ZONE  = 68   # RSI > 68 超买
RSI_WATCH_HIGH = 62   # RSI > 62 注意回调

# 涨跌阈值
GAIN_WATCH    = 3.0   # 当日涨幅 >3% 强势
LOSS_WATCH    = -2.5  # 当日跌幅 >2.5% 弱势
LOSS_REJECT   = -4.0  # 当日跌幅 >4%  出货嫌疑

# 量比阈值
VOL_BREAKOUT  = 1.5   # 量比 >1.5 放量
VOL_STRONG    = 2.5   # 量比 >2.5 明显放量

# 趋势得分权重（5项各20分）
TREND_UP_PCT  = 0.40  # 价格高于均线的数量权重
TREND_DIR     = 0.30  # 均线方向向上权重
TREND_ORDER   = 0.30  # 多头排列权重


# ═══════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════

def cjk_width(s: str) -> int:
    w = 0
    for ch in str(s):
        cp = ord(ch)
        if (0x1100 <= cp <= 0x115f or 0x2329 <= cp <= 0x232a
                or 0x2e80 <= cp <= 0xa4cf and cp != 0x303f
                or 0xac00 <= cp <= 0xd7a3
                or 0xf900 <= cp <= 0xfaff
                or 0xfe10 <= cp <= 0xfe19
                or 0xfe30 <= cp <= 0xfe6f
                or 0xff00 <= cp <= 0xff60
                or 0xffe0 <= cp <= 0xffe6
                or 0x20000 <= cp <= 0x2fffd
                or 0x30000 <= cp <= 0x3fffd):
            w += 2
        else:
            w += 1
    return w


def pad(s: str, w: int, align: str = "<") -> str:
    sw = cjk_width(s)
    p = w - sw
    if p <= 0:
        return s
    if align == ">":
        return " " * p + s
    return s + " " * p


def is_tty() -> bool:
    return sys.stdout.isatty()


def strip_ansi(s: str) -> str:
    import re
    return re.sub(r"\x1b\[[0-9;]*[a-zA-Z]|\x1b\[K", "", s)


# ═══════════════════════════════════════════════════════════════
# 数据类
# ═══════════════════════════════════════════════════════════════

@dataclass
class StockSignal:
    code: str
    name: str
    # 历史数据指标
    price: float       = 0.0
    last_close: float = 0.0
    change_pct: float  = 0.0
    # 均线
    ma5: float  = 0.0
    ma10: float = 0.0
    ma20: float = 0.0
    ma60: float = 0.0
    ma5_dir: int = 0    # 1=up, -1=down, 0=flat
    ma10_dir: int = 0
    ma20_dir: int = 0
    ma60_dir: int = 0
    # 技术指标
    rsi: float = 50.0
    macd_hist: float = 0.0  # DIF - DEA
    # 实时数据
    vol_ratio: float = 0.0   # 当日量比
    vol_today: int   = 0     # 当日累计成交量
    vol_5d_avg: int  = 0    # 5日均量
    # 信号评分
    trend_score: float = 50.0  # 0~100
    signal_score: float = 50.0 # 综合买卖评分 0~100
    signal_label: str   = "观望"  # 买入/观望/卖出/关注
    signal_reason: str   = ""
    error: str = ""


# ═══════════════════════════════════════════════════════════════
# 均线/指标计算
# ═══════════════════════════════════════════════════════════════

def calc_ma(closes: np.ndarray, period: int) -> Optional[float]:
    if len(closes) < period:
        return None
    return float(np.mean(closes[-period:]))


def ma_direction(closes: np.ndarray, period: int) -> int:
    if len(closes) < period + 5:
        return 0
    now = calc_ma(closes, period)
    ago = calc_ma(closes[:-5], period)
    if now is None or ago is None:
        return 0
    if now > ago * 1.001:
        return 1
    if now < ago * 0.999:
        return -1
    return 0


def calc_macd(closes: np.ndarray, fast=12, slow=26, signal=9):
    """返回 (macd_hist, dif, dea) —— 向量化 O(n) 实现"""
    n = len(closes)
    if n < slow + signal:
        return 0.0, 0.0, 0.0

    def ema_vec(arr, period):
        """向量化 EMA """
        alpha = 2.0 / (period + 1)
        result = np.zeros_like(arr, dtype=np.float64)
        result[0] = arr[0]
        for i in range(1, len(arr)):
            result[i] = alpha * arr[i] + (1 - alpha) * result[i - 1]
        return result

    ema_fast = ema_vec(closes, fast)
    ema_slow = ema_vec(closes, slow)
    dif_arr = ema_fast - ema_slow
    dea_arr = ema_vec(dif_arr, signal)

    # 返回最新值
    macd_hist = (dif_arr[-1] - dea_arr[-1]) * 2
    return float(macd_hist), float(dif_arr[-1]), float(dea_arr[-1])


def compute_indicators(code: str, hist_df) -> dict:
    """从历史K线计算均线+RSI+MACD，返回dict"""
    if hist_df is None or len(hist_df) < 66:
        return {}
    closes = hist_df["close"].values
    vols   = hist_df["volume"].values  # 手数

    ma5  = calc_ma(closes, 5)
    ma10 = calc_ma(closes, 10)
    ma20 = calc_ma(closes, 20)
    ma60 = calc_ma(closes, 60)
    if None in [ma5, ma10, ma20, ma60]:
        return {}

    d5  = ma_direction(closes, 5)
    d10 = ma_direction(closes, 10)
    d20 = ma_direction(closes, 20)
    d60 = ma_direction(closes, 60)

    rsi = compute_rsi_scalar(closes, 14)
    macd_hist, dif, dea = calc_macd(closes)

    # 5日均量
    vol_5d_avg = int(np.mean(vols[-5:]))

    return {
        "ma5": ma5, "ma10": ma10, "ma20": ma20, "ma60": ma60,
        "d5": d5, "d10": d10, "d20": d20, "d60": d60,
        "rsi": rsi,
        "macd_hist": macd_hist, "dif": dif, "dea": dea,
        "vol_5d_avg": vol_5d_avg,
        "last_close": float(closes[-1]),
        "closes": closes,
    }


def trend_score_from_indicators(
    price: float, ma5: float, ma10: float, ma20: float, ma60: float,
    d5: int, d10: int, d20: int, d60: int,
) -> float:
    """趋势评分 0~100 —— 双向评分，基准50，牛熊各有加减"""
    score = 50.0

    # 1. 价格 vs 均线（±16分）
    for ma, w in [(ma5, 4), (ma10, 4), (ma20, 4), (ma60, 4)]:
        if price > ma:
            score += w
        else:
            score -= w

    # 2. 均线排列（±12分）
    if ma5 > ma10 > ma20 > ma60:
        score += 12
    elif ma5 > ma10 > ma20:
        score += 6
    elif ma5 > ma10:
        score += 3
    elif ma5 < ma10 < ma20 < ma60:
        score -= 12
    elif ma5 < ma10 < ma20:
        score -= 6
    elif ma5 < ma10:
        score -= 3

    # 3. 均线方向（±12分）
    for d, w in [(d5, 4), (d10, 3), (d20, 3), (d60, 2)]:
        if d == 1:
            score += w
        elif d == -1:
            score -= w

    # 4. 乖离修正（发散+3，粘合-3）
    if ma5 > ma20:
        spread = (ma5 - ma20) / ma20 * 100
        score += min(spread, 10) * 0.3  # 最多+3
    elif ma5 < ma20 * 0.97:
        score -= 3

    return max(min(score, 100.0), 0.0)


def generate_signal(
    price: float, last_close: float, change_pct: float,
    ma5: float, ma10: float, ma20: float, ma60: float,
    d5: int, d10: int, d20: int, d60: int,
    rsi: float, macd_hist: float,
    vol_ratio: float,
) -> tuple[float, str]:
    """返回 (综合评分, 理由) —— 标签由 assign_signal_labels 统一分配"""

    # 趋势得分
    trend = trend_score_from_indicators(
        price, ma5, ma10, ma20, ma60, d5, d10, d20, d60)

    # 涨跌修正（当日大幅下跌降分）
    if change_pct < LOSS_REJECT:
        trend *= 0.5
    elif change_pct < LOSS_WATCH:
        trend *= 0.8

    # RSI 修正
    if rsi < RSI_BUY_ZONE:
        trend += (RSI_BUY_ZONE - rsi) / RSI_BUY_ZONE * 15  # 超卖加分
    elif rsi > RSI_SELL_ZONE:
        trend -= (rsi - RSI_SELL_ZONE) / (100 - RSI_SELL_ZONE) * 15  # 超买减分

    # MACD 柱修正
    if macd_hist > 0:
        trend += min(macd_hist / last_close * 300, 8)
    else:
        trend -= min(abs(macd_hist) / last_close * 300, 8)

    # 量比加分
    if vol_ratio > VOL_STRONG and change_pct > 0:
        trend += 5
    elif vol_ratio > VOL_BREAKOUT and change_pct > 0:
        trend += 3

    score = round(min(max(trend, 0), 100), 1)

    # 理由（利好 + 利空）
    reasons = []
    # 利空因素
    if price < ma60:
        reasons.append(f"价<MA60")
    if ma5 < ma10 and ma10 < ma20 and ma20 < ma60:
        reasons.append("空头排列")
    elif ma5 < ma10:
        reasons.append("MA5<MA10")
    if d5 == -1 and d10 == -1 and d20 == -1:
        reasons.append("均线全向下")
    elif d5 == -1:
        reasons.append("MA5向下")
    # 利好因素
    if rsi < RSI_BUY_ZONE:
        reasons.append(f"RSI={rsi:.0f}超卖")
    elif rsi > RSI_SELL_ZONE:
        reasons.append(f"RSI={rsi:.0f}超买")
    if change_pct > GAIN_WATCH:
        reasons.append(f"当日涨+{change_pct:.1f}%")
    elif change_pct < LOSS_WATCH:
        reasons.append(f"当日跌{change_pct:.1f}%")
    if vol_ratio > VOL_STRONG:
        reasons.append(f"量比{vol_ratio:.1f}x放量")
    elif vol_ratio > VOL_BREAKOUT:
        reasons.append(f"量比{vol_ratio:.1f}x")
    if d5 == 1 and d10 == 1:
        reasons.append("MA5/10向上")
    if macd_hist > 0:
        reasons.append("MACD多头")
    elif macd_hist < 0:
        reasons.append("MACD空头")
    reason_str = " ".join(reasons) if reasons else ""

    return score, reason_str


def assign_signal_labels(results: list[StockSignal]):
    """按批次百分位分配信号标签：买入(top 20%), 关注(60~80%), 观望(30~60%), 卖出(bottom 30%)"""
    valid_sigs = [s for s in results if not s.error]
    if not valid_sigs:
        return
    scores = [s.signal_score for s in valid_sigs]
    n = len(scores)
    if n < 5:
        # 太少股票，用绝对阈值
        for s in valid_sigs:
            if s.signal_score >= 60:
                s.signal_label = "买入"
            elif s.signal_score <= 30:
                s.signal_label = "卖出"
            else:
                s.signal_label = "观望"
        return

    sorted_scores = sorted(scores)
    p80 = sorted_scores[max(0, min(n - 1, int(n * 0.80)))]
    p60 = sorted_scores[max(0, min(n - 1, int(n * 0.60)))]
    p30 = sorted_scores[max(0, min(n - 1, int(n * 0.30)))]

    for s in valid_sigs:
        score = s.signal_score
        is_hot = (s.rsi > RSI_WATCH_HIGH
                  or s.change_pct > GAIN_WATCH
                  or s.vol_ratio > VOL_BREAKOUT)
        # 安全垫：MACD多头或当日上涨的最差也是"观望"
        safe = (s.change_pct > 0 or s.macd_hist > 0)
        if score >= p80:
            s.signal_label = "买入"
        elif score >= p60 and is_hot:
            s.signal_label = "关注"
        elif score >= p60:
            s.signal_label = "观望"
        elif score <= p30 and not safe:
            s.signal_label = "卖出"
        else:
            s.signal_label = "观望"


# ═══════════════════════════════════════════════════════════════
# mootdx 实时行情
# ═══════════════════════════════════════════════════════════════

def get_moo_client():
    return MooQuotes.factory(market="std")


def fetch_realtime(client, code: str, market: int) -> Optional[dict]:
    """返回 dict 或 None"""
    try:
        result = client.client.get_security_quotes([(market, code)])
        if result:
            return dict(result[0])
    except Exception:
        pass
    return None


def auto_market(code: str) -> int:
    if code.startswith(("6", "9")):
        return 1
    return 0


# ═══════════════════════════════════════════════════════════════
# BLK 文件写入
# ═══════════════════════════════════════════════════════════════

def write_blk(signals: list, path: str, top_n: int = 30):
    """按信号评分降序写 BLK（量比排名股池）"""
    valid = [s for s in signals if not s.error]
    valid.sort(key=lambda s: s.signal_score, reverse=True)
    top = valid[:top_n]

    lines = ["", ]
    for s in top:
        blk_code = f"{auto_market(s.code)}{s.code}"
        lines.append(blk_code)

    new_content = "\r\n".join(lines) + "\r\n"
    p = Path(path)
    try:
        old = p.read_bytes() if p.exists() else b""
    except Exception:
        old = b""
    if new_content.encode("utf-8") != old:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(new_content.encode("utf-8"))


# ═══════════════════════════════════════════════════════════════
# 加载股票列表
# ═══════════════════════════════════════════════════════════════

def load_stocks(path: str) -> list[dict]:
    p = Path(path)
    if not p.is_absolute():
        p = WORKSPACE / path

    if p.suffix.lower() in (".ebk",):
        raw = p.read_bytes().decode("ascii", errors="ignore")
        codes = [t[1:] for t in raw.split("\r\n") if len(t) == 7 and t[0] in "01"]
        codes = list(dict.fromkeys(codes))
        names = load_stock_names()
        return [{"code": c, "name": names.get(c, "")} for c in codes]

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


# ═══════════════════════════════════════════════════════════════
# 主扫描函数
# ═══════════════════════════════════════════════════════════════

def scan_stocks(stocks: list[dict], client, top_n: int = 0,
                blk_path: str = "", blk_top: int = 30,
                show_fail: bool = False) -> list[StockSignal]:
    """
    对每只股票：
      1. 加载历史K线 → 计算均线/RSI/MACD
      2. 实时行情   → 当日涨跌幅/量比
      3. 综合评分   → 买卖信号

    blk_top: BLK 写入前N名（独立于 --top 显示参数）
    show_fail=True 时，获取实时数据失败的股票也会显示（标为错误）
    """
    results = []

    # ── 批量获取实时行情（分块50只/组，减少丢弃）──
    quotes_map = {}
    BATCH_CHUNK = 50
    for i in range(0, len(stocks), BATCH_CHUNK):
        chunk = [(auto_market(s["code"]), s["code"]) for s in stocks[i:i + BATCH_CHUNK]]
        try:
            raw = client.client.get_security_queries(chunk) if hasattr(client.client, 'get_security_queries') else client.client.get_security_quotes(chunk)
            for q in (raw or []):
                quotes_map[q["code"]] = dict(q)
        except Exception:
            pass

    # ── 遗漏的逐个补齐 ──
    for stock in stocks:
        code = stock["code"]
        if code in quotes_map:
            continue
        mkt = auto_market(code)
        for _ in range(2):
            try:
                raw = client.client.get_security_quotes([(mkt, code)])
                if raw:
                    quotes_map[code] = dict(raw[0])
                    break
            except Exception:
                pass

    for stock in stocks:
        code = stock["code"]
        name = stock["name"]
        mkt = auto_market(code)
        sig = StockSignal(code=code, name=name)

        # ── 实时行情 ────────────────────────────────────
        quote = quotes_map.get(code)
        if quote is None:
            if show_fail:
                sig.error = "获取实时数据失败"
                results.append(sig)
            continue

        # ── 历史数据 ────────────────────────────────────
        hist = None
        try:
            hist = load_qfq_history(code)
        except Exception:
            pass

        ind = compute_indicators(code, hist) if hist is not None else {}
        if not ind:
            sig.error = "历史数据不足"
            results.append(sig)
            continue

        sig.ma5 = ind["ma5"]
        sig.ma10 = ind["ma10"]
        sig.ma20 = ind["ma20"]
        sig.ma60 = ind["ma60"]
        sig.d5 = ind["d5"]
        sig.d10 = ind["d10"]
        sig.d20 = ind["d20"]
        sig.d60 = ind["d60"]
        sig.rsi = ind["rsi"]
        sig.macd_hist = ind["macd_hist"]
        sig.last_close = ind["last_close"]
        sig.vol_5d_avg = ind["vol_5d_avg"]

        # 实时行情已在上方获取，此处直接使用
        price = quote.get("price", 0.0) or 0.0
        last_close = quote.get("last_close", ind["last_close"]) or ind["last_close"]
        sig.price = price
        sig.last_close = last_close

        if last_close > 0 and price > 0:
            sig.change_pct = (price - last_close) / last_close * 100

        # 当日累计成交量 → 股（mootdx返回手，×100 对齐 qfq 的股）
        vol_today = int(quote.get("vol", 0) or 0) * 100
        sig.vol_today = vol_today

        # 量比 = 当日预估全天量 / 5日均量
        cur_minute = _current_minute_index()
        if cur_minute > 5 and sig.vol_5d_avg > 0:
            est_full = vol_today / cur_minute * 240 if cur_minute > 0 else vol_today
            sig.vol_ratio = round(est_full / sig.vol_5d_avg, 2)
        elif sig.vol_5d_avg > 0:
            sig.vol_ratio = round(vol_today / sig.vol_5d_avg, 2)
        else:
            sig.vol_ratio = 0.0

        # ── 信号评分 ─────────────────────────────────────
        score, reason = generate_signal(
            price, last_close, sig.change_pct,
            sig.ma5, sig.ma10, sig.ma20, sig.ma60,
            sig.d5, sig.d10, sig.d20, sig.d60,
            sig.rsi, sig.macd_hist, sig.vol_ratio,
        )
        sig.signal_score = score
        sig.signal_reason = reason
        results.append(sig)

    # ── 按百分位统一分配信号标签 ──
    assign_signal_labels(results)

    # ── 排序：信号优先（买入>关注>观望>卖出），同级别按评分降序 ──
    LABEL_ORDER = {"买入": 0, "关注": 1, "观望": 2, "卖出": 3, "错误": 4}
    results.sort(key=lambda s: (LABEL_ORDER.get(s.signal_label, 5),
                                -s.signal_score if s.signal_label != "错误" else 999))

    # ── BLK 输出（使用独立 blk_top，基于全量结果，不受 --top 截断影响）──
    if blk_path:
        write_blk(results, blk_path, blk_top)

    if top_n > 0:
        results = results[:top_n]

    return results


def _current_minute_index() -> int:
    """返回当前已交易分钟数（0~240）"""
    now = datetime.now()
    h, m = now.hour, now.minute
    total = h * 60 + m
    if 9 * 60 + 30 <= total <= 11 * 60 + 29:
        return total - (9 * 60 + 30) + 1
    if 13 * 60 <= total <= 14 * 60 + 59:
        return total - (13 * 60) + 121
    if total > 14 * 60 + 59:
        return 240
    return 0


# ═══════════════════════════════════════════════════════════════
# 渲染引擎
# ═══════════════════════════════════════════════════════════════

# 列宽定义
W = {
    "code":  7, "name":  8, "signal": 10, "score": 10,
    "price": 10, "chg":   8, "volr":  6,
    "ma5":   6, "ma10":  6, "ma20":  6, "ma60":  6,
    "rsi":   6, "macd":  8,
    "reason": 50,
}


def render(signals: list, cur_time: datetime):
    now_str = cur_time.strftime("%H:%M:%S")

    # ── 标题 ──
    sig_counts = {}
    for s in signals:
        sig_counts[s.signal_label] = sig_counts.get(s.signal_label, 0) + 1

    title = (
        f"{CLEAR_LINE} 实时信号监控  {now_str}"
        f"   买入:{sig_counts.get('买入',0)}  关注:{sig_counts.get('关注',0)}"
        f"  观望:{sig_counts.get('观望',0)}  卖出:{sig_counts.get('卖出',0)}"
    )
    pad_total = SEP_TOTAL - cjk_width(strip_ansi(title))
    header = title + " " * max(0, pad_total) + "\n"
    sep = "─" * SEP_TOTAL + "\n"

    # ── 列头 ──
    def h(label: str, key: str) -> str:
        w = W.get(key, 10)
        return pad(label, w, "<")

    col_hdr = (
        h("代码", "code")
        + h("名称", "name")
        + h("信号", "signal")
        + h("评分", "score")
        + h("现价", "price")
        + h("涨跌%", "chg")
        + h("量比", "volr")
        + h(" MA5", "ma5")
        + h("  MA10", "ma10")
        + h("    MA20", "ma20")
        + h("    MA60", "ma60")
        + h("    RSI", "rsi")
        + h("    MACD", "macd")
        + h("      理由", "reason")
    )
    hdr_line = f"{CLEAR_LINE}{col_hdr}\n"

    # ── 数据行 ──
    def sig_icon(label: str) -> str:
        return {"买入": "🟢", "关注": "🟡", "观望": "🔵",
                "卖出": "🔴", "错误": "⚠️"}.get(label, "⚪")

    rows = []
    for s in signals:
        if s.error:
            row = (
                f"{pad(s.code, W['code'])}"
                f"{pad(s.name, W['name'])}"
                f"{'⚠️':<6}"
                f"{'':>5}  {s.error}"
            )
            rows.append(f"{CLEAR_LINE}{row}\n")
            continue

        icon = sig_icon(s.signal_label)
        score_str = f"{s.signal_score:.0f}"
        chg_str = f"{s.change_pct:+.1f}%"
        volr_str = f"{s.vol_ratio:.1f}x" if s.vol_ratio > 0 else "N/A"

        def pf(v: float, w: int) -> str:
            return f"{v:>{w}.2f}" if w > 0 else ""

        row = (
            f"{pad(s.code, W['code'])}"
            f"{pad(s.name, W['name'])}"
            f"{icon}{s.signal_label:<4}"
            f"{score_str:>5}  "
            f"{pf(s.price, W['price'])}  "
            f"{chg_str:>7}  "
            f"{volr_str:>6}  "
            f"{pf(s.ma5, W['ma5'])}  "
            f"{pf(s.ma10, W['ma10'])}  "
            f"{pf(s.ma20, W['ma20'])}  "
            f"{pf(s.ma60, W['ma60'])}  "
            f"{s.rsi:>5.0f}  "
            f"{s.macd_hist:>+7.3f}  "
            f"{s.signal_reason[:W['reason']]}"
        )
        rows.append(f"{CLEAR_LINE}{row}\n")

    data_block = "".join(rows)
    footer = f"{CLEAR_LINE}{'─' * SEP_TOTAL}\n"

    output = header + sep + hdr_line + sep + data_block + footer

    if is_tty():
        sys.stdout.write(CLEAR_SCREEN)
        sys.stdout.write(output)
    else:
        sys.stdout.write("\n" + "=" * 60 + "\n" + strip_ansi(output))
    sys.stdout.flush()


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="mootdx 实时股票买卖信号监控"
    )
    parser.add_argument("--file", "-f",
                        default="output/signallist.EBK",
                        help="股票列表文件（默认 output/signallist.EBK）")
    parser.add_argument("--interval", "-i", type=float, default=3,
                        help="轮询间隔（秒），默认3秒")
    parser.add_argument("--count", "-c", type=int, default=0,
                        help="查询次数，0=无限循环")
    parser.add_argument("--top", "-t", type=int, default=0,
                        help="显示前N名（默认全部），设为0则显示全部")
    parser.add_argument("--blk", "-b",
                        default="/mnt/d/new_tdx/T0002/blocknew/SSTSGP.blk",
                        help="BLK 输出路径（默认 /mnt/d/new_tdx/T0002/blocknew/SSTSGP.blk）")
    parser.add_argument("--blk-top", type=int, default=30,
                        help="BLK 写入前N名，默认30")
    parser.add_argument("--show-fail", action="store_true",
                        help="显示获取实时数据失败的股票（默认隐藏）")
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    print(f"📂 加载股票列表: {args.file}", flush=True)
    stocks = load_stocks(args.file)
    if not stocks:
        print("错误：股票列表为空", file=sys.stderr)
        sys.exit(1)
    print(f"✅ 共 {len(stocks)} 只股票\n", flush=True)

    # 抑制 tdxpy 日志
    import tdxpy.logger as _tdx_log
    _tdx_log.logger.setLevel(logging.CRITICAL)
    for _h in _tdx_log.logger.handlers:
        _h.setLevel(logging.CRITICAL)

    client = get_moo_client()
    count  = 0

    try:
        while True:
            count += 1
            loop_start = time.time()
            cur_time = datetime.now()

            signals = scan_stocks(
                stocks, client,
                top_n=args.top if args.top > 0 else 0,
                blk_path=args.blk,
                blk_top=args.blk_top,
                show_fail=args.show_fail,
            )
            render(signals, cur_time)

            if args.count > 0 and count >= args.count:
                print(f"\n已完成 {count} 次扫描，退出。\n", flush=True)
                break

            elapsed = time.time() - loop_start
            if elapsed < args.interval:
                time.sleep(args.interval - elapsed)

    except KeyboardInterrupt:
        print(f"\n监控已停止（Ctrl+C）。\n", flush=True)


if __name__ == "__main__":
    main()
