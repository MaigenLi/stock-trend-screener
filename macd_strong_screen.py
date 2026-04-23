#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MACD 趋势启动型选股（增强版）
====================
独立脚本，扫描全市场符合 MACD 启动形态的股票。

MACD 信号条件（信号日须同时满足）：
  1. MACD > 0（多方市场）
  2. MACD > 0 天数 ≤ 11
  3. MACD>0 区间内，2/3 以上为上涨日
  4. 信号日涨幅 > -3%
  5. 3日涨幅 5% - 22% 之间（避免过度上涨）
  6. DIF 连续2日上涨（DIF[T] > DIF[T-1] > DIF[T-2]）
  7. DEA 上涨（DEA[T] > DEA[T-1]）
  8. MACD 连续两天红柱（MACD[T] > 0 且 MACD[T-1] > 0）
  9. 收盘价 > MA20 > MA60（确保中长期趋势向上）

增强特性：
- 数据要求：≥80根K线（指标更稳定）
- RSI计算：Wilder平滑（标准通达信算法）
- 换手率：优先使用true_turnover，否则估算
- 评分稳定：dif/dea硬上限防止爆炸
- 红柱统计：准确计数（无20根上限）

输出：~/stock_reports/macd_screen_YYYY-MM-DD.txt
"""

import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd

WORKSPACE = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(WORKSPACE))

from gain_turnover import normalize_prefixed, load_stock_names, get_stock_name


def _rpad(s: str, width: int) -> str:
    return s + " " * max(0, width - len(s))


def _lpad(s: str, width: int) -> str:
    return " " * max(0, width - len(s)) + s


DEFAULT_WORKERS = 8
DEFAULT_MIN_TURNOVER = 5.0   # 5日均换手率下限（%）
DEFAULT_MIN_AMOUNT = 1e8     # 20日均成交额下限（元）
DEFAULT_MARKET_DAYS = 21     # 市场21日涨幅计算天数


# ─────────────────────────────────────────────────────────
# MACD 计算
# ─────────────────────────────────────────────────────────
def _compute_macd(closes: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9):
    """计算 MACD，返回 (dif, dea, macd) 三个序列。"""
    ema_fast = pd.Series(closes).ewm(span=fast, adjust=False).mean().values
    ema_slow = pd.Series(closes).ewm(span=slow, adjust=False).mean().values
    dif = ema_fast - ema_slow
    dea = pd.Series(dif).ewm(span=signal, adjust=False).mean().values
    macd = 2.0 * (dif - dea)
    return dif, dea, macd


def _count_consecutive_positive_macd(macd: np.ndarray, idx: int) -> int:
    """从 idx 往回数，连续红柱（MACD>0）的天数。"""
    count = 0
    for i in range(idx, -1, -1):
        if macd[i] > 0:
            count += 1
        else:
            break
    return count


# ─────────────────────────────────────────────────────────
# 单股 MACD 评估
# ─────────────────────────────────────────────────────────
def evaluate_macd_stock(code: str, df: pd.DataFrame, min_turnover: float,
                         min_amount: float) -> dict | None:
    """
    评估单只股票是否满足 MACD 启动条件。
    df：前复权日线，已按日期升序排列。
    返回 dict 或 None。
    """
    if len(df) < 80:  # 改为80根K线确保指标稳定性
        return None

    close = df["close"].values
    amount = df["amount"].values
    volume = df["volume"].values

    dif, dea, macd = _compute_macd(close)

    idx = len(dif) - 1  # 最新一根

    # ── 硬过滤 ────────────────────────────────────────
    # 1. DIF > 0
    if dif[idx] <= 0:
        return None

    # 2. DIF 连续2日上涨
    if not (dif[idx] > dif[idx - 1] > dif[idx - 2]):
        return None

    # 3. DEA 上涨
    if not (dea[idx] > dea[idx - 1]):
        return None

    # 4. MACD 连续两天红柱
    if not (macd[idx] > 0 and macd[idx - 1] > 0):
        return None

    # 5. MACD > 0 天数 ≤ 7，且 2/3 以上为上涨日
    red_days = _count_consecutive_positive_macd(macd, idx)
    if red_days > 11:
        return None
    # 统计 MACD>0 区间的上涨天数（start≥1，防止 close[i-1] 越界到 close[-1]）
    up_count = 0
    start = max(1, idx - red_days + 1)
    for i in range(start, idx + 1):
        day_gain = (close[i] / close[i - 1] - 1.0) * 100.0 if close[i - 1] > 0 else 0.0
        if day_gain > 0:
            up_count += 1
    if up_count < (red_days * 2 // 3):
        return None

    # ── 基础指标 ─────────────────────────────────────
    # 信号日涨幅 > -3%，3日涨幅 5% - 22% 之间
    gain1 = (close[idx] / close[idx - 1] - 1.0) * 100.0 if close[idx - 1] > 0 else 0.0
    gain3 = (close[idx] / close[idx - 3] - 1.0) * 100.0 if close[idx - 3] > 0 else 0.0
    if gain1 <= -3.0:
        return None
    if gain3 <= 5.0 or gain3 >= 22.0:  # 5% ≤ gain3 ≤ 22%
        return None
    # 5日均换手率（优先用 true_turnover，否则降级用成交额估算）
    if "true_turnover" in df.columns:
        true_to = df["true_turnover"].astype(float).values
        avg_turnover_5 = float(np.nanmean(true_to[idx - 4:idx + 1])) if idx >= 4 else float(np.nanmean(true_to[:idx + 1]))
    else:
        avg_turnover_5 = 0.0
        for i in range(5, len(df)):
            amt_sum = float(amount[i - 4:i + 1].sum())
            vol_sum = float(volume[i - 4:i + 1].sum())
            avg_turnover_5 = vol_sum / amt_sum * 100.0 if amt_sum > 0 else 0.0
        avg_turnover_5 = avg_turnover_5 if not np.isnan(avg_turnover_5) else 0.0
    if avg_turnover_5 < min_turnover:
        return None

    # 20日均成交额
    avg_amount_20 = float(np.nanmean(amount[idx - 19:idx + 1])) if idx >= 19 else float(np.nanmean(amount[:idx + 1]))
    if avg_amount_20 < min_amount:
        return None

    # 涨跌幅
    gain5 = (close[idx] / close[idx - 5] - 1.0) * 100.0 if close[idx - 5] > 0 else 0.0
    gain20 = (close[idx] / close[idx - 20] - 1.0) * 100.0 if close[idx - 20] > 0 else 0.0

    # RSI-14（Wilder 平滑，EMA-like，与通达信标准一致）
    delta = np.diff(close)
    gain = np.clip(delta, 0, None)
    loss = np.clip(-delta, 0, None)
    rsi_arr = np.full(len(close), np.nan, dtype=float)
    avg_g = float(gain[:14].mean())
    avg_l = float(loss[:14].mean())
    rs = avg_g / avg_l if avg_l > 1e-12 else 100.0
    rsi_arr[13] = 100.0 - 100.0 / (1.0 + rs)
    for i in range(14, len(gain)):
        avg_g = (avg_g * 13.0 + gain[i]) / 14.0
        avg_l = (avg_l * 13.0 + loss[i]) / 14.0
        rs = avg_g / avg_l if avg_l > 1e-12 else 100.0
        rsi_arr[i] = 100.0 - 100.0 / (1.0 + rs)
    rsi = float(rsi_arr[idx]) if not np.isnan(rsi_arr[idx]) else 50.0

    # MA
    ma5 = float(pd.Series(close).rolling(5).mean().iloc[idx])
    ma10 = float(pd.Series(close).rolling(10).mean().iloc[idx])
    ma20 = float(pd.Series(close).rolling(20).mean().iloc[idx])
    ma60 = float(pd.Series(close).rolling(60).mean().iloc[idx])

    # 更严格的趋势过滤：close > ma20 > ma60 确保中长期趋势向上
    if not (close[idx] > ma20 > ma60):
        return None

    # 综合评分：DIF强度 × 红柱天数奖励
    # DIF/DEA 硬上限防止爆炸，再用 min(..., 20) 确保稳定
    dif_strength = min(dif[idx] / abs(dea[idx]), 20.0) if dea[idx] != 0 else 0.0
    red_bonus = max(0, 12 - red_days) * 5  # 红柱越少奖励越高
    score = round(min(100, dif_strength * 4.0 + red_bonus + 50), 1)

    return {
        "code": normalize_prefixed(code),
        "dif": round(float(dif[idx]), 4),
        "dea": round(float(dea[idx]), 4),
        "macd": round(float(macd[idx]), 4),
        "red_days": red_days,
        "score": score,
        "close": round(float(close[idx]), 2),
        "gain1": round(gain1, 2),
        "gain3": round(gain3, 2),
        "gain5": round(gain5, 2),
        "gain20": round(gain20, 2),
        "rsi": round(rsi, 1),
        "avg_turnover_5": round(avg_turnover_5, 2),
        "avg_amount_20": round(avg_amount_20 / 1e8, 2),  # 亿
        "ma5": round(ma5, 2),
        "ma10": round(ma10, 2),
        "ma20": round(ma20, 2),
        "ma60": round(ma60, 2),  # 新增MA60
    }


# ─────────────────────────────────────────────────────────
# 全市场扫描
# ─────────────────────────────────────────────────────────
def scan_market(codes: list, min_turnover: float, min_amount: float,
                max_workers: int, target_date: datetime | None) -> list[dict]:
    from gain_turnover import load_qfq_history

    results = []
    t0 = time.time()
    total = len(codes)

    def work(code: str) -> dict | None:
        c = normalize_prefixed(code)
        end_date = target_date.strftime("%Y-%m-%d") if target_date else None
        df = load_qfq_history(c, end_date=end_date, adjust="qfq", refresh=False)
        if df is None or df.empty:
            return None
        if target_date is not None:
            df = df[df["date"] <= pd.Timestamp(target_date.date())].reset_index(drop=True)
        if df.empty:
            return None
        result = evaluate_macd_stock(c, df, min_turnover, min_amount)
        if result is not None:
            name = get_stock_name(c, load_stock_names()) or ""
            result["name"] = name
        return result

    done = [0]

    def log_progress(futures):
        for f in as_completed(futures):
            done[0] += 1
            if done[0] % 200 == 0 or done[0] == total:
                eta = (time.time() - t0) / done[0] * (total - done[0])
                print(f"  进度: {done[0]}/{total} ({done[0]*100//total}%) ETA={eta:.0f}s", flush=True)

    print(f"📋 全市场扫描: {total} 只")
    print(f"🚀 开始 MACD 筛选（workers={max_workers}）...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(work, c): c for c in codes}
        log_progress(futures)
        for future in as_completed(futures):
            r = future.result()
            if r is not None:
                results.append(r)

    print(f"✅ 扫描完成: {len(results)}/{total} 只通过，用时 {time.time()-t0:.1f}s")
    return results


# ─────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MACD 趋势启动型选股")
    parser.add_argument("--date", type=str, default=None, help="信号日期 YYYY-MM-DD（复盘用）")
    parser.add_argument("--min-turnover", type=float, default=DEFAULT_MIN_TURNOVER, help=f"5日均换手率下限/%%（默认{DEFAULT_MIN_TURNOVER}）")
    parser.add_argument("--min-amount", type=float, default=1.0, help=f"20日均成交额下限/亿元（默认1.0）")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help=f"并行线程数（默认{DEFAULT_WORKERS}）")
    parser.add_argument("--codes", nargs="+", default=None, help="指定股票代码列表（跳过全市场）")
    parser.add_argument("--output", type=str, default=None, help="输出文件路径")
    args = parser.parse_args()

    target_date = None
    if args.date:
        target_date = datetime.strptime(args.date, "%Y-%m-%d")
        print(f"\n📅 复盘模式: {args.date}")

    # 输出路径
    date_str = target_date.strftime("%Y-%m-%d") if target_date else datetime.now().strftime("%Y-%m-%d")
    output_path = Path(args.output) if args.output else Path.home() / "stock_reports" / f"macd_screen_{date_str}.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 市场行情（不做止损，只展示）
    if target_date:
        try:
            from stock_trend.trend_strong_screen import get_market_gain, INDEX_CODES
            market = get_market_gain(INDEX_CODES, days=DEFAULT_MARKET_DAYS, target_date=target_date)
            print(f"📈 市场{DEFAULT_MARKET_DAYS}日涨幅: {market:.2f}%")
        except Exception:
            market = 0.0

    # 股票范围
    if args.codes:
        codes = args.codes
        print(f"\n📊 MACD 筛选（指定 {len(codes)} 只）")
    else:
        from stock_trend.rps_strong_screen import get_all_stock_codes
        codes = get_all_stock_codes()
        print(f"\n📊 MACD 筛选（全市场 {len(codes)} 只）")

    # 扫描
    results = scan_market(
        codes=codes,
        min_turnover=args.min_turnover,
        min_amount=args.min_amount * 1e8,
        max_workers=args.workers,
        target_date=target_date,
    )

    if not results:
        print("\n⚠️  无符合 MACD 启动形态的股票")
        sys.exit(0)

    # 按 score 降序
    results.sort(key=lambda x: x["score"], reverse=True)

    # 打印 & 写入
    print(f"\n{'='*72}")
    print(f"📊 MACD 趋势启动型 {date_str}（共 {len(results)} 只）")
    print("=" * 160)

    lines = []
    lines.append(f"📊 MACD 趋势启动型 {date_str}（共 {len(results)} 只）")
    lines.append("=" * 160)

    # 列宽（字符数）：统一用 _rpad 保证对齐
    _COLS = [10, 8, 12, 6, 6, 8, 10, 10, 6, 8]
    header = "\t".join([
        _rpad('代码',10), _rpad('名称',8), _rpad('信号日',12),
        _rpad('评分',6), _rpad('红柱天',6), _rpad('收盘',8),
        _rpad('当日涨幅',10), _rpad('3日涨幅',10), _rpad('RSI',6), _rpad('换手%',8),
    ])
    print(header)
    lines.append(header)
    print("-" * 160)
    lines.append("-" * 160)

    for r in results:
        code = r["code"]
        name = r["name"]
        score = r["score"]
        gain3 = r["gain3"]
        red_days = r["red_days"]
        close = r["close"]
        gain1 = r["gain1"]
        rsi = r["rsi"]
        turnover = r["avg_turnover_5"]


        row = "\t".join([
            _rpad(code,10), _rpad(name,8), _rpad(date_str,12),
            _rpad(str(score),6), _rpad(str(red_days),6), _rpad(str(close),8),
            _rpad(f'{gain1:+.2f}%',10), _rpad(f'{gain3:+.2f}%',10), _rpad(str(rsi),6), _rpad(str(turnover),8),
        ])
        print(row)
        lines.append(row)

    print("=" * 160)
    lines.append("=" * 160)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"💾 结果已写入: {output_path}")
