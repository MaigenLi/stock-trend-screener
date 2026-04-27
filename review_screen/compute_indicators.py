#!/usr/bin/env python3
"""
预计算所有股票的日线指标
========================

对 .cache/qfq_daily/*.csv 中的每只股票，按日计算：
- 移动平均线：MA5/10/20/60
- RSI(14)
- MACD（DIF/DEA/红柱）
- 波段检测（涨段起止点、涨跌幅、均量）
- 量能指标：量比、涨跌量比
- 趋势指标：20日涨幅、stop_loss_ref

输出：.cache/indicators/{code}_indicators.json
每只股票一个文件，包含该股票所有日期的指标序列

用法：
    python compute_indicators.py --workers 8
    python compute_indicators.py --codes 000001 000002  # 只计算指定股票
    python compute_indicators.py --refresh  # 强制重新计算（跳过已有文件）
"""

import sys
import json
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd

WORKSPACE = Path(__file__).parent.parent.parent.resolve()
CACHE_DIR = WORKSPACE / ".cache"
QFQ_DIR = CACHE_DIR / "qfq_daily"
INDICATORS_DIR = CACHE_DIR / "indicators"

INDICATORS_DIR.mkdir(parents=True, exist_ok=True)

MIN_DAYS = 80  # 至少需要80根K线才能计算完整指标

COLUMNS = ["date", "open", "high", "low", "close", "volume", "amount",
           "turnover", "outstanding_share", "true_turnover"]


def compute_ma(close: np.ndarray, period: int) -> np.ndarray:
    """移动平均"""
    out = np.full_like(close, np.nan, dtype=np.float64)
    for i in range(period - 1, len(close)):
        out[i] = np.nanmean(close[i - period + 1:i + 1])
    return out


def compute_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """RSI（ Wilder 平滑法）"""
    out = np.full_like(close, np.nan, dtype=np.float64)
    if len(close) < period + 1:
        return out
    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    if avg_loss == 0:
        out[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        out[period] = 100 - 100 / (1 + rs)
    for i in range(period + 1, len(close)):
        delta = deltas[i - 1]
        gain = delta if delta > 0 else 0.0
        loss = -delta if delta < 0 else 0.0
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        if avg_loss == 0:
            out[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            out[i] = 100 - 100 / (1 + rs)
    return out


def compute_macd(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
    """MACD（DIF/DEA/柱）"""
    ema_fast = np.full_like(close, np.nan)
    ema_slow = np.full_like(close, np.nan)
    dif = np.full_like(close, np.nan)
    dea = np.full_like(close, np.nan)
    macd_hist = np.full_like(close, np.nan)

    if len(close) < slow + signal:
        return dif, dea, macd_hist

    # EMA初始化
    ema_fast[slow - 1] = np.mean(close[:slow])
    ema_slow[slow - 1] = np.mean(close[:slow])

    alpha_fast = 2 / (fast + 1)
    alpha_slow = 2 / (slow + 1)

    for i in range(slow, len(close)):
        ema_fast[i] = ema_fast[i - 1] * (1 - alpha_fast) + close[i] * alpha_fast
        ema_slow[i] = ema_slow[i - 1] * (1 - alpha_slow) + close[i] * alpha_slow
        dif[i] = ema_fast[i] - ema_slow[i]

    # DEA（signal line）
    dea[slow + signal - 2] = np.mean(dif[slow - 1:slow + signal - 1])
    alpha_dea = 2 / (signal + 1)
    for i in range(slow + signal - 1, len(close)):
        dea[i] = dea[i - 1] * (1 - alpha_dea) + dif[i] * alpha_dea
        macd_hist[i] = (dif[i] - dea[i]) * 2

    return dif, dea, macd_hist


def detect_volume_price_wave(
    close: np.ndarray,
    volume: np.ndarray,
    ma20: np.ndarray,
    ma60: np.ndarray,
    lookback: int = 60,
) -> dict:
    """
    波段检测（简化版，用于回测）
    返回最近一个完整波段的信息
    """
    if len(close) < lookback or np.isnan(close[-1]):
        return {"wave_count": 0, "up_waves": [], "down_waves": []}

    # 从 lookback 位置往前扫
    start = max(0, len(close) - lookback - 1)
    prices = close[start:]
    vols = volume[start:]

    up_waves = []
    down_waves = []
    in_up = prices[0] < prices[-1]  # 粗判断方向

    # 简化：找最近的高低点
    max_idx = np.argmax(prices)
    min_idx = np.argmin(prices)

    if max_idx > min_idx:
        # 先跌后涨 → 一个下跌波段 + 一个上涨波段
        down_waves.append({
            "start": start, "peak": start + max_idx,
            "end": start + min_idx,
            "gain": (prices[min_idx] / prices[max_idx] - 1) * 100,
        })
        up_waves.append({
            "start": start + min_idx, "end": start + len(prices) - 1,
            "gain": (prices[-1] / prices[min_idx] - 1) * 100,
        })
    else:
        # 先涨后跌 → 一个上涨波段 + 一个下跌波段
        up_waves.append({
            "start": start, "end": start + max_idx,
            "gain": (prices[max_idx] / prices[0] - 1) * 100,
        })
        down_waves.append({
            "start": start + max_idx, "end": start + len(prices) - 1,
            "gain": (prices[-1] / prices[max_idx] - 1) * 100,
        })

    return {
        "wave_count": len(up_waves) + len(down_waves),
        "up_waves": up_waves,
        "down_waves": down_waves,
    }


def compute_volume_metrics(volume: np.ndarray, lookback: int = 20) -> dict:
    """量能指标"""
    if len(volume) < lookback:
        return {"vol_ratio": 1.0, "vol_up_vs_down": 1.0, "vol_up_days": 0}

    vol_ma = np.nanmean(volume[-lookback:])
    vol_now = volume[-1] if not np.isnan(volume[-1]) else vol_ma
    vol_ratio = vol_now / vol_ma if vol_ma > 0 else 1.0

    # 涨跌量比（近5日）
    gains = np.diff(volume[-(lookback - 1):])
    up_days = np.sum(gains > 0)
    down_days = np.sum(gains < 0)

    vol_up = np.nanmean(gains[gains > 0]) if np.any(gains > 0) else 0
    vol_down = abs(np.nanmean(gains[gains < 0])) if np.any(gains < 0) else 0
    vol_up_vs_down = vol_up / vol_down if vol_down > 0 else 1.0

    return {
        "vol_ratio": vol_ratio,
        "vol_up_vs_down": vol_up_vs_down,
        "vol_up_days_ratio": up_days / (up_days + down_days) if (up_days + down_days) > 0 else 0.5,
    }


def compute_wave_quality_score(
    close: np.ndarray,
    up_waves: list,
    down_waves: list,
) -> float:
    """
    波段质量评分（0-20分）
    - 上涨波段是否创新高（+5）
    - 涨跌幅度比（+5）
    - 回调是否浅（+5）
    - 上涨日占比（+5）
    """
    if not up_waves:
        return 0.0

    score = 0.0
    last_up = up_waves[-1] if up_waves else None

    # 上涨波段高度
    if last_up and last_up.get("gain", 0) > 10:
        score += 3.0
    elif last_up and last_up.get("gain", 0) > 5:
        score += 1.5

    # 下跌是否浅
    if down_waves:
        last_down = down_waves[-1]
        dd = abs(last_down.get("gain", 0))
        if dd < 5:
            score += 4.0
        elif dd < 10:
            score += 2.0

    # 上涨日占比（近10日）
    if len(close) >= 11:
        price_changes = np.diff(close[-11:])
        up_ratio = np.sum(price_changes > 0) / 10
        score += up_ratio * 4.0

    return min(score, 12.0)


def compute_stop_loss_ref(close: np.ndarray, ma20: np.ndarray, lookback: int = 20) -> float:
    """止损参考价：近20日最低点（或MA20，取较近者）"""
    if len(close) < lookback:
        return close[-1] * 0.97

    low_20 = np.nanmin(close[-lookback:])
    ref_ma20 = ma20[-1] if not np.isnan(ma20[-1]) else close[-1]
    # 取较高者（止损更紧）
    return max(low_20 * 1.002, ref_ma20 * 0.98)


def compute_all_indicators_for_stock(df: pd.DataFrame, code: str) -> list[dict]:
    """
    对一只股票计算所有日期的指标
    每行返回一个 dict（供后续回测使用）
    """
    if len(df) < MIN_DAYS:
        return []

    # 字段解析
    close = df["close"].values.astype(np.float64)
    open_ = df["open"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    volume = df["volume"].values.astype(np.float64)
    dates = df["date"].values

    # 预处理 NaN
    close = np.nan_to_num(close, nan=close[-1])
    volume = np.nan_to_num(volume, nan=0)

    # 计算基础指标
    ma5 = compute_ma(close, 5)
    ma10 = compute_ma(close, 10)
    ma20 = compute_ma(close, 20)
    ma60 = compute_ma(close, 60)
    rsi = compute_rsi(close, 14)
    dif, dea, macd_hist = compute_macd(close)

    results = []

    for i in range(60, len(close)):  # 至少需要60根K线
        if np.isnan(close[i]):
            continue

        date = str(dates[i])

        # 20日涨幅
        gain20 = (close[i] / close[i - 20] - 1) * 100 if i >= 20 and not np.isnan(close[i - 20]) else 0.0

        # MA分离度
        ma20_val = ma20[i] if not np.isnan(ma20[i]) else close[i]
        ma60_val = ma60[i] if not np.isnan(ma60[i]) else close[i]
        ma_sep = (ma20_val / ma60_val - 1) * 100 if ma60_val > 0 else 0

        # 量能
        vol_metrics = compute_volume_metrics(volume, 20)
        vol_ratio = vol_metrics["vol_ratio"]
        vol_up_vs_down = vol_metrics["vol_up_vs_down"]

        # 波段
        wave_info = detect_volume_price_wave(
            close[:i + 1], volume[:i + 1],
            ma20[:i + 1], ma60[:i + 1], lookback=40
        )
        up_waves = wave_info.get("up_waves", [])
        down_waves = wave_info.get("down_waves", [])
        wave_quality = compute_wave_quality_score(close[:i + 1], up_waves, down_waves)

        # 止损参考
        stop_loss_ref = compute_stop_loss_ref(close[:i + 1], ma20[:i + 1], 20)

        # RSI
        rsi_val = rsi[i] if not np.isnan(rsi[i]) else 50.0

        # MA5偏离
        ma5_val = ma5[i] if not np.isnan(ma5[i]) else close[i]
        ma5_dist = (close[i] / ma5_val - 1) * 100

        # 今日涨幅
        gain1 = (close[i] / close[i - 1] - 1) * 100 if i >= 1 and not np.isnan(close[i - 1]) else 0.0

        # MACD红柱（近3日）
        macd_red = 1 if macd_hist[i] > 0 else 0
        macd_red_3 = int(macd_hist[i] > 0) + int(macd_hist[i - 1] > 0) + int(macd_hist[i - 2] > 0) if i >= 2 else 0

        # 波段结构
        max_price = np.nanmax(high[i - 5:i + 1])
        min_price = np.nanmin(low[i - 5:i + 1])
        structure_score = (max_price / min_price - 1) * 100 if min_price > 0 else 0

        results.append({
            "code": code,
            "date": date,
            "close": round(float(close[i]), 2),
            "open": round(float(open_[i]), 2),
            "high": round(float(high[i]), 2),
            "low": round(float(low[i]), 2),
            "volume": float(volume[i]),
            "ma5": round(float(ma5_val), 2),
            "ma10": round(float(ma10[i]), 2),
            "ma20": round(float(ma20_val), 2),
            "ma60": round(float(ma60_val), 2),
            "ma_sep": round(ma_sep, 3),
            "rsi": round(rsi_val, 1),
            "gain1": round(gain1, 2),
            "gain20": round(gain20, 2),
            "vol_ratio": round(vol_ratio, 3),
            "vol_up_vs_down": round(vol_up_vs_down, 3),
            "wave_quality": round(wave_quality, 2),
            "stop_loss_ref": round(stop_loss_ref, 2),
            "ma5_distance_pct": round(ma5_dist, 2),
            "macd_red_days": macd_red_3,
            "structure_score": round(structure_score, 2),
            "dif": round(float(dif[i]), 4) if not np.isnan(dif[i]) else 0.0,
            "dea": round(float(dea[i]), 4) if not np.isnan(dea[i]) else 0.0,
            "macd_hist": round(float(macd_hist[i]), 4) if not np.isnan(macd_hist[i]) else 0.0,
        })

    return results


def process_stock_file(csv_path: Path, force: bool = False) -> tuple[str, int]:
    """
    处理单只股票文件
    Returns: (code, num_days_processed)
    """
    code = csv_path.stem.replace("_qfq", "")
    out_path = INDICATORS_DIR / f"{code}_indicators.json"

    if out_path.exists() and not force:
        # 检查是否已包含最新日期
        try:
            with open(out_path) as f:
                existing = json.load(f)
            if existing and existing[-1]["date"] >= "2026-04-23":
                return code, len(existing)
        except Exception:
            pass

    try:
        df = pd.read_csv(csv_path, usecols=COLUMNS)
        df = df.sort_values("date").reset_index(drop=True)

        results = compute_all_indicators_for_stock(df, code)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False)

        return code, len(results)
    except Exception as e:
        return code, 0


def get_all_codes() -> list[str]:
    """获取所有股票代码"""
    codes = []
    for f in QFQ_DIR.glob("*_qfq.csv"):
        code = f.stem.replace("_qfq", "")
        codes.append(code)
    return sorted(codes)


def main():
    parser = argparse.ArgumentParser(description="预计算所有股票日线指标")
    parser.add_argument("--workers", type=int, default=8, help="并行线程数")
    parser.add_argument("--codes", nargs="+", default=None, help="只计算指定股票")
    parser.add_argument("--refresh", action="store_true", help="强制重新计算")
    parser.add_argument("--limit", type=int, default=0, help="限制股票数量（用于测试）")
    args = parser.parse_args()

    if args.codes:
        csv_files = [QFQ_DIR / f"{c}_qfq.csv" for c in args.codes]
        csv_files = [f for f in csv_files if f.exists()]
        print(f"📊 计算 {len(csv_files)} 只指定股票")
    else:
        csv_files = sorted(QFQ_DIR.glob("*_qfq.csv"))
        if args.limit > 0:
            csv_files = csv_files[:args.limit]
        print(f"📊 预计算 {len(csv_files)} 只股票指标")

    start = datetime.now()
    done = 0
    errors = 0
    total_days = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_stock_file, f, args.refresh): f
            for f in csv_files
        }

        for future in as_completed(futures):
            code, n_days = future.result()
            done += 1
            if n_days > 0:
                total_days += n_days
            else:
                errors += 1

            if done % 200 == 0 or done == len(csv_files):
                elapsed = (datetime.now() - start).total_seconds()
                speed = done / elapsed if elapsed > 0 else 0
                eta = (len(csv_files) - done) / speed if speed > 0 else 0
                print(f"  进度: {done}/{len(csv_files)}  已计算{total_days}个交易日  速度:{speed:.0f}只/秒  ETA:{eta:.0f}秒")

    elapsed = (datetime.now() - start).total_seconds()
    print(f"\n✅ 完成: {done} 只股票  {total_days} 个交易日  耗时{elapsed:.1f}秒  错误{errors}只")
    print(f"💾 输出: {INDICATORS_DIR}/")


if __name__ == "__main__":
    main()
