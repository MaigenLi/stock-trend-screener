#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
涨幅-换手策略公共模块（前复权版）

核心升级：
1. 统一使用前复权日线（AkShare + 本地缓存）
2. 拆分“信号窗口”和“质量窗口”
3. 统一筛选与回测评分口径
4. 回测支持 T+1 开盘买入、持有 N 个交易日后收盘卖出
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import akshare as ak


def _disp_w(s: str) -> int:
    """计算字符串的显示宽度 (ASCII=1, CJK=2)."""
    return sum(2 if (ord(c) >> 11) else 1 for c in s)


def _rpad(s: str, width: int) -> str:
    """左对齐，按显示宽度补足宽度 (for text columns)."""
    d = _disp_w(s)
    return s + " " * max(0, width - d)


def _lpad(s: str, width: int) -> str:
    """右对齐，按显示宽度补足宽度 (for numeric columns)."""
    d = _disp_w(s)
    return " " * max(0, width - d) + s
import numpy as np
import pandas as pd

WORKSPACE = Path(__file__).parent.parent.resolve()
CACHE_DIR = WORKSPACE / "stock_trend" / ".cache" / "qfq_daily"
STOCK_CODES_FILE = Path.home() / "stock_code" / "results" / "stock_codes.txt"
STOCK_NAMES_FILE = Path.home() / "stock_code" / "results" / "all_stock_names_final.json"

_CACHE_LOCKS: dict[str, threading.Lock] = {}
_CACHE_LOCKS_GUARD = threading.Lock()


@dataclass
class StrategyConfig:
    signal_days: int = 2
    min_gain: float = 2.0
    max_gain: float = 7.0
    quality_days: int = 10
    min_turnover: float = 1.5
    min_amount: float = 1e8
    score_threshold: float = 60.0
    adjust: str = "qfq"
    max_extension_pct: float = 10.0
    min_history_days: int = 90


def get_lock(key: str) -> threading.Lock:
    with _CACHE_LOCKS_GUARD:
        if key not in _CACHE_LOCKS:
            _CACHE_LOCKS[key] = threading.Lock()
        return _CACHE_LOCKS[key]


def normalize_symbol(code: str) -> str:
    code = str(code).strip().lower()
    if code.startswith(("sh", "sz", "bj")):
        return code[-6:]
    return code


def normalize_prefixed(code: str) -> str:
    code = str(code).strip().lower()
    if code.startswith(("sh", "sz", "bj")):
        return code
    if len(code) == 6 and code.isdigit():
        if code.startswith(("60", "68", "90")):
            return f"sh{code}"
        if code.startswith(("00", "20", "30")):
            return f"sz{code}"
        if code.startswith(("43", "83", "87", "92")):
            return f"bj{code}"
    return code


def get_all_stock_codes() -> List[str]:
    if not STOCK_CODES_FILE.exists():
        raise FileNotFoundError(f"股票代码文件不存在: {STOCK_CODES_FILE}")
    codes: List[str] = []
    with open(STOCK_CODES_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            codes.append(normalize_prefixed(line))
    return list(dict.fromkeys(codes))


def load_stock_names() -> Dict[str, str]:
    names: Dict[str, str] = {}
    if not STOCK_NAMES_FILE.exists():
        return names
    try:
        with open(STOCK_NAMES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        stocks = data.get("stocks", {}) if isinstance(data, dict) else {}
        for code, info in stocks.items():
            if not isinstance(info, dict):
                continue
            name = info.get("name", "未知")
            names[code.lower()] = name
            pure = info.get("code", "")
            if pure:
                names[pure.lower()] = name
    except Exception:
        pass
    return names


def get_stock_name(code: str, names_cache: Dict[str, str]) -> str:
    c = normalize_prefixed(code).lower()
    if c in names_cache:
        return names_cache[c]
    pure = c[-6:]
    return names_cache.get(pure, "未知")


def _cache_path(code: str, adjust: str = "qfq") -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    pure = normalize_symbol(code)
    return CACHE_DIR / f"{pure}_{adjust}.csv"


def _fetch_ak_qfq(code: str, adjust: str = "qfq") -> pd.DataFrame:
    """优先用更稳定的 stock_zh_a_daily，失败时回退 stock_zh_a_hist。"""
    pure = normalize_symbol(code)
    prefixed = normalize_prefixed(code)

    # 1) 优先 daily 接口
    last_error = None
    for _ in range(3):
        try:
            df = ak.stock_zh_a_daily(
                symbol=prefixed,
                start_date="19900101",
                end_date="21000101",
                adjust=adjust,
            )
            if df is not None and not df.empty:
                out = df.copy()
                out["date"] = pd.to_datetime(out["date"])
                out["turnover"] = pd.to_numeric(out.get("turnover"), errors="coerce") * 100.0
                for c in ["open", "high", "low", "close", "volume", "amount"]:
                    out[c] = pd.to_numeric(out.get(c), errors="coerce")
                out = out[["date", "open", "high", "low", "close", "volume", "amount", "turnover"]]
                out = out.dropna(subset=["date", "open", "high", "low", "close"]).sort_values("date").reset_index(drop=True)
                return out
        except Exception as e:
            last_error = e
            time.sleep(0.5)

    # 2) 回退 hist 接口
    for _ in range(2):
        try:
            df = ak.stock_zh_a_hist(
                symbol=pure,
                period="daily",
                start_date="19900101",
                end_date="20500101",
                adjust=adjust,
            )
            if df is None or df.empty:
                return pd.DataFrame()
            col_map = {
                "日期": "date",
                "股票代码": "symbol",
                "开盘": "open",
                "收盘": "close",
                "最高": "high",
                "最低": "low",
                "成交量": "volume",
                "成交额": "amount",
                "换手率": "turnover",
            }
            keep = [c for c in col_map if c in df.columns]
            df = df[keep].rename(columns={k: v for k, v in col_map.items() if k in keep}).copy()
            if "date" not in df.columns:
                return pd.DataFrame()
            df["date"] = pd.to_datetime(df["date"])
            for c in ["open", "high", "low", "close", "volume", "amount", "turnover"]:
                if c not in df.columns:
                    df[c] = np.nan
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df[["date", "open", "high", "low", "close", "volume", "amount", "turnover"]]
            df = df.dropna(subset=["date", "open", "high", "low", "close"]).sort_values("date").reset_index(drop=True)
            return df
        except Exception as e:
            last_error = e
            time.sleep(0.5)

    return pd.DataFrame()



def load_qfq_history(
    code: str,
    start_date: Optional[str | pd.Timestamp] = None,
    end_date: Optional[str | pd.Timestamp] = None,
    adjust: str = "qfq",
    refresh: bool = False,
    max_age_hours: float = 12.0,
) -> pd.DataFrame:
    """加载前复权日线。优先本地缓存，缺失或过期时从 AkShare 拉取。"""
    path = _cache_path(code, adjust=adjust)
    lock = get_lock(path.name)

    with lock:
        use_cache = path.exists() and not refresh
        if use_cache and max_age_hours > 0:
            age_hours = (time.time() - path.stat().st_mtime) / 3600.0
            if age_hours > max_age_hours and end_date is None:
                use_cache = False

        if use_cache:
            try:
                df = pd.read_csv(path, parse_dates=["date"])
            except Exception:
                df = pd.DataFrame()
        else:
            df = _fetch_ak_qfq(code, adjust=adjust)
            if not df.empty:
                df.to_csv(path, index=False, encoding="utf-8")
            elif path.exists():
                # 拉取失败时退回缓存
                try:
                    df = pd.read_csv(path, parse_dates=["date"])
                except Exception:
                    df = pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    if start_date is not None:
        start_ts = pd.Timestamp(start_date)
        df = df[df["date"] >= start_ts]
    if end_date is not None:
        end_ts = pd.Timestamp(end_date)
        df = df[df["date"] <= end_ts]
    return df.sort_values("date").reset_index(drop=True)


def rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(arr), np.nan, dtype=float)
    if len(arr) >= window:
        out[window - 1:] = np.convolve(arr, np.ones(window) / window, mode="valid")
    return out


def compute_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    close = np.asarray(close, dtype=float)
    out = np.full(len(close), np.nan, dtype=float)
    if len(close) <= period:
        return out
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.empty_like(gain)
    avg_loss = np.empty_like(loss)
    avg_gain[:period] = np.nan
    avg_loss[:period] = np.nan
    first_gain = gain[:period].mean()
    first_loss = loss[:period].mean()
    prev_g, prev_l = first_gain, first_loss
    rs = prev_g / prev_l if prev_l > 1e-12 else np.inf
    out[period] = 100.0 - 100.0 / (1.0 + rs)
    for i in range(period, len(gain)):
        prev_g = (prev_g * (period - 1) + gain[i]) / period
        prev_l = (prev_l * (period - 1) + loss[i]) / period
        rs = prev_g / prev_l if prev_l > 1e-12 else np.inf
        out[i + 1] = 100.0 - 100.0 / (1.0 + rs)
    return out


@dataclass
class PreparedData:
    df: pd.DataFrame
    dates: np.ndarray
    open_: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    amount: np.ndarray
    turnover: np.ndarray
    gains: np.ndarray
    ma5: np.ndarray
    ma10: np.ndarray
    ma20: np.ndarray
    ma60: np.ndarray
    avg_amount_5: np.ndarray
    avg_amount_20: np.ndarray
    avg_turnover_5: np.ndarray
    avg_turnover_20: np.ndarray
    rsi14: np.ndarray


@dataclass
class SignalResult:
    code: str
    name: str
    signal_date: str
    score: float
    total_gain_window: float
    close: float
    avg_amount_20: float
    avg_turnover_5: float
    rsi14: float
    ma5: float
    ma10: float
    ma20: float
    ma60: float
    extension_pct: float
    subscores: Dict[str, float]
    details: Dict[str, float | int | str]


def prepare_data(df: pd.DataFrame) -> Optional[PreparedData]:
    if df is None or df.empty or len(df) < 80:
        return None
    x = df.sort_values("date").reset_index(drop=True).copy()
    dates = x["date"].dt.strftime("%Y-%m-%d").values
    open_ = x["open"].astype(float).values
    high = x["high"].astype(float).values
    low = x["low"].astype(float).values
    close = x["close"].astype(float).values
    volume = x["volume"].astype(float).values
    amount = x["amount"].astype(float).values
    turnover = x["turnover"].astype(float).values if "turnover" in x.columns else np.full(len(x), np.nan)

    gains = np.full(len(x), np.nan, dtype=float)
    gains[1:] = (close[1:] / np.where(close[:-1] > 0, close[:-1], np.nan) - 1.0) * 100.0

    ma5 = rolling_mean(close, 5)
    ma10 = rolling_mean(close, 10)
    ma20 = rolling_mean(close, 20)
    ma60 = rolling_mean(close, 60)
    avg_amount_5 = rolling_mean(amount, 5)
    avg_amount_20 = rolling_mean(amount, 20)
    avg_turnover_5 = rolling_mean(turnover, 5)
    avg_turnover_20 = rolling_mean(turnover, 20)
    rsi14 = compute_rsi(close, 14)

    return PreparedData(
        df=x,
        dates=dates,
        open_=open_,
        high=high,
        low=low,
        close=close,
        volume=volume,
        amount=amount,
        turnover=turnover,
        gains=gains,
        ma5=ma5,
        ma10=ma10,
        ma20=ma20,
        ma60=ma60,
        avg_amount_5=avg_amount_5,
        avg_amount_20=avg_amount_20,
        avg_turnover_5=avg_turnover_5,
        avg_turnover_20=avg_turnover_20,
        rsi14=rsi14,
    )


def evaluate_signal(prepared: PreparedData, idx: int, config: StrategyConfig) -> Optional[dict]:
    min_idx = max(config.signal_days + 1, config.quality_days, 60, config.min_history_days)
    if idx < min_idx:
        return None

    signal_gains = prepared.gains[idx - config.signal_days + 1: idx + 1]
    quality_gains = prepared.gains[idx - config.quality_days + 1: idx + 1]
    if np.isnan(signal_gains).any() or np.isnan(quality_gains).any():
        return None

    if not np.all((signal_gains >= config.min_gain) & (signal_gains <= config.max_gain)):
        return None

    avg_amt20 = prepared.avg_amount_20[idx]
    avg_to5 = prepared.avg_turnover_5[idx]
    avg_to20 = prepared.avg_turnover_20[idx]
    if np.isnan(avg_amt20) or avg_amt20 < config.min_amount:
        return None
    if config.min_turnover > 0:
        if np.isnan(avg_to5) or avg_to5 < config.min_turnover:
            return None

    close = prepared.close[idx]
    ma5 = prepared.ma5[idx]
    ma10 = prepared.ma10[idx]
    ma20 = prepared.ma20[idx]
    ma60 = prepared.ma60[idx]
    if np.isnan(ma5) or np.isnan(ma10) or np.isnan(ma20) or np.isnan(ma60):
        return None

    ma5_prev = prepared.ma5[idx - 1]
    ma10_prev = prepared.ma10[idx - 1]
    if np.isnan(ma5_prev) or np.isnan(ma10_prev):
        return None

    gain20 = (prepared.close[idx] / prepared.close[idx - 20] - 1.0) * 100.0
    extension_pct = (close / ma20 - 1.0) * 100.0 if ma20 > 0 else 999.0
    rsi = prepared.rsi14[idx] if not np.isnan(prepared.rsi14[idx]) else 50.0
    amount_ratio_5_20 = prepared.avg_amount_5[idx] / avg_amt20 if avg_amt20 > 0 and not np.isnan(prepared.avg_amount_5[idx]) else 1.0
    gain10 = (prepared.close[idx] / prepared.close[idx - 10] - 1.0) * 100.0

    # 强制趋势过滤，升级版核心约束
    # ma5 >= ma10（允许0.5%容差），ma5和ma10均上涨; ma20位置不限
    if not (close > ma5 >= ma10 * 0.995):
        return None
    if ma5 <= ma5_prev or ma10 <= ma10_prev:
        return None
    if gain10 <= 0:
        return None
    if extension_pct > config.max_extension_pct:
        return None
    if rsi >= 82:
        return None

    # 评分系统（总分100）
    score = 0.0
    subscores: Dict[str, float] = {}

    # 1) 信号稳定性 20
    gain_std = float(np.std(signal_gains, ddof=0))
    stability = max(0.0, min(20.0, 20.0 - gain_std * 5.0))
    subscores["stability"] = round(stability, 2)
    score += stability

    # 2) 信号强度 10, 越靠近区间中值越好，避免太弱/太热
    mid = (config.min_gain + config.max_gain) / 2.0
    mean_gain = float(np.mean(signal_gains))
    strength = max(0.0, 10.0 - abs(mean_gain - mid) * 3.0)
    subscores["signal_strength"] = round(strength, 2)
    score += strength

    # 3) 趋势质量 25
    trend = 0.0
    trend += 10.0  # close > ma5 > ma10 已过滤
    if ma20 > prepared.ma20[idx - 10]:
        trend += 7.0
    if gain20 >= 5:
        trend += 5.0
    elif gain20 > 0:
        trend += 3.0
    if 0 <= extension_pct <= 8:
        trend += 3.0
    subscores["trend"] = round(trend, 2)
    score += trend

    # 4) 成交活跃度 15
    liquidity = 0.0
    if avg_amt20 >= config.min_amount * 2:
        liquidity += 8.0
    elif avg_amt20 >= config.min_amount:
        liquidity += 5.0
    if config.min_turnover <= 0:
        liquidity += 7.0
    else:
        if avg_to5 >= config.min_turnover * 1.5:
            liquidity += 7.0
        elif avg_to5 >= config.min_turnover:
            liquidity += 5.0
    subscores["liquidity"] = round(liquidity, 2)
    score += liquidity

    # 5) 量能配合 15
    volume_quality = 0.0
    if 0.9 <= amount_ratio_5_20 <= 2.5:
        volume_quality += 8.0
    elif 0.7 <= amount_ratio_5_20 <= 3.5:
        volume_quality += 5.0
    q_start = idx - config.quality_days + 1
    up_mask = quality_gains > 0
    down_mask = ~up_mask
    recent_amount = prepared.amount[q_start: idx + 1]
    if up_mask.any() and down_mask.any():
        up_amt = float(np.nanmean(recent_amount[up_mask]))
        down_amt = float(np.nanmean(recent_amount[down_mask]))
        if up_amt > down_amt * 1.15:
            volume_quality += 7.0
        elif up_amt > down_amt:
            volume_quality += 4.0
    else:
        volume_quality += 5.0
    subscores["volume_quality"] = round(volume_quality, 2)
    score += volume_quality

    # 6) K线质量 5
    candle_quality = 0.0
    opens = prepared.open_[idx - config.signal_days + 1: idx + 1]
    highs = prepared.high[idx - config.signal_days + 1: idx + 1]
    lows = prepared.low[idx - config.signal_days + 1: idx + 1]
    closes = prepared.close[idx - config.signal_days + 1: idx + 1]
    body = np.abs(closes - opens)
    full_range = np.maximum(highs - lows, 1e-6)
    upper_shadow = np.maximum(highs - np.maximum(opens, closes), 0)
    body_ratio = float(np.nanmean(body / full_range))
    upper_ratio = float(np.nanmean(upper_shadow / full_range))
    candle_quality += min(3.0, body_ratio * 6.0)
    candle_quality += max(0.0, min(2.0, 2.0 - upper_ratio * 5.0))
    subscores["candle_quality"] = round(candle_quality, 2)
    score += candle_quality

    # 7) RSI/不过热 10
    rsi_score = 0.0
    if 45 <= rsi <= 72:
        rsi_score = 10.0
    elif 38 <= rsi < 45:
        rsi_score = 7.0
    elif 72 < rsi <= 78:
        rsi_score = 6.0
    elif 78 < rsi < 82:
        rsi_score = 2.0
    subscores["rsi_health"] = round(rsi_score, 2)
    score += rsi_score

    if score < config.score_threshold:
        return None

    details = {
        "mean_gain_signal": round(mean_gain, 3),
        "gain_std_signal": round(gain_std, 3),
        "gain10_pct": round(gain10, 2),
        "gain20_pct": round(gain20, 2),
        "avg_amount_20": round(avg_amt20 / 1e8, 2),
        "avg_turnover_5": round(float(avg_to5), 2) if not np.isnan(avg_to5) else None,
        "avg_turnover_20": round(float(avg_to20), 2) if not np.isnan(avg_to20) else None,
        "amount_ratio_5_20": round(float(amount_ratio_5_20), 2),
        "extension_pct": round(extension_pct, 2),
        "rsi14": round(float(rsi), 2),
    }

    return {
        "score": round(score, 2),
        "subscores": subscores,
        "details": details,
        "total_gain_window": round(float(np.sum(signal_gains)), 2),
        "close": round(float(close), 2),
        "signal_date": str(prepared.dates[idx]),
        "rsi14": round(float(rsi), 2),
        "avg_amount_20": round(avg_amt20 / 1e8, 2),
        "avg_turnover_5": round(float(avg_to5), 2) if not np.isnan(avg_to5) else None,
        "ma5": round(float(ma5), 2),
        "ma10": round(float(ma10), 2),
        "ma20": round(float(ma20), 2),
        "ma60": round(float(ma60), 2),
        "extension_pct": round(float(extension_pct), 2),
    }


def evaluate_latest_signal(code: str, name: str, df: pd.DataFrame, config: StrategyConfig) -> Optional[SignalResult]:
    prepared = prepare_data(df)
    if prepared is None:
        return None
    idx = len(prepared.df) - 1
    result = evaluate_signal(prepared, idx, config)
    if result is None:
        return None
    return SignalResult(
        code=normalize_prefixed(code),
        name=name,
        signal_date=result["signal_date"],
        score=result["score"],
        total_gain_window=result["total_gain_window"],
        close=result["close"],
        avg_amount_20=result["avg_amount_20"],
        avg_turnover_5=result["avg_turnover_5"] or 0.0,
        rsi14=result["rsi14"],
        ma5=result["ma5"],
        ma10=result["ma10"],
        ma20=result["ma20"],
        ma60=result["ma60"],
        extension_pct=result["extension_pct"],
        subscores=result["subscores"],
        details=result["details"],
    )


def format_signal_results(results: List[SignalResult], title: str) -> str:
    lines = []
    lines.append("=" * 150)
    lines.append(f"📊 {title}（共 {len(results)} 只）")
    lines.append("=" * 150)
    lines.append(
        f"{_rpad('代码',10)} {_rpad('名称',8)} {_rpad('日期',12)} {_rpad('总分',6)} {_rpad('窗口涨幅',9)} "
        f"{_rpad('20日额(亿)',10)} {_rpad('5日换手',8)} {_rpad('RSI',6)} {_rpad('偏离MA20',9)} {_rpad('收盘',7)}"
    )
    lines.append("-" * 150)
    for r in results:
        name = r.name or ''
        code = r.code or ''
        signal_date = r.signal_date or ''
        lines.append(
            f"{_rpad(code,10)} {_rpad(name,8)} {_rpad(signal_date,12)} {_lpad(f'{r.score:.1f}',6)} "
            f"{_lpad(f'{r.total_gain_window:+.2f}%',9)} {_lpad(f'{r.avg_amount_20:.2f}',10)} "
            f"{_lpad(f'{r.avg_turnover_5:.2f}%',8)} {_lpad(f'{r.rsi14:.1f}',6)} "
            f"{_lpad(f'{r.extension_pct:+.2f}%',9)} {_lpad(f'{r.close:.2f}',7)}"
        )
    lines.append("-" * 150)
    lines.append("评分构成: 稳定性20 + 信号强度10 + 趋势25 + 流动性15 + 量能15 + K线5 + RSI10 = 100")
    return "\n".join(lines)
