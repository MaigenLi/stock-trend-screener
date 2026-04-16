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
CACHE_DIR = WORKSPACE / ".cache" / "qfq_daily"
STOCK_CODES_FILE = Path.home() / "stock_code" / "results" / "stock_codes.txt"
STOCK_NAMES_FILE = Path.home() / "stock_code" / "results" / "all_stock_names_final.json"
FUNDAMENTAL_CACHE_DIR = WORKSPACE / ".cache" / "fundamental"
FUNDAMENTAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── 板块缓存 ──────────────────────────────────────────────
SECTOR_CACHE_DIR = WORKSPACE / ".cache" / "sector"
SECTOR_MAP_FILE = SECTOR_CACHE_DIR / "stock_sector_map.json"
SECTOR_TOP_FILE  = SECTOR_CACHE_DIR / "top_sectors.json"
SECTOR_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _fetch_sina_sectors() -> tuple[list[dict], list[dict]]:
    """从新浪获取所有行业板块及涨跌幅，返回 (top_list, bottom_list)。"""
    try:
        import re, urllib.request
        url = "https://vip.stock.finance.sina.com.cn/q/view/newFLJK.php?param=class"
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://finance.sina.com.cn"
        })
        with urllib.request.urlopen(req, timeout=10) as resp:
            txt = resp.read().decode("gbk", errors="replace")
        pattern = re.compile(r'"(gn_\w+)":"([^"]+)"')
        matches = pattern.findall(txt)
        boards = []
        for key, value in matches:
            parts = value.split(",")
            if len(parts) < 5:
                continue
            try:
                name = parts[1]
                pct = float(parts[4]) if parts[4].replace(".", "", 1).replace("-", "", 1).isdigit() else 0
                boards.append({"name": name, "pct": pct, "code": key})
            except (ValueError, IndexError):
                continue
        boards.sort(key=lambda x: x["pct"], reverse=True)
        return boards[:15], boards[-5:][::-1]
    except Exception as e:
        print(f"⚠️ 新浪板块获取失败: {e}")
        return [], []


def get_top_sectors(n: int = 15, use_cache: bool = True) -> set[str]:
    """
    获取今日热门板块（涨跌幅前N名），返回板块名称集合。
    缓存文件: top_sectors.json（有效期24小时）
    """
    if use_cache and SECTOR_TOP_FILE.exists():
        age_hours = (time.time() - SECTOR_TOP_FILE.stat().st_mtime) / 3600.0
        if age_hours < 24:
            try:
                data = json.loads(SECTOR_TOP_FILE.read_text(encoding="utf-8"))
                return set(data["top_sectors"])
            except Exception:
                pass
    top, _ = _fetch_sina_sectors()
    top_names = {b["name"] for b in top[:n]}
    SECTOR_TOP_FILE.write_text(
        json.dumps({"top_sectors": list(top_names)}, ensure_ascii=False),
        encoding="utf-8"
    )
    return top_names


def get_stock_sector_map() -> dict[str, str]:
    """
    加载股票→板块映射表（从缓存文件），不存在则返回空dict。
    缓存文件: stock_sector_map.json
    """
    if not SECTOR_MAP_FILE.exists():
        return {}
    try:
        return json.loads(SECTOR_MAP_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_stock_sector_map(mapping: dict[str, str]):
    """保存股票→板块映射表到缓存文件。"""
    try:
        existing = get_stock_sector_map()
        existing.update(mapping)
        SECTOR_MAP_FILE.write_text(json.dumps(existing, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        print(f"⚠️ 板块映射缓存保存失败: {e}")


def fetch_stock_sector_from_sina(code: str) -> str | None:
    """
    通过新浪API查询单只股票的所属行业板块，返回板块名称，失败返回None。
    使用 Sina 行业板块（160+板块，每板块1只代表股票）构建逆向映射。
    """
    try:
        import re, urllib.request
        url = "https://vip.stock.finance.sina.com.cn/q/view/newFLJK.php?param=class"
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://finance.sina.com.cn"
        })
        with urllib.request.urlopen(req, timeout=10) as resp:
            txt = resp.read().decode("gbk", errors="replace")
        pattern = re.compile(r'"(gn_\w+)":"([^"]+)"')
        matches = pattern.findall(txt)
        # 逆向: stock_code → sector_name
        for key, value in matches:
            parts = value.split(",")
            if len(parts) < 9:
                continue
            stock_code = parts[8].strip()   # 代表股票代码
            sector_name = parts[1].strip()  # 板块名称
            if stock_code == code:
                return sector_name
        return None
    except Exception:
        return None


def resolve_stock_sector(code: str, cache: dict[str, str]) -> tuple[str | None, dict[str, str]]:
    """
    获取股票所属板块（优先用缓存，否则尝试查询Sina）。
    返回 (sector_name, updated_cache)
    """
    code = normalize_prefixed(code)
    if code in cache:
        return cache[code], cache
    sector = fetch_stock_sector_from_sina(code)
    if sector:
        new_cache = dict(cache)
        new_cache[code] = sector
        save_stock_sector_map({code: sector})
        return sector, new_cache
    return None, cache


_CACHE_LOCKS: dict[str, threading.Lock] = {}
_CACHE_LOCKS_GUARD = threading.Lock()


@dataclass
class StrategyConfig:
    signal_days: int = 2
    min_gain: float = 2.0
    max_gain: float = 7.0
    quality_days: int = 20
    min_turnover: float = 1.5
    min_amount: float = 1e8
    score_threshold: float = 60.0
    adjust: str = "qfq"
    max_extension_pct: float = 16.0
    min_history_days: int = 90
    check_fundamental: bool = False   # 是否检查基本面（亏损/PE为负扣分）
    sector_bonus: bool = False         # 是否开启热门板块加分
    sector_top_n: int = 15             # 前N名板块视为热门板块
    sector_bonus_pts: float = 8.0      # 热门板块加分分值
    check_volume_surge: bool = False  # 是否检查质量窗口内明显放量（默认关闭）
    volume_surge_ratio: float = 1.8  # 放量倍数阈值（默认1.8倍）


@dataclass
class FundamentalData:
    """基本面数据（最新季度报告）。"""
    code: str
    eps: float          # 摊薄每股收益（元）
    roe: float          # 净资产收益率（%）
    gross_margin: float # 销售毛利率（%）
    net_profit: float   # 净利润（元）
    is_profitable: bool # 是否盈利
    pe: float           # 市盈率（动态，可能为nan）
    report_date: str     # 最新报告日期


# ── 基本面扣分常量 ───────────────────────────────────
FUNDAMENTAL_PENALTY_LOSS = 10     # 亏损（EPS<0）扣10分
FUNDAMENTAL_PENALTY_NEGATIVE_PE = 10  # PE为负（亏损股）扣10分


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



def _fundamental_cache_path(code: str) -> Path:
    """基本面缓存路径（JSON，每季度更新一次）。"""
    pure = normalize_symbol(code)
    return FUNDAMENTAL_CACHE_DIR / f"{pure}.json"


def _fetch_fundamental(code: str) -> Optional[FundamentalData]:
    """
    从 AkShare 获取单只股票基本面数据（最新季度）。
    数据源：stock_financial_analysis_indicator（东方财富）
    """
    pure = normalize_symbol(code)
    last_error = None
    for _ in range(2):
        try:
            df = ak.stock_financial_analysis_indicator(symbol=pure, start_year="2023")
            if df is None or df.empty:
                last_error = "empty"
                time.sleep(0.3)
                continue
            df = df.sort_values("日期", ascending=False).reset_index(drop=True)
            latest = df.iloc[0]

            # EPS（摊薄每股收益，取最新有效值）
            eps_col = next((c for c in df.columns if "摊薄每股收益" in c), None)
            eps_series = df[eps_col].dropna() if eps_col else pd.Series(dtype=float)
            eps = float(eps_series.iloc[-1]) if not eps_series.empty else 0.0

            # ROE（净资产收益率，取最新有效值）
            roe_col = next((c for c in df.columns if "净资产收益率" in c and "加权" not in c), None)
            roe_series = df[roe_col].dropna() if roe_col else pd.Series(dtype=float)
            roe = float(roe_series.iloc[-1]) if not roe_series.empty else 0.0

            # 毛利率（取最新有效值）
            gm_col = next((c for c in df.columns if "销售毛利率" in c), None)
            gm_series = df[gm_col].dropna() if gm_col else pd.Series(dtype=float)
            gross_margin = float(gm_series.iloc[-1]) if not gm_series.empty else 0.0

            # 净利润（单位：元 → 转为亿元）
            # 优先：扣除非经常性损益后的净利润(元)，其次归属母公司净利润(元)
            np_col = next(
                (c for c in df.columns if c in ("扣除非经常性损益后的净利润(元)", "归属母公司净利润(元)", "净利润(元)")),
                None
            )
            net_profit = float(latest[np_col]) / 1e8 if np_col and pd.notna(latest.get(np_col)) else 0.0

            # PE（估算：收盘价 / EPS）
            pe = float("nan")

            # 盈利判断：EPS>0
            is_profitable = eps > 0
            report_date = str(latest.get("日期", ""))

            return FundamentalData(
                code=normalize_prefixed(code),
                eps=eps,
                roe=roe,
                gross_margin=gross_margin,
                net_profit=net_profit,
                is_profitable=is_profitable,
                pe=pe,
                report_date=report_date,
            )
        except Exception as e:
            last_error = e
            time.sleep(0.3)
    return None


def load_fundamental_data(code: str, refresh: bool = False) -> Optional[FundamentalData]:
    """
    加载基本面数据（缓存优先，过期时间设为90天）。
    """
    path = _fundamental_cache_path(code)
    if not refresh and path.exists():
        age_days = (time.time() - path.stat().st_mtime) / 86400.0
        if age_days < 90:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                return FundamentalData(**raw)
            except Exception:
                pass
    # 缓存未命中，尝试从网络获取
    data = _fetch_fundamental(code)
    if data is not None:
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(asdict(data), f, ensure_ascii=False)
        except Exception:
            pass
    return data



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
    fundamental_penalty: int = 0
    fundamental_eps: Optional[float] = None
    fundamental_roe: Optional[float] = None
    fundamental_is_profitable: Optional[bool] = None
    fundamental_report_date: Optional[str] = None
    sector_name: Optional[str] = None
    sector_bonus_applied: float = 0.0
    limit_up_bonus: float = 0.0


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


def evaluate_signal(prepared: PreparedData, idx: int, config: StrategyConfig,
                  code: Optional[str] = None, fundamental: Optional[FundamentalData] = None,
                  top_sectors: Optional[set[str]] = None,
                  stock_sector_map: Optional[dict[str, str]] = None) -> Optional[dict]:
    min_idx = max(config.signal_days + 1, config.quality_days, 60, config.min_history_days)
    if idx < min_idx:
        return None

    signal_gains = prepared.gains[idx - config.signal_days + 1: idx + 1]
    quality_gains = prepared.gains[idx - config.quality_days + 1: idx + 1]
    if np.isnan(signal_gains).any() or np.isnan(quality_gains).any():
        return None

    # ── 质量窗口过滤（可选）：近quality_days个交易日内必须有明显放量区间
    if config.check_volume_surge and config.quality_days >= 5:
        q_start = idx - config.quality_days + 1
        quality_amounts = prepared.amount[q_start: idx + 1]
        if len(quality_amounts) >= 5:
            # 滑动窗口：1-5 vs 6-10，2-6 vs 7-11，3-7 vs 8-12，...
            # 只要有任意一组满足放大倍数，即为放量
            n = len(quality_amounts)
            found_surge = False
            for i in range(n - 9):   # 需要i+9 < n，即i <= n-10
                prev_avg = float(np.nanmean(quality_amounts[i:i+5]))
                curr_avg = float(np.nanmean(quality_amounts[i+5:i+10]))
                if not (np.isnan(prev_avg) or np.isnan(curr_avg) or prev_avg <= 0):
                    if curr_avg >= prev_avg * config.volume_surge_ratio:
                        found_surge = True
                        break
            if not found_surge:
                return None

    # 信号窗口过滤：允许 1/3 的交易日不满足 min_gain（days >= 3 时）
    # 但最后一个交易日必须满足 > -3%（允许小幅回调但不超过 -3%）
    below_min = (signal_gains < config.min_gain).sum()
    max_allowed_below = config.signal_days // 3 if config.signal_days >= 3 else 0
    last_day_ok = (signal_gains[-1] > -3.0) if len(signal_gains) > 0 else True
    if not (last_day_ok and below_min <= max_allowed_below and np.all(signal_gains <= config.max_gain)):
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
    # max_extension 动态化为 signal_days * max_gain
    dynamic_max_ext = config.signal_days * config.max_gain
    if extension_pct > dynamic_max_ext:
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
    # ── 上升形态加分 ─────────────────────────────────
    # MA5 持续上涨
    ma5_5d_ago = prepared.ma5[idx - 5]
    if not np.isnan(ma5_5d_ago) and ma5 > ma5_5d_ago * 1.01:   # 5日内上涨 >1%
        trend += 3.0
    elif not np.isnan(ma5_5d_ago) and ma5 > ma5_5d_ago:
        trend += 1.5
    # MA10 持续上涨
    ma10_5d_ago = prepared.ma10[idx - 5]
    if not np.isnan(ma10_5d_ago) and ma10 > ma10_5d_ago * 1.01:  # 5日内上涨 >1%
        trend += 3.0
    elif not np.isnan(ma10_5d_ago) and ma10 > ma10_5d_ago:
        trend += 1.5
    # MA20 加速上涨
    ma20_10d_ago = prepared.ma20[idx - 10]
    if not np.isnan(ma20_10d_ago) and ma20 > ma20_10d_ago * 1.02:  # 10日内上涨 >2%
        trend += 4.0
    elif not np.isnan(ma20_10d_ago) and ma20 > ma20_10d_ago:
        trend += 2.0
    # MA60 向上拐点（从下跌转为上涨）
    ma60_10d_ago = prepared.ma60[idx - 10]
    ma60_5d_ago = prepared.ma60[idx - 5]
    if not np.isnan(ma60_10d_ago) and not np.isnan(ma60_5d_ago):
        if ma60 > ma60_5d_ago > ma60_10d_ago:   # MA60 加速向上
            trend += 3.0
        elif ma60 > ma60_10d_ago:               # MA60 整体向上
            trend += 1.5
    # 20日涨幅趋势
    if gain20 >= 5:
        trend += 5.0
    elif gain20 > 0:
        trend += 3.0
    # 偏离MA20适中（不过度偏离）
    if 0 <= extension_pct <= 8:
        trend += 3.0
    subscores["trend"] = round(min(trend, 40.0), 2)   # 上限40
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

    # ── 近10日涨停加分 ─────────────────────────────────
    # 涨停阈值：前复权数据中单日涨幅 >= 9.5%（留0.5%容差）
    recent_gains = prepared.gains[idx - 9: idx + 1]   # 最近10个交易日（含今日）
    limit_up_bonus = 0.0
    if len(recent_gains) >= 10 and (~np.isnan(recent_gains)).sum() >= 10:
        if np.any(recent_gains >= 9.5):
            limit_up_bonus = 10.0
            score += limit_up_bonus

    # ── 基本面扣分（亏损 / PE为负）──────────────────────
    fundamental_penalty = 0
    if config.check_fundamental and fundamental is not None:
        if not fundamental.is_profitable:
            fundamental_penalty = FUNDAMENTAL_PENALTY_LOSS
        score -= fundamental_penalty

    # ── 热门板块加分 ────────────────────────────────────
    sector_name = None
    sector_bonus_applied = 0.0
    if config.sector_bonus and top_sectors and code and stock_sector_map:
        c = normalize_prefixed(code)
        sector_name = stock_sector_map.get(c)
        if sector_name and sector_name in top_sectors:
            sector_bonus_applied = config.sector_bonus_pts
            score += sector_bonus_applied

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
        "fundamental_penalty": fundamental_penalty,
        "fundamental_eps": round(fundamental.eps, 3) if fundamental is not None else None,
        "fundamental_roe": round(fundamental.roe, 2) if fundamental is not None else None,
        "fundamental_is_profitable": fundamental.is_profitable if fundamental is not None else None,
        "fundamental_report_date": fundamental.report_date if fundamental is not None else None,
        "sector_name": sector_name,
        "sector_bonus_applied": sector_bonus_applied,
        "limit_up_bonus": limit_up_bonus,
    }


def evaluate_latest_signal(code: str, name: str, df: pd.DataFrame, config: StrategyConfig,
                         fundamental: Optional[FundamentalData] = None,
                         top_sectors: Optional[set[str]] = None,
                         stock_sector_map: Optional[dict[str, str]] = None) -> Optional[SignalResult]:
    prepared = prepare_data(df)
    if prepared is None:
        return None
    # 基本面按需加载（缓存90天）
    if config.check_fundamental and fundamental is None:
        fundamental = load_fundamental_data(code)
    idx = len(prepared.df) - 1
    result = evaluate_signal(prepared, idx, config, code=code, fundamental=fundamental,
                             top_sectors=top_sectors, stock_sector_map=stock_sector_map)
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
        fundamental_penalty=result["fundamental_penalty"],
        fundamental_eps=result["fundamental_eps"],
        fundamental_roe=result["fundamental_roe"],
        fundamental_is_profitable=result["fundamental_is_profitable"],
        fundamental_report_date=result["fundamental_report_date"],
        sector_name=result.get("sector_name"),
        sector_bonus_applied=result.get("sector_bonus_applied", 0.0),
        limit_up_bonus=result.get("limit_up_bonus", 0.0),
    )


def format_signal_results(results: List[SignalResult], title: str) -> str:
    lines = []
    lines.append("=" * 160)
    lines.append(f"📊 {title}（共 {len(results)} 只）")
    lines.append("=" * 160)
    # 主表（技术面）— 列间用 \t 分隔，兼容 webchat
    col_spec = (
        f"{_rpad('代码',10)}\t{_rpad('名称',8)}\t{_rpad('日期',12)}\t{_rpad('总分',6)}\t{_rpad('窗口涨幅',9)}\t"
        f"{_rpad('20日额(亿)',10)}\t{_rpad('5日换手',8)}\t{_rpad('RSI',6)}\t{_rpad('偏离MA20',9)}\t"
        f"{_rpad('收盘',7)}\t{_rpad('EPS',7)}\t{_rpad('ROE%%',7)}\t{_rpad('盈利',5)}\t{_rpad('扣分',5)}"
    )
    lines.append(col_spec)
    lines.append("-" * 160)
    for r in results:
        name = r.name or ""
        code = r.code or ""
        signal_date = r.signal_date or ""
        eps_str = f"{r.fundamental_eps:.3f}" if r.fundamental_eps is not None else "-"
        roe_str = f"{r.fundamental_roe:.2f}" if r.fundamental_roe is not None else "-"
        profit_str = "✓" if r.fundamental_is_profitable else ("✗" if r.fundamental_is_profitable is False else "-")
        # 扣分列：基本面扣分 或 板块加分
        if r.sector_bonus_applied > 0:
            penalty_str = f"+{int(r.sector_bonus_applied)}({r.sector_name})"
        elif r.fundamental_penalty:
            penalty_str = f"-{r.fundamental_penalty}"
        else:
            penalty_str = "-"
        # 附加信息：板块加分 或 涨停加分
        extras = []
        if r.sector_bonus_applied > 0:
            extras.append(f"+{int(r.sector_bonus_applied)}({r.sector_name})")
        if r.limit_up_bonus > 0:
            extras.append(f"+{int(r.limit_up_bonus)}涨停")
        if extras:
            penalty_str = " ".join(extras)
        row = (
            f"{_rpad(code,10)}\t{_rpad(name,8)}\t{_rpad(signal_date,12)}\t{_lpad(f'{r.score:.1f}',6)}\t"
            f"{_lpad(f'{r.total_gain_window:+.2f}%',9)}\t{_lpad(f'{r.avg_amount_20:.2f}',10)}\t"
            f"{_lpad(f'{r.avg_turnover_5:.2f}%',8)}\t{_lpad(f'{r.rsi14:.1f}',6)}\t"
            f"{_lpad(f'{r.extension_pct:+.2f}%',9)}\t{_lpad(f'{r.close:.2f}',7)}\t"
            f"{_lpad(eps_str,7)}\t{_lpad(roe_str,7)}\t{_lpad(profit_str,5)}\t{_lpad(penalty_str,8)}"
        )
        lines.append(row)
    lines.append("-" * 160)
    bonus_parts = []
    if any(r.sector_bonus_applied > 0 for r in results):
        bonus_parts.append("热门板块+8")
    if any(r.limit_up_bonus > 0 for r in results):
        bonus_parts.append("近10日涨停+10")
    bonus_note = (" + " + " + ".join(bonus_parts)) if bonus_parts else ""
    lines.append(f"评分: 稳定性20 + 信号强度10 + 趋势25 + 流动性15 + 量能15 + K线5 + RSI10{bonus_note}")
    # 基本面详情（亏损股摘要）
    loss_stocks = [r for r in results if r.fundamental_penalty > 0]
    if loss_stocks:
        lines.append(f"\n⚠️  亏损/微利股（已扣分，共 {len(loss_stocks)} 只）：")
        lines.append(f"{'代码':<10}\t{'名称':<8}\t{'报告期':<12}\t{'EPS(元)':>8}\t{'ROE%%':>7}\t{'扣分':>5}")
        for r in loss_stocks:
            lines.append(
                f"{r.code:<10}\t{r.name:<8}\t"
                f"{(r.fundamental_report_date or '-'):<12}\t"
                f"{(r.fundamental_eps if r.fundamental_eps is not None else 0):>8.3f}\t"
                f"{(r.fundamental_roe if r.fundamental_roe is not None else 0):>7.2f}\t"
                f"{r.fundamental_penalty:>5}"
            )
    return "\n".join(lines)
