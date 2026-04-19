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
    max_gain: float = 10.0
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
                # 真实换手率：vol × 100 / outstanding（%），用先乘后除避免浮点精度损失
                if "outstanding_share" in out.columns:
                    out["outstanding_share"] = pd.to_numeric(out["outstanding_share"], errors="coerce")
                    vol = out["volume"].astype(float).values
                    out_s = out["outstanding_share"].astype(float).values
                    out["true_turnover"] = np.where(
                        out_s > 0,
                        np.round(vol * 100.0 / out_s, 2),
                        np.nan
                    )
                    out["turnover"] = np.round(out["turnover"].values, 2)
                else:
                    out["outstanding_share"] = np.nan
                    out["true_turnover"] = np.nan
                out = out[["date", "open", "high", "low", "close", "volume", "amount",
                           "turnover", "outstanding_share", "true_turnover"]]
                out = out.dropna(subset=["date", "open", "high", "low", "close"]).sort_values("date").reset_index(drop=True)
                # daily 返回空数据（如某些科创板股票）时跳过，继续走 hist 回退
                if out.empty:
                    last_error = ValueError("daily returned empty DataFrame")
                else:
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
            # hist 接口无 outstanding_share，降级写 NaN
            df["outstanding_share"] = np.nan
            df["true_turnover"] = np.nan
            df = df[["date", "open", "high", "low", "close", "volume", "amount",
                     "turnover", "outstanding_share", "true_turnover"]]
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
    """加载前复权日线。优先本地缓存，缺失或过期时从 AkShare 拉取。

    回测用法：显式传入 end_date（截止日），函数内部完成防未来数据泄漏截断。
    实时用法：不传 end_date，自动取缓存最新日期（已验证无未来数据）。
    """
    path = _cache_path(code, adjust=adjust)
    lock = get_lock(path.name)

    with lock:
        use_cache = path.exists() and not refresh
        if use_cache and max_age_hours > 0:
            age_hours = (time.time() - path.stat().st_mtime) / 3600.0
            if age_hours > max_age_hours:
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

    # 统一时间类型
    df["date"] = pd.to_datetime(df["date"])

    # 确定截止时间戳
    if end_date is not None:
        end_ts = pd.Timestamp(end_date)
    else:
        # 实时模式：默认截到缓存已有最新日期（已保证无未来数据）
        end_ts = pd.Timestamp(df["date"].max())

    # ★ 第一步：截断 end_date（防止未来数据泄漏，必须在最前面）
    #max_raw_date = df["date"].max()
    df = df[df["date"] <= end_ts]

    # ★ 第二步：立即检查截断是否有效（截断后仍有数据超出 end_ts 则报错）
    if not df.empty and df["date"].max() > end_ts:
        raise RuntimeError(
            f"未来数据泄漏检测：max_date={df['date'].max()} > end_date={end_ts}，"
            f"缓存 {path.name} 包含 end_date 之后的未来数据，请删除缓存后重试"
        )

    # ★ 第三步：截断 start_date
    if start_date is not None:
        start_ts = pd.Timestamp(start_date)
        if start_ts > end_ts:
            raise ValueError(
                f"参数错误：start_date ({start_ts}) > end_date ({end_ts})"
            )
        df = df[df["date"] >= start_ts]

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
    turnover: np.ndarray      # 原始换手率（AkShare）
    true_turnover: np.ndarray  # 真实换手率：volume / (outstanding * 10000) * 100
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
    top_risk_penalty: float = 0.0    # 见顶风险扣分（P0）
    top_risk_reason: str = ""         # 见顶风险原因
    rsi_tier: str = ""   # 🟢低位/🟡健康/🔴高位/高位热/❌超买
    rsi_momentum: float = 0.0   # RSI今日 - RSI昨日（RSI动量）
    rsi_momentum_score: float = 0.0  # RSI动量得分
    volume_accel: float = 0.0   # 今日量 / 昨日量（量能加速度）
    volume_accel_score: float = 0.0  # 量能加速度得分


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
    turnover = (np.round(x["turnover"].astype(float).values, 2)
                if "turnover" in x.columns else np.full(len(x), np.nan))

    # 真实换手率：vol × 100 / outstanding（%），先乘后除避免浮点精度损失
    if "outstanding_share" in x.columns and "true_turnover" not in x.columns:
        outstanding = x["outstanding_share"].astype(float).values
        true_turnover = np.where(
            outstanding > 0,
            np.round(volume * 100.0 / outstanding, 2),
            np.round(turnover, 2)  # 降级：无法计算时用原始换手率
        )
    elif "true_turnover" in x.columns:
        true_turnover = np.round(x["true_turnover"].astype(float).values, 2)
    else:
        true_turnover = np.round(turnover.copy(), 2)  # 历史缓存无 outstanding 时降级

    gains = np.full(len(x), np.nan, dtype=float)
    gains[1:] = (close[1:] / np.where(close[:-1] > 0, close[:-1], np.nan) - 1.0) * 100.0

    ma5 = rolling_mean(close, 5)
    ma10 = rolling_mean(close, 10)
    ma20 = rolling_mean(close, 20)
    ma60 = rolling_mean(close, 60)
    avg_amount_5 = rolling_mean(amount, 5)
    avg_amount_20 = rolling_mean(amount, 20)
    avg_turnover_5 = rolling_mean(true_turnover, 5)
    avg_turnover_20 = rolling_mean(true_turnover, 20)
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
        true_turnover=true_turnover,
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
    # 逻辑：质量窗口内，任意连续 surge_days 日均 true_turnover >= 前 surge_days 日均 × 1.30
    # 例：--days 2 → 窗口1-2 vs 3-4, 窗口2-3 vs 4-5, ..., 均满足则通过
    if config.check_volume_surge and config.quality_days >= 5:
        q_start = idx - config.quality_days + 1
        quality_to = prepared.true_turnover[q_start: idx + 1]
        surge_days = max(2, int(round(config.volume_surge_ratio)))  # 2.0=2天, 3.0=3天
        found_surge = False
        for i in range(len(quality_to) - surge_days * 2 + 1):
            # 窗口A: quality_to[i : i+surge_days]  vs 窗口B: quality_to[i+surge_days : i+surge_days*2]
            window_a = float(np.nanmean(quality_to[i: i + surge_days]))
            window_b = float(np.nanmean(quality_to[i + surge_days: i + surge_days * 2]))
            if (not np.isnan(window_a) and not np.isnan(window_b)
                    and window_b > 0 and window_a >= window_b * 1.30):
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
    rsi_prev = prepared.rsi14[idx-1] if idx >= 1 and not np.isnan(prepared.rsi14[idx-1]) else rsi
    rsi_momentum = float(rsi - rsi_prev)
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

    # ── P0: 见顶风险过滤（6项见顶信号 + RECOVER 化解）─────────────
    top_risk_penalty, top_risk_reason = check_top_risk(prepared, idx)

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

    # ── P1: 形态质量加分 ─────────────────────────────────────────
    form_bonus = 0.0

    # CONSIST: 10日内 ≥ 7天收盘在 MA10 上方（稳定性）
    n_consist = min(config.signal_days, 10)
    if n_consist >= 7 and idx >= n_consist:
        c_arr = prepared.close[idx - n_consist + 1: idx + 1]
        ma10_arr = prepared.ma10[idx - n_consist + 1: idx + 1]
        above_ma10 = int(np.sum(c_arr > ma10_arr))
        if above_ma10 >= 7:
            form_bonus += 5.0
            subscores["consist_pass"] = True
        else:
            subscores["consist_pass"] = False
    else:
        subscores["consist_pass"] = False

    # F_HL: 低点抬升（近5日低点 > 前5日低点 > 再前5日低点）
    if idx >= 15:
        low_0_4  = float(np.nanmin(prepared.low[idx - 4: idx + 1]))
        low_5_9  = float(np.nanmin(prepared.low[idx - 9: idx - 4]))
        low_10_14 = float(np.nanmin(prepared.low[idx - 14: idx - 9]))
        if low_0_4 > low_5_9 > low_10_14:
            form_bonus += 5.0
            subscores["f_hl_pass"] = True
        else:
            subscores["f_hl_pass"] = False
    else:
        subscores["f_hl_pass"] = False

    # F_BP: 平台突破（近15日振幅 < 15%，放量突破前高，量 > MA20量 × 1.3）
    if idx >= 16:
        close_15 = prepared.close[idx - 15: idx]
        range_15 = (float(np.nanmax(close_15)) - float(np.nanmin(close_15))) / float(np.nanmax(close_15)) * 100.0
        high_15 = float(np.nanmax(close_15))
        ma20_amt_today = float(prepared.avg_amount_20[idx]) if not np.isnan(prepared.avg_amount_20[idx]) else 0.0
        ma20_amt_prior = float(prepared.avg_amount_20[idx - 1]) if not np.isnan(prepared.avg_amount_20[idx - 1]) else ma20_amt_today
        vol_3d = float(np.nanmean(prepared.volume[idx - 2: idx + 1]))
        today_vol = float(prepared.volume[idx])
        avg_vol = max(vol_3d, today_vol)
        if range_15 < 15.0 and close > high_15 and ma20_amt_prior > 0 and avg_vol >= ma20_amt_prior * 1.3:
            form_bonus += 5.0
            subscores["f_bp_pass"] = True
        else:
            subscores["f_bp_pass"] = False
    else:
        subscores["f_bp_pass"] = False

    subscores["form_bonus"] = form_bonus
    score += form_bonus

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

    # ── RSI 分层扣分（tiered penalty）────────────────────────────────
    # RSI 45~65: 健康，上涨持续性好 → 不扣分
    # RSI 65~72: 偏强 → 扣5分
    # RSI 72~75: 高位区 → 扣10分
    # RSI 75~78: 过热预警 → 扣15分
    # RSI 78~82: 过热 → 扣25分（但不强排除，因为趋势延续性强）
    # RSI > 82: 超买 → calc 已有过滤，这里不会到
    rsi_tier = ""
    if rsi < 50:
        rsi_tier = "🔵低位"
    elif 50 <= rsi <= 65:
        rsi_tier = "🟢健康"
    elif 65 < rsi <= 72:
        rsi_tier = "🟡偏强"
        score -= 5
    elif 72 < rsi <= 75:
        rsi_tier = "🔴高位"
        score -= 10
    elif 75 < rsi <= 78:
        rsi_tier = "高位热"
        score -= 15
    elif 78 < rsi <= 82:
        rsi_tier = "高位热"
        score -= 25
    else:
        rsi_tier = "❌超买"
    # Store tier penalty value for subscores record
    tier_penalty_map = {"🟡偏强": -5.0, "🔴高位": -10.0, "高位热": -25.0, "❌超买": -40.0}
    subscores["rsi_tier_penalty"] = tier_penalty_map.get(rsi_tier, 0.0)

    # ── RSI 动量加分（加速上涨中的股继续涨）───────────────────────
    # RSI动量 = RSI今日 - RSI昨日
    # RSI动量 > +5: 加速中 → +5分
    # RSI动量 > +8: 强烈加速 → +10分
    # RSI动量 < -5: 开始回落 → 扣5分
    rsi_momentum_score = 0.0
    if rsi_momentum > 8:
        rsi_momentum_score = 10.0
    elif rsi_momentum > 5:
        rsi_momentum_score = 5.0
    elif rsi_momentum < -5:
        rsi_momentum_score = -5.0
    subscores["rsi_momentum"] = round(rsi_momentum, 2)
    subscores["rsi_momentum_score"] = round(rsi_momentum_score, 2)
    score += rsi_momentum_score

    # ── 量能加速度（连续放量 = 主力参与）─────────────────────────
    # vol_accel = 今日成交量 / 昨日成交量
    vol_today = float(prepared.volume[idx]) if idx >= 0 else 1.0
    vol_yesterday = float(prepared.volume[idx-1]) if idx >= 1 else vol_today
    vol_accel = vol_today / vol_yesterday if vol_yesterday > 0 else 1.0
    volume_accel_score = 0.0
    if vol_accel >= 2.0:
        volume_accel_score = 8.0   # 今日量是昨日2倍以上，强力放量
    elif vol_accel >= 1.5:
        volume_accel_score = 5.0   # 明显放量
    elif vol_accel >= 1.2:
        volume_accel_score = 2.0   #温和放量
    subscores["volume_accel"] = round(vol_accel, 2)
    subscores["volume_accel_score"] = round(volume_accel_score, 2)
    score += volume_accel_score

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

    # ── 见顶风险扣分 ─────────────────────────────────────────────
    if top_risk_penalty < 0:
        score += top_risk_penalty   # e.g. -30（6项全触发）
        subscores["top_risk_penalty"] = top_risk_penalty
        subscores["top_risk_reason"] = top_risk_reason

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
        "top_risk_penalty": subscores.get("top_risk_penalty", 0.0),
        "top_risk_reason": subscores.get("top_risk_reason", "无见顶信号"),
        "rsi_tier": rsi_tier,
        "rsi_momentum": round(rsi_momentum, 2),
        "rsi_momentum_score": round(rsi_momentum_score, 2),
        "volume_accel": round(vol_accel, 2),
        "volume_accel_score": round(volume_accel_score, 2),
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
        top_risk_penalty=result.get("top_risk_penalty", 0.0),
        top_risk_reason=result.get("top_risk_reason", ""),
        rsi_tier=result.get("rsi_tier", "🟢健康"),
        rsi_momentum=result.get("rsi_momentum", 0.0),
        rsi_momentum_score=result.get("rsi_momentum_score", 0.0),
        volume_accel=result.get("volume_accel", 0.0),
        volume_accel_score=result.get("volume_accel_score", 0.0),
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
        f"{_rpad('收盘',7)}\t{_rpad('扣分',5)}"
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
            f"{_lpad(f'{r.extension_pct:+.2f}%',9)}\t{_lpad(f'{r.close:.2f}',7)}\t{_lpad(penalty_str,8)}"
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


# ─────────────────────────────────────────────────────────────────────────────
# P0: 见顶风险过滤（6项见顶信号 + RECOVER 化解机制）
# ─────────────────────────────────────────────────────────────────────────────
def _compute_macd(closes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (dif, dea, macd)，用标准参数 EMA(12,26,9)。"""
    ema12 = _ema(closes, 12)
    ema26 = _ema(closes, 26)
    dif = ema12 - ema26
    dea = _ema(dif, 9)
    macd = (dif - dea) * 2.0   # 柱状图
    return dif, dea, macd


def _ema(series: np.ndarray, n: int) -> np.ndarray:
    """指数移动平均，pandas风格。"""
    import pandas as pd
    s = pd.Series(series)
    return s.ewm(span=n, adjust=False).mean().to_numpy()


def check_top_risk(prepared: PreparedData, idx: int) -> tuple[float, str]:
    """
    检查6项见顶风险信号，返回 (penalty, reason)。
    penalty < 0 表示有风险；reason 描述风险内容。
    如果满足 RECOVER 条件（强势反包），penalty = 0。
    """
    PENALTY_PER_SIGNAL = -10.0   # 每个信号扣分
    MAX_LOOKBACK = 20            # 最大回溯天数

    lookback = min(idx, MAX_LOOKBACK)
    if lookback < 5:
        return 0.0, ""

    close_arr = prepared.close[idx - lookback: idx + 1]
    open_arr  = prepared.open_[idx - lookback: idx + 1]
    high_arr  = prepared.high[idx - lookback: idx + 1]
    low_arr   = prepared.low[idx - lookback: idx + 1]
    vol_arr   = prepared.volume[idx - lookback: idx + 1]

    today_close = float(close_arr[-1])
    today_open  = float(open_arr[-1])
    today_vol   = float(vol_arr[-1])
    today_gain  = float(prepared.gains[idx]) if not np.isnan(prepared.gains[idx]) else 0.0

    # 基础指标
    ma5_vol  = float(np.nanmean(vol_arr[-6:-1]))   # 前5日均量（不含今日）
    ma5_vol_y = float(np.nanmean(vol_arr[-6:-1]))  # 同上（用于昨日5日均量）
    vol_arr_for_ma5 = vol_arr[:-1]  # 排除今日
    ma5_vol_calc = float(np.nanmean(vol_arr_for_ma5[-5:])) if len(vol_arr_for_ma5) >= 5 else today_vol
    ma5_vol_y_calc = float(np.nanmean(vol_arr_for_ma5[-6:-1])) if len(vol_arr_for_ma5) >= 6 else ma5_vol_calc

    # 今日换手
    today_to = float(prepared.true_turnover[idx]) if not np.isnan(prepared.true_turnover[idx]) else 0.0

    # RECOVER 条件：涨幅>3% AND 收盘>MA20 AND 量>MA5均量
    ma20_today = float(prepared.ma20[idx]) if not np.isnan(prepared.ma20[idx]) else 0.0
    ma5_vol_today_ref = float(np.nanmean(vol_arr[-6:-1]))  # 前5日均量（含昨）
    recover = (today_gain > 3.0) and (today_close > ma20_today > 0) and (today_vol > ma5_vol_today_ref)

    total_penalty = 0.0
    reasons = []

    # ── 信号3: 缩量新高（3日内）────────────────────────────────
    # 股价 >= 20日最高 × 0.98 AND 量 < 10日均量 × 0.6
    lookback3 = min(lookback, 3)
    bl_found = False
    for d in range(lookback3):
        c = float(close_arr[d])   # d=0 是最早天，d=lookback-1 是今天
        h20 = float(np.nanmax(close_arr[d:min(d+20, len(close_arr))]))
        vol_ma10 = float(np.nanmean(vol_arr[d:min(d+10, len(vol_arr))]))
        if c >= h20 * 0.98 and vol_ma10 > 0 and float(vol_arr[d]) < vol_ma10 * 0.6:
            bl_found = True
            break
    if bl_found:
        total_penalty += PENALTY_PER_SIGNAL
        reasons.append("缩量新高")

    # ── 信号4: 高位放量大阴线（5日内）──────────────────────────
    # 跌幅 > 5% AND 量 > 昨日5日均量 × 1.5
    lookback5 = min(lookback, 5)
    dayin_found = False
    for d in range(lookback5):
        if d == 0:
            continue
        prev_close = float(close_arr[d - 1])
        curr_close = float(close_arr[d])
        if prev_close <= 0:
            continue
        drop_pct = (prev_close - curr_close) / prev_close * 100.0
        vol_prev_ma5 = float(np.nanmean(vol_arr[max(0,d-5):d])) if d >= 5 else float(np.nanmean(vol_arr[:d])) if d > 0 else today_vol
        if drop_pct > 5.0 and float(vol_arr[d]) > vol_prev_ma5 * 1.5:
            dayin_found = True
            break
    if dayin_found:
        total_penalty += PENALTY_PER_SIGNAL
        reasons.append("放量大阴")

    # ── 信号5: 高位连续阴线（5日内阴线>=3天）───────────────────
    # NO_LY: SUM(CLOSE<OPEN, 5) < 3（即阴线<3天才安全，等价于≥3则风险）
    lookback5 = min(lookback, 5)
    close_arr_5 = close_arr[-lookback5:] if lookback5 > 0 else close_arr
    open_arr_5  = open_arr[-lookback5:] if lookback5 > 0 else open_arr
    red_days = int(np.sum(close_arr_5 < open_arr_5))
    if red_days >= 3:
        total_penalty += PENALTY_PER_SIGNAL
        reasons.append(f"连阴{red_days}天")

    # ── 信号6: MACD高位死叉（5日内，DIF在零轴上方死叉）──────────
    lookback5 = min(lookback, 5)
    if lookback5 >= 2:
        macd_close = prepared.close[idx - lookback5: idx + 1]
        if len(macd_close) >= 27:
            dif, dea, _ = _compute_macd(macd_close)
            macd_found = False
            for d in range(1, len(dif)):
                # DIF 从上方（前一周期>0）下穿 DEA（当前<0）
                if dif[d-1] > 0 and dif[d] < dea[d]:
                    macd_found = True
                    break
            if macd_found:
                total_penalty += PENALTY_PER_SIGNAL
                reasons.append("MACD高位死叉")

    # ── 信号①: 放量滞涨（5日内）────────────────────────────────
    # 量是前5日均量2倍以上 AND 涨幅 < 1%
    lookback5 = min(lookback, 5)
    zz_found = False
    for d in range(lookback5):
        if d == 0:
            continue
        vol_ma5_prior = float(np.nanmean(vol_arr[max(0,d-5):d])) if d >= 5 else float(np.nanmean(vol_arr[:d])) if d > 0 else today_vol
        gain_d = float(prepared.gains[idx - lookback5 + d]) if not np.isnan(prepared.gains[idx - lookback5 + d]) else 0.0
        if vol_ma5_prior > 0 and float(vol_arr[d]) >= vol_ma5_prior * 2.0 and gain_d < 1.0:
            zz_found = True
            break
    if zz_found:
        total_penalty += PENALTY_PER_SIGNAL
        reasons.append("放量滞涨")

    # ── 信号②: 长上影线（3日内）────────────────────────────────
    # 上影 >= 实体×2 AND 上影 > 股价×1% AND 振幅 > 2%
    lookback3 = min(lookback, 3)
    yx_found = False
    for d in range(lookback3):
        o = float(open_arr[d]); c = float(close_arr[d])
        h = float(high_arr[d]); l = float(low_arr[d])
        body = abs(c - o)
        upper_shadow = max(h - max(o, c), 0)
        full_range = max(h - l, 1e-6)
        if (body > 0 and upper_shadow >= body * 2.0
                and upper_shadow > today_close * 0.01
                and full_range / today_close * 100.0 > 2.0):
            yx_found = True
            break
    if yx_found:
        total_penalty += PENALTY_PER_SIGNAL
        reasons.append("长上影")

    reason_str = "; ".join(reasons) if reasons else "无见顶信号"

    # RECOVER: 强势反包化解所有风险
    if recover and total_penalty < 0:
        total_penalty = 0.0
        reason_str = "RECOVER化解"

    return total_penalty, reason_str
