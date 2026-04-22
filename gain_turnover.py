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
import urllib.request
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
    score_threshold: float = 40.0
    adjust: str = "qfq"
    max_extension_pct: float = 16.0
    min_history_days: int = 90
    check_fundamental: bool = False   # 是否检查基本面（亏损/PE为负扣分）
    sector_bonus: bool = False         # 是否开启热门板块加分
    sector_top_n: int = 15             # 前N名板块视为热门板块
    sector_bonus_pts: float = 8.0      # 热门板块加分分值
    cross_rps_days: int = 20          # 截面RPS计算周期（默认20日）
    use_cross_rps: bool = True       # 是否启用截面RPS加分（默认开启）

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


def _fetch_tencent_realtime_today(code: str) -> Optional[pd.DataFrame]:
    """从腾讯财经获取今日实时行情，返回兼容 .day 格式的 DataFrame（若市场未收盘或失败返回 None）。"""
    try:
        url = f"https://qt.gtimg.cn/q={code}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            txt = resp.read().decode("gbk", errors="replace")
        # 格式：v_sh600036="1~name~..."，需去掉前缀
        body = txt.split("=", 1)[1].strip('"; \n')
        fields = body.split("~")
        if len(fields) < 38:
            return None
        price = float(fields[3])
        prev_close = float(fields[4])
        open_ = float(fields[5])
        vol_lots = float(fields[6])   # 手（1手=100股）
        amount_wan = float(fields[8])  # 万元
        amount = amount_wan * 1e4       # 元
        # amount = vol_lots * 100 * price（验算用）
        vol = vol_lots * 100          # 股数
        high = float(fields[33])
        low = float(fields[34])
        ts = fields[30]
        today_str = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}"
        # turnover：换手率需要流动股本，腾讯数据无此字段，留空由 AkShare 补充
        df = pd.DataFrame([{
            "date": pd.Timestamp(today_str),
            "open": open_,
            "high": high,
            "low": low,
            "close": price,
            "volume": vol,
            "amount": amount,
            "change_pct": (price - prev_close) / prev_close * 100 if prev_close > 0 else 0.0,
        }])
        return df
    except Exception as e:
        print(f"[DEBUG _fetch_tencent] EXCEPTION: {e}")
        import traceback; traceback.print_exc()
        return None


# ── 腾讯实时批量预取（供 load_qfq_history 调用）──────────────────────────────
_tencent_batch_cache: dict[str, pd.DataFrame] = {}
_tencent_batch_loaded = False


def _prefetch_tencent_realtime(codes: list[str], chunk_size: int = 50) -> None:
    """批量从腾讯获取今日行情（分块请求避免 URL过长），结果存入 _tencent_batch_cache。"""
    global _tencent_batch_cache, _tencent_batch_loaded
    if not codes or _tencent_batch_loaded:
        return
    # 分块请求，每块 50 只股票（URL 约 450 字符，安全）
    for i in range(0, len(codes), chunk_size):
        chunk = codes[i:i + chunk_size]
        try:
            url = "https://qt.gtimg.cn/q=" + ",".join(chunk)
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                txt = resp.read().decode("gbk", errors="replace")
            for line in txt.strip().split("\n"):
                if "=" not in line:
                    continue
                prefix_part = line.split("=")[0]
                code_key = prefix_part.replace("v_", "") if prefix_part.startswith("v_") else None
                if code_key is None or not code_key.startswith(("sz", "sh")):
                    continue
                body = line.split("=", 1)[1].strip('"; \n')
                fields = body.split("~")
                if len(fields) < 38:
                    continue
                try:
                    price = float(fields[3])
                    prev_close = float(fields[4])
                    open_ = float(fields[5])
                    vol_lots = float(fields[6])
                    amount_wan = float(fields[8])
                    amount = amount_wan * 1e4
                    vol = vol_lots * 100
                    high = float(fields[33])
                    low = float(fields[34])
                    ts = fields[30]
                    today_str = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}"
                    _tencent_batch_cache[code_key] = pd.DataFrame([{
                        "date": pd.Timestamp(today_str),
                        "open": open_,
                        "high": high,
                        "low": low,
                        "close": price,
                        "volume": vol,
                        "amount": amount,
                        "change_pct": (price - prev_close) / prev_close * 100 if prev_close > 0 else 0.0,
                    }])
                except Exception:
                    continue
        except Exception:
            continue
    _tencent_batch_loaded = True


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
            # age_hours = (time.time() - path.stat().st_mtime) / 3600.0
            # if age_hours > max_age_hours:
            #     use_cache = False
            # 复盘指定日期时：若缓存已包含目标日期则用缓存，否则联网拉取
            if end_date is not None:
                cached = pd.read_csv(path, parse_dates=["date"]) if path.exists() else pd.DataFrame()
                if not cached.empty and cached["date"].max() >= pd.Timestamp(end_date):
                    use_cache = True
                else:
                    use_cache = False   # 缓存不含目标日期，联网拉取

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
        # ── 腾讯实时数据补充（盘中用）：若目标日期是今天且缓存只有昨天，尝试从腾讯获取今日数据 ──
        today_ts = pd.Timestamp.today().normalize()
        if end_ts == today_ts and not df.empty:
            cache_max = df["date"].max().normalize()
            if cache_max < today_ts:
                # 优先用批量预取缓存，否则回退到单股补获
                rt = _tencent_batch_cache.get(normalize_prefixed(code))
                if rt is None or rt.empty:
                    rt = _fetch_tencent_realtime_today(code)
                if rt is not None and not rt.empty:
                    rt = rt.copy()
                    rt["date"] = pd.to_datetime(rt["date"])
                    if rt.iloc[0]["date"] == today_ts and not df[df["date"] == today_ts].shape[0]:
                        df = pd.concat([df, rt], ignore_index=True)
                        df = df.sort_values("date").reset_index(drop=True)
    else:
        # 实时模式：默认截到缓存已有最新日期
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
    sector_name: Optional[str] = None
    sector_bonus_applied: float = 0.0
    limit_up_bonus: float = 0.0
    rsi_tier: str = "🟢健康"
    cross_rps: float = 50.0          # 截面RPS（0~100，候选股范围内排名）
    cross_rps_bonus: float = 0.0     # 截面RPS加分（0~10）

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
                  stock_sector_map: Optional[dict[str, str]] = None,
                  cross_rps: Optional[dict[str, float]] = None) -> Optional[dict]:
    min_idx = max(config.signal_days + 1, config.quality_days, 60, config.min_history_days)
    if idx < min_idx:
        return None

    signal_gains = prepared.gains[idx - config.signal_days + 1: idx + 1]
    quality_gains = prepared.gains[idx - config.quality_days + 1: idx + 1]
    if np.isnan(signal_gains).any() or np.isnan(quality_gains).any():
        return None

    # 信号窗口软扣分（不硬过滤）
    # 扣分项：末日军缩超标、低于min_gain天数超限、存在超max_gain
    signal_penalty = 0.0
    if config.min_gain <= 0:
        below_min = 0
    else:
        below_min = (signal_gains < config.min_gain).sum()
    max_allowed_below = config.signal_days // 3 if config.signal_days >= 3 else 0
    last_day_ok = (signal_gains[-1] > -3.0) if len(signal_gains) > 0 else True
    if not last_day_ok:
        signal_penalty += 5.0   # 末日军缩>-3%
    if below_min > max_allowed_below:
        signal_penalty += 3.0   # 信号窗口低于min_gain天数超限
    if not np.all(signal_gains <= config.max_gain):
        signal_penalty += 2.0   # 存在涨幅超max_gain

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

    # 强制趋势过滤（仅保留核心约束）
    if not (close > ma5 >= ma10 * 0.995):
        return None
    # 均线排列确认（短期趋势确认）
    if not (ma5 > ma10 and ma10 > prepared.ma20[idx - 1] * 1.0):
        return None

    # ── 简化见顶过滤（改为软扣分，不硬过滤）──────────────────────
    rejected, reject_reason = _simplified_top_filter(prepared, idx)
    if rejected:
        signal_penalty += 5.0   # 见顶风险扣5分

    # 线性评分模型（基础分100，扣分制）
    subscores: Dict[str, float] = {}

    # 基础分 = 100，扣除见顶风险等软扣分
    score = max(round(100.0 - signal_penalty, 2), 0.0)

    subscores["signal_penalty"] = round(signal_penalty, 2)

    if score < config.score_threshold:
        return None

    # Compute stats for details (linear model removed these variables)
    mean_gain = float(np.mean(signal_gains))
    gain_std = float(np.std(signal_gains, ddof=0))
    amount_ratio_5_20_val = float(prepared.avg_amount_5[idx] / avg_amt20) if avg_amt20 > 0 and not np.isnan(prepared.avg_amount_5[idx]) else 1.0

    details = {
        "mean_gain_signal": round(mean_gain, 3),
        "gain_std_signal": round(gain_std, 3),
        "gain10_pct": round(gain10, 2),
        "gain20_pct": round(gain20, 2),
        "avg_amount_20": round(avg_amt20 / 1e8, 2),
        "avg_turnover_5": round(float(avg_to5), 2) if not np.isnan(avg_to5) else None,
        "avg_turnover_20": round(float(avg_to20), 2) if not np.isnan(avg_to20) else None,
        "amount_ratio_5_20": round(amount_ratio_5_20_val, 2),
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
        "sector_name": None,
        "sector_bonus_applied": 0.0,
        "limit_up_bonus": limit_up_bonus,
        "rsi_tier": "🟢健康",
    }

def evaluate_latest_signal(code: str, name: str, df: pd.DataFrame, config: StrategyConfig,
                         fundamental: Optional[FundamentalData] = None,
                         top_sectors: Optional[set[str]] = None,
                         stock_sector_map: Optional[dict[str, str]] = None,
                         cross_rps: Optional[dict[str, float]] = None) -> Optional[SignalResult]:
    prepared = prepare_data(df)
    if prepared is None:
        return None
    # 基本面按需加载（缓存90天）
    if config.check_fundamental and fundamental is None:
        fundamental = load_fundamental_data(code)
    idx = len(prepared.df) - 1
    result = evaluate_signal(prepared, idx, config, code=code, fundamental=fundamental,
                             top_sectors=top_sectors, stock_sector_map=stock_sector_map,
                             cross_rps=cross_rps)
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
        sector_name=result.get("sector_name"),
        sector_bonus_applied=result.get("sector_bonus_applied", 0.0),
        limit_up_bonus=result.get("limit_up_bonus", 0.0),
        rsi_tier=result.get("rsi_tier", "🟢健康"),

        cross_rps=result.get("subscores", {}).get("cross_rps", 50.0),
        cross_rps_bonus=result.get("subscores", {}).get("cross_rps_bonus", 0.0),
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
        # 扣分列
        if r.sector_bonus_applied > 0:
            penalty_str = f"+{int(r.sector_bonus_applied)}({r.sector_name})"
        elif r.limit_up_bonus > 0:
            penalty_str = f"+{int(r.limit_up_bonus)}(涨停)"
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
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 简化见顶过滤（3项核心）（6项见顶信号 + RECOVER 化解机制）
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

def _simplified_top_filter(prepared: PreparedData, idx: int) -> tuple[bool, str]:
    """
    简化见顶过滤（仅2项）：
    - 放量大阴：跌幅>5% 且 量>MA5量×1.5（5日内）
    - MACD高位死叉：DIF在零轴上方下穿DEA（5日内）
    """
    if idx < 5:
        return False, ""
    curr_close = float(prepared.close[idx])
    prev_close = float(prepared.close[idx - 1]) if idx >= 1 else curr_close
    vol_curr = float(prepared.volume[idx])
    vol_ma5 = float(np.nanmean(prepared.volume[max(0, idx - 5):idx])) if idx >= 5 else float(np.nanmean(prepared.volume[:idx]))
    if prev_close > 0:
        drop_pct = (prev_close - curr_close) / prev_close * 100.0
        if drop_pct > 5.0 and vol_curr > vol_ma5 * 1.5:
            return True, "放量大阴"
    if idx >= 22:
        macd_close = prepared.close[idx - 22: idx + 1]
        if len(macd_close) >= 27:
            dif, dea, _ = _compute_macd(macd_close)
            for d in range(1, len(dif)):
                if dif[d] < dea[d] and dif[d-1] >= dea[d-1]:
                    return True, "MACD高位死叉"
    return False, ""


def diagnose_rejection(prepared: PreparedData, idx: int, config: StrategyConfig) -> list[str]:
    """诊断 evaluate_signal 失败原因，返回所有不满足的条件列表。"""
    reasons = []
    min_idx = max(config.signal_days + 1, config.quality_days, 60, config.min_history_days)
    if idx < min_idx:
        reasons.append(f"历史数据不足(idx={idx}<{min_idx})")
        return reasons

    signal_gains = prepared.gains[idx - config.signal_days + 1: idx + 1]
    quality_gains = prepared.gains[idx - config.quality_days + 1: idx + 1]
    if np.isnan(signal_gains).any() or np.isnan(quality_gains).any():
        reasons.append("信号/质量窗口含NaN")

    signal_penalty_d = 0.0
    if config.min_gain > 0:
        below_min = (signal_gains < config.min_gain).sum()
        max_allowed_below = config.signal_days // 3 if config.signal_days >= 3 else 0
        last_day_ok = (signal_gains[-1] > -3.0) if len(signal_gains) > 0 else True
        if not last_day_ok:
            signal_penalty_d += 5.0
            reasons.append(f"信号窗口末日军缩>-3%（扣5分）")
        if below_min > max_allowed_below:
            signal_penalty_d += 3.0
            reasons.append(f"信号窗口低于min_gain天数超限（扣3分）")
        if not np.all(signal_gains <= config.max_gain):
            signal_penalty_d += 2.0
            reasons.append(f"信号窗口存在涨幅>{config.max_gain}%%（扣2分）")
    elif config.min_gain <= 0:
        pass  # 不限制最低涨幅

    avg_amt20 = prepared.avg_amount_20[idx]
    if np.isnan(avg_amt20) or avg_amt20 < config.min_amount:
        reasons.append(f"20日均成交额不足({avg_amt20/1e8:.1f}亿<{config.min_amount/1e8:.1f}亿)")

    avg_to5 = prepared.avg_turnover_5[idx]
    if config.min_turnover > 0 and (np.isnan(avg_to5) or avg_to5 < config.min_turnover):
        reasons.append(f"5日均换手率不足({avg_to5:.2f}%<{config.min_turnover}%)")

    ma5 = prepared.ma5[idx]; ma10 = prepared.ma10[idx]
    ma20 = prepared.ma20[idx]; ma60 = prepared.ma60[idx]
    if np.isnan(ma5) or np.isnan(ma10) or np.isnan(ma20) or np.isnan(ma60):
        reasons.append("均线数据缺失(MA5/MA10/MA20/MA60)")

    close = prepared.close[idx]
    if not (close > ma5 >= ma10 * 0.995):
        reasons.append(f"均线多头排列不符(收盘{close:.2f} ma5{ma5:.2f} ma10{ma10:.2f})")

    rsi = prepared.rsi14[idx] if not np.isnan(prepared.rsi14[idx]) else 50.0
    if rsi >= 82:
        reasons.append(f"RSI={rsi:.1f}≥82超买过滤")

    rejected_top, reason_top = _simplified_top_filter(prepared, idx)
    if rejected_top:
        signal_penalty_d += 5.0
        reasons.append(f"见顶风险({reason_top})（扣5分）")

    # 评分（独立计算，不修改prepared）
    gain10 = float((prepared.close[idx] / prepared.close[idx - 10] - 1.0) * 100.0) if idx >= 10 else 0.0
    ma20_val = float(prepared.ma20[idx])
    ma60_val = float(prepared.ma60[idx]) if not np.isnan(prepared.ma60[idx]) else 0.0
    ma20_above_ma60 = 1.0 if (ma20_val > ma60_val > 0) else 0.0
    ma5_above_ma10 = 0.5 if (ma5 > ma10) else 0.0
    trend = ma20_above_ma60 + ma5_above_ma10
    if idx >= 60:
        low_60 = float(np.nanmin(prepared.low[idx-60:idx]))
        high_60 = float(np.nanmax(prepared.high[idx-60:idx]))
        range_60 = high_60 - low_60
        position = (close - low_60) / range_60 * 100.0 if range_60 > 0 else 50.0
    else:
        position = 50.0
    position_score = max(100.0 - position * 0.5, 0.0)
    W1, W4 = 1.0, 1.0
    score = W1 * trend + W4 * (position_score / 100.0 * 5.0)
    score = max(round(score / (W1 + W4) * 20.0 - signal_penalty_d, 2), 0.0)
    if score < config.score_threshold:
        reasons.append(f"评分不足({score:.1f}<{config.score_threshold})")

    return reasons

    """
    简化见顶过滤（仅3项）：
    - 放量大阴：跌幅>5% 且 量>MA5量×1.5（5日内）
    - MACD高位死叉：DIF在零轴上方下穿DEA（5日内）
    - 长上影线：上影>=实体×2 且 上影>股价×1% 且 振幅>2%（3日内）
    返回 (rejected, reason)。
    """
    if idx < 5:
        return False, ""
    # ── 放量大阴（当日跌幅>5% 且 量>MA5×1.5）────────────────────
    curr_close = float(prepared.close[idx])
    prev_close = float(prepared.close[idx - 1]) if idx >= 1 else curr_close
    vol_curr = float(prepared.volume[idx])
    vol_ma5 = float(np.nanmean(prepared.volume[max(0, idx - 5):idx])) if idx >= 5 else float(np.nanmean(prepared.volume[:idx]))
    if prev_close > 0:
        drop_pct = (prev_close - curr_close) / prev_close * 100.0   # 正数=下跌
        if drop_pct > 5.0 and vol_curr > vol_ma5 * 1.5:
            return True, "放量大阴"

    # ── MACD高位死叉（5日内）──────────────────────────────────
    if idx >= 22:
        macd_close = prepared.close[idx - 22: idx + 1]
        if len(macd_close) >= 27:
            dif, dea, _ = _compute_macd(macd_close)
            for d in range(1, len(dif)):
                if dif[d] < dea[d] and dif[d-1] >= dea[d-1]:
                    return True, "MACD高位死叉"

    # ── 长上影线（3日内）+ 化解：次日阳线实体覆盖 ──────────────
    for d in range(min(3, idx + 1)):
        o = float(prepared.open_[idx - d])
        c = float(prepared.close[idx - d])
        h = float(prepared.high[idx - d])
        lo = float(prepared.low[idx - d])
        today_close = float(prepared.close[idx])
        body = abs(c - o)
        upper_shadow = max(h - max(o, c), 0)
        full_range = max(h - lo, 1e-6)
        if not (body > 0 and upper_shadow >= body * 2.0
                and upper_shadow > today_close * 0.01
                and full_range / today_close * 100.0 > 2.0):
            continue

        # 化解：被检验日(D日，即idx-d)的次日(D+1，即idx-d+1)阳线实体覆盖
        next_offset = d - 1    # D+1相对idx的偏移（d=0→-1=无效；d=1→0=今日；d=2→1=昨日）
        if next_offset < 0 or next_offset >= idx:
            continue  # 今日(D-0)无化解数据，跳过，继续检查昨日(D-1)
        next_o = float(prepared.open_[idx - next_offset])
        next_c = float(prepared.close[idx - next_offset])
        next_body = next_c - next_o
        if next_body > 0 and next_o <= c and next_c >= o:
            continue   # 被化解，不拒绝
        reasons.append(f"见顶风险(长上影)")

    return False, ""


def _compute_cross_sectional_rps(
    prepared_data: dict[str, tuple[PreparedData, int]],
    days: int = 20,
) -> dict[str, float]:
    """
    对候选股票按过去 N 日收益率排名，返回每只股票的截面 RPS（0~100）。
    prepared_data: {code: (prepared_obj, latest_idx)}
    """
    if len(prepared_data) < 2:
        return {code: 50.0 for code in prepared_data}

    # 计算每只股票的 N 日收益率
    returns: dict[str, float] = {}
    for code, (prep, idx) in prepared_data.items():
        if idx < days:
            returns[code] = -999.0
        else:
            curr = float(prep.close[idx])
            past = float(prep.close[idx - days])
            if past > 0:
                returns[code] = (curr / past - 1.0) * 100.0
            else:
                returns[code] = -999.0

    # 排序（升序）：收益最低 → RPS=0，收益最高 → RPS=100
    sorted_codes = sorted(returns.keys(), key=lambda c: returns[c])
    n = len(sorted_codes)
    rps_dict: dict[str, float] = {}
    for rank, code in enumerate(sorted_codes):
        rps_dict[code] = round((rank / max(n - 1, 1)) * 100.0, 2)

    return rps_dict
