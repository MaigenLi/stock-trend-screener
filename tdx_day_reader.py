"""
通达信 .day 文件解析接口
=========================
从本地离线目录读取股票日K线数据

数据目录:
  上海: /mnt/d/new_tdx/vipdoc/sh/lday/  (sh60xxxx, sh688xxx)
  深圳: /mnt/d/new_tdx/vipdoc/sz/lday/  (sz00xxxx, sz30xxxx)
"""

import struct
import datetime
import time as time_module
from pathlib import Path
from typing import Literal, Optional, Dict

import numpy as np

# ── 数据目录 ──────────────────────────────────────────────
TDX_DATA_SH_DIR = Path("/mnt/d/new_tdx/vipdoc/sh/lday")
TDX_DATA_SZ_DIR = Path("/mnt/d/new_tdx/vipdoc/sz/lday")

# 通达信 .day 文件格式（每记录 32 字节，小端序）:
#   0-3:   date    (uint32, YYYYMMDD)
#   4-7:   open    (uint32, 价格×100)
#   8-11:  high    (uint32, 价格×100)
#   12-15: low     (uint32, 价格×100)
#   16-19: close   (uint32, 价格×100)
#   20-23: amount  (float,  IEEE-754 单精度，成交额，元)
#   24-27: volume  (uint32, 成交量×100，即股数)
#   28-31: reserved (uint32, 保留)
RECORD_SIZE = 32


# ── 工具函数 ──────────────────────────────────────────────

def _normalize_code(code: str) -> tuple[str, str]:
    """解析代码，返回 (市场前缀, 纯代码)"""
    code = code.strip().lower()
    if code.startswith("sh"):
        return ("sh", code[2:])
    elif code.startswith("sz"):
        return ("sz", code[2:])
    if len(code) == 6:
        first = code[0]
        if first in ("0", "3"):
            return ("sz", code)
        elif first in ("6", "8"):
            return ("sh", code)   # 6=沪主版 8=沪科创板
    raise ValueError(f"无法识别的股票代码: {code!r}")


def _find_file(code: str) -> Path:
    """根据代码找到对应市场的 .day 文件"""
    market, pure = _normalize_code(code)
    if market == "sh":
        day_dir = TDX_DATA_SH_DIR
    else:
        day_dir = TDX_DATA_SZ_DIR
    fp = day_dir / f"{market}{pure}.day"
    if not fp.exists():
        raise FileNotFoundError(f"找不到数据文件: {fp}")
    return fp


def _parse_record(chunk: bytes) -> dict:
    """解析单条 32 字节记录"""
    date, open_, high, low, close, amount, volume, _ = struct.unpack('<IIIIIfII', chunk)
    return {
        'date':   datetime.datetime.strptime(str(date), '%Y%m%d'),
        'open':   open_ / 100.0,
        'high':   high / 100.0,
        'low':    low / 100.0,
        'close':  close / 100.0,
        'amount': float(amount),   # IEEE-754 float，成交额（元）
        'volume': volume / 100.0,   # 原始为股数
    }


def _load_all_records(file_path: Path) -> list:
    """从文件加载全部记录（按日期升序）"""
    with open(file_path, 'rb') as f:
        data = f.read()
    records = []
    for i in range(0, len(data), RECORD_SIZE):
        chunk = data[i:i + RECORD_SIZE]
        if len(chunk) < RECORD_SIZE:
            break
        records.append(_parse_record(chunk))
    return records


def _filter_by_date(records: list, end_date, days: Optional[int]) -> list:
    """按截止日期和数量过滤记录"""
    if end_date is None or end_date == "today":
        end_dt = datetime.datetime.now()
    elif isinstance(end_date, str):
        end_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    elif isinstance(end_date, datetime.date):
        end_dt = datetime.datetime.combine(end_date, datetime.time())
    else:
        raise TypeError(f"end_date 类型不支持: {type(end_date)}")

    filtered = [r for r in records if r['date'] <= end_dt]

    if days is not None:
        filtered = filtered[-days:]

    result = []
    for r in filtered:
        result.append({
            'date':   r['date'].strftime('%Y-%m-%d'),
            'open':   round(r['open'], 2),
            'high':   round(r['high'], 2),
            'low':    round(r['low'], 2),
            'close':  round(r['close'], 2),
            'volume': round(r['volume'], 0),
            'amount': round(r['amount'], 2),
        })
    return result


# ── 公开 API ──────────────────────────────────────────────

def read_tdx_kline(
    code: str,
    days: Optional[int] = None,
    end_date: Optional[Literal["today"] | str | datetime.date] = None,
) -> list[dict]:
    """
    读取通达信日K线数据

    Parameters
    ----------
    code : str
        股票代码，支持纯代码 "600036" / 带前缀 "sh600036" / "sz000001"
    days : int, optional
        获取最近多少天（从 end_date 往前算），不指定则返回全部
    end_date : str or date, optional
        截止日期，默认为今天

    Returns
    -------
    list[dict]
        每条: date, open, high, low, close, volume, amount（升序）
    """
    fp = _find_file(code)
    all_records = _load_all_records(fp)
    return _filter_by_date(all_records, end_date, days)


def print_kline(code: str, days: int = 10,
                end_date: Optional[str] = None) -> list[dict]:
    """打印 K 线数据的便捷函数"""
    data = read_tdx_kline(code, days=days, end_date=end_date or "today")
    if not data:
        print(f"未找到数据: {code}")
        return []

    print(f"\n{'='*80}")
    print(f"股票代码: {code}  |  最近 {len(data)} 天  |  截止: {data[-1]['date']}")
    print(f"{'='*80}")
    print(f"{'日期':<12} {'开盘':>8} {'最高':>8} {'最低':>8} {'收盘':>8} {'成交量(手)':>12} {'成交额':>14}")
    print("-" * 80)
    for r in data:
        print(f"{r['date']:<12} {r['open']:>8.2f} {r['high']:>8.2f} {r['low']:>8.2f} "
              f"{r['close']:>8.2f} {r['volume']:>12.0f} {r['amount']:>14.2f}")
    print("-" * 80)
    return data


def preload_all_klines(
    codes: list,
    days: int = 80,
    workers: int = 30,
    progress: bool = True,
) -> Dict[str, list]:
    """
    一次性预加载多只股票的 K 线数据。
    返回 {code: [records...]}，失败返回空列表。
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _load_one(code: str) -> tuple[str, list]:
        try:
            return (code, read_tdx_kline(code, days=days))
        except Exception:
            return (code, [])

    result = {}
    total = len(codes)
    done = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_load_one, c): c for c in codes}
        for fut in as_completed(futures):
            code, recs = fut.result()
            result[code] = recs
            done += 1
            if progress and done % 500 == 0:
                print(f"  预加载进度: {done}/{total}", flush=True)
    return result


if __name__ == "__main__":
    import sys
    code = sys.argv[1] if len(sys.argv) > 1 else "sh600862"
    days = int(sys.argv[2]) if len(sys.argv) > 2 else 10

    t0 = time_module.time()
    data = print_kline(code, days=days)
    print(f"\n⏱️  耗时: {time_module.time() - t0:.3f}s")
    if not data:
        print("⚠️  未获取到数据")