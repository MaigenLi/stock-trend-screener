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
    """
    根据代码找到对应市场的 .day 文件。
    当用户明确指定了 sh/sz 前缀但文件实际在另一个市场目录时，
    自动回退查找（处理通达信目录里 sh/sz 前缀与数字段不严格对应的情况）。
    """
    market, pure = _normalize_code(code)

    # 先查对应市场目录
    if market == "sh":
        dirs = [TDX_DATA_SH_DIR, TDX_DATA_SZ_DIR]
    else:
        dirs = [TDX_DATA_SZ_DIR, TDX_DATA_SH_DIR]

    for day_dir in dirs:
        fp = day_dir / f"{market}{pure}.day"
        if fp.exists():
            return fp

    # 回退：前缀正确但文件在另一个市场
    other = "sz" if market == "sh" else "sh"
    fp = (TDX_DATA_SZ_DIR if other == "sz" else TDX_DATA_SH_DIR) / f"{other}{pure}.day"
    if fp.exists():
        return fp

    raise FileNotFoundError(f"找不到数据文件: {code} (尝试过 sh/{market}{pure}.day, sz/{other}{pure}.day)")


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
    print(f"{'日期':<8} {'开盘':>8} {'最高':>6} {'最低':>6} {'收盘':>6} {'成交量(手)':>12} {'成交额':>6}")
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


def get_all_tdx_codes() -> list[str]:
    """
    扫描上海和深圳数据目录，返回所有股票代码列表（带前缀）。

    规则:
      sh60xxxx  → 上海主板
      sh688xxx  → 科创板
      sz00xxxx  → 深圳主板
      sz30xxxx  → 创业板

    Returns
    -------
    list[str]
        按市场+代码排序的完整列表，例: ['sh600000', 'sh600004', ..., 'sz000001', ...]
    """
    codes: list[str] = []

    sh_dir = Path("/mnt/d/new_tdx/vipdoc/sh/lday")
    sz_dir = Path("/mnt/d/new_tdx/vipdoc/sz/lday")

    if sh_dir.is_dir():
        # sh60xxxx: 上海主板（固定60前缀，5位数字）
        for fp in sorted(sh_dir.glob("sh60????.day")):
            codes.append(fp.stem)          # 'sh600000'
        for fp in sorted(sh_dir.glob("sh688???.day")):
            codes.append(fp.stem)          # 'sh688xxx' 科创板

    if sz_dir.is_dir():
        for fp in sorted(sz_dir.glob("sz00????.day")):
            codes.append(fp.stem)          # 'sz000xxx'
        for fp in sorted(sz_dir.glob("sz30????.day")):
            codes.append(fp.stem)          # 'sz300xxx'

    return codes


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="通达信日K线查看器")
    parser.add_argument("code", nargs="?", default="sh600862", help="股票代码（默认 sh600862）")
    parser.add_argument("--days", "-d", type=int, default=10, help="天数（默认 10）")
    parser.add_argument("--all", "-a", action="store_true", help="全市场扫描：列出所有股票最近N天K线")
    parser.add_argument("--limit", "-l", type=int, default=0, help="配合 --all 使用，最多显示多少只（0=不限）")
    parser.add_argument("--output", "-o", type=str, default=None, help="输出文件路径（默认打印到终端）")
    args = parser.parse_args()

    t0 = time_module.time()

    if args.all:
        all_codes = get_all_tdx_codes()
        total = len(all_codes)
        limit = args.limit if args.limit > 0 else total
        codes_to_show = all_codes[:limit]

        import sys
        out = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout
        sep = "=" * 100

        out.write(f"\n{'='*100}\n")
        out.write(f"  全市场扫描  |  共 {total} 只  |  显示前 {limit} 只  |  每天 {args.days} 根K线\n")
        out.write(f"{'='*100}\n\n")

        done = 0
        for code in codes_to_show:
            try:
                data = read_tdx_kline(code, days=args.days)
                if not data:
                    continue
                last = data[-1]
                out.write(f"{sep}\n")
                out.write(f"  [{done+1:>4}/{limit}] {code}  最近 {len(data)} 天  "
                          f"| 最新: {last['date']}  收 {last['close']:>8.2f}  "
                          f"| 高 {last['high']:>8.2f}  低 {last['low']:>8.2f}  "
                          f"| 成交额 {last['amount']/1e8:>10.2f} 亿\n")
                out.write(f"{'-'*100}\n")
                out.write(f"  {'日期':<12} {'开盘':>8} {'最高':>8} {'最低':>8} {'收盘':>8} {'成交量(手)':>12} {'成交额':>14}\n")
                out.write(f"{'-'*100}\n")
                for r in data:
                    out.write(f"  {r['date']:<12} {r['open']:>8.2f} {r['high']:>8.2f} "
                              f"{r['low']:>8.2f} {r['close']:>8.2f} "
                              f"{r['volume']:>12.0f} {r['amount']:>14.2f}\n")
                out.write(f"\n")
            except Exception:
                pass
            done += 1
            if done % 500 == 0:
                sys.stderr.write(f"  已处理 {done}/{limit} 只 ...\n")
                sys.stderr.flush()

        if out is not sys.stdout:
            out.close()
            print(f"✅ 已保存到 {args.output}")
    else:
        data = print_kline(args.code, days=args.days)
        if not data:
            print("⚠️  未获取到数据")

    print(f"\n⏱️  耗时: {time_module.time() - t0:.3f}s")