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

WORKSPACE = Path.home() / ".openclaw/workspace"

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

    raise FileNotFoundError(f"找不到数据文件: {code}")


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


def get_all_tdx_codes() -> list[str]:
    """
    扫描上海和深圳数据目录，返回所有股票代码列表（带前缀）。

    规则:
      sh60xxxx  → 上海主板（固定 sh60 前缀，排除 ETF/指数）
      sh688xxx  → 科创板
      sz00xxxx  → 深圳主板
      sz30xxxx  → 创业板

    Returns
    -------
    list[str]
        按市场+代码排序的完整列表
    """
    codes: list[str] = []
    sh_dir = Path("/mnt/d/new_tdx/vipdoc/sh/lday")
    sz_dir = Path("/mnt/d/new_tdx/vipdoc/sz/lday")

    if sh_dir.is_dir():
        for fp in sorted(sh_dir.glob("sh60????.day")):
            codes.append(fp.stem)
        for fp in sorted(sh_dir.glob("sh688???.day")):
            codes.append(fp.stem)

    if sz_dir.is_dir():
        for fp in sorted(sz_dir.glob("sz00????.day")):
            codes.append(fp.stem)
        for fp in sorted(sz_dir.glob("sz30????.day")):
            codes.append(fp.stem)

    return codes


# ═══════════════════════════════════════════════════════════════
#  股票信息 CSV 维护（名称 + 流通股本）
# ═══════════════════════════════════════════════════════════════

def update_stock_info_csv(csv_path: Optional[Path] = None,
                          refresh: bool = False,
                          batch_size: int = 50,
                          progress: bool = True) -> dict:
    """
    读取/创建 stock_info.csv，维护 code / name / outstanding_share 三列。

    对比规则（任一不等则更新该行）：
      - 无该代码 → 追加新行
      - 有该代码但 name 或 outstanding_share 不同 → 修改该行

    Parameters
    ----------
    csv_path : Path, optional
        CSV 文件路径（默认 WORKSPACE/.cache/stock_info.csv）
    refresh : bool
        True 时跳过 AkShare 网络请求，复用 CSV 中已有数据做对比
        （用于快速检查 CSV 是否已包含所有代码）
    batch_size : int
        每批联网获取的股票数量（AkShare 接口限制）
    progress : bool
        打印进度

    Returns
    -------
    dict {code: {"name": str, "outstanding_share": float, "updated": bool}}
    """
    import csv as csvlib
    import akshare as ak
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if csv_path is None:
        csv_path = WORKSPACE / ".cache" / "stock_info.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # ── 1. 读取已有 CSV ─────────────────────────────────
    existing: dict[str, dict] = {}
    if csv_path.exists():
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csvlib.DictReader(f)
            for row in reader:
                code = row.get("code", "").strip()
                if not code:
                    continue
                try:
                    existing[code] = {
                        "name": row.get("name", "").strip(),
                        "outstanding_share": float(row.get("outstanding_share", 0)),
                    }
                except (ValueError, TypeError):
                    continue

    if progress:
        print(f"  已有记录: {len(existing)} 条")

    # ── 2. 获取全市场代码 ───────────────────────────────
    all_codes = get_all_tdx_codes()
    total = len(all_codes)
    if progress:
        print(f"  TDX 代码总数: {total} 条")

    # ── 3. 从 AkShare 获取名称 ─────────────────────────
    name_map: dict[str, str] = {}
    if not refresh:
        if progress:
            print("  联网获取股票名称...")
        try:
            # 上海主板
            df_sh = ak.stock_info_sh_name_code()
            for _, row in df_sh.iterrows():
                code = str(row["证券代码"]).zfill(6)
                name_map[f"sh{code}"] = str(row["证券简称"]).strip()
            # 深圳
            df_sz = ak.stock_info_sz_name_code()
            for _, row in df_sz.iterrows():
                code = str(row["A股代码"]).zfill(6)
                name_map[f"sz{code}"] = str(row["A股简称"]).strip()
            # 科创板
            df_kc = ak.stock_zh_kcb_spot()
            for _, row in df_kc.iterrows():
                raw = str(row["代码"]).replace("sh", "")
                name_map[f"sh{raw}"] = str(row["名称"]).strip()
            if progress:
                print(f"  名称获取完成: {len(name_map)} 条")
        except Exception as e:
            print(f"  ⚠️  名称获取失败: {e}，复用 CSV 已有名称")
            for code, info in existing.items():
                if code not in name_map:
                    name_map[code] = info["name"]

    # ── 4. 从 AkShare 获取 outstanding_share ────────────
    outstanding_map: dict[str, float] = {}
    failed: list[str] = []

    def _fetch_outstanding(code: str) -> tuple[str, float | None]:
        try:
            df = ak.stock_zh_a_daily(symbol=code, adjust="qfq")
            if not df.empty and "outstanding_share" in df.columns:
                val = float(df["outstanding_share"].iloc[-1])
                return (code, round(val, 1))
        except Exception:
            pass
        return (code, None)

    def _do_fetch(codes_batch: list[str]) -> None:
        nonlocal outstanding_map, failed
        for i in range(0, len(codes_batch), batch_size):
            batch = codes_batch[i:i + batch_size]
            with ThreadPoolExecutor(max_workers=min(8, len(batch))) as pool:
                futures = {pool.submit(_fetch_outstanding, c): c for c in batch}
                for fut in as_completed(futures):
                    code, val = fut.result()
                    if val is not None:
                        outstanding_map[code] = val
                    else:
                        failed.append(code)
            if progress and (i + batch_size) % 500 == 0:
                print(f"    进度 {i + batch_size}/{len(codes_batch)} ...")

    if refresh:
        # refresh=True: 全量重新获取所有股票
        if progress:
            print(f"  [refresh] 全量获取 outstanding_share ({len(all_codes)} 只)...")
        _do_fetch(list(all_codes))
        if progress:
            print(f"  [refresh] 完成: {len(outstanding_map)} 只成功, {len(failed)} 只失败")
    else:
        # 非 refresh: 获取 CSV 中 outstanding=0 的缺失项，以及非零但可能异常的值
        # 获取 CSV 中从未有过的股票（outstanding 列为空/0）
        codes_need_fetch = [c for c in all_codes
                           if existing.get(c, {}).get("outstanding_share", 0) == 0]
        if progress:
            print(f"  联网获取流通股本（outstanding=0 补获取，{len(codes_need_fetch)} 只）...")
        if codes_need_fetch:
            _do_fetch(codes_need_fetch)
            if progress:
                print(f"  outstanding 获取完成: {len(outstanding_map)} 只成功, {len(failed)} 只失败")
                if failed[:5]:
                    print(f"  失败样例: {failed[:5]}")

    # ── 5. 合并结果 ────────────────────────────────────
    results: dict[str, dict] = {}
    updated_rows: list[dict] = []
    new_count = 0
    upd_count = 0
    same_count = 0

    for code in all_codes:
        name = name_map.get(code, existing.get(code, {}).get("name", "未知"))
        outstanding = outstanding_map.get(
            code, existing.get(code, {}).get("outstanding_share", 0.0))
        if outstanding:
            outstanding = round(outstanding, 1)
        old = existing.get(code)

        if old is None:
            results[code] = {"name": name, "outstanding_share": outstanding, "updated": True}
            updated_rows.append({"code": code, "name": name, "outstanding_share": outstanding})
            new_count += 1
        elif old["name"] != name or old["outstanding_share"] != outstanding:
            results[code] = {"name": name, "outstanding_share": outstanding, "updated": True}
            updated_rows.append({"code": code, "name": name, "outstanding_share": outstanding})
            upd_count += 1
        else:
            results[code] = {"name": old["name"], "outstanding_share": old["outstanding_share"],
                              "updated": False}
            updated_rows.append({"code": code, "name": old["name"],
                                  "outstanding_share": old["outstanding_share"]})
            same_count += 1

    # ── 6. 写回 CSV ────────────────────────────────────
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csvlib.DictWriter(f, fieldnames=["code", "name", "outstanding_share"])
        writer.writeheader()
        writer.writerows(updated_rows)

    if progress:
        print(f"\n  ✅ CSV 写入完成: {csv_path}")
        print(f"     总计 {len(updated_rows)} 条 | 新增 {new_count} | 更新 {upd_count} | 无变化 {same_count}")

    return results


def load_stock_info_csv(csv_path: Optional[Path] = None) -> dict[str, dict]:
    """读取 stock_info.csv，返回 {code: {name, outstanding_share}}"""
    if csv_path is None:
        csv_path = WORKSPACE / ".cache" / "stock_info.csv"
    result: dict[str, dict] = {}
    if not csv_path.exists():
        return result
    import csv as csvlib
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csvlib.DictReader(f)
        for row in reader:
            code = row.get("code", "").strip()
            if not code:
                continue
            try:
                result[code] = {
                    "name": row.get("name", "").strip(),
                    "outstanding_share": float(row.get("outstanding_share", 0)),
                }
            except (ValueError, TypeError):
                continue
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="通达信日K线查看器")
    parser.add_argument("code", nargs="?", default="sh600862", help="股票代码（默认 sh600862）")
    parser.add_argument("--days", "-d", type=int, default=10, help="天数（默认 10）")
    parser.add_argument("--all", "-a", action="store_true", help="全市场扫描：列出所有股票最近N天K线")
    parser.add_argument("--limit", "-l", type=int, default=0, help="配合 --all 使用，最多显示多少只（0=不限）")
    parser.add_argument("--output", "-o", type=str, default=None, help="输出文件路径")
    parser.add_argument("--update-info", "-u", action="store_true", help="更新 stock_info.csv（名称+流通股本）")
    parser.add_argument("--refresh", "-r", action="store_true", help="配合 --update-info，强制联网重新获取所有 outstanding_share")
    args = parser.parse_args()

    t0 = time_module.time()

    if args.update_info:
        update_stock_info_csv(refresh=args.refresh)
    elif args.all:
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