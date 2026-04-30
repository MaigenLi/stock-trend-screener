#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
refresh_sector_cache.py

刷新 screen_strat.py 使用的两个缓存文件：
1. .cache/sector/sector_hotspot.json      板块名 -> 当日涨跌幅
2. .cache/sector/stock_sector_map.json    股票代码(6位) -> 板块名

默认：只刷新 sector_hotspot.json
可选：加 --refresh-map 时，重建 stock_sector_map.json

用法：
  ~/.venv/bin/python stock_trend/refresh_sector_cache.py
  ~/.venv/bin/python stock_trend/refresh_sector_cache.py --refresh-map
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

import akshare as ak

WORKSPACE = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(WORKSPACE))

from stock_trend import sector_hotspot as sh

SECTOR_CACHE_DIR = WORKSPACE / ".cache" / "sector"
SECTOR_CACHE_DIR.mkdir(parents=True, exist_ok=True)

SECTOR_HOTSPOT_FILE = SECTOR_CACHE_DIR / "sector_hotspot.json"
STOCK_SECTOR_MAP_FILE = SECTOR_CACHE_DIR / "stock_sector_map.json"


def refresh_sector_hotspot() -> dict[str, float]:
    """刷新板块热点缓存，返回 板块名 -> 涨跌幅。"""
    df = sh.get_sector_spot(force_refresh=True)
    if df is None or df.empty:
        raise RuntimeError("板块热点为空，无法刷新 sector_hotspot.json")

    hotspot = {
        str(row["name"]).strip(): float(row["change_pct"])
        for _, row in df.iterrows()
    }
    SECTOR_HOTSPOT_FILE.write_text(
        json.dumps(hotspot, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    try:
        sh._sector_spot_cache = df.copy()
        sh._sector_spot_ts = time.time()
        sh.save_to_disk()
    except Exception:
        pass

    return hotspot


def rebuild_stock_sector_map() -> dict[str, str]:
    """重建股票 -> 板块映射缓存。"""
    df = sh.get_sector_spot(force_refresh=True)
    if df is None or df.empty:
        raise RuntimeError("板块列表为空，无法重建 stock_sector_map.json")

    result: dict[str, str] = {}
    sectors = df[["label", "name"]].drop_duplicates().to_dict("records")

    for i, row in enumerate(sectors, 1):
        label = str(row["label"]).strip()
        name = str(row["name"]).strip()
        try:
            buf = io.StringIO()
            with contextlib.redirect_stderr(buf):
                detail = ak.stock_sector_detail(sector=label)
            if detail is None or detail.empty:
                continue

            code_col = None
            for col in ["股票代码", "code", "代码"]:
                if col in detail.columns:
                    code_col = col
                    break
            if code_col is None:
                continue

            for code in detail[code_col].dropna():
                c = str(code).strip().upper()
                if len(c) == 6 and c.isdigit():
                    result[c] = name
        except Exception:
            continue

        if i % 20 == 0 or i == len(sectors):
            print(f"   映射进度: {i}/{len(sectors)} -> {len(result)}只")

    STOCK_SECTOR_MAP_FILE.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    try:
        sh._stock_sector_map = dict(result)
        sh._stock_sector_map_ts = time.time()
        sh.save_to_disk()
    except Exception:
        pass

    return result


def main():
    parser = argparse.ArgumentParser(
        description="刷新 screen_strat.py 使用的板块热点缓存和股票-板块映射缓存"
    )
    parser.add_argument(
        "--refresh-map",
        action="store_true",
        help="同时重建 stock_sector_map.json（较慢，不建议每天执行）",
    )
    args = parser.parse_args()

    print("1) 刷新板块热点缓存...")
    hotspot = refresh_sector_hotspot()
    print(f"   已写入: {SECTOR_HOTSPOT_FILE} ({len(hotspot)}个板块)")

    if args.refresh_map:
        print("2) 重建股票-板块映射缓存...")
        mapping = rebuild_stock_sector_map()
        print(f"   已写入: {STOCK_SECTOR_MAP_FILE} ({len(mapping)}只)")
    else:
        print("2) 跳过股票-板块映射缓存（未指定 --refresh-map）")

    print("完成")


if __name__ == "__main__":
    main()
