#!/usr/bin/env python3
"""
构建日期索引的合并指标文件
================================

将 .cache/indicators/*.json 合并为一个日期索引文件：
{
  "2026-04-23": {
    "000001": {indicator_row},
    "000002": {indicator_row},
    ...
  },
  ...
}

只包含 2024-01-01 之后的数据（约559天 × 5196只 = 290万条）
输出：.cache/indicators_merged.json（预计 1-2GB）

用法：
    python build_merged_index.py  # 一次性构建
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

WORKSPACE = Path(__file__).parent.parent.parent.resolve()
INDICATORS_DIR = WORKSPACE / ".cache" / "indicators"
OUT_PATH = WORKSPACE / ".cache" / "indicators_merged.json"
CUTOFF = "2024-01-01"


def process_stock(fpath: Path) -> tuple[str, dict]:
    """处理单只股票，返回 {date: row} 的子字典"""
    code = fpath.stem.replace("_indicators", "")
    try:
        with open(fpath) as f:
            data = json.load(f)
        index = {}
        for row in data:
            if row["date"] >= CUTOFF:
                index[row["date"]] = row
        return code, index
    except Exception:
        return code, {}


def main():
    print(f"📂 扫描指标文件...", flush=True)
    files = sorted(INDICATORS_DIR.glob("*_indicators.json"))
    print(f"   找到 {len(files)} 只股票", flush=True)

    start = datetime.now()
    merged = {}

    # 分批处理，每批200只股票，逐步写入
    batch_size = 200
    done = 0

    for batch_start in range(0, len(files), batch_size):
        batch = files[batch_start:batch_start + batch_size]
        with ProcessPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(process_stock, f): f for f in batch}
            for future in as_completed(futures):
                code, index = future.result()
                # 合并到主字典
                for date, row in index.items():
                    if date not in merged:
                        merged[date] = {}
                    merged[date][code] = row
                done += 1

        if (done % 500 == 0) or done == len(files):
            elapsed = (datetime.now() - start).total_seconds()
            print(f"  进度: {done}/{len(files)}  已合并{len(merged)}天  耗时{elapsed:.0f}秒", flush=True)

    # 排序日期
    sorted_dates = sorted(merged.keys())
    sorted_merged = {d: merged[d] for d in sorted_dates}

    print(f"\n💾 写入 {OUT_PATH}（{len(sorted_dates)}天 × 约{len(merged.get(sorted_dates[-1], {}))}只）...", flush=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(sorted_merged, f, ensure_ascii=False)

    size_gb = OUT_PATH.stat().st_size / 1024**3
    elapsed = (datetime.now() - start).total_seconds()
    print(f"✅ 完成: {len(sorted_dates)}天  大小{size_gb:.2f}GB  耗时{elapsed:.0f}秒", flush=True)


if __name__ == "__main__":
    main()
