#!/usr/bin/env python3
"""
补录未映射股票的行业分类（方案B）
用法: python3 replenish_sector_map.py

原理:
  1. 读取当前 stock_sector_map.json（2953只）
  2. 读取 stock_info_a_code_name（全市场5511只）
  3. 对未映射的股票逐只调用腾讯 stock_individual_info_em
  4. 写入 stock_sector_map_enriched.json（合并结果）

预计耗时: ~7分钟（可随时 Ctrl+C 中断，已映射的不会重复请求）
预计新增覆盖: 视腾讯接口返回情况，约能补几十到几百只
"""

import json, time, akshare as ak, pandas as pd
from pathlib import Path

CACHE_DIR = Path.home() / ".openclaw/workspace/.cache/sector"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── 读取已有映射 ──────────────────────────────────────────
existing = {}
emap_path = CACHE_DIR / "stock_sector_map.json"
if emap_path.exists():
    existing = json.loads(emap_path.read_text())
print(f"已有映射: {len(existing)} 只")

# ── 读取全市场股票 ─────────────────────────────────────────
all_stocks = ak.stock_info_a_code_name()
all_codes = all_stocks["code"].tolist()
print(f"全市场股票: {len(all_codes)} 只")

# ── 找出未映射的 ──────────────────────────────────────────
unmapped = [c for c in all_codes if c not in existing]
print(f"未映射股票: {len(unmapped)} 只，开始补录...\n")

# ── 逐只查询行业 ──────────────────────────────────────────
new_map = {}
failed = []
done = 0

for i, code in enumerate(unmapped, 1):
    try:
        info = ak.stock_individual_info_em(symbol=code)
        row = info[info["item"] == "行业"]
        industry = row["value"].values[0] if not row.empty else None
        if industry:
            new_map[code] = industry
        done += 1
    except Exception:
        failed.append(code)

    # 每50只打印进度
    if i % 50 == 0:
        print(f"  进度 {i}/{len(unmapped)}  本轮新增 {len(new_map)}  失败 {len(failed)}")

print(f"\n补录完成: 新增 {len(new_map)} 只行业映射")

# ── 合并并写入 ──────────────────────────────────────────
enriched = {**existing, **new_map}
out_path = CACHE_DIR / "stock_sector_map_enriched.json"
out_path.write_text(json.dumps(enriched, ensure_ascii=False, indent=2))
print(f"合并写入: {out_path}  (共 {len(enriched)} 只)")

# ── 摘要 ──────────────────────────────────────────
mapped_new = len(enriched)
pct = mapped_new / len(all_codes) * 100
print(f"\n覆盖率: {mapped_new}/{len(all_codes)} = {pct:.1f}%")
if failed:
    print(f"查询失败: {len(failed)} 只（前10: {failed[:10]}）")

# ── 如果有新增，写入正式文件 ──────────────────────────────
if new_map:
    confirm = input(f"\n有 {len(new_map)} 只新增行业映射，输入 'yes' 写入正式文件 stock_sector_map.json: ")
    if confirm.strip().lower() == "yes":
        emap_path.write_text(json.dumps(enriched, ensure_ascii=False, indent=2))
        print(f"已写入正式文件: {emap_path}")
