#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
公共工具函数（供各模块共享）
"""

# Wave dataclass 属性别名映射（dict key → dataclass attribute）
_WAVE_KEY_MAP = {
    "start_idx": "start",   # dict: start_idx → Wave.start
    "end_idx": "end",       # dict: end_idx   → Wave.end
    "len": "days",          # dict: len        → Wave.days
    "price_change": "pct",  # dict: price_change → Wave.pct
}


def w_get(obj, key: str):
    """
    统一访问 dict 和 Wave/dataclass 对象的字段。

    - dict：直接用 obj[key] 取值（KeyError 表示 key 真的不存在）
    - dataclass/object：先按 key 直接取，再按别名映射表取

    Wave dataclass 字段别名（兼容旧代码）：
      start_idx → start, end_idx → end, len → days, price_change → pct

    Returns:
        字段值，key 不存在时返回 None（不会抛异常）
    """
    if isinstance(obj, dict):
        try:
            return obj[key]
        except KeyError:
            return None
    # 先按 key 直接取
    val = getattr(obj, key, None)
    if val is not None:
        return val
    # 再按别名映射表取
    mapped_key = _WAVE_KEY_MAP.get(key)
    if mapped_key is not None:
        return getattr(obj, mapped_key, None)
    return None


def find_ascending_start(ups: list, default: int | None = 0) -> int | None:
    """
    扫描找到第一个连续递增三联 ups[i] < ups[i+1] < ups[i+2]
    返回 u1 的索引 i，之前的波段全部丢弃不参与评分。

    Args:
        ups: 按时间顺序排列的上涨波段列表
        default: 找不到时返回的默认值。
                 - 传 0（默认）：找不到则从第一段开始评分（宽容，scorer.py 用）
                 - 传 None：找不到则返回 None（严格，filter_rules.py 用）

    Returns:
        第一个 u1 的索引，或 default（找不到时）
    """
    for i in range(len(ups) - 2):
        h0 = w_get(ups[i], "wave_high")
        h1 = w_get(ups[i + 1], "wave_high")
        h2 = w_get(ups[i + 2], "wave_high")
        if h0 is not None and h1 is not None and h2 is not None and h0 < h1 < h2:
            return i
    return default
