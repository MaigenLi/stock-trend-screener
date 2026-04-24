#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
缓存模块（完全兼容原 gain_turnover）
"""

import pickle
from pathlib import Path

# ⚠️ 保持原路径
WORKSPACE = Path(__file__).parent.parent.resolve()
CACHE_DIR = WORKSPACE / ".cache" / "qfq_daily"
CACHE_ROOT = Path("../../.cache")


def _get_cache_path(namespace: str, key: str):
    d = CACHE_ROOT / namespace
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{key}.pkl"


def load_cache(namespace: str, key: str):
    path = _get_cache_path(namespace, key)

    if not path.exists():
        return None

    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def save_cache(namespace: str, key: str, data):
    path = _get_cache_path(namespace, key)

    with open(path, "wb") as f:
        pickle.dump(data, f)
