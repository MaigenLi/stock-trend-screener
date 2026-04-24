#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
缓存系统（带简单文件锁）
"""

import os
import pickle
import time
from pathlib import Path


CACHE_ROOT = Path(".cache")
CACHE_ROOT.mkdir(exist_ok=True)


def _get_path(namespace: str, key: str):
    d = CACHE_ROOT / namespace
    d.mkdir(exist_ok=True)
    return d / f"{key}.pkl"


def load_cache(namespace: str, key: str, max_age: int = None):
    path = _get_path(namespace, key)

    if not path.exists():
        return None

    if max_age:
        if time.time() - path.stat().st_mtime > max_age:
            return None

    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def save_cache(namespace: str, key: str, data):
    path = _get_path(namespace, key)

    tmp_path = str(path) + ".tmp"

    with open(tmp_path, "wb") as f:
        pickle.dump(data, f)

    os.replace(tmp_path, path)  # 原子替换（防止写坏）


# =========================
# 简单文件锁（防并发写）
# =========================

def acquire_lock(lock_path: Path, timeout=10):
    start = time.time()

    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
            return fd
        except FileExistsError:
            if time.time() - start > timeout:
                raise TimeoutError("lock timeout")
            time.sleep(0.1)


def release_lock(fd, lock_path: Path):
    os.close(fd)
    os.remove(lock_path)
