"""
sector_hotspot.py — 市场/板块热点实时数据

数据源：
  1. 新浪行业板块实时（stock_sector_spot）— 主力接口，实时
  2. 东财个股人气榜（stock_hot_rank_em）— 补充接口

缓存策略：
  sector_spot  每 5 分钟刷新（盘中实时）
  stock_hot    每 10 分钟刷新
"""

import time
import json
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import akshare as ak

CACHE_DIR = Path.home() / ".openclaw/workspace/.cache/sector_hotspot"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────
# 全局缓存（进程内共享，有锁保护）
# ─────────────────────────────────────────
_lock = threading.RLock()
_sector_spot_cache: Optional[pd.DataFrame] = None
_sector_spot_ts: float = 0.0          # 上次拉取时间戳
_hot_rank_cache: Optional[pd.DataFrame] = None
_stock_sector_map: Optional[dict[str, str]] = None   # code → label
_stock_sector_map_ts: float = 0.0
_hot_rank_ts: float = 0.0

SECTOR_CACHE_TTL  = 300   # 5 分钟
HOT_CACHE_TTL     = 600   # 10 分钟
SECTOR_MAP_TTL   = 3600   # 个股→板块映射缓存 1 小时

# 空缓存默认骨架（确保列结构正确，避免 KeyError）
_EMPTY_SECTOR_DF = pd.DataFrame(columns=[
    "label","name","count","change_pct","change_amt","total_volume","total_amount"])
_EMPTY_HOT_DF = pd.DataFrame(columns=["rank","code","name","price","change_amt","change_pct"])


# ═══════════════════════════════════════════════════════════
# 内部：低级拉取接口
# ═══════════════════════════════════════════════════════════

def _fetch_sector_spot() -> pd.DataFrame:
    """从新浪拉行业板块实时数据（3次重试）。"""
    last_err = None
    for attempt in range(3):
        try:
            return _fetch_sector_spot_once()
        except Exception as e:
            last_err = e
            if attempt < 2:
                time.sleep(1.0 * (attempt + 1))
    raise last_err


def _fetch_sector_spot_once() -> pd.DataFrame:
    """单次拉取（供重试调用）。"""
    import io, sys, contextlib
    f = io.StringIO()
    with contextlib.redirect_stderr(f):
        df = ak.stock_sector_spot()
    df = df.rename(columns={
        "label":      "label",
        "板块":        "name",
        "公司家数":    "count",
        "涨跌幅":      "change_pct",   # %
        "涨跌额":      "change_amt",
        "总成交量":    "total_volume",
        "总成交额":    "total_amount",
    })
    # 统一列：保留有用的
    out = df[["label", "name", "count", "change_pct", "change_amt",
              "total_volume", "total_amount"]].copy()
    out["change_pct"] = pd.to_numeric(out["change_pct"], errors="coerce")
    out["change_amt"] = pd.to_numeric(out["change_amt"], errors="coerce")
    out["total_volume"] = pd.to_numeric(out["total_volume"], errors="coerce")
    out["total_amount"] = pd.to_numeric(out["total_amount"], errors="coerce")
    out = out.dropna(subset=["change_pct"]).sort_values("change_pct", ascending=False).reset_index(drop=True)
    return out


def _fetch_hot_rank() -> pd.DataFrame:
    """从东财拉个股人气榜。"""
    df = ak.stock_hot_rank_em()
    df = df.rename(columns={
        "当前排名":  "rank",
        "代码":      "code",
        "股票名称":  "name",
        "最新价":    "price",
        "涨跌额":    "change_amt",
        "涨跌幅":    "change_pct",
    })
    out = df[["rank", "code", "name", "price", "change_amt", "change_pct"]].copy()
    out["change_pct"]  = pd.to_numeric(out["change_pct"], errors="coerce")
    out["change_amt"]  = pd.to_numeric(out["change_amt"], errors="coerce")
    out["price"]       = pd.to_numeric(out["price"], errors="coerce")
    out = out.sort_values("rank").reset_index(drop=True)
    return out

def _fetch_stock_sector_map() -> dict[str, str]:
    """
    从新浪板块详情接口构建个股→板块映射。
    返回 dict：{代码(大写): label}。
    注意：tqdm 进度条会被临时抑制（stderr）。
    """
    import io, sys, contextlib
    result: dict[str, str] = {}
    sectors = _fetch_sector_spot()
    if sectors.empty:
        return result
    for _, row in sectors.iterrows():
        label = str(row["label"])
        try:
            # 抑制 akshare 内部 tqdm 进度条输出
            f = io.StringIO()
            with contextlib.redirect_stderr(f):
                detail = ak.stock_sector_detail(sector=label)
            if detail is None or detail.empty:
                continue
            for col in ["股票代码", "code", "代码"]:
                if col in detail.columns:
                    for code in detail[col].dropna():
                        result[str(code).strip().upper()] = label
                    break
        except Exception:
            continue
    return result


def get_stock_sector_map(force_refresh: bool = False) -> dict[str, str]:
    """
    获取个股→板块映射 dict {code: label}，缓存1小时。
    """
    global _stock_sector_map, _stock_sector_map_ts
    now = time.time()
    if not force_refresh and _stock_sector_map is not None and (now - _stock_sector_map_ts) < SECTOR_MAP_TTL:
        return _stock_sector_map
    try:
        _stock_sector_map = _fetch_stock_sector_map()
        _stock_sector_map_ts = now
    except Exception:
        if _stock_sector_map is None:
            _stock_sector_map = {}
    return _stock_sector_map


def get_stock_sector(code: str) -> Optional[str]:
    """查询某股票所属板块 label，不存在返回 None。"""
    m = get_stock_sector_map()
    return m.get(code.strip().upper())




# ═══════════════════════════════════════════════════════════
# 公开：带缓存的读取接口
# ═══════════════════════════════════════════════════════════════════════

def get_sector_spot(force_refresh: bool = False) -> pd.DataFrame:
    """
    获取新浪行业板块实时数据（有缓存）。

    Returns DataFrame，列：
      label, name, count, change_pct, change_amt, total_volume, total_amount
    按 change_pct 降序排列。
    """
    global _sector_spot_cache, _sector_spot_ts

    now = time.time()
    with _lock:
        if (not force_refresh
            and _sector_spot_cache is not None
            and (now - _sector_spot_ts) < SECTOR_CACHE_TTL):
            return _sector_spot_cache.copy()

        try:
            _sector_spot_cache = _fetch_sector_spot()
            _sector_spot_ts = now
        except Exception:
            if _sector_spot_cache is None:
                _sector_spot_cache = _EMPTY_SECTOR_DF.copy()

    return _sector_spot_cache.copy()


def get_hot_rank(force_refresh: bool = False) -> pd.DataFrame:
    """
    获取东财个股人气榜（有缓存）。

    Returns DataFrame，列：rank, code, name, price, change_amt, change_pct
    按 rank 升序。
    """
    global _hot_rank_cache, _hot_rank_ts

    now = time.time()
    with _lock:
        if (not force_refresh
            and _hot_rank_cache is not None
            and (now - _hot_rank_ts) < HOT_CACHE_TTL):
            return _hot_rank_cache.copy()

        try:
            _hot_rank_cache = _fetch_hot_rank()
            _hot_rank_ts = now
        except Exception:
            if _hot_rank_cache is None:
                _hot_rank_cache = _EMPTY_HOT_DF.copy()

    return _hot_rank_cache.copy()


# ═══════════════════════════════════════════════════════════
# 高层接口：热点判断
# ═══════════════════════════════════════════════════════════════════════

def get_top_sectors(n: int = 15) -> pd.DataFrame:
    """
    获取涨幅前 N 名行业板块。

    Returns DataFrame（n 行），列：label, name, change_pct, ...
    """
    df = get_sector_spot()
    if df.empty:
        return pd.DataFrame()
    return df.head(n).copy()


def is_sector_hot(sector_label: str, top_n: int = 15) -> bool:
    """
    判断某板块（label）是否属于当日热点（前 top_n 名）。
    """
    df = get_top_sectors(n=top_n)
    if df.empty:
        return False
    return sector_label in df["label"].values


def get_sector_hot_rank(sector_label: str) -> Optional[int]:
    """
    获取某板块的当日热度排名（1=最热），不在前100返回 None。
    """
    df = get_sector_spot()
    if df.empty:
        return None
    idx = df[df["label"] == sector_label].index
    if len(idx) == 0:
        return None
    rank = int(idx[0]) + 1
    return rank if rank <= 100 else None


def get_stock_hot_rank(code: str) -> Optional[int]:
    """
    获取某股票的人气榜排名（1=最热），返回 None 表示不在榜上。
    """
    df = get_hot_rank()
    if df.empty:
        return None
    row = df[df["code"].str.upper() == code.upper()]
    if row.empty:
        return None
    return int(row.iloc[0]["rank"])


def get_stock_hot_score(code: str) -> float:
    """
    综合人气得分（0~100）。

    逻辑：
      - 人气榜排名得分：rank 1 → 100分，每低10名 -5分
      - 人气涨幅得分：change_pct > 0 → +10分
    """
    rank = get_stock_hot_rank(code)
    if rank is None:
        return 0.0

    hot_score = max(0.0, 100.0 - (rank - 1) * 0.5)  # rank1=100, rank201=0

    # 人气涨幅加分
    df = get_hot_rank()
    row = df[df["code"].str.upper() == code.upper()]
    if not row.empty:
        chg = float(row.iloc[0]["change_pct"])
        if chg > 0:
            hot_score += min(chg, 10)  # 最多加10分

    return round(hot_score, 2)


def get_sector_change_pct(sector_label: str) -> float:
    """获取某板块当日涨跌幅（%），不存在返回 NaN。"""
    df = get_sector_spot()
    row = df[df["label"] == sector_label]
    if row.empty:
        return np.nan
    return float(row.iloc[0]["change_pct"])


# ═══════════════════════════════════════════════════════════
# 文件缓存：持久化到磁盘（用于进程重启后快速加载）
# ═══════════════════════════════════════════════════════════════════════

SECTOR_FILE  = CACHE_DIR / "sector_spot.json"
HOT_FILE    = CACHE_DIR / "hot_rank.json"
SECTOR_MAP_FILE = CACHE_DIR / "sector_map.json"


def _json_serializer(obj):
    """numpy 类型序列化 hook for json.dumps."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"{type(obj)} not serializable")


def save_to_disk():
    """手动保存当前缓存到磁盘（供后续进程使用）。"""
    global _sector_spot_cache, _hot_rank_cache, _stock_sector_map

    with _lock:
        if _sector_spot_cache is not None and not _sector_spot_cache.empty:
            SECTOR_FILE.parent.mkdir(parents=True, exist_ok=True)
            SECTOR_FILE.write_text(
                json.dumps({
                    "ts":     _sector_spot_ts,
                    "data":   _sector_spot_cache.to_dict(orient="records"),
                }, default=_json_serializer, ensure_ascii=False),
                encoding="utf-8"
            )
        if _hot_rank_cache is not None and not _hot_rank_cache.empty:
            HOT_FILE.parent.mkdir(parents=True, exist_ok=True)
            HOT_FILE.write_text(
                json.dumps({
                    "ts":   _hot_rank_ts,
                    "data": _hot_rank_cache.to_dict(orient="records"),
                }, default=_json_serializer, ensure_ascii=False),
                encoding="utf-8"
            )
        if _stock_sector_map:
            SECTOR_MAP_FILE.parent.mkdir(parents=True, exist_ok=True)
            SECTOR_MAP_FILE.write_text(
                json.dumps({
                    "ts":   _stock_sector_map_ts,
                    "data": _stock_sector_map,
                }, default=_json_serializer, ensure_ascii=False),
                encoding="utf-8"
            )

def load_from_disk():
    """从磁盘加载缓存到内存（进程启动时调用）。"""
    global _sector_spot_cache, _sector_spot_ts
    global _hot_rank_cache, _hot_rank_ts
    global _stock_sector_map, _stock_sector_map_ts

    if SECTOR_FILE.exists():
        try:
            raw = json.loads(SECTOR_FILE.read_text(encoding="utf-8"))
            _sector_spot_ts  = float(raw["ts"])
            _sector_spot_cache = pd.DataFrame(raw["data"])
        except Exception:
            pass

    if HOT_FILE.exists():
        try:
            raw = json.loads(HOT_FILE.read_text(encoding="utf-8"))
            _hot_rank_ts   = float(raw["ts"])
            _hot_rank_cache = pd.DataFrame(raw["data"])
        except Exception:
            pass
    if SECTOR_MAP_FILE.exists():
        try:
            raw = json.loads(SECTOR_MAP_FILE.read_text(encoding="utf-8"))
            _stock_sector_map_ts = float(raw["ts"])
            _stock_sector_map = dict(raw["data"])
        except Exception:
            pass


# 进程启动时自动加载磁盘缓存
load_from_disk()


# ═══════════════════════════════════════════════════════════
# CLI 测试
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="板块热点数据")
    parser.add_argument("--sectors", action="store_true", help="显示前15热点板块")
    parser.add_argument("--hot",      action="store_true", help="显示人气榜前20")
    parser.add_argument("--refresh",  action="store_true", help="强制刷新")
    parser.add_argument("--test",    type=str, default="", help="查询某板块/股票热点")
    args = parser.parse_args()

    if args.sectors:
        df = get_sector_spot(force_refresh=args.refresh)
        print(f"共 {len(df)} 个板块，按涨跌幅排序：")
        print(df[["name", "change_pct", "count"]].head(20).to_string(index=False))

    elif args.hot:
        df = get_hot_rank(force_refresh=args.refresh)
        print(f"人气榜共 {len(df)} 只，按排名：")
        if not df.empty:
            print(df.head(20).to_string(index=False))
        else:
            print("（人气榜接口暂时无法访问）")

    elif args.test:
        q = args.test.upper()
        # 尝试作为板块
        sec = get_sector_change_pct(q)
        if not np.isnan(sec):
            rank = get_sector_hot_rank(q)
            print(f"板块 {q}: 涨跌幅={sec:+.2f}%, 热度排名={rank}")
        # 尝试作为股票
        rk = get_stock_hot_rank(q)
        if rk is not None:
            score = get_stock_hot_score(q)
            print(f"股票 {q}: 人气排名={rk}, 人气得分的={score}")
        if np.isnan(sec) and rk is None:
            print(f"未找到: {q}")

    else:
        # 默认：显示概览
        df_sec = get_sector_spot(force_refresh=args.refresh)
        df_hot = get_hot_rank(force_refresh=args.refresh)
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{now_str}] 板块数={len(df_sec)}, 人气榜={len(df_hot)}")
        print("\n━━━ 热点板块 Top10 ━━━")
        print(df_sec[["name","change_pct"]].head(10).to_string(index=False))
        print("\n━━━ 人气榜 Top10 ━━━")
        if not df_hot.empty:
            print(df_hot[["rank","code","name","change_pct"]].head(10).to_string(index=False))
        else:
            print("（人气榜接口暂时无法访问，可稍后重试）")
