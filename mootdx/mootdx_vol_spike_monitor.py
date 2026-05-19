#!/usr/bin/env python3
"""
mootdx_vol_spike_monitor.py
===========================
全市场分钟级放量监控 — 找出相对上一分钟放量 N 倍的股票

对标 mootdx_full_volume_scan.py 的全市场扫描架构：
  1. 启动时并发预加载昨日分时数据
  2. 每轮用批量 get_security_quotes 获取今日实时量
  3. 同分钟位对比：今日当前累计量 vs 昨日同时刻累计量
  4. 放量倍数 = 今日累计 / 昨日累计

用法：
  python3 mootdx_vol_spike_monitor.py                      # 全市场，5倍阈值
  python3 mootdx_vol_spike_monitor.py --ratio 3 --top 30    # 3倍阈值
  python3 mootdx_vol_spike_monitor.py --avg-days 3          # 前3日均值基准
  python3 mootdx_vol_spike_monitor.py --once              # 单次扫描
"""

import argparse, sys, time, json
from pathlib import Path
from datetime import datetime, date as date_type, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

WORKSPACE = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(WORKSPACE))
from stock_trend import gain_turnover as gt

# ─────────────────────────────────────────────────────────────
# 常量
# ─────────────────────────────────────────────────────────────
TOTAL_MINUTES    = 240
BATCH_SIZE        = 80
_DEFAULT_WORKERS  = 8
DEFAULT_RATIO     = 5.0
DEFAULT_TOP       = 40
DEFAULT_AVG_DAYS  = 1
MIN_PRELOADED   = 500   # 预加载有效只数低于此值时重试
OUTPUT_DIR        = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 通达信连接
# ─────────────────────────────────────────────────────────────
_TDX_CLIENT = None


def get_tdx_client():
    global _TDX_CLIENT
    if _TDX_CLIENT is None:
        from mootdx.quotes import Quotes
        _TDX_CLIENT = Quotes.factory(market="std")
    return _TDX_CLIENT


def _suppress_logs():
    try:
        import tdxpy.logger as _log
        _log.logger.setLevel(999)
        for h in _log.logger.handlers:
            h.setLevel(999)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────
# 市场判断
# ─────────────────────────────────────────────────────────────

def auto_market(code: str) -> int:
    """根据代码判断市场：沪=1，深=0"""
    c = code.lower()
    for pfx in ("sh", "sz", "bj"):
        if c.startswith(pfx):
            c = c[2:]
            break
    return 1 if (c.startswith("6") or c.startswith("9")) else 0


def _raw_code(code: str) -> str:
    """去掉 sh/sz/bj 前缀，返回纯代码"""
    raw = code.lower()
    for pfx in ("sh", "sz", "bj"):
        if raw.startswith(pfx):
            raw = raw[2:]
            break
    return raw


# ─────────────────────────────────────────────────────────────
# 分时数据获取
# ─────────────────────────────────────────────────────────────

def _fetch_day_minute(code: str, day_date: date_type, client) -> list | None:
    """
    获取单只股票单日分时累计量数组（到每分钟为止的累计量）。
    返回 list[int]（240个累计量），失败返回None。
    """
    _suppress_logs()
    mkt = auto_market(code)
    date_int = day_date.year * 10000 + day_date.month * 100 + day_date.day
    try:
        bars = client.client.get_history_minute_time_data(mkt, code, date_int)
        if not bars or len(bars) < 200:
            return None
        cum = []
        running = 0
        for b in bars:
            running += int(b.get("vol") or 0)
            cum.append(running)
        return cum if cum else None
    except Exception:
        return None


def get_last_trading_days(n: int = 10) -> list[date_type]:
    """获取最近N个交易日（倒序，周末跳过）"""
    today = datetime.now().date()
    days = []
    cur = today - timedelta(days=1)
    while len(days) < n and cur.year >= 2020:
        if cur.weekday() < 5:
            days.append(cur)
        cur -= timedelta(days=1)
    return days


# ─────────────────────────────────────────────────────────────
# 预加载昨日分时数据（并发）
# ─────────────────────────────────────────────────────────────

def preload_avg_bars(all_codes: list, names_cache: dict,
                     last_n_days: list[date_type], avg_days: int = 1,
                     max_workers: int = _DEFAULT_WORKERS) -> dict:
    """
    并发预加载全市场前 N 日分时均值。
    返回 {raw_code: {"cum_avg": [240 floats], "cum_days": [list of 240-list],
                   "name": str, "last_close": float, "yesterday_total": float}}
    """
    result  = {}
    lock    = __import__("threading").Lock()
    done    = [0]
    total   = len(all_codes)
    t0      = time.time()
    days    = last_n_days[:avg_days]
    client  = get_tdx_client()
    day_str = ", ".join(str(d) for d in days)

    def _worker(code: str):
        raw = _raw_code(code)
        mkt = auto_market(code)

        all_cums = []
        for day in days:
            cum = _fetch_day_minute(raw, day, client)
            if cum:
                all_cums.append(cum)

        if not all_cums:
            return

        valid_days = len(all_cums)
        min_len    = min(len(c) for c in all_cums)
        avg_cum    = [sum(c[i] for c in all_cums) / valid_days for i in range(min_len)]
        if len(avg_cum) < TOTAL_MINUTES:
            avg_cum += [avg_cum[-1]] * (TOTAL_MINUTES - len(avg_cum))

        last_close       = 0.0
        yesterday_total  = 0.0
        try:
            c  = gt.normalize_prefixed(code)
            df = gt.load_qfq_history(c, refresh=False)
            if df is not None and len(df) >= 2:
                last_close      = float(df["close"].iloc[-2])
                yesterday_total = float(df["volume"].iloc[-2]) if "volume" in df.columns else 0.0
        except Exception:
            pass

        with lock:
            result[raw] = {
                "cum_avg":         avg_cum,
                "cum_days":        all_cums,
                "name":            names_cache.get(code, ""),
                "last_close":      last_close,
                "yesterday_total": yesterday_total,
                "valid_days":      valid_days,
            }
            done[0] += 1
            if done[0] % 500 == 0 or done[0] == total:
                print(f"\r  进度: {done[0]}/{total} ({done[0]/total*100:.0f}%)", end="", flush=True)

    workers = min(max_workers, len(all_codes))
    print(f"📥 预加载分时均值（{avg_days}日: {day_str}，{total}只，{workers}线程）...")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_worker, code): code for code in all_codes}
        for future in as_completed(futures):
            pass
    print(f"\n  完成 {len(result)} 只有效，耗时 {time.time()-t0:.1f}s")
    return result


# ─────────────────────────────────────────────────────────────
# 批量获取今日实时行情
# ─────────────────────────────────────────────────────────────

def fetch_all_realtime(all_codes: list) -> tuple[dict, int, int]:
    """
    通过 mootdx 分块批量获取全市场实时行情。
    返回 (quotes_dict, total_requested, n_failed)
    """
    client   = get_tdx_client()
    quotes   = {}
    total    = len(all_codes)
    n_fail   = 0

    for start in range(0, total, BATCH_SIZE):
        chunk  = all_codes[start:start + BATCH_SIZE]
        batch  = [(auto_market(c), _raw_code(c)) for c in chunk]
        try:
            bresult = client.client.get_security_quotes(batch)
            returned = set()
            for q in (bresult or []):
                code = q.get("code", "").lower()
                quotes[code] = {
                    "price":      float(q.get("price")     or 0),
                    "open":       float(q.get("open")       or 0),
                    "vol":        int(q.get("vol")          or 0),
                    "last_close": float(q.get("last_close") or 0),
                }
                returned.add(code)
            # 统计本批次已请求但未返回的
            for c in chunk:
                rc = _raw_code(c)
                if rc not in returned:
                    n_fail += 1
        except Exception:
            n_fail += len(chunk)

    return quotes, total, n_fail


# ─────────────────────────────────────────────────────────────
# 量比计算（分钟精度）
# ─────────────────────────────────────────────────────────────

def compute_ratio_from_minute(code_raw: str, quote: dict, cur_idx: int,
                              yesterday_data: dict | None) -> float | None:
    """
    计算放量倍数：今日当前累计量 / 昨日同时刻累计量
    cur_idx: 当前分钟索引（0-based）
    """
    today_vol = quote.get("vol", 0)
    if today_vol <= 0:
        return None

    if yesterday_data is not None:
        # 优先用日均分时
        if "cum_avg" in yesterday_data:
            cum    = yesterday_data["cum_avg"]
            cap    = min(cur_idx, len(cum) - 1)
            if cap >= 0:
                avg_cum = cum[cap]
                if avg_cum > 0:
                    return today_vol / avg_cum
        # 兼容单日 cum
        elif "cum" in yesterday_data:
            cum  = yesterday_data["cum"]
            cap  = min(cur_idx, len(cum) - 1)
            if cap >= 0:
                yes_cum = cum[cap]
                if yes_cum > 0:
                    return today_vol / yes_cum

    # 回退：日线比例
    yesterday_total = (yesterday_data.get("yesterday_total", 0)
                       if yesterday_data else 0)
    if yesterday_total > 0:
        day_frac = (cur_idx + 1) / TOTAL_MINUTES if cur_idx >= 0 else 1.0
        expected = yesterday_total * day_frac
        if expected > 0:
            return today_vol / expected

    return None


# ─────────────────────────────────────────────────────────────
# 上一分钟放大计算（核心新增逻辑）
# ─────────────────────────────────────────────────────────────

def compute_prev_ratio(code_raw: str, quote: dict, cur_idx: int,
                       yesterday_data: dict | None) -> float | None:
    """
    计算相对上一分钟的放量倍数：今日当前量 / 上一分钟量
    这里用昨日同时刻的前一分钟作为代理基准。
    如果 cur_idx == 0，返回 None（上一天没有可比数据）。
    """
    if cur_idx < 1:
        return None

    today_vol = quote.get("vol", 0)
    if today_vol <= 0:
        return None

    if yesterday_data is not None:
        if "cum_avg" in yesterday_data:
            cum     = yesterday_data["cum_avg"]
            cap_cur  = min(cur_idx,     len(cum) - 1)
            cap_prev = min(cur_idx - 1,  len(cum) - 1)
            if cap_cur >= 0 and cap_prev >= 0:
                cur_avg  = cum[cap_cur]
                prev_avg = cum[cap_prev]
                if prev_avg > 0:
                    return today_vol / prev_avg
        elif "cum" in yesterday_data:
            cum      = yesterday_data["cum"]
            cap_cur  = min(cur_idx,     len(cum) - 1)
            cap_prev = min(cur_idx - 1, len(cum) - 1)
            if cap_cur >= 0 and cap_prev >= 0:
                cur_avg  = cum[cap_cur]
                prev_avg = cum[cap_prev]
                if prev_avg > 0:
                    return today_vol / prev_avg

    return None


# ─────────────────────────────────────────────────────────────
# 扫描一轮
# ─────────────────────────────────────────────────────────────

def scan_cycle(quotes: dict, yesterday: dict, cur_idx: int,
               ratio_threshold: float = DEFAULT_RATIO,
               top: int = DEFAULT_TOP,
               use_prev: bool = False) -> list[dict]:
    """
    全市场一轮扫描，返回放量股票列表。
    use_prev=True 时用上一分钟对比模式，False 时用昨日同时刻对比模式。
    """
    results  = []
    mode_lbl = "上一分钟" if use_prev else "昨日同时"

    for code_raw, q in quotes.items():
        today_vol = q.get("vol", 0)
        if today_vol <= 0:
            continue

        yd  = yesterday.get(code_raw)
        r   = (compute_prev_ratio(code_raw, q, cur_idx, yd)
               if use_prev
               else compute_ratio_from_minute(code_raw, q, cur_idx, yd))
        if r is None:
            continue
        if r < ratio_threshold:
            continue

        name       = (yd["name"]       if yd else "")
        last_close = (yd["last_close"] if yd else 0.0)
        price      = q.get("price", 0.0)
        gain_day   = ((price - last_close) / last_close * 100
                      if last_close and last_close > 0 else 0.0)

        results.append({
            "code":       code_raw,
            "name":       name,
            "price":      round(price, 2),
            "vol":        today_vol,
            "last_close": round(last_close, 2),
            "gain_day":   round(gain_day, 2),
            "ratio":      round(r, 2),
        })

    results.sort(key=lambda x: -x["ratio"])
    return results[:top]


# ─────────────────────────────────────────────────────────────
# 渲染
# ─────────────────────────────────────────────────────────────

def current_minute_index(cur_time: datetime) -> int:
    total   = cur_time.hour * 3600 + cur_time.minute * 60 + cur_time.second
    AM_START = 9 * 3600 + 30 * 60
    AM_END   = 11 * 3600 + 29 * 60
    PM_START = 13 * 3600
    PM_END   = 15 * 3600 - 1
    if total < AM_START:
        return -1
    if total <= AM_END:
        return (total - AM_START) // 60
    if total < PM_START:
        return -1
    if total <= PM_END:
        return (total - PM_START) // 60 + 120
    return TOTAL_MINUTES - 1


def minute_to_str(idx: int) -> str:
    if idx < 0:
        return "--:--"
    secs = (34200 + idx * 60) if idx < 120 else (46800 + (idx - 120) * 60)
    return f"{secs // 3600:02d}:{(secs % 3600) // 60:02d}"


def render_top(results: list, cur_time: datetime, cur_idx: int,
               ratio: float, cycle: int, elapsed: float,
               total: int, n_fail: int, mode_lbl: str):
    now_str  = cur_time.strftime("%H:%M:%S")
    min_lbl  = minute_to_str(cur_idx)
    n_ok     = total - n_fail
    print(f"\n{'='*75}")
    print(f"⏰ {now_str}  [{min_lbl}]  #{cur_idx+1}/240  |  第{cycle}轮  |  耗时{elapsed:.1f}s")
    print(f"📡 获取: {n_ok}/{total} 只  失败: {n_fail} 只  |  🔍 {mode_lbl} 阈值≥{ratio}x  找到 {len(results)} 只")
    print(f"{'代码':10s}{'名称':8s}{'价格':>8s}{'今日量':>12s}{'倍数':>8s}{'当日涨幅':>9s}")
    print("-"*75)
    for r in results:
        print(
            f"{r['code']:10s}"
            f"{r['name'][:6]:8s}"
            f"{r['price']:8.2f}"
            f"{r['vol']:12.0f}"
            f"{r['ratio']:7.1f}x"
            f"{r['gain_day']:+8.2f}%"
        )
    print()


# ─────────────────────────────────────────────────────────────
# 主循环
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ratio",    type=float, default=DEFAULT_RATIO,
                        help=f"放量倍数阈值（默认{DEFAULT_RATIO}x）")
    parser.add_argument("--top",       type=int,   default=DEFAULT_TOP,
                        help=f"显示前N只（默认{DEFAULT_TOP}）")
    parser.add_argument("--avg-days", type=int,   default=DEFAULT_AVG_DAYS,
                        help=f"参比基准天数（默认{DEFAULT_AVG_DAYS}）")
    parser.add_argument("--workers",   type=int,   default=_DEFAULT_WORKERS,
                        help=f"预加载线程数（默认{_DEFAULT_WORKERS}）")
    parser.add_argument("--interval",  type=int,   default=60,
                        help="刷新间隔（秒，默认60）")
    parser.add_argument("--once",     action="store_true",
                        help="单次扫描后退出")
    parser.add_argument("--prev",    action="store_true",
                        help="用上一分钟对比模式（默认用昨日同时刻）")
    args = parser.parse_args()

    _suppress_logs()

    # ── 1. 预加载（含重试）──────────────────────────────
    all_codes   = gt.get_all_stock_codes()
    names_cache = gt.load_stock_names()
    last_n_days = get_last_trading_days(n=10)

    yesterday = {}
    retry_wait = 10  # 秒
    for attempt in range(1, 3):
        yesterday = preload_avg_bars(all_codes, names_cache, last_n_days,
                                     avg_days=args.avg_days,
                                     max_workers=args.workers)
        n_pre = len(yesterday)
        if n_pre >= MIN_PRELOADED:
            break
        if attempt == 1:
            print(f"⚠️  分时数据 {n_pre} 只不足，等待 {retry_wait}s 后重试...")
            time.sleep(retry_wait)

    if not yesterday:
        print("❌ 预加载失败（通达信连接问题），退出")
        sys.exit(1)

    n_pre = len(yesterday)
    if n_pre < MIN_PRELOADED:
        print(f"⚠️  分时数据 {n_pre} 只（< {MIN_PRELOADED}），其余用日线比例回退")

    # ── 2. 全市场代码列表 ────────────────────────────────
    raw_codes = [_raw_code(c) for c in all_codes]

    mode_lbl = "上一分钟" if args.prev else "昨日同时"
    print(f"\n{'='*50}")
    print(f"  全市场分钟级放量监控")
    print(f"  模式: {mode_lbl}对比  倍率阈值: {args.ratio}x")
    print(f"  预加载: {len(yesterday)} 只  |  全市场: {len(raw_codes)} 只")
    print(f"{'='*50}\n")

    # ── 3. 扫描循环 ─────────────────────────────────────
    cycle       = 0
    t_start     = time.time()
    cur_idx_ref = -1  # 上一次的 minute index，避免重复输出

    while True:
        cycle  += 1
        t0     = time.time()
        cur    = datetime.now()

        # 计算当前分钟索引
        cur_idx = current_minute_index(cur)
        if cur_idx < 0:
            print(f"[{cur.strftime('%H:%M')}] 非交易时段，跳过...")
            time.sleep(args.interval)
            continue

        # 跳过同一分钟内重复扫描
        if cur_idx == cur_idx_ref and not args.once:
            time.sleep(5)
            continue
        cur_idx_ref = cur_idx

        # 批量获取今日行情
        quotes, total_req, n_fail = fetch_all_realtime(raw_codes)

        # 扫描
        results = scan_cycle(
            quotes, yesterday, cur_idx,
            ratio_threshold=args.ratio,
            top=args.top,
            use_prev=args.prev,
        )

        elapsed = time.time() - t0
        render_top(results, cur, cur_idx, args.ratio, cycle, elapsed,
                   len(quotes), n_fail, mode_lbl)

        # 保存
        out = OUTPUT_DIR / f"vol_spike_{cur.strftime('%Y-%m-%d_%H%M')}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        if args.once:
            break

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
