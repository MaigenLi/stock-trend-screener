#!/usr/bin/env python3
"""
遗漏股参数敏感性分析
=====================
对于每天涨幅>5%但未被 triple_screen 选中的股票：
1. 分析被 Step1 排除的具体原因
2. 测试小幅放宽参数是否能捕获

用法：python analyze_missed.py
"""
import sys, time, re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, str(Path(__file__).parent))

from gain_turnover import load_qfq_history
from rps_strong_screen import scan_rps, get_all_stock_codes, calc_stock_rps
from datetime import datetime

REPORTS_DIR = Path.home() / "stock_reports"

def get_gain5_codes(trade_date: str) -> set[str]:
    """获取指定日期涨幅>5%的股票代码"""
    prev_date_map = {
        "2026-04-14": "2026-04-13",
        "2026-04-15": "2026-04-14",
        "2026-04-16": "2026-04-15",
        "2026-04-17": "2026-04-16",
    }
    prev_date = prev_date_map[trade_date]

    codes = set()
    for p in REPORTS_DIR.glob(f"gain5_{trade_date}*.txt"):
        for line in p.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0].startswith(('sh', 'sz', 'bj')):
                codes.add(parts[0].lower())
    return codes

def scan_gain5(trade_date: str, prev_date: str) -> set[str]:
    """扫描指定日期涨幅>5%的股票，写入缓存文件"""
    cache_file = REPORTS_DIR / f"gain5_{trade_date}.txt"
    if cache_file.exists():
        codes = set()
        for line in cache_file.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0].startswith(('sh', 'sz', 'bj')):
                codes.add(parts[0].lower())
        return codes

    all_codes = get_all_stock_codes()
    print(f"\n  扫描 {trade_date} 涨幅>5%（全市场 {len(all_codes)} 只）...")

    def check_gain(code):
        try:
            df = load_qfq_history(code, end_date=trade_date, adjust="qfq")
            if df is None or len(df) < 2:
                return None
            dates = df['date'].values
            prices = df['close'].values
            idx_t, idx_p = None, None
            for i in range(len(dates)-1, -1, -1):
                d = str(dates[i])[:10]
                if d == trade_date: idx_t = i
                elif d == prev_date: idx_p = i
            if idx_p is None or idx_t is None or idx_t != idx_p + 1:
                return None
            gain = (prices[idx_t] / prices[idx_p] - 1) * 100
            return (code, round(gain, 2)) if gain > 5.0 else None
        except:
            return None

    results = []
    done = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(check_gain, c): c for c in all_codes}
        for f in as_completed(futures):
            r = f.result()
            if r: results.append(r)
            done += 1
            if done % 1000 == 0:
                print(f"    {done}/{len(all_codes)}")

    results.sort(key=lambda x: -x[1])
    cache_file.write_text(
        "\n".join(f"{c}  {g:+.2f}%" for c, g in results),
        encoding="utf-8"
    )
    print(f"  → {len(results)} 只涨幅>5%，已缓存")
    return {r[0] for r in results}


def parse_triple_screen(path: Path) -> set[str]:
    """解析 triple_screen 文件，返回股票代码集合"""
    codes = set()
    code_pat = re.compile(r"^(sh|sz|bj)(\d{6})$")
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) >= 2:
            m = code_pat.match(parts[0].lower())
            if m:
                codes.add(f"{m.group(1)}{m.group(2)}")
    return codes


def step1_check(code: str, sig_date: str, relaxed: bool = False) -> dict:
    """
    用 calc_stock_rps 做 Step1 检查。
    返回各项指标和通过/失败原因。
    """
    try:
        result = calc_stock_rps(code, target_date=datetime.strptime(sig_date, "%Y-%m-%d"))
        if result is None:
            return {"code": code, "status": "no_data", "reason": "无数据或被过滤"}

        rsi = result.get("rsi", 0)
        composite = result.get("composite", 0)
        rps20 = result.get("ret20_rps", 0)
        ret20 = result.get("ret20", 0)
        ret5 = result.get("ret5", 0)
        avg_turn5 = result.get("avg_turnover_5", 0)

        # 当前门槛
        C_RPS = 70 if relaxed else 80
        C_RPS20 = 65 if relaxed else 75
        C_RET20_MAX = 50 if relaxed else 40
        C_RET5_MAX = 25 if relaxed else 20
        C_RSI_LOW = 45 if relaxed else 50
        C_RSI_HIGH = 85 if relaxed else 82
        C_TURNOVER = 2.0 if relaxed else 3.0

        failures = []
        if composite < C_RPS: failures.append(f"RPS综={composite:.0f}<{C_RPS}")
        if rsi < C_RSI_LOW or rsi > C_RSI_HIGH: failures.append(f"RSI={rsi:.0f}不在[{C_RSI_LOW},{C_RSI_HIGH}]")
        if rps20 < C_RPS20: failures.append(f"RPS20={rps20:.0f}<{C_RPS20}")
        if ret20 > C_RET20_MAX: failures.append(f"20日涨={ret20:.0f}%>{C_RET20_MAX}%")
        if ret20 < -10: failures.append(f"20日涨={ret20:.0f}<-10%")
        if ret5 > C_RET5_MAX: failures.append(f"近5日={ret5:.0f}%>{C_RET5_MAX}%")
        if avg_turn5 < C_TURNOVER: failures.append(f"换手={avg_turn5:.1f}%<{C_TURNOVER}%")

        return {
            "code": code,
            "status": "pass" if not failures else "fail",
            "failures": failures,
            "rsi": rsi,
            "composite": composite,
            "rps20": rps20,
            "ret20": ret20,
            "ret5": ret5,
            "avg_turn5": avg_turn5,
            "relaxed": relaxed,
        }
    except Exception as e:
        return {"code": code, "status": "error", "reason": str(e)}


def analyze_day(trade_date: str, prev_date: str):
    """分析单个交易日"""
    triple_file = REPORTS_DIR / f"triple_screen_{prev_date}.txt"
    gain5_cache = REPORTS_DIR / f"gain5_{trade_date}.txt"

    print(f"\n{'='*60}")
    print(f"📊 {trade_date} 分析 | triple_screen信号日: {prev_date}")
    print(f"{'='*60}")

    # 获取涨幅>5%的股票
    gain5_codes = scan_gain5(trade_date, prev_date)
    if not gain5_codes:
        print(f"  无涨幅>5%数据，跳过")
        return

    # triple_screen 选中的股票
    if not triple_file.exists():
        print(f"  ⚠️ 找不到 {triple_file.name}，跳过")
        return
    selected = parse_triple_screen(triple_file)

    # 交集和遗漏
    in_selected = gain5_codes & selected
    missed = gain5_codes - selected

    print(f"  {trade_date}涨幅>5%: {len(gain5_codes)} 只")
    print(f"  triple_screen({prev_date})选中: {len(selected)} 只")
    print(f"  交集（已捕获）: {len(in_selected)} 只")
    print(f"  遗漏（未捕获）: {len(missed)} 只")

    if not missed:
        return

    # Step1 分析（当前门槛）
    print(f"\n  ── Step1 原因分析（当前门槛）──")
    step1_pass = []
    step1_fail_reasons = {}
    all_codes = list(missed)

    def check_wrapper(code):
        return step1_check(code, prev_date, relaxed=False)

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(check_wrapper, c): c for c in all_codes}
        done = 0
        for f in as_completed(futures):
            r = f.result()
            if r["status"] == "pass":
                step1_pass.append(r["code"])
            else:
                key = r.get("failures", [r.get("reason", "unknown")])
                key_str = "; ".join(key) if isinstance(key, list) else str(key)
                step1_fail_reasons[key_str] = step1_fail_reasons.get(key_str, 0) + 1
            done += 1
            if done % 50 == 0:
                print(f"    Step1检查进度: {done}/{len(missed)}")

    for reason, cnt in sorted(step1_fail_reasons.items(), key=lambda x: -x[1]):
        print(f"    [{cnt:3}只] {reason}")

    # 放宽参数后的 Step1（只测 fail 的）
    if step1_fail_reasons:
        print(f"\n  ── 放宽参数后（放宽版 Step1）──")
        # 宽松门槛: RPS综≥70, RPS20≥65, 20日涨≤50%, 近5日≤25%, RSI[45,85], 换手≥2%
        relax_pass = []
        fail_still = {}

        def check_relaxed(code):
            return step1_check(code, prev_date, relaxed=True)

        with ThreadPoolExecutor(max_workers=8) as ex:
            futures = {ex.submit(check_relaxed, c): c for c in all_codes}
            done = 0
            for f in as_completed(futures):
                r = f.result()
                if r["status"] == "pass":
                    relax_pass.append(r["code"])
                else:
                    key = "; ".join(r.get("failures", [r.get("reason", "unknown")]))
                    fail_still[key] = fail_still.get(key, 0) + 1
                done += 1
                if done % 50 == 0:
                    print(f"    宽松检查进度: {done}/{len(missed)}")

        newly_passed = set(relax_pass) - set(step1_pass)
        print(f"    当前门槛通过: {len(step1_pass)} 只")
        print(f"    放宽后通过: {len(relax_pass)} 只")
        print(f"    新增通过: {len(newly_passed)} 只")

        # 剩余被挡在 Step2/Step3 的
        still_fail = {k: v for k, v in fail_still.items()}
        if still_fail:
            print(f"\n    放宽后仍被排除的原因：")
            for reason, cnt in sorted(still_fail.items(), key=lambda x: -x[1]):
                print(f"      [{cnt:3}只] {reason}")

        if newly_passed:
            print(f"\n    放宽后新增捕获（{len(newly_passed)} 只）：")
            for code in sorted(newly_passed):
                r = step1_check(code, prev_date, relaxed=True)
                print(f"      {code}  RPS综={r['composite']:.0f}  RSI={r['rsi']:.0f}  5日={r['ret5']:.0f}%  换手={r['avg_turn5']:.1f}%")


def main():
    analyses = [
        ("2026-04-14", "2026-04-13"),
        ("2026-04-15", "2026-04-14"),
        ("2026-04-16", "2026-04-15"),
        ("2026-04-17", "2026-04-16"),
    ]

    for trade_date, prev_date in analyses:
        analyze_day(trade_date, prev_date)


if __name__ == "__main__":
    main()
