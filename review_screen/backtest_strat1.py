#!/usr/bin/env python3
"""
策略1回测（优化版v2）
T日出信号 → T+1开盘买入 → T+5收盘卖出
使用 pandas ewm 向量化计算 EMA/MACD
"""
import sys, json, time, argparse
from pathlib import Path
import numpy as np
import pandas as pd

WORKSPACE = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(WORKSPACE))
QFQ_DIR = WORKSPACE / ".cache" / "qfq_daily"
OUT_DIR  = WORKSPACE / "stock_reports"; OUT_DIR.mkdir(exist_ok=True)

# ── 数据加载 ─────────────────────────────────────────────
print("📂 加载数据...", flush=True)
_all = {}
for f in QFQ_DIR.glob("*_qfq.csv"):
    code = f.stem.replace("_qfq","")
    try:
        df = pd.read_csv(f)
        df = df.sort_values("date").reset_index(drop=True)
        _all[code] = df
    except: pass
print(f"✅ {len(_all)}只", flush=True)

all_dates = sorted(_all[list(_all.keys())[0]]["date"].tolist())

# ── 指标预计算（向量化）───────────────────────────────────
def precompute(df):
    n = len(df)
    if n < 70: return None
    closes = df["close"].values.astype(float)
    turnovers = df["turnover"].values.astype(float) if "turnover" in df.columns else np.zeros(n)

    s = pd.Series(closes)
    ma5  = s.rolling(5).mean().values
    ma10 = s.rolling(10).mean().values
    ma20 = s.rolling(20).mean().values
    ma60 = s.rolling(60).mean().values

    # 均线方向（5日前 vs 现在）
    ma5_5d   = pd.Series(ma5).shift(5).values
    ma10_5d  = pd.Series(ma10).shift(5).values
    ma20_5d  = pd.Series(ma20).shift(5).values
    ma60_5d  = pd.Series(ma60).shift(5).values
    dir5  = np.where((~np.isnan(ma5)) & (~np.isnan(ma5_5d))  & (ma5  > ma5_5d *1.001), 1,
                 np.where((~np.isnan(ma5)) & (~np.isnan(ma5_5d))  & (ma5  < ma5_5d *0.999), -1, 0))
    dir10 = np.where((~np.isnan(ma10))& (~np.isnan(ma10_5d)) & (ma10 > ma10_5d*1.001), 1,
                 np.where((~np.isnan(ma10))&(~np.isnan(ma10_5d))&(ma10 < ma10_5d*0.999), -1, 0))
    dir20 = np.where((~np.isnan(ma20))& (~np.isnan(ma20_5d)) & (ma20 > ma20_5d*1.001), 1,
                 np.where((~np.isnan(ma20))&(~np.isnan(ma20_5d))&(ma20 < ma20_5d*0.999), -1, 0))
    dir60 = np.where((~np.isnan(ma60))& (~np.isnan(ma60_5d)) & (ma60 > ma60_5d*1.001), 1,
                 np.where((~np.isnan(ma60))&(~np.isnan(ma60_5d))&(ma60 < ma60_5d*0.999), -1, 0))

    # MACD (EMA方式)
    ema_fast = s.ewm(span=12, adjust=False).mean().values
    ema_slow = s.ewm(span=26, adjust=False).mean().values
    dif = ema_fast - ema_slow
    dea = pd.Series(dif).ewm(span=9, adjust=False).mean().values
    macd = (dif - dea) * 2

    # 5日涨幅
    gain5 = np.full(n, np.nan)
    gain5[5:] = (closes[5:] / closes[:-5] - 1) * 100

    return {
        "close": closes, "turnover": turnovers,
        "ma5": ma5, "ma10": ma10, "ma20": ma20, "ma60": ma60,
        "dir5": dir5, "dir10": dir10, "dir20": dir20, "dir60": dir60,
        "macd": macd, "dif": dif, "dea": dea, "gain5": gain5,
        "open": df["open"].values.astype(float),
    }

print("⚙️  预计算指标...", flush=True)
t0 = time.time()
pc = {}
for i, (code, df) in enumerate(_all.items()):
    if i % 1000 == 0: print(f"  {i}/{len(_all)}", flush=True)
    result = precompute(df)
    if result is not None:
        result["date"] = df["date"].tolist()
        pc[code] = result
print(f"✅ {len(pc)}只完成  ({time.time()-t0:.0f}秒)", flush=True)

# ── 回测 ─────────────────────────────────────────────────
date_to_idx = {d: i for i, d in enumerate(all_dates)}

def check_conditions(pcd, idx):
    try:
        if not (pcd["close"][idx] > pcd["ma5"][idx]):          return False
        if not (pcd["dir5"][idx]==1 and pcd["dir10"][idx]==1 and pcd["dir20"][idx]==1 and pcd["dir60"][idx]==1): return False
        if not (pcd["ma5"][idx]>pcd["ma10"][idx]>pcd["ma20"][idx]>pcd["ma60"][idx]): return False
        if not (pcd["macd"][idx]>0 and pcd["dif"][idx]>0 and pcd["dea"][idx]>0): return False
        if not (pcd["gain5"][idx]>5):                          return False
        t5 = np.nanmean(pcd["turnover"][idx-4:idx+1]) if idx>=4 else pcd["turnover"][idx]
        if t5 < 10.0: return False
        return True
    except: return False

def run_backtest(dates, si, ei):
    signals = []
    for di in range(si, min(ei+1, len(dates)-5)):
        d = dates[di]
        buy_date  = dates[di+1]
        sell_date = dates[di+5]
        bi = date_to_idx.get(buy_date)
        si_idx = date_to_idx.get(sell_date)
        if bi is None or si_idx is None: continue
        for code, pcd in pc.items():
            if di < 65 or di >= len(pcd["close"]): continue
            if not check_conditions(pcd, di): continue
            buy_p  = pcd["open"][bi]    if bi    < len(pcd["open"])  else None
            sell_p = pcd["close"][si_idx] if si_idx < len(pcd["close"]) else None
            if buy_p is None or sell_p is None: continue
            if buy_p<=0 or sell_p<=0: continue
            ret = (sell_p - buy_p) / buy_p * 100
            signals.append({
                "code": code, "signal_date": d, "buy_date": buy_date,
                "sell_date": sell_date, "buy_price": round(buy_p,2),
                "sell_price": round(sell_p,2), "return_pct": round(ret,3),
                "win": ret > 0,
            })
    return signals

# ── 报告 ─────────────────────────────────────────────────
def print_report(signals, label):
    if not signals:
        print(f"\n{label}: 无信号", flush=True)
        return None
    rets = [s["return_pct"] for s in signals]
    win_rate = sum(1 for r in rets if r>0)/len(rets)*100
    avg_ret  = np.mean(rets)
    std_ret  = np.std(rets)
    sharpe   = (avg_ret*252/5)/(std_ret*np.sqrt(252/5)) if std_ret>0 else 0
    print(f"\n{'='*60}", flush=True)
    print(f"{label}", flush=True)
    print(f"  信号数: {len(signals)}  胜率: {win_rate:.1f}%  均收益: {avg_ret:+.3f}%", flush=True)
    print(f"  最大盈利: {max(rets):+.2f}%  最大亏损: {min(rets):+.2f}%", flush=True)
    buckets = [(-50,-20),(-20,-10),(-10,-5),(-5,0),(0,5),(5,10),(10,20),(20,50)]
    for lo, hi in buckets:
        cnt = sum(1 for r in rets if lo<=r<hi)
        bar = '█' * int(cnt/len(rets)*50) if len(rets)>0 else ''
        print(f"  [{lo:>4},{hi:>3})%  {cnt:>4}  {bar}", flush=True)
    return {"signals":len(signals),"win_rate":round(win_rate,2),"avg_ret":round(avg_ret,4),
            "max_win":round(max(rets),2),"max_loss":round(min(rets),2),"sharpe":round(sharpe,4)}

# ── 主程序 ───────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--train-end",  default="2025-09-30")
parser.add_argument("--valid-start",default="2025-10-01")
parser.add_argument("--valid-end",  default="2026-04-25")
args = parser.parse_args()

end_idx   = len(all_dates)-1
train_end_idx   = all_dates.index(args.train_end)   if args.train_end   in all_dates else 0
valid_start_idx = all_dates.index(args.valid_start) if args.valid_start in all_dates else train_end_idx+1
valid_end_idx   = all_dates.index(args.valid_end)   if args.valid_end   in all_dates else end_idx

print(f"训练集: {all_dates[0]} ~ {args.train_end} ({train_end_idx+1}个交易日)", flush=True)
print(f"验证集: {args.valid_start} ~ {args.valid_end} ({valid_end_idx-valid_start_idx+1}个交易日)", flush=True)

ts = time.time()
train_signals = run_backtest(all_dates, 0, train_end_idx)
print(f"训练集扫描完成 ({time.time()-ts:.0f}秒)", flush=True)
ts2 = time.time()
valid_signals = run_backtest(all_dates, valid_start_idx, valid_end_idx)
print(f"验证集扫描完成 ({time.time()-ts2:.0f}秒)", flush=True)

train_stats = print_report(train_signals, f"【训练集】 {all_dates[0]} ~ {args.train_end}")
valid_stats = print_report(valid_signals, f"【验证集】 {args.valid_start} ~ {args.valid_end}")

def clean_for_json(obj):
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clean_for_json(x) for x in obj]
    if hasattr(obj, 'item') and callable(obj.item):
        return obj.item()
    return obj

out = {"train": train_stats, "valid": valid_stats,
       "train_signals": clean_for_json(train_signals[:200]),
       "valid_signals": clean_for_json(valid_signals[:200])}
with open(OUT_DIR/"backtest_strat1.json","w") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(f"\n💾 已保存: backtest_strat1.json", flush=True)
