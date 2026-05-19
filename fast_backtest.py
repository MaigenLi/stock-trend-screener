#!/usr/bin/env python3
"""
screen_trend 快速回测 — 每个股票只加载一次，处理所有日期
"""
import sys, json, math, time, re, smtplib, os
from pathlib import Path
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
from gain_turnover import load_raw_history, rolling_mean, get_all_stock_codes

# ── 邮件 ──────────────────────────────────────────
def send_email(to_addr, subject, body):
    env = {}
    with open(os.path.expanduser('~/.openclaw/.env')) as f:
        for line in f:
            line = line.strip()
            if '=' in line and not line.startswith('#'):
                k, v = line.split('=', 1)
                env[k] = v
    user = env.get('QQ_EMAIL', 'maigenmuzi@qq.com')
    pwd = env.get('QQ_PASS', '')
    if not pwd:
        print("No email credentials found")
        return
    msg = MIMEMultipart()
    msg['From'] = user
    msg['To'] = to_addr
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'html', 'utf-8'))
    try:
        server = smtplib.SMTP('smtp.qq.com', 587)
        server.starttls()
        server.login(user, pwd)
        server.sendmail(user, [to_addr], msg.as_string())
        server.quit()
        print(f"Email sent to {to_addr}")
    except Exception as e:
        print(f"Email error: {e}")

# ── 固定参数（来自 commit 94ea69e）───────────────
TURN_BASE, TURN_FLOOR, TURN_CAP_REF, TURN_POWER = 3.0, 1.5, 100.0, 0.48
GAIN5_MIN, GAIN5_MAX = 1.5, 25.0
GAIN20_MIN, GAIN20_MAX = 7.0, 60.0
MA5_DAILY = 0.3
CLOSE_NEAR_HIGH_RATIO = 0.95
VOL_RATIO5_MAX = 1.10
SLOPE_MA5_ATR = 0.05
SLOPE_MA10_ATR = 0.05
SLOPE_MA20_ATR = 0.10
SLOPE_MA5_ATR_MAX = 0.685
SLOPE_MA10_ATR_MAX = 0.531
SLOPE_MA20_ATR_MAX = 0.431

def get_turnover_min(cap_yi):
    if cap_yi <= 0: return TURN_BASE
    return max(TURN_FLOOR, TURN_BASE * (TURN_CAP_REF / max(cap_yi, 1)) ** TURN_POWER)

def check_ma_fast(close, turnover, outs, atr_pct, dates, i, slope_cache):
    """快速版本，只在需要时计算斜率"""
    n = len(close)
    if i < 20 or n < 67: return None

    ma5 = rolling_mean(close, 5)
    ma10 = rolling_mean(close, 10)
    ma20 = rolling_mean(close, 20)
    ma60 = rolling_mean(close, 60)

    # 只在需要的索引计算斜率
    needed = set()
    for di in [i, i-1, i-2]:
        if di >= 2: needed.add(di)
    # ma5_dir5 需要 i 和 i-4
    for di in [i, i-4]:
        if di >= 2: needed.add(di)

    for di in needed:
        if di not in slope_cache:
            x = np.array([0.,1.,2.])
            for arr, key_prefix in [(ma5,'m5'),(ma10,'m10'),(ma20,'m20')]:
                if di < 2: continue
                y = arr[di-2:di+1].astype(float)
                ym = y.mean()
                num = np.sum((x-x.mean())*(y-ym))
                den = np.sum((x-x.mean())**2)
                slope = (num/den)/(ym+1e-12)*100.0 if den > 1e-12 else 0.0
                slope_cache[(key_prefix, di)] = slope / (atr_pct[di] + 1e-12)

    def get_slope(key_prefix, di):
        return slope_cache.get((key_prefix, di), -999.0)

    ma5_slope_atr_i = get_slope('m5', i)
    ma10_slope_atr_i = get_slope('m10', i)
    ma20_slope_atr_i = get_slope('m20', i)

    # ── 条件① MA排列 ─────────────────────────────
    if not (ma5[i] > ma10[i] > ma20[i] and ma5[i] > ma60[i] and ma10[i] > ma60[i]):
        return None
    if ma20[i] < ma60[i] and not (ma5[i] > ma5[i-1]):
        return None
    if close[i] < ma5[i] and close[i-1] < ma5[i-1] and not (ma5[i] > ma5[i-1]):
        return None

    # ── 条件②-1 斜率/ATR ──────────────────────────
    if not (SLOPE_MA5_ATR <= ma5_slope_atr_i < SLOPE_MA5_ATR_MAX): return None
    if not (SLOPE_MA10_ATR <= ma10_slope_atr_i < SLOPE_MA10_ATR_MAX): return None
    if not (SLOPE_MA20_ATR <= ma20_slope_atr_i < SLOPE_MA20_ATR_MAX): return None

    # ── 条件②-2 日变 ────────────────────────────
    ma5_d = (ma5[i]/ma5[i-1]-1)*100 if i>=1 else -999
    ma10_d = (ma10[i]/ma10[i-1]-1)*100 if i>=1 else -999
    ma20_d = (ma20[i]/ma20[i-1]-1)*100 if i>=1 else -999
    if not (ma5_d >= MA5_DAILY and ma10_d >= 0.4 and ma20_d >= 0.4): return None

    # ── 条件③ 市值 ──────────────────────────────
    cap = close[i] * outs[i] / 1e8 if (outs is not None and not np.isnan(outs[i]) and outs[i]>0) else 30.0
    if cap < 25.0: return None

    # ── 条件④ 换手率 ───────────────────────────
    turn_avg = np.mean(turnover[i-4:i+1])
    turn_now = float(turnover[i])
    turn_prev = float(turnover[i-1]) if i>=1 else 0.0
    gain_day = (close[i]/close[i-1]-1)*100 if i>=1 else 0.0
    turn_thresh = get_turnover_min(cap)
    if not (turn_avg >= turn_thresh or (turn_now > TURN_FLOOR and turn_now > turn_prev and gain_day >= 3)):
        return None

    # ── 条件⑤ 20日涨幅 ─────────────────────────
    gain20 = (close[i]/close[i-20]-1)*100
    gain5 = (close[i]/close[i-5]-1)*100
    ok5 = GAIN20_MAX >= gain20 >= GAIN20_MIN
    if not ok5:
        if not (GAIN20_MAX >= gain20 >= GAIN20_MIN and gain5 >= 5.0):
            return None

    # ── 条件⑥ 当日涨幅 ──────────────────────────
    if not (-2.0 <= gain_day <= 20.1): return None

    # ── 条件⑦ 收盘近20日最高 ───────────────────
    high20 = float(np.max(np.array([close[max(0,i-19):i+1]])))
    if close[i] < high20 * CLOSE_NEAR_HIGH_RATIO: return None

    # ── 条件⑧ 5日涨幅 ──────────────────────────
    if not (GAIN5_MIN <= gain5 <= GAIN5_MAX): return None

    # ── 条件⑨ 量比 ─────────────────────────────
    vol5_avg_t = np.mean(turnover[i-4:i+1])
    vol5_prev_t = np.mean(turnover[i-9:i-4]) if i >= 10 else np.mean(turnover[max(0,i-9):i])
    vr5 = vol5_avg_t / (vol5_prev_t + 1e-12)
    if vr5 > VOL_RATIO5_MAX: return None

    # ── 辅助指标 ────────────────────────────────
    deltas = np.diff(close)
    gains_w = np.where(deltas>0, deltas, 0)
    losses_w = np.where(deltas<0, -deltas, 0)
    avg_g = np.mean(gains_w[max(0,i-15):i])
    avg_l = np.mean(losses_w[max(0,i-15):i])
    rsi = 100 - 100/(1 + avg_g/avg_l) if avg_l > 0 else 100
    ma5_dir5 = ma5[i] - ma5[i-4] if i >= 4 else 0
    ma20_dir5 = ma20[i] - ma20[i-4] if i >= 4 else 0

    return {
        'gain5': round(gain5, 1),
        'gain20': round(gain20, 1),
        'gain_day': round(gain_day, 1),
        'rsi': round(rsi, 1),
        'vr5': round(vr5, 3),
        'ma5_dir5': round(ma5_dir5, 2),
        'ma20_dir5': round(ma20_dir5, 2),
        'mktcap': round(cap, 0),
        'vol_avg5': round(np.mean(turnover[i-4:i+1]), 2),
    }

def simulate_buy(code, buy_date, buy_price, dates_list, close_arr, ma5_arr):
    """模拟卖出：MA5下跌或持满25日"""
    dm = {d: i for i, d in enumerate(dates_list)}
    if buy_date not in dm: return None
    start_i = dm[buy_date]
    n = len(close_arr)
    for i in range(start_i, min(start_i + 25, n)):
        hold_days = i - start_i + 1
        if hold_days >= 25:
            return (str(dates_list[i])[:10], float(close_arr[i]), hold_days,
                    (float(close_arr[i]) / buy_price - 1) * 100, "25日卖出")
        if i > start_i and ma5_arr[i] < ma5_arr[i-1]:
            return (str(dates_list[i])[:10], float(close_arr[i]), hold_days,
                    (float(close_arr[i]) / buy_price - 1) * 100, "MA5卖出")
    return None

# ── 主流程 ────────────────────────────────────────
START = "2025-12-17"
END = "2026-04-30"
NAMES = {}

print("加载股票名称...")
try:
    with open('/home/lyc/stock_code/results/all_stock_names_final.json') as f:
        d = json.load(f)
    stocks = d.get('stocks', {})
    NAMES = {k: v['name'] for k, v in stocks.items()}
except:
    pass

ALL_CODES = list(NAMES.keys())
print(f"股票数量: {len(ALL_CODES)}")

print("构建日历...")
df0 = load_raw_history(ALL_CODES[0], end_date=END)
all_dates = sorted([str(d)[:10] for d in df0['date'].values])
CAL = [d for d in all_dates if START <= d <= END]
print(f"日历: {len(CAL)} 天 ({CAL[0]} ~ {CAL[-1]})")

# 扫描日期：每3天一次
SCAN_DATES = [CAL[i] for i in range(0, len(CAL), 3)]
print(f"扫描日期: {len(SCAN_DATES)} 个")

# ── 阶段1: 收集信号 ──────────────────────────────
print("\n阶段1: 扫描信号...")
t0 = time.time()
all_signals = []  # [(date, code, metrics_dict), ...]

for ci, code in enumerate(ALL_CODES):
    if ci % 500 == 0:
        print(f"  进度: {ci}/{len(ALL_CODES)} ({time.time()-t0:.0f}s)")

    try:
        df = load_raw_history(code, end_date=END)
        if df is None or len(df) < 67: continue
        close = df['close'].values.astype(float)
        vol = df['true_turnover'].values.astype(float)
        outs = df['outstanding_share'].values.astype(float) if 'outstanding_share' in df.columns else None
        dates_arr = np.array([str(d)[:10] for d in df['date'].values])
        dm = {d: i for i, d in enumerate(dates_arr)}

        # ATR% — 简化版：日内振幅均值/收盘
        high_arr = df['high'].values.astype(float)
        low_arr = df['low'].values.astype(float)
        tr = np.zeros(len(close))
        tr[0] = high_arr[0] - low_arr[0]
        for i in range(1, len(close)):
            tr[i] = max(high_arr[i]-low_arr[i], abs(high_arr[i]-close[i-1]), abs(low_arr[i]-close[i-1]))
        atr_raw = np.zeros(len(close))
        atr_raw[0] = tr[0]
        for i in range(1, len(close)):
            atr_raw[i] = (atr_raw[i-1]*13 + tr[i]) / 14
        atr_pct_full = atr_raw / (close + 1e-12) * 100

        slope_cache = {}
        for scan_date in SCAN_DATES:
            if scan_date not in dm: continue
            i = dm[scan_date]
            m = check_ma_fast(close, vol, outs, atr_pct_full, dates_arr, i, slope_cache)
            if m:
                m['code'] = code
                m['name'] = NAMES.get(code, '')
                all_signals.append((scan_date, code, m))
    except:
        pass

print(f"  找到 {len(all_signals)} 个信号 in {time.time()-t0:.0f}s")

# ── 阶段2: 验证买入条件 ─────────────────────────
print("\n阶段2: 验证买入条件...")
buy_trades = []

for scan_date, code, m in all_signals:
    try:
        df = load_raw_history(code, end_date=END)
        if df is None: continue
        close = df['close'].values.astype(float)
        open_ = df['open'].values.astype(float)
        dates_arr = [str(d)[:10] for d in df['date'].values]
        dm = {d: i for i, d in enumerate(dates_arr)}

        if scan_date not in dm: continue
        i = dm[scan_date]
        if i >= len(df) - 1: continue  # 无下一天

        next_i = i + 1
        open_pct = (open_[next_i] / close[i] - 1) * 100
        if not (-2.0 <= open_pct <= 5.0): continue

        buy_price = float(open_[next_i])
        buy_date = dates_arr[next_i]

        m2 = dict(m)
        m2['scan_date'] = scan_date
        m2['buy_date'] = buy_date
        m2['buy_price'] = round(buy_price, 2)
        m2['open_pct'] = round(open_pct, 2)
        buy_trades.append(m2)
    except:
        pass

print(f"  有效买入: {len(buy_trades)} 笔")

# ── 阶段3: 模拟持股 ─────────────────────────────
print("\n阶段3: 模拟持股...")
results = []

for m in buy_trades:
    try:
        df = load_raw_history(m['code'], end_date=END)
        if df is None: continue
        close = df['close'].values.astype(float)
        open_ = df['open'].values.astype(float)
        ma5 = rolling_mean(close, 5)
        dates_arr = [str(d)[:10] for d in df['date'].values]

        res = simulate_buy(m['code'], m['buy_date'], m['buy_price'], dates_arr, close, ma5)
        if res:
            sell_date, sell_price, hold_days, ret_pct, reason = res
            m3 = dict(m)
            m3['sell_date'] = sell_date
            m3['sell_price'] = round(sell_price, 2)
            m3['hold_days'] = hold_days
            m3['ret_pct'] = round(ret_pct, 2)
            m3['sell_reason'] = reason
            results.append(m3)
    except:
        pass

print(f"  完成: {len(results)} 笔交易")

# ── 统计分析 ──────────────────────────────────────
print(f"\n{'='*60}")
print(f"总交易: {len(results)} 笔")
if not results:
    print("无交易结果！")
else:
    rets = [t['ret_pct'] for t in results]
    wins = [t for t in results if t['ret_pct'] > 0]
    losses = [t for t in results if t['ret_pct'] <= 0]
    print(f"盈利: {len(wins)} ({len(wins)*100/len(results):.1f}%)")
    print(f"亏损: {len(losses)} ({len(losses)*100/len(results):.1f}%)")
    if rets:
        print(f"平均收益: {np.mean(rets):.2f}%")
        print(f"中位数: {np.median(rets):.2f}%")
        print(f"最大盈利: {max(rets):.2f}%")
        print(f"最大亏损: {min(rets):.2f}%")
        pos_rets = [r for r in rets if r > 0]
        neg_rets = [r for r in rets if r < 0]
        print(f"盈亏比: {np.mean(pos_rets):.2f}% / {abs(np.mean(neg_rets)):.2f}%")
        std = np.std(rets)
        sharpe = (np.mean(rets) - 0.03) / std * math.sqrt(len(rets)) if std > 0 else 0
        print(f"夏普比: {sharpe:.2f}")

    print(f"\n按持股天数:")
    for d in [5,10,15,20,25]:
        g = [t for t in results if t['hold_days'] <= d]
        if g:
            avg = np.mean([t['ret_pct'] for t in g])
            wr = len([t for t in g if t['ret_pct']>0])/len(g)*100
            print(f"  ≤{d}日: {len(g)}笔, avg={avg:.2f}%, win={wr:.0f}%")

    print(f"\n按gain20分组:")
    for lo, hi in [(0,20),(20,40),(40,60),(60,999)]:
        g = [t for t in results if lo<=t['gain20']<hi]
        if g:
            avg = np.mean([t['ret_pct'] for t in g])
            wr = len([t for t in g if t['ret_pct']>0])/len(g)*100
            print(f"  {lo}~{hi}%: {len(g)}笔, avg={avg:.2f}%, win={wr:.0f}%")

    print(f"\n按RSI分组:")
    for lo, hi in [(0,70),(70,80),(80,999)]:
        g = [t for t in results if lo<=t['rsi']<hi]
        if g:
            avg = np.mean([t['ret_pct'] for t in g])
            wr = len([t for t in g if t['ret_pct']>0])/len(g)*100
            print(f"  {lo}~{hi}: {len(g)}笔, avg={avg:.2f}%, win={wr:.0f}%")

    print(f"\n按ma5_dir5分组:")
    for lo, hi in [(-999,3),(3,5),(5,8),(8,999)]:
        g = [t for t in results if lo<=t['ma5_dir5']<hi]
        if g:
            avg = np.mean([t['ret_pct'] for t in g])
            wr = len([t for t in g if t['ret_pct']>0])/len(g)*100
            print(f"  {lo:.0f}~{hi:.0f}: {len(g)}笔, avg={avg:.2f}%, win={wr:.0f}%")

    print(f"\n按vr5分组:")
    for lo, hi in [(0,0.9),(0.9,1.0),(1.0,1.1),(1.1,999)]:
        g = [t for t in results if lo<=t['vr5']<hi]
        if g:
            avg = np.mean([t['ret_pct'] for t in g])
            wr = len([t for t in g if t['ret_pct']>0])/len(g)*100
            print(f"  {lo:.1f}~{hi:.1f}: {len(g)}笔, avg={avg:.2f}%, win={wr:.0f}%")

    print(f"\nTop10赚钱:")
    for t in sorted(results, key=lambda x: -x['ret_pct'])[:10]:
        print(f"  {t['code']} {t['name']} 信号日gain20={t['gain20']}% RSI={t['rsi']} vr5={t['vr5']} ma5_d5={t['ma5_dir5']} 买入{t['buy_date']} 持{t['hold_days']}日 {t['ret_pct']:+.2f}%")

    print(f"\nTop10亏钱:")
    for t in sorted(results, key=lambda x: x['ret_pct'])[:10]:
        print(f"  {t['code']} {t['name']} 信号日gain20={t['gain20']}% RSI={t['rsi']} vr5={t['vr5']} ma5_d5={t['ma5_dir5']} 买入{t['buy_date']} 持{t['hold_days']}日 {t['ret_pct']:+.2f}%")

    # 保存
    out = {'summary': {
        'period': f'{START} ~ {END}',
        'total_trades': len(results),
        'win_rate': len(wins)*100/len(results),
        'avg_return': round(np.mean(rets), 2),
        'median_return': round(np.median(rets), 2),
        'max_win': round(max(rets), 2),
        'max_loss': round(min(rets), 2),
    }, 'trades': results}
    with open('output/backtest_trend_results.json', 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # 发送邮件
    body = f"""
<html><body>
<h2>screen_trend 策略回测报告</h2>
<p><b>回测区间</b>: {START} ~ {END}</p>
<p><b>扫描日期数</b>: {len(SCAN_DATES)} 个 (每3天扫一次)</p>
<p><b>信号总数</b>: {len(all_signals)} 个 → 有效买入 {len(buy_trades)} 笔 → 完成 {len(results)} 笔交易</p>

<h3>总体统计</h3>
<table border='1' cellpadding='6'>
<tr><td>总交易</td><td>{len(results)} 笔</td></tr>
<tr><td>盈利/亏损</td><td>{len(wins)} ({len(wins)*100/len(results):.1f}%) / {len(losses)} ({len(losses)*100/len(results):.1f}%)</td></tr>
<tr><td>平均收益</td><td>{np.mean(rets):.2f}%</td></tr>
<tr><td>中位数收益</td><td>{np.median(rets):.2f}%</td></tr>
<tr><td>最大盈利</td><td>{max(rets):.2f}%</td></tr>
<tr><td>最大亏损</td><td>{min(rets):.2f}%</td></tr>
<tr><td>夏普比</td><td>{sharpe:.2f}</td></tr>
</table>

<h3>按持股天数</h3>
<table border='1' cellpadding='6'>
<tr><th>天数</th><th>笔数</th><th>平均收益</th><th>胜率</th></tr>
"""
    for d in [5,10,15,20,25]:
        g = [t for t in results if t['hold_days'] <= d]
        if g:
            avg = np.mean([t['ret_pct'] for t in g])
            wr = len([t for t in g if t['ret_pct']>0])/len(g)*100
            body += f"<tr><td>≤{d}日</td><td>{len(g)}</td><td>{avg:.2f}%</td><td>{wr:.0f}%</td></tr>\n"
    body += "</table>\n"

    body += "<h3>按信号日gain20分组</h3><table border='1' cellpadding='6'><tr><th>gain20</th><th>笔数</th><th>平均收益</th><th>胜率</th></tr>\n"
    for lo, hi in [(0,20),(20,40),(40,60),(60,999)]:
        g = [t for t in results if lo<=t['gain20']<hi]
        if g:
            avg = np.mean([t['ret_pct'] for t in g])
            wr = len([t for t in g if t['ret_pct']>0])/len(g)*100
            body += f"<tr><td>{lo}~{hi}%</td><td>{len(g)}</td><td>{avg:.2f}%</td><td>{wr:.0f}%</td></tr>\n"
    body += "</table>\n"

    body += "<h3>按RSI分组</h3><table border='1' cellpadding='6'><tr><th>RSI</th><th>笔数</th><th>平均收益</th><th>胜率</th></tr>\n"
    for lo, hi in [(0,70),(70,80),(80,999)]:
        g = [t for t in results if lo<=t['rsi']<hi]
        if g:
            avg = np.mean([t['ret_pct'] for t in g])
            wr = len([t for t in g if t['ret_pct']>0])/len(g)*100
            body += f"<tr><td>{lo}~{hi}</td><td>{len(g)}</td><td>{avg:.2f}%</td><td>{wr:.0f}%</td></tr>\n"
    body += "</table>\n"

    body += "<h3>按ma5_dir5分组</h3><table border='1' cellpadding='6'><tr><th>ma5_dir5</th><th>笔数</th><th>平均收益</th><th>胜率</th></tr>\n"
    for lo, hi in [(-999,3),(3,5),(5,8),(8,999)]:
        g = [t for t in results if lo<=t['ma5_dir5']<hi]
        if g:
            avg = np.mean([t['ret_pct'] for t in g])
            wr = len([t for t in g if t['ret_pct']>0])/len(g)*100
            body += f"<tr><td>{lo:.0f}~{hi:.0f}</td><td>{len(g)}</td><td>{avg:.2f}%</td><td>{wr:.0f}%</td></tr>\n"
    body += "</table>\n"

    body += "<h3>按vr5分组</h3><table border='1' cellpadding='6'><tr><th>vr5</th><th>笔数</th><th>平均收益</th><th>胜率</th></tr>\n"
    for lo, hi in [(0,0.9),(0.9,1.0),(1.0,1.1),(1.1,999)]:
        g = [t for t in results if lo<=t['vr5']<hi]
        if g:
            avg = np.mean([t['ret_pct'] for t in g])
            wr = len([t for t in g if t['ret_pct']>0])/len(g)*100
            body += f"<tr><td>{lo:.1f}~{hi:.1f}</td><td>{len(g)}</td><td>{avg:.2f}%</td><td>{wr:.0f}%</td></tr>\n"
    body += "</table>\n"

    body += "<h3>Top10赚钱交易</h3><table border='1' cellpadding='6'><tr><th>代码</th><th>名称</th><th>gain20</th><th>RSI</th><th>vr5</th><th>ma5_d5</th><th>买入日</th><th>持股</th><th>收益</th></tr>\n"
    for t in sorted(results, key=lambda x: -x['ret_pct'])[:10]:
        body += f"<tr><td>{t['code']}</td><td>{t['name']}</td><td>{t['gain20']}%</td><td>{t['rsi']}</td><td>{t['vr5']}</td><td>{t['ma5_dir5']}</td><td>{t['buy_date']}</td><td>{t['hold_days']}日</td><td>{t['ret_pct']:+.2f}%</td></tr>\n"
    body += "</table>\n"

    body += "<h3>Top10亏钱交易</h3><table border='1' cellpadding='6'><tr><th>代码</th><th>名称</th><th>gain20</th><th>RSI</th><th>vr5</th><th>ma5_d5</th><th>买入日</th><th>持股</th><th>收益</th></tr>\n"
    for t in sorted(results, key=lambda x: x['ret_pct'])[:10]:
        body += f"<tr><td>{t['code']}</td><td>{t['name']}</td><td>{t['gain20']}%</td><td>{t['rsi']}</td><td>{t['vr5']}</td><td>{t['ma5_dir5']}</td><td>{t['buy_date']}</td><td>{t['hold_days']}日</td><td>{t['ret_pct']:+.2f}%</td></tr>\n"
    body += "</table>\n"

    body += f"<p>详细数据: output/backtest_trend_results.json</p></body></html>"

    send_email('maigenmuzi@qq.com', 'screen_trend 策略回测报告 2026-05-19', body)
    print("\n邮件已发送!")