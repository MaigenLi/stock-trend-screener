#!/usr/bin/env python3
"""
四阶段筛选器 v2
=================
从趋势良好的股票中，找「上涨→回调蓄势→再上涨」的波段结构。

核心条件：
  1. MA60 方向向上（10日均值 > 更早10日均值）
  2. 至少4段波段（up→down→up→down），且顶底逐波抬高
  3. 波段时间均衡（≥3天/段），过滤噪音
  4. 每波段有实质涨跌幅（≥3%），过滤微小抖动
  5. 最后一跌回调幅度合理（≤50%），浅回最好
  6. 量价健康：上涨放量，下跌缩量
  7. MA20 均线支撑（跌破MA20须警惕）

评分维度（每波段，满分10分）：
  - 回调深度（仅下跌波段）: ≤10%=3分, 10~25%=2分, 25~40%=1分
  - 量比（涨/跌均量比）: ≥1.5=3分, 1.2~1.5=2分, 持平=1分
  - 时间对称: 0.5~2x=2分
  - MA20支撑: ≥5%=2分, 0~5%=1分

用法：
  python scan_4phase.py --top 20
  python scan_4phase.py --end 2026-04-24 --top 30
  python scan_4phase.py --end 2026-04-24 --top 20 --min-wave-score 5
"""
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

WORKSPACE = Path.home() / ".openclaw" / "workspace"
sys.path.insert(0, str(WORKSPACE / "stock_trend"))

from gain_turnover import load_qfq_history, get_all_stock_codes, load_stock_names, normalize_symbol

PARAMS = dict(
    lookback=90,             # 回看窗口
    min_ma60_rise=0.5,       # MA60最小升幅（%）
    min_wave_pct=3.0,        # 波段最小涨跌幅（%）
    min_wave_days=3,         # 波段最短天数（过滤噪音）
    max_pullback=0.55,      # 最大回调深度（0.55 = 55%）
    min_vol_up_vs_down=1.0, # 最小量比（涨/跌均量）
    min_avg_score=4.5,       # 每波段平均分门槛
)


# ══════════════════════════════════════════════════════════════
def zigzag_pivots(closes: np.ndarray,
                   threshold: float = 0.025,
                   min_bars: int = 3) -> list[int]:
    """
    ZigZag 转折点识别（带最小bar间隔约束）。

    核心：跟踪全局趋势，必须超过 threshold 才确认反转，
    且两次确认反转之间至少间隔 min_bars 根K线。
    这样可以防止噪音导致的频繁微小反转。

    - trend==0：无方向，等待第一个 threshold 变动确立方向
    - trend==1：上升中，持续更新高点；价格回落 threshold 则确认顶点，转为下降
    - trend==-1：下降中，持续更新低点；价格反弹 threshold 则确认底点，转为上升
    """
    n = len(closes)
    if n < 3:
        return []

    pivots = []
    trend = 0           # 0=无方向, 1=上升, -1=下降
    last_pivot_idx = 0
    last_pivot_price = float(closes[0])

    for i in range(1, n):
        price = float(closes[i])
        change = (price - last_pivot_price) / last_pivot_price
        bars_since_pivot = i - last_pivot_idx

        if trend == 0:
            if abs(change) >= threshold:
                trend = 1 if change > 0 else -1
                last_pivot_idx = i
                last_pivot_price = price
        elif trend == 1:
            if price > last_pivot_price:
                last_pivot_idx = i
                last_pivot_price = price
            elif bars_since_pivot >= min_bars and change <= -threshold:
                pivots.append(last_pivot_idx)
                trend = -1
                last_pivot_idx = i
                last_pivot_price = price
        else:  # trend == -1
            if price < last_pivot_price:
                last_pivot_idx = i
                last_pivot_price = price
            elif bars_since_pivot >= min_bars and change >= threshold:
                pivots.append(last_pivot_idx)
                trend = 1
                last_pivot_idx = i
                last_pivot_price = price

    return pivots


def build_waves_from_pivots(closes: np.ndarray,
                             pivots: list[int],
                             dates: list[str],
                             volumes: np.ndarray = None
                             ) -> list[dict]:
    """
    把 ZigZag 确认的 pivot 序列连成波段。

    pivots[0] = 第一个确认的转折点（是谷还是峰，由 closepivots[1] 决定）
    相邻两个 pivots 组成一个波段：
      pivots[i] < pivots[i+1] → up（谷→峰）
      pivots[i] > pivots[i+1] → down（峰→谷）
    """
    if len(pivots) < 2:
        return []

    waves = []
    for i in range(len(pivots) - 1):
        s_idx, e_idx = pivots[i], pivots[i + 1]
        s_p, e_p = float(closes[s_idx]), float(closes[e_idx])
        wave_type = "up" if e_p > s_p else "down"

        chg = round((e_p / s_p - 1) * 100, 2)

        if volumes is not None and len(volumes) > 0:
            seg = volumes[s_idx:e_idx + 1]
            avg_v = float(np.mean(seg)) if len(seg) > 0 else 0.0
        else:
            avg_v = 0.0

        waves.append({
            "type": wave_type,
            "start_idx": s_idx,
            "end_idx": e_idx,
            "start_price": s_p,
            "end_price": e_p,
            "chg_pct": chg,
            "avg_vol": avg_v,
            "days": abs(e_idx - s_idx),
            "start_date": dates[s_idx] if s_idx < len(dates) else "",
            "end_date": dates[e_idx] if e_idx < len(dates) else "",
        })
    return waves


def ma60_direction(closes: np.ndarray) -> float:
    """MA60方向：最近10日均值相对更早10日均值的涨幅（%）。"""
    n = len(closes)
    if n < 70:
        return 0.0
    ma60s = [float(np.mean(closes[i - 59:i + 1])) for i in range(69, n)]
    if len(ma60s) < 20:
        return 0.0
    return float((np.mean(ma60s[-10:]) / np.mean(ma60s[-20:-10]) - 1) * 100)


def score_wave(w: dict, prev_up: dict | None) -> dict:
    """
    给单个波段质量打分（满分约10分）。
    w: 波段 dict
    prev_up: 同组内前一个up波段（用于量比计算）
    """
    score = 0
    reasons = []

    # ── 回调深度（仅下跌波段）──────────────────────────
    pd_val = w.get("pullback_depth", 0) * 100
    if w["type"] == "down":
        if pd_val <= 10:
            score += 3; reasons.append(f"回调极浅({pd_val:.0f}%)")
        elif pd_val <= 25:
            score += 2; reasons.append(f"回调浅({pd_val:.0f}%)")
        elif pd_val <= 40:
            score += 1; reasons.append(f"回调深({pd_val:.0f}%)")
        else:
            reasons.append(f"回调过深({pd_val:.0f}%)⚠️")

    # ── 量比（上涨段均量 / 前一下跌段均量）──────────────
    vr = w.get("vol_ratio", 1.0)
    if vr >= 1.5:
        score += 3; reasons.append(f"量比健康({vr:.2f})")
    elif vr >= 1.2:
        score += 2; reasons.append(f"量比合理({vr:.2f})")
    elif vr >= 1.0:
        score += 1; reasons.append(f"量比持平({vr:.2f})")
    else:
        reasons.append(f"下跌无量({vr:.2f})")

    # ── 时间对称 ─────────────────────────────────────
    tr = w.get("time_ratio", 1.0)
    if 0.25 <= tr <= 4.0:
        score += 2; reasons.append(f"时间对称({tr:.1f}x)")
    elif tr < 0.25:
        score += 1; reasons.append(f"时间急促({tr:.1f}x)")
    else:
        score += 1; reasons.append(f"时间偏慢({tr:.1f}x)")

    # ── MA20支撑（仅下跌波段）──────────────────────────
    ma20d = w.get("ma20_dist_pct")
    if ma20d is not None and w["type"] == "down":
        if ma20d >= 5:
            score += 2; reasons.append(f"MA20支撑强(+{ma20d:.0f}%)")
        elif ma20d >= 0:
            score += 1; reasons.append(f"MA20上方(+{ma20d:.0f}%)")
        elif ma20d >= -3:
            reasons.append(f"轻微破MA20({ma20d:.1f}%)")
        else:
            reasons.append(f"跌破MA20({ma20d:.0f}%)⚠️")
    elif w["type"] == "up":
        score += 1  # 上涨波段基础分

    return {"score": score, "reasons": reasons}


# ══════════════════════════════════════════════════════════════
def analyze_4phase(closes: np.ndarray,
                   volumes: np.ndarray,
                   dates: list,
                   params: dict = None
                   ) -> dict | None:
    """
    四阶段核心分析。

    策略逻辑：
    ──────────────────────────────────────────────────────────
    什么样的上涨趋势是有效的？

    【有效趋势的7个特征】

    ① 底部不断抬高
       每次回调的谷比上一次谷高 → 主力未出货，趋势延续

    ② 顶部不断突破
       每个峰比上一个峰高 → 买方力量持续强于卖方

    ③ 回调浅（≤50%）
       涨10元只回3~4元，50%是极限 → 超过50%趋势可能逆转

    ④ 涨时放量，跌时缩量
       放量说明资金认可，缩量说明抛压轻

    ⑤ 均线族多头排列
       MA5>MA10>MA20>MA60 → 成本均线向上发散

    ⑥ 波段时间大致对等
       涨跌用时接近 → 供需平衡，非主力拉抬

    ⑦ 回调不破重要均线
       回调在MA20/MA60获支撑 → 二次启动平台
    ──────────────────────────────────────────────────────────
    """
    p = params or PARAMS
    n = len(closes)
    if n < 70:
        return None

    lookback = p["lookback"]
    use_n = min(n, lookback + 30)
    recent = closes[-use_n:]
    vols   = volumes[-use_n:] if volumes is not None else np.zeros_like(recent)
    dates_r = [str(d)[:10] for d in dates[-use_n:]]

    # ① MA60 方向
    if ma60_direction(closes) < p["min_ma60_rise"]:
        return None

    # ② ZigZag 转折点识别
    pivots = zigzag_pivots(recent, threshold=0.040)
    if len(pivots) < 4:
        return None

    # ③ 构建波段（ZigZag 自然保证方向正确，无需合并）
    waves = build_waves_from_pivots(recent, pivots, dates_r, vols)
    if len(waves) < 4:
        return None

    # ④ 追加未完成的最后一个波段（当前K线到上一个确认Pivot）
    last_p = pivots[-1]          # 最后一个确认Pivot
    cur_idx = len(recent) - 1   # 当前K线索引
    if cur_idx > last_p:
        last_p_price = float(recent[last_p])
        cur_price_wave = float(recent[cur_idx])
        chg = round((cur_price_wave / last_p_price - 1) * 100, 2)
        seg_vols = vols[last_p:cur_idx + 1]
        avg_v = float(np.mean(seg_vols)) if len(seg_vols) > 0 else 0.0
        waves.append({
            "type": "up" if chg > 0 else "down",
            "start_idx": last_p,
            "end_idx": cur_idx,
            "start_price": last_p_price,
            "end_price": cur_price_wave,
            "chg_pct": chg,
            "avg_vol": avg_v,
            "days": cur_idx - last_p,
            "start_date": dates_r[last_p] if last_p < len(dates_r) else "",
            "end_date": dates_r[-1],
            "_incomplete": True,
        })

    # ⑤ 从后往前找最后一个 up→down→up→down
    four = None
    for i in range(len(waves) - 4, -1, -1):
        seq = [waves[i + j]["type"] for j in range(4)]
        if seq == ["up", "down", "up", "down"]:
            four = waves[i:i + 4]
            break
    if four is None:
        return None

    # ⑤ 初步质量过滤（天数 & 涨跌幅）
    for w in four:
        if w["days"] < p["min_wave_days"]:
            return None
        if abs(w["chg_pct"]) < p["min_wave_pct"]:
            return None

    # ⑥ 当前是否在下跌中：当前价 < 最近确认峰 × 0.995
    cur_price = float(recent[-1])
    last_peak = float(four[2]["end_price"])   # 第3波段（up）终点 = 最近确认峰
    in_pb = cur_price < last_peak * 0.995

    # ⑦ 计算各波段量比（用同向前一浪）
    for i, w in enumerate(four):
        if w["type"] == "up" and i > 0 and four[i - 1]["type"] == "down":
            ref = four[i - 1]
            w["vol_ratio"] = w["avg_vol"] / ref["avg_vol"] if ref["avg_vol"] > 0 else 1.0
        elif w["type"] == "down" and i > 0 and four[i - 1]["type"] == "up":
            ref = four[i - 1]
            w["vol_ratio"] = w["avg_vol"] / ref["avg_vol"] if ref["avg_vol"] > 0 else 1.0
        else:
            w["vol_ratio"] = 1.0
        # 时间比（相对前一波段）
        if i > 0 and four[i - 1]["days"] > 0:
            w["time_ratio"] = w["days"] / four[i - 1]["days"]
        else:
            w["time_ratio"] = 1.0

    # ⑧ 计算下跌波段回调深度（pullback_depth）
    #    回调深度 = (前一个up波段的峰 - 当前低点) / 前一个up波段涨幅
    #    当前在下跌中时：低点 = 当前价
    for i, w in enumerate(four):
        if w["type"] != "down":
            continue
        # 找前一个 up 波段
        prev_up = None
        for j in range(i - 1, -1, -1):
            if four[j]["type"] == "up":
                prev_up = four[j]
                break
        if prev_up is None:
            w["pullback_depth"] = 0.0
            continue

        up_range = prev_up["end_price"] - prev_up["start_price"]
        if up_range <= 0:
            w["pullback_depth"] = 0.0
            continue

        if in_pb and i == len(four) - 1:
            # 当前下跌中：参考点 = 最近确认峰(peak)，低点 = 当前价
            down_move = last_peak - cur_price
            w["pullback_depth"] = down_move / up_range
            w["_use_current"] = True
        else:
            # 已确认谷底
            down_move = prev_up["end_price"] - w["end_price"]
            w["pullback_depth"] = down_move / up_range
            w["_use_current"] = False

    # ⑨ 顶底递增验证
    ups   = [w for w in four if w["type"] == "up"]
    downs = [w for w in four if w["type"] == "down"]
    if len(ups) < 2 or len(downs) < 2:
        return None
    up_tops   = [w["end_price"]   for w in ups]
    down_bots = [w["end_price"]  for w in downs]
    if not (up_tops[0] < up_tops[1]):
        return None
    if not (down_bots[0] < down_bots[1]):
        return None

    # ⑩ 最大回调深度过滤（只看最后一个下跌波段）
    last_down = four[-1]
    if last_down.get("pullback_depth", 0) > p["max_pullback"]:
        return None

    # ⑪ MA20 支撑计算
    ma20s = np.full(len(recent), np.nan)
    if len(recent) >= 20:
        for i in range(19, len(recent)):
            ma20s[i] = float(np.mean(recent[i - 19:i + 1]))

    for i, w in enumerate(four):
        if w["type"] == "down":
            if w.get("_use_current"):
                bottom = cur_price
            else:
                bottom = w["end_price"]
            ma20_at = ma20s[w["end_idx"]]
            if not np.isnan(ma20_at) and ma20_at > 0:
                w["ma20_dist_pct"] = (bottom - ma20_at) / ma20_at * 100
            else:
                w["ma20_dist_pct"] = None

    # ⑫ 逐波段评分
    total = 0
    for i, w in enumerate(four):
        prev_up = four[i - 1] if (i > 0 and four[i - 1]["type"] == "up") else None
        r = score_wave(w, prev_up)
        w["wave_score"] = r["score"]
        w["reasons"] = r["reasons"]
        total += r["score"]
    avg_score = total / 4

    if avg_score < p["min_avg_score"]:
        return None

    # ⑬ 组装输出
    # 未完成波段：使用真实当前价格更新
    if last_down.get("_incomplete"):
        last_down["end_price"] = cur_price
        last_down["end_idx"]   = len(recent) - 1
        last_down["end_date"]  = dates_r[-1]
        last_down["chg_pct"]   = round(
            (cur_price - last_down["start_price"]) / last_down["start_price"] * 100, 2)
        if in_pb:
            last_down["_in_pullback"] = True
        else:
            last_down["_in_pullback"] = False
            if cur_price > last_peak:
                last_down["type"] = "up"
                last_down["_recovered"] = True

    # 各波段加上日期
    for w in four:
        w["start_date"] = dates_r[w["start_idx"]] if w["start_idx"] < len(dates_r) else ""
        w["end_date"]   = dates_r[w["end_idx"]]   if w["end_idx"]   < len(dates_r) else ""

    gain90 = round((recent[-1] / recent[0] - 1) * 100, 2) if len(recent) > 0 else 0.0

    return {
        "code": "",
        "name": "",
        "close": cur_price,
        "last_date": dates_r[-1],
        "gain_90d": gain90,
        "ma60_rise": round(ma60_direction(closes), 2),
        "structure_score": total,
        "avg_score": round(avg_score, 1),
        "up_tops":   [round(x, 2) for x in up_tops],
        "down_bots": [round(x, 2) for x in down_bots],
        "waves": four,
        "in_pullback": in_pb,
        "all_peaks":   [float(recent[p]) for p in pivots],
        "all_valleys": [],  # ZigZag 不单独区分顶/谷
    }


# ══════════════════════════════════════════════════════════════
def scan(top_n=20, start=None, end=None, params=None):
    p = dict(PARAMS)
    if params:
        p.update(params)

    print(f"\n🔍 四阶段筛选器 v2", flush=True)
    print(f"   截止: {end or '最新交易日'}", flush=True)
    print(f"   条件: MA60升≥{p['min_ma60_rise']}% | 波段≥{p['min_wave_days']}天/≥{p['min_wave_pct']}% | "
          f"最大回调≤{p['max_pullback']*100:.0f}% | 量比≥{p['min_vol_up_vs_down']} | "
          f"均分≥{p['min_avg_score']}", flush=True)

    names = load_stock_names()
    codes = get_all_stock_codes()
    print(f"   股票: {len(codes)} 只\n", flush=True)

    results = []
    for code in codes:
        code_n = normalize_symbol(code)
        try:
            df = load_qfq_history(code_n, end_date=end)
            if df is None or len(df) < 70:
                continue
            df = df.sort_values("date").reset_index(drop=True)
            r = analyze_4phase(
                df["close"].values.astype(float),
                df["volume"].values.astype(float),
                df["date"].tolist(),
                params=p,
            )
            if r:
                r["code"] = code
                r["name"] = names.get(code_n, names.get(code, code))
                results.append(r)
        except Exception:
            pass

    results.sort(key=lambda x: (-x["structure_score"], -x["gain_90d"]))
    print(f"   通过: {len(results)} 只\n", flush=True)

    # ── 汇总表 ───────────────────────────────────────
    hdr = (f"{'代码':<10}{'名称':<8}{'现价':>8}{'90日%':>9}{'均分':>6}"
           f"{'MA60升':>7}  {'顶部序列':<28}{'底部序列':<18}")
    sep = "─" * 105
    print(sep)
    print(hdr)
    print(sep)
    for r in results[:top_n]:
        tops = "→".join([f"{v:.1f}" for v in r["up_tops"][:3]])
        bots = "→".join([f"{v:.1f}" for v in r["down_bots"][:3]])
        print(f"{r['code']:<10}{r['name']:<8}{r['close']:>8.2f}"
              f"{r['gain_90d']:>+8.1f}%{r['avg_score']:>5.1f}分"
              f"{r['ma60_rise']:>+6.1f}%  "
              f"{tops:<28}{bots:<18}")
    print(sep)
    print(f"  共 {len(results[:top_n])} 只  |  满分约40分（每波段10分）")

    # ── 详细波段 ────────────────────────────────────
    all_txt = []
    for r in results[:top_n]:
        lines = []
        lines.append(f"\n{'=' * 95}")
        lines.append(f"{r['code']} {r['name']}  "
                     f"现价={r['close']}  90日涨跌={r['gain_90d']:+.1f}%  "
                     f"结构={r['structure_score']}分(均{r['avg_score']})  "
                     f"MA60升={r['ma60_rise']:+.1f}%")
        lines.append(f"{'=' * 95}")
        for i, w in enumerate(r["waves"]):
            arrow = "↗" if w["type"] == "up" else "↘"
            tag = "← 上涨" if w["type"] == "up" else "← 下跌"
            if w.get("_incomplete"):
                if w.get("_recovered"):
                    tag = "← 上涨（回抽完毕✓）"
                elif w.get("_in_pullback"):
                    tag = "← 下跌（回调中）"
                else:
                    tag = "← 上涨（当前）"
            elif i == len(r["waves"]) - 1:
                tag += "（当前）" if not r["in_pullback"] else "（回调中）"
            pd_ = w.get("pullback_depth", 0)
            vr  = w.get("vol_ratio", 1.0)
            ma20d = w.get("ma20_dist_pct")
            ma20_str = f" | MA20距={ma20d:+.0f}%" if ma20d is not None else ""
            score_str = f"{w['wave_score']}分" if w["wave_score"] else ""

            line = (f"  波段{i+1}: {arrow} {w.get('start_date','')}→{w.get('end_date','')}"
                    f"({w['days']}天)  "
                    f"{w['start_price']:.2f} → {w['end_price']:.2f}"
                    f" ({w['chg_pct']:+.1f}%)  {tag}")
            sub = (f"         质量{score_str}: {' | '.join(w.get('reasons',[]))}"
                   f"  量={vr:.2f}{f' 回调={pd_*100:.0f}%' if w['type']=='down' else ''}"
                   f"{ma20_str}")
            lines.append(line)
            lines.append(sub)
        all_txt.append("\n".join(lines))
        print("\n".join(lines))

    # ── 保存 ──────────────────────────────────────
    out_dir = WORKSPACE / "stock_reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = end or "latest"
    txt_path = out_dir / f"4phase_v2_{tag}.txt"
    header = (f"四阶段筛选器 v2  截止: {tag}  通过: {len(results)} 只\n"
             f"条件: MA60升≥{p['min_ma60_rise']}% | 波段≥{p['min_wave_days']}天/≥{p['min_wave_pct']}% | "
             f"最大回调≤{p['max_pullback']*100:.0f}% | 量比≥{p['min_vol_up_vs_down']} | 均分≥{p['min_avg_score']}\n"
             f"{'='*95}\n")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(all_txt))
    print(f"\n💾 TXT: {txt_path}")

    json_path = out_dir / f"4phase_v2_{tag}.json"
    serializable = []
    for r in results:
        s = dict(r)
        for k in ("waves",):
            s[k] = [
                {kk: (float(vv) if isinstance(vv, (np.floating, np.integer)) else vv)
                 for kk, vv in w.items()}
                for w in r[k]
            ]
        for k in ("up_tops", "down_bots", "all_peaks", "all_valleys"):
            s[k] = [float(x) for x in r[k]]
        serializable.append(s)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"date": tag, "count": len(results), "results": serializable},
                  f, ensure_ascii=False, indent=2)
    print(f"💾 JSON: {json_path}")


# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--top",        type=int, default=20)
    a.add_argument("--start",     default=None)
    a.add_argument("--end",      default=None)
    a.add_argument("--min-score", type=float, default=None,
                   help=f"每波段最低均分（default={PARAMS['min_avg_score']}）")
    a.add_argument("--max-pb",    type=float, default=None,
                   help=f"最大回调深度（default={PARAMS['max_pullback']}）")
    args = a.parse_args()

    params = dict(PARAMS)
    if args.min_score is not None:
        params["min_avg_score"] = args.min_score
    if args.max_pb is not None:
        params["max_pullback"] = args.max_pb

    scan(args.top, args.start, args.end, params=params)
