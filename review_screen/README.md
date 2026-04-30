# review_screen/ — screen_double 专用子模块

> 本目录为 `screen_double.py` 的配套模块，提供波段识别、指标计算、评分规则等内部组件。

---

## 目录结构

```
review_screen/
├── screen_double.py       # ⭐ 主入口（双层过滤策略扫描器）
├── screen_trend.py        # ⭐ 单层趋势筛选器（备选）
├── eval_double.py         # ⭐ screen_double 信号收益评估
├── indicators.py          # ⭐ 指标计算（MA/MACD/RSI/均线方向）
├── scorer.py              # ⭐ 评分引擎（Layer2 六维评分）
├── filter_rules.py        # ⭐ 过滤规则（Layer1 七条件）
├── scan_4phase.py         # ⭐ 四相位波段识别
└── analyze_T1_winners.py  # T+1 赢家特征分析
```

---

## 核心文件说明

### ⭐ indicators.py
**技术指标计算库**，被 `screen_double.py` 调用。

| 函数 | 说明 |
|------|------|
| `calc_ma(closes, period)` | 简单移动平均 |
| `calc_ema(closes, period)` | 指数移动平均 |
| `calc_macd(closes)` | MACD（DIF/DEA/MACD柱） |
| `ma_direction(closes, period)` | 均线方向（5日内均值 vs 5日前均值） |
| `calc_rsi(closes, period=14)` | RSI（Wilder 平滑，统一复用 gain_turnover.compute_rsi_scalar） |

---

### ⭐ scorer.py
**Layer2 六维评分引擎**。

对 Layer1 通过的股票，按以下维度评分（满分85分）：

| 维度 | 满分 | 逻辑 |
|------|------|------|
| RSI健康区间 | 20 | 55~70=20分，48~55/70~75=14分 |
| 板块动量 | 15 | 热点板块+15，非热点按涨幅加分 |
| 偏离MA20 | 15 | 0~15%=15分，15~25%=8分 |
| 换手率质量 | 10 | ≥10%=10分，≥5%=5分 |
| 5日涨幅健康 | 15 | 8~15%=15分，5~8%/15~25%=9分 |
| 全市场RPS综合 | 5 | RPS/100×5 |

> 注：RSI>82 超买时扣分；偏离MA20>25% 直接拒绝；5日涨幅<2% 或 >35% 直接拒绝。

---

### ⭐ filter_rules.py
**Layer1 七条件过滤规则**。

| # | 条件 | 说明 |
|---|------|------|
| 1 | MA5>MA10>MA20>MA60 | 均线多头排列 |
| 2 | MA5/MA10/MA20/MA60 方向全向上 | 5日均值 > 5日前均值×1.001 |
| 3 | 收盘价 > MA10 | 价格站在中期均线上方 |
| 4 | MACD>0 且 DIF>0 且 DEA>0 | 多头排列，非金叉 |
| 5 | 5日涨幅 > 5% **或**（>2% 且近5日≥3天>MA5） | 涨幅要求（正常/宽松二选一） |
| 6 | 5日均换手率 ≥ 市值门槛 | 市值≥500亿≥1%，≥100亿≥3%，≥30亿≥5%，≥20亿≥8%，＜20亿≥10% |
| 7 | 信号日数据窗口 ≥ 66根K线 | 距上市>65天，排除新股 |

---

### ⭐ scan_4phase.py
**四相位波段识别引擎**。

识别一只股票在一定时间窗口内的四个标准波段：
- u1：第一个明确上涨段
- d2：-u1后的回调
- u3：第二上涨段
- d4：-u3后的回调

核心函数：
- `detect_volume_price_wave(df, target_date)` — 识别波段
- `score_stock(u1, d2, u3, d4, ...)` — 波段质量评分
- `find_best_entry(u1, d2, u3)` — 寻找最优买入点

---

### ⭐ eval_double.py
**screen_double 信号收益评估**（位于本目录，供上层调用）。

```bash
python eval_double.py --date 2026-04-29
```

---

### analyze_T1_winners.py
**T+1 赢家特征分析**。

分析 screen_double 选出的股票在 T+1 的表现，统计特征规律。

---

## 与 screen_double 的调用关系

```
screen_double.py（主入口）
  ├── indicators.py         ← MA/MACD/RSI/均线方向
  ├── filter_rules.py       ← Layer1 七条件
  ├── scorer.py             ← Layer2 六维评分
  └── scan_4phase.py        ← 波段识别（可选）
```
