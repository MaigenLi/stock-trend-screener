# gain_turnover 策略文档

## 目录结构

```
stock_trend/
├── gain_turnover.py              # 策略核心（共用模块）
├── gain_turnover_screen.py      # 选股筛选器
├── gain_turnover_backtest.py   # 历史回测
├── gain_turnover_optimize.py    # 参数寻优
└── trend_strong_screen.py       # 趋势强势股筛选器
```

---

## 一、gain_turnover 策略（连续温和上涨选股）

### 核心理念
捕捉上升趋势中连续 N 天温和上涨（每日涨幅在某个区间内）的股票，适合寻找启动点。

### 选股逻辑

**硬门槛（全部满足方可入选）：**
1. 信号窗口：最近 N 天，每日涨幅在 `[min_gain, max_gain]`
2. `close > ma5 >= ma10`（均线多头，允许 0.5% 容差）
3. ma5、ma10 当日值均高于前日值（均线向上）
4. 10 日涨幅 > 0
5. RSI(14) < 82
6. 偏离 MA20 < max_extension%
7. 20 日均成交额 ≥ 1 亿
8. 5 日均换手率 ≥ min_turnover

**评分体系（满分 100）：**

| 维度 | 满分 | 说明 |
|------|------|------|
| 信号稳定性 | 20 | 涨幅标准差越小越好 |
| 信号强度 | 10 | 均值越接近区间中值越好 |
| 趋势质量 | 25 | 多头排列 + ma20上涨 + 20日涨幅 |
| 成交活跃度 | 15 | 成交额和换手率 |
| 量能配合 | 15 | 5日/20日成交额比值 + 涨跌量能 |
| K线质量 | 5 | 实体占比高 + 上影线短 |
| RSI健康 | 10 | 45~72 得满分，≥82 过滤 |

---

### 筛选器 — gain_turnover_screen.py

筛选全市场符合条件的目标股票。

```bash
# 默认参数（全市场筛选，Top50）
~/.venv/bin/python gain_turnover_screen.py

# 指定股票
~/.venv/bin/python gain_turnover_screen.py --codes sh600036 sz300819

# 调整参数
~/.venv/bin/python gain_turnover_screen.py \
    --days 2 \
    --min-gain 2.0 --max-gain 7.0 \
    --quality-days 10 \
    --turnover 1.5 \
    --score-threshold 60 \
    --top-n 30

# 复盘指定日期（使用该日期的数据）
~/.venv/bin/python gain_turnover_screen.py --date 2026-04-10

# 输出到文件
~/.venv/bin/python gain_turnover_screen.py -o ~/stock_reports/my_screen.txt
```

**默认参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--days` | 2 | 信号窗口天数 |
| `--min-gain` | 2.0% | 每日涨幅下限 |
| `--max-gain` | 7.0% | 每日涨幅上限 |
| `--quality-days` | 10 | 质量窗口天数 |
| `--turnover` | 1.5% | 5日均换手率下限 |
| `--score-threshold` | 60 | 评分门槛 |
| `--max-extension` | 10% | 偏离MA20上限 |
| `--adjust` | qfq | 前复权 |
| `--top-n` | 50 | 返回前N只 |
| `--workers` | 8 | 并行线程数 |

---

### 回测器 — gain_turnover_backtest.py

对历史数据进行回测，评估策略表现。

```bash
# 默认参数回测（2024-01-01 至 2025-12-31）
~/.venv/bin/python gain_turnover_backtest.py

# 自定义参数
~/.venv/bin/python gain_turnover_backtest.py \
    --days 2 --min-gain 2.0 --max-gain 7.0 \
    --quality-days 10 --turnover 1.5 \
    --score-threshold 60 \
    --hold 3 \
    --start 2024-01-01 --end 2025-12-31

# 输出交易明细
~/.venv/bin/python gain_turnover_backtest.py -o trades.json
```

**默认参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--days` | 2 | 信号窗口天数 |
| `--min-gain` | 2.0% | 每日涨幅下限 |
| `--max-gain` | 7.0% | 每日涨幅上限 |
| `--quality-days` | 10 | 质量窗口天数 |
| `--turnover` | 1.5% | 5日均换手率下限 |
| `--score-threshold` | 60 | 评分门槛 |
| `--hold` | 3 | 持有交易日数 |
| `--max-picks-per-day` | 3 | 每个信号日最多买几只 |
| `--buy-slip` | 0.5% | 买入滑点 |
| `--sell-slip` | 0.5% | 卖出滑点 |
| `--commission` | 0.03% | 单边佣金 |
| `--tax` | 0.1% | 卖出印花税 |

---

### 寻优器 — gain_turnover_optimize.py

网格搜索最佳参数组合。

```bash
# 全量寻优（所有参数范围）
~/.venv/bin/python gain_turnover_optimize.py \
    --start 2024-01-01 --end 2025-12-31

# 自定义范围
~/.venv/bin/python gain_turnover_optimize.py \
    --start 2024-01-01 --end 2025-12-31 \
    --days 2,3 \
    --min-gain 1.8,2.0,2.2 \
    --max-gain 5.0,6.0 \
    --quality-days 10,15 \
    --turnover 1.0,1.5,2.0 \
    --score-threshold 60,70 \
    --hold 3,5 \
    --max-extension 8,10
```

**参数范围说明：**
- 所有参数用逗号分隔多个值
- 组合数 = 各参数值数量的乘积
- 建议先用宽范围粗搜，再用窄范围精搜

---

## 二、趋势强势股筛选 — trend_strong_screen.py

独立于 gain_turnover 的另一个策略，专注趋势强度。

### 核心理念
寻找均线多头排列、量能配合良好的趋势强势股。

### 选股逻辑

**硬门槛：**
1. 20日均成交额 ≥ 5000万
2. RSI(14) ≤ 88（>88 直接过滤）
3. 相对强弱 < -10% 直接过滤（个股弱于市场过多）

**评分体系（满分100）：**

| 维度 | 权重 | 说明 |
|------|------|------|
| 趋势质量 | 50% | 价格在均线上方 + 均线多头 + 发散度 + 斜率 |
| 动量 | 30% | 20日涨幅 + 10日涨幅 + 创新高 |
| 量价 | 20% | 量比 + 成交额放大 + 量价配合 |

**RSI 惩罚：**
- RSI 75~82：扣 20 分
- RSI 82~88：扣 40 分
- RSI > 88：直接过滤

**相对强弱：**
- 相对强弱 < -10%：直接过滤
- 相对强弱 < -5%：动量得分打 5 折

### 使用方法

```bash
# 默认（全市场 Top30）
~/.venv/bin/python trend_strong_screen.py

# 严格模式（评分≥80）
~/.venv/bin/python trend_strong_screen.py --strict

# 指定日期复盘
~/.venv/bin/python trend_strend_screen.py --date 2026-04-10

# 指定股票
~/.venv/bin/python trend_strong_screen.py --codes sh600036 sz300568
```

---

## 三、数据说明

所有脚本使用 **AkShare 前复权日线数据**：
- 数据来源：`ak.stock_zh_a_daily()`（Tushare/Emoney）
- 本地缓存：`.cache/qfq_daily/*.csv`（首次运行后生成）
- **不依赖通达信本地 .day 文件**

---

## 四、参数推荐

### gain_turnover 稳健版（默认）
```
--days 2 --min-gain 2.0 --max-gain 7.0 --quality-days 10 --turnover 1.5 --hold 3
```

### gain_turnover 进取版
```
--days 2 --min-gain 2.0 --max-gain 7.0 --quality-days 10 --turnover 2.0 --hold 5
```
> 进取版 Sharpe 更高，但最大亏损 -28.57%，实盘需配合止损。
