# stock_trend 脚本参考手册

> 生成时间：2026-04-16  
> 文档版本：v1.2  
> 共收录 11 个脚本/模块
>
> **v1.2 更新**：近10日涨停+10；亏损-10；信号窗口最后一天>-3%容忍；trend上限40分；热门板块+8

---

## 目录

1. [gain_turnover.py](#1-核心策略公共模块-gain_turnoverpy) — 策略核心库（不可直接执行）
2. [gain_turnover_screen.py](#2-选股筛选器-gain_turnover_screenpy) — 每日选股
3. [gain_turnover_backtest.py](#3-回测脚本-gain_turnover_backtestpy) — 历史回测
4. [gain_turnover_optimize.py](#4-参数寻优-gain_turnover_optimizepy) — 参数优化
5. [signal_validator.py](#5-信号验证器-l1-signal_validatorpy) — 第一层进化
6. [feedback_tracker.py](#6-反馈追踪器-l2-feedback_trackerpy) — 第二层进化
7. [score_evolution.py](#7-评分进化器-l3-score_evolutionpy) — 第三层进化
8. [cache_qfq_daily.py](#8-前复权日线缓存-cache_qfq_dailypy) — 数据预缓存
9. [cache_fundamental.py](#9-基本面数据缓存-cache_fundamentalpy) — 基本面预缓存
10. [closing_report.py](#10-收盘报告生成器-closing_reportpy) — 每日收盘报告
11. [trend_strong_screen.py](#11-趋势强势股筛选器-trend_strong_screenpy) — 另类选股策略

---

## 1. 核心策略公共模块 `gain_turnover.py`

### 概述

**不是可执行脚本**，是 `gain_turnover_screen.py`、`gain_turnover_backtest.py` 等脚本共用的策略核心库。包含：

- `StrategyConfig` — 参数配置数据类
- `FundamentalData` — 基本面数据类
- `evaluate_signal()` / `evaluate_latest_signal()` — 信号评估函数
- `load_qfq_history()` — 前复权日线加载（含本地缓存）
- `load_stock_names()` / `get_stock_name()` — 股票名称查询
- `compute_rsi()` / `rolling_mean()` — 技术指标计算

### StrategyConfig 默认参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `signal_days` | 2 | 信号窗口天数（连续 N 天满足涨幅条件） |
| `min_gain` | 2.0 | 每日涨幅下限（%） |
| `max_gain` | 7.0 | 每日涨幅上限（%） |
| `quality_days` | 10 | 质量窗口天数（评估中期趋势） |
| `min_turnover` | 1.5 | 5日平均换手率下限（%） |
| `min_amount` | 1e8 | 20日平均成交额下限（元，默认 1 亿） |
| `score_threshold` | 60.0 | 评分门槛（满分 100） |
| `adjust` | `"qfq"` | 复权方式：`"qfq"` 前复权 / `""` 不复权 / `"hfq"` 后复权 |
| `max_extension_pct` | 16.0 | 股价距 MA20 最大偏离（%），超过则过滤 |
| `min_history_days` | 90 | 最小历史数据天数 |
| `check_fundamental` | `False` | 是否开启基本面检查（亏损股扣 20 分） |

### 评分体系（满分 100）

| 维度 | 满分 | 规则 |
|------|------|------|
| 信号稳定性 | 20 | N 天涨幅标准差越小越高 |
| 信号强度 | 10 | 越靠近 `[min_gain, max_gain]` 中值越高 |
| 趋势质量 | 25 | MA5>MA10 + MA20 向上 + 20日涨幅 > 0 + 偏离度适中 |
| 成交活跃度 | 15 | 20 日均成交额、5 日均换手率 |
| 量能配合 | 15 | 5 日/20 日均成交额比值 + 上涨日放量 |
| K 线质量 | 5 | 实体占比高、上影线短 |
| RSI/不过热 | 10 | 45~72 满分，>78 开始扣分，≥82 直接过滤 |
| 基本面扣分 | -10 | EPS<0（亏损）扣 10 分 |
| 近10日涨停 | +10 | 近10个交易日内有涨停（≥9.5%）加 10 分 |
| 热门板块加分 | +8 | 属于当日涨幅前15名板块加 8 分（需 `--sector-bonus`） |

### 数据缓存路径

- 前复权日线：`~/.openclaw/workspace/.cache/qfq_daily/`
- 基本面数据：`~/.openclaw/workspace/.cache/fundamental/`

---

## 2. 选股筛选器 `gain_turnover_screen.py`

### 概述

每日收盘后运行（建议 16:00 后），从全市场 5000+ 只股票中筛选满足 **涨幅区间 + 趋势质量** 双窗口条件的股票，按评分排序输出。

### 使用方法

```bash
# 默认参数（全市场，Top100）
python gain_turnover_screen.py

# 严格参数（前10只）
python gain_turnover_screen.py --top-n 10 --score-threshold 80

# 宽松参数（Top50）
python gain_turnover_screen.py --top-n 50 --min-gain 1.5 --max-gain 8.0

# 复盘指定日期
python gain_turnover_screen.py --date 2026-04-08 --top-n 30

# 指定股票代码
python gain_turnover_screen.py --codes sz002990 sz000967 sh600036

# 开启基本面检查（亏损股扣20分）
python gain_turnover_screen.py --check-fundamental --top-n 50

# 强制刷新缓存
python gain_turnover_screen.py --refresh-cache

# 指定输出文件
python gain_turnover_screen.py --output /path/to/output.txt
```

### 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--days` | int | 2 | 信号窗口天数；≥3天时允许 1/3 天不满足 min_gain |
| `--min-gain` | float | 2.0 | 每日涨幅最小值（%） |
| `--max-gain` | float | 7.0 | 每日涨幅最大值（%） |
| `--quality-days` | int | 10 | 质量窗口天数 |
| `--turnover` | float | 1.5 | 5 日平均换手率下限（%） |
| `--min-volume` | float | 1e8 | 20 日平均成交额下限（元） |
| `--score-threshold` | float | 60.0 | 评分门槛 |
| `--adjust` | str | `qfq` | 复权方式：`qfq` / `hfq` / 空 |
| `--top-n` | int | 100 | 返回前 N 只 |
| `--workers` | int | 8 | 并行线程数 |
| `--codes` | list | None | 指定股票代码（覆盖全市场） |
| `--date` | str | None | 截止日期 `YYYY-MM-DD`（复盘用） |
| `--refresh-cache` | flag | False | 强制刷新前复权缓存 |
| `--check-fundamental` | flag | False | 开启基本面检查（亏损股扣10分） |
| `--sector-bonus` | flag | False | 开启热门板块加分（当日涨幅前15名板块内股票+8分） |
| `--output` / `-o` | str | None | 输出文件路径 |

> **注意**：`--max-extension` 已移除，距 MA20 最大偏离改为动态计算：`max_extension = days × max_gain`
> 例如 `days=3, max_gain=8.0` → max_extension = 24%

### 输出示例

```
sz002990  盛视科技  2026-04-08  88.7  +4.92%  5.12  16.94%  62.5  +7.75%  22.75
sz000967  盈峰环境   2026-04-08  82.3  +3.21%  8.45  12.30%  58.2  +5.12%  15.60
```

列顺序：`代码` `名称` `信号日` `评分` `窗口涨幅` `20日均额(亿)` `5日换手(%)` `RSI14` `偏离MA20(%)` `收盘价`

### 输出文件

默认保存到 `~/stock_reports/daily_screen_{日期}.txt`

---

## 3. 回测脚本 `gain_turnover_backtest.py`

### 概述

对 `gain_turnover_screen.py` 筛选逻辑进行历史回测，验证策略在指定时间区间的表现。回测口径：**T 日出信号 → T+1 开盘价买入 → 持有 N 个交易日后收盘卖出**。

### 使用方法

```bash
# 默认参数（2020-01-01 至今，持有3天）
python gain_turnover_backtest.py

# 短期持有（持有2天）
python gain_turnover_backtest.py --hold 2

# 宽松参数回测
python gain_turnover_backtest.py --days 2 --min-gain 1.5 --max-gain 8.0 --score-threshold 50

# 指定时间区间
python gain_turnover_backtest.py --start 2024-01-01 --end 2025-12-31

# 指定股票
python gain_turnover_backtest.py --codes sz002990 sz000967

# 强制刷新缓存
python gain_turnover_backtest.py --refresh-cache

# 保存交易明细到指定路径
python gain_turnover_backtest.py --output /path/to/trades.csv
```

### 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--days` | int | 2 | 信号窗口天数 |
| `--min-gain` | float | 2.0 | 每日涨幅最小值（%） |
| `--max-gain` | float | 7.0 | 每日涨幅最大值（%） |
| `--quality-days` | int | 10 | 质量窗口天数 |
| `--turnover` | float | 1.5 | 5 日平均换手率下限（%） |
| `--min-volume` | float | 1e8 | 20 日平均成交额下限（元） |
| `--score-threshold` | float | 60.0 | 评分门槛 |
| `--max-extension` | float | 16.0 | 距 MA20 最大偏离（%） |
| `--adjust` | str | `qfq` | 复权方式 |
| `--hold` | int | 3 | 持有交易日数 |
| `--start` | str | `2020-01-01` | 回测开始日期 |
| `--end` | str | `2026-04-08` | 回测结束日期 |
| `--workers` | int | 8 | 并行线程数 |
| `--max-picks-per-day` | int | 5 | 每个信号日最多买几只 |
| `--buy-slip` | float | 0.001 | 买入滑点（千分之一） |
| `--sell-slip` | float | 0.001 | 卖出滑点（千分之一） |
| `--commission` | float | 0.0003 | 单边佣金（万三） |
| `--tax` | float | 0.001 | 卖出印花税（千分之一） |
| `--codes` | list | None | 指定股票 |
| `--refresh-cache` | flag | False | 强制刷新缓存 |
| `--output` / `-o` | str | None | 交易明细 CSV 输出路径 |

### 回测指标说明

| 指标 | 说明 |
|------|------|
| 胜率 | 正收益交易笔数 / 总交易笔数 |
| 平均收益 | 所有交易平均收益率（% / 笔） |
| 夏普比率 | (年化收益 - 无风险利率) / 年化波动率 |
| 最大亏损 | 所有交易中最大单笔亏损（%） |
| 期望年化 | 平均收益 / 持有天数 × 252 |
| 盈亏比 | 平均盈利 / 平均亏损（绝对值） |

### 按评分分组

回测输出会自动按信号评分分组显示（0-60 / 60-70 / 70-80 / 80+），观察高分信号是否对应更高胜率。

---

## 4. 参数寻优 `gain_turnover_optimize.py`

### 概述

对 `gain_turnover` 策略进行网格搜索参数寻优，从大量参数组合中找到历史表现最优的参数集。寻优结果写入 JSON 文件，可直接用于回测和实盘。

### 使用方法

```bash
# 默认寻优（2024-01-01 ~ 2025-12-31）
python gain_turnover_optimize.py

# 密集参数网格
python gain_turnover_optimize.py \
  --start 2024-01-01 --end 2025-12-31 \
  --days 2,3 --min-gain 1.5,2.0,2.5 \
  --max-gain 5.0,6.0,7.0,9.0 \
  --quality-days 10,15,20,30 \
  --turnover 0,1.5,3.0 \
  --score-threshold 50,60,70 \
  --hold 3,5 \
  --max-extension 8,10,12,16

# 提高最小交易数门槛
python gain_turnover_optimize.py --min-trades 50 --top-k 10
```

### 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--start` | str | `2024-01-01` | 寻优开始日期 |
| `--end` | str | `2025-12-31` | 寻优结束日期 |
| `--days` | str | `"2,3"` | 信号窗口候选值（逗号分隔） |
| `--min-gain` | str | `"1.5,2.0"` | 最小涨幅候选值（%） |
| `--max-gain` | str | `"5,6"` | 最大涨幅候选值（%） |
| `--quality-days` | str | `"15,20"` | 质量窗口候选值 |
| `--turnover` | str | `"0,1.5,3"` | 换手率候选值（%） |
| `--score-threshold` | str | `"50,60,70"` | 评分门槛候选值 |
| `--hold` | str | `"3,5"` | 持有天数候选值 |
| `--max-extension` | str | `"8,10,12"` | 偏离 MA20 上限候选值（%） |
| `--max-picks-per-day` | int | 3 | 每信号日最大持仓数 |
| `--min-volume` | float | 1e8 | 成交额下限（元） |
| `--adjust` | str | `qfq` | 复权方式 |
| `--workers` | int | 8 | 并行线程数 |
| `--min-trades` | int | 30 | 有效组合最低交易数门槛 |
| `--top-k` | int | 20 | 输出 Top K 组合 |
| `--codes` | list | None | 指定股票 |
| `--refresh-cache` | flag | False | 强制刷新缓存 |
| `--output` / `-o` | str | None | 输出 JSON 路径 |

### 排序规则

先保证满足 `min-trades` 约束，再按：**Sharpe → 平均收益 → 胜率 → 交易数** 依次排序。

### 输出文件格式（JSON）

```json
{
  "meta": {
    "start": "2024-01-01",
    "end": "2025-12-31",
    "adjust": "qfq",
    "codes": 5195,
    "combos": 96,
    "min_trades": 30,
    "max_picks_per_day": 3
  },
  "top": [ /* Top 20 组合 */ ],
  "all": [ /* 所有 96 个组合 */ ]
}
```

每个组合字段：`days`, `min_gain`, `max_gain`, `quality_days`, `turnover`, `score_threshold`, `hold`, `max_extension`, `trades`, `win_rate`, `avg_ret`, `med_ret`, `std_ret`, `max_loss`, `sharpe`, `expected_annual`

---

## 5. 信号验证器（L1） `signal_validator.py`

### 概述

自我进化策略第一层。读取昨日选股结果，用今日真实行情验证信号质量。**真实收益定义：T+1 开盘价买入 → T+1 收盘价卖出（持有 1 天）。**

### 使用方法

```bash
# 自动找昨日选股文件并验证
python signal_validator.py

# 指定输入文件
python signal_validator.py --input daily_screen_2026-04-14.txt

# 指定个股验证
python signal_validator.py --codes sz002990 sz000967 --names "盛视科技" "盈峰环境"

# 输出到指定文件
python signal_validator.py --output validation_result.txt
```

### 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--input` / `-i` | str | 自动查找 | 昨日选股输出文件路径 |
| `--codes` | list | None | 指定股票代码（覆盖文件输入） |
| `--names` | list | None | 对应股票名称 |
| `--output` / `-o` | str | None | 报告输出路径 |

### 验证指标

| 指标 | 说明 |
|------|------|
| 真实收益（ret_actual） | T+1 开盘买入 → T+1 收盘卖出 |
| 参收益（ret_signal） | 信号日收盘 → T+1 收盘（代理指标） |
| 高涨收益（ret_high） | 信号日收盘 → T+1 日内高点 |
| 触发 +3%/+5%/+7%/+10% | 日内高点是否突破相应阈值 |
| 止损 | T+1 收盘 ≤ 信号收盘 × 0.98 |

### 质量评分标准

| 评分 | 评价 | 条件 |
|------|------|------|
| ≥ 85 | 🟢 优秀 | 真实收益 ≥ 6% 或 高分+止盈 |
| 70 ~ 84 | 🔵 良好 | 真实收益 ≥ 3% ~ 6% |
| 55 ~ 69 | 🟡 及格 | 真实收益 ≥ 0% ~ 3% |
| < 55 | 🔴 失效 | 触发止损 或 真实收益 < 0% |

#### 真实收益评分（基础分 50）

| 真实收益 | 加分 |
|---------|------|
| ≥ +6.0% | **+35** |
| ≥ +5.0% | **+30** |
| ≥ +4.0% | **+25** |
| ≥ +3.0% | **+20** |
| ≥ +2.0% | **+15** |
| ≥ 0% | **+8** |
| < 0% | **-20** |

#### 止盈 / 止损 / 跳空

| 项目 | 条件 | 加分/扣分 |
|------|------|---------|
| 止盈 | 日内高点 ≥ 信号收盘 × 1.05 | **+10** |
| 止损 | 收盘价 ≤ 信号收盘 × 0.98 | **-15** |
| 跳空高开 | ≥ +9% | **-10** |
| 跳空高开 | ≥ +5% | **-5** |

### 输出文件

`~/stock_reports/signal_validation_{日期}.txt`

---

## 6. 反馈追踪器（L2） `feedback_tracker.py`

### 概述

自我进化策略第二层。将 `signal_validator.py` 的验证结果追加到 CSV 数据库，记录每个信号的完整生命周期，供第三层（L3）进化分析使用。

### 使用方法

```bash
# 追加今日验证结果到数据库
python feedback_tracker.py

# 指定输入文件
python feedback_tracker.py --input daily_screen_2026-04-14.txt

# 显示数据库统计（不追加）
python feedback_tracker.py --stats
```

### 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--input` / `-i` | str | 自动查找 | 指定选股文件 |
| `--codes` | list | None | 指定股票 |
| `--names` | list | None | 对应名称 |
| `--stats` | flag | False | 仅显示统计，不追加数据 |

### CSV 字段说明

| 字段 | 说明 |
|------|------|
| `signal_date` | 信号日期 |
| `code` / `name` | 股票代码和名称 |
| `signal_close` | 信号日收盘价 |
| `verified_date` | T+1 验证日期 |
| `open_verified` / `close_verified` | T+1 开收价格 |
| `ret_actual` | T+1 真实收益（%） |
| `hit_3pct` / `hit_5pct` / `hit_7pct` / `hit_10pct` | 是否触发止盈 |
| `stop_loss` | 是否触发止损（-2%） |
| `quality_score` | 质量评分 |
| `exit_date` / `exit_price` | 最终卖出日期/价格 |
| `hold_days` | 持有天数 |
| `ret_exit` | 最终持仓收益（%） |
| `max_retrace` | 最大回撤（%） |
| `closed` | 是否已平仓 |

### 数据文件

`~/stock_reports/feedback_tracker.csv`

---

## 7. 评分进化器（L3） `score_evolution.py`

### 概述

自我进化策略第三层。每周（建议周一 09:00）分析 `feedback_tracker.csv` 中积累的历史信号，评估各参数组合表现，输出量化的参数调整建议，供人工确认后生效。

### 使用方法

```bash
# 默认运行（分析 + 输出建议 + 记录历史）
python score_evolution.py

# 降低最小样本量门槛（数据不足时强制分析）
python score_evolution.py --min-samples 10

# 指定输出文件
python score_evolution.py --output evolution_report_custom.txt
```

### 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--min-samples` | int | 20 | 最小样本量门槛（低于则仅参考） |
| `--output` / `-o` | str | None | 报告输出路径 |

### 参数调整建议类型

| 建议 | 触发条件 |
|------|----------|
| 提高 `score_threshold` | 高评分(≥70) 胜率 > 低评分(55-65) 胜率 +10% |
| 放宽 `max_extension` | 止损率 > 15% |
| 收紧 `max_extension` | 止损率 < 5% |
| 缩短 `hold_days` | 持有 ≤3 天平均收益 > 持有 >3 天 |
| 降低 `max_gain` 上限 | +7% 触发后平均收益 ≈ +5% 触发 |

### 进化历史文件

`~/stock_reports/evolution_history.csv`，包含每次进化建议的记录，`confirmed` 字段供人工标记是否采纳。

### 手动生效建议

当置信度 ≥ 70% 时，报告会输出具体命令，例如：

```bash
sed -i 's/16.0.*# max_extension/18.0  # max_extension/' gain_turnover.py
```

---

## 8. 前复权日线缓存 `cache_qfq_daily.py`

### 概述

每个交易日 **16:10** 定时执行，将全市场股票前复权日线数据刷新到本地缓存，确保次日选股、回测使用最新数据。支持增量刷新（只刷新过期缓存）。

### 使用方法

```bash
# 默认：增量刷新（缓存超过6小时未更新则重新拉取）
python cache_qfq_daily.py

# 自定义过期阈值（4小时）
python cache_qfq_daily.py --max-age-hours 4

# 指定股票
python cache_qfq_daily.py --codes sz002990 sz000967

# 强制刷新全部（含有效缓存）
python cache_qfq_daily.py --refresh

# 预览哪些会被请求（不实际拉取）
python cache_qfq_daily.py --dry-run

# 详细输出
python cache_qfq_daily.py --verbose
```

### 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--codes` | list | 全市场 | 指定股票 |
| `--workers` | int | 16 | 并行线程数 |
| `--max-age-hours` | float | 6.0 | 缓存超过 N 小时视为过期 |
| `--refresh` | flag | False | 强制刷新所有（含未过期） |
| `--dry-run` | flag | False | 预览哪些会刷新，不实际请求 |
| `-v` / `--verbose` | flag | False | 显示每只股票数据条数 |

### 定时任务示例（crontab）

```cron
# 每个交易日 16:10 执行
10 16 * * 1-5 cd /home/lyc/.openclaw/workspace/stock_trend && ~/.venv/bin/python cache_qfq_daily.py >> ~/stock_reports/cache_qfq_daily.log 2>&1
```

### 缓存文件

`~/.openclaw/workspace/.cache/qfq_daily/{代码}_qfq.csv`

---

## 9. 基本面数据缓存 `cache_fundamental.py`

### 概述

每日 **16:00** 定时执行，将全市场股票基本面数据（EPS、ROE、毛利率、净利润等）预加载到本地缓存，供次日筛选器直接读取，无需盘中重复请求网络。

### 使用方法

```bash
# 默认：增量预热（30天未更新才刷新）
python cache_fundamental.py

# 自定义过期阈值（10天）
python cache_fundamental.py --max-age-days 10

# 指定股票
python cache_fundamental.py --codes sz002990

# 强制刷新全部
python cache_fundamental.py --refresh

# 预览哪些会被请求
python cache_fundamental.py --dry-run

# 详细输出（显示每只股票 EPS/ROE）
python cache_fundamental.py --verbose
```

### 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--codes` | list | 全市场 | 指定股票 |
| `--workers` | int | 8 | 并行线程数 |
| `--max-age-days` | int | 30 | 缓存超过 N 天视为过期 |
| `--refresh` | flag | False | 强制刷新所有 |
| `--dry-run` | flag | False | 预览模式 |
| `-v` / `--verbose` | flag | False | 详细输出 |

### 定时任务示例（crontab）

```cron
# 每个交易日 16:00 执行
0 16 * * 1-5 cd /home/lyc/.openclaw/workspace/stock_trend && ~/.venv/bin/python cache_fundamental.py >> ~/stock_reports/cache_fundamental.log 2>&1
```

### 缓存文件

`~/.openclaw/workspace/.cache/fundamental/{代码}.json`

### 基本面字段

| 字段 | 说明 |
|------|------|
| `eps` | 摊薄每股收益（元） |
| `roe` | 净资产收益率（%） |
| `gross_margin` | 销售毛利率（%） |
| `net_profit` | 净利润（亿元） |
| `is_profitable` | 是否盈利（EPS > 0） |
| `report_date` | 最新报告日期 |

---

## 10. 收盘报告生成器 `closing_report.py`

### 概述

每个交易日 **17:00** 自动运行，生成 PDF 格式的完整收盘报告，包含：大盘指数、板块涨跌、策略数据库统计、今日选股结果、昨日前瞻信号验证。报告自动发送到 QQ 邮箱。

### 使用方法

```bash
# 默认：生成 PDF + 发送邮件
python closing_report.py

# 仅生成 PDF，不发送邮件
python closing_report.py --no-email

# 仅运行选股，不生成完整报告
python closing_report.py --screen-only

# 自定义选股数量
python closing_report.py --top-n 30
```

### 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--top-n` | int | 50 | 选股数量 |
| `--no-email` | flag | False | 仅生成 PDF，不发送邮件 |
| `--screen-only` | flag | False | 仅运行选股，不生成报告 |

### 邮件配置

邮件账户信息从 `~/.openclaw/.env` 读取，格式：

```
QQ_EMAIL=your_email@qq.com
QQ_PASS=your授权码
```

### 报告内容

| 板块 | 内容 |
|------|------|
| 大盘指数 | 上证指数、深证成指、创业板指、沪深300 实时行情 |
| 行业板块 | 新浪行业板块（175个）涨跌幅 TOP/BOTTOM |
| 策略数据库 | 累计信号数、胜率、平均收益 |
| 今日选股 | `gain_turnover_screen.py` 结果表格 |
| 信号验证 | 昨日信号 T+1 真实收益彩色评价 |

### 定时任务示例（crontab）

```cron
# 每个交易日 17:00 执行
0 17 * * 1-5 cd /home/lyc/.openclaw/workspace/stock_trend && ~/.venv/bin/python closing_report.py >> ~/stock_reports/closing_report.log 2>&1
```

### 输出文件

- PDF：`~/stock_reports/closing_report_{日期}.pdf`
- 文本：`~/stock_reports/signal_validation_{日期}.txt`

---

## 11. 趋势强势股筛选器 `trend_strong_screen.py`

### 概述

与 `gain_turnover_screen.py` 完全独立的另一套选股策略。基于价格趋势、动量、量价配合三维度评分，筛选处于明确上升趋势的强势股。**RSI 超买过滤 + 相对强弱调整**是 v2 核心改进。

### 评分体系（三维度 × 100分制）

| 维度 | 权重 | 子维度 |
|------|------|--------|
| 趋势质量 | 50% | 价格在均线上方(40) + 均线多头排列(32) + 均线发散度(20) + 5日斜率(8) |
| 动量强度 | 30% | 20日涨幅(35) + 10日涨幅(25) + 创20日新高(40)，再按相对强弱打折 |
| 量价配合 | 20% | 量比(35) + 成交额放大(35) + 量价同向配合(30) |

### RSI 过滤规则

| RSI 范围 | 处理 |
|----------|------|
| > 88 | 直接过滤（不入选） |
| 82 ~ 88 | 扣 40 分 |
| 75 ~ 82 | 扣 20 分 |

### 相对强弱调整

以四大指数（上证/深证/沪深300/创业板）等权平均的近 21 日涨幅为基准：

| 相对强弱 | 处理 |
|----------|------|
| < -10% | 直接过滤（个股弱于市场太明显） |
| -10% ~ -5% | 动量得分打 5 折 |
| ≥ -5% | 无调整 |

### 使用方法

```bash
# 默认：全市场 Top30，评分阈值 50
python trend_strong_screen.py

# 前50只
python trend_strong_screen.py --top-n 50

# 提高门槛
python trend_strong_screen.py --score-threshold 60

# 复盘指定日期
python trend_strong_screen.py --date 2026-04-08

# 指定股票
python trend_strong_screen.py --codes sz002990 sz000967

# 调整最低成交额
python trend_strong_screen.py --min-volume 1e8

# 指定线程数
python trend_strong_screen.py --workers 40
```

### 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--top-n` | int | 30 | 返回前 N 只 |
| `--score-threshold` | float | 50.0 | 评分门槛 |
| `--min-volume` | float | 5e7 | 最低成交额（元，默认 5000 万） |
| `--codes` | list | None | 指定股票（覆盖全市场） |
| `--workers` | int | 30 | 并行线程数 |
| `--date` | str | None | 截止日期 `YYYY-MM-DD`（复盘用），可用 `today` |

### 评分说明

```
总分 = 趋势×50% + 动量×30% + 量价×20% - RSI惩罚
v2改进：RSI>88过滤，RSI>82扣40分，RSI>75扣20分；相对强弱<-10%过滤，<-5%动量5折
```

---

## 附：完整定时任务配置（crontab）

```cron
# ── 工作日 16:00 ──────────────────────────────────
# 基本面数据缓存（先执行，确保收盘数据可用）
0  16 * * 1-5 cd /home/lyc/.openclaw/workspace/stock_trend && \
  ~/.venv/bin/python cache_fundamental.py >> ~/stock_reports/cache_fundamental.log 2>&1

# ── 工作日 16:10 ──────────────────────────────────
# 前复权日线缓存
10 16 * * 1-5 cd /home/lyc/.openclaw/workspace/stock_trend && \
  ~/.venv/bin/python cache_qfq_daily.py >> ~/stock_reports/cache_qfq_daily.log 2>&1

# ── 工作日 17:00 ──────────────────────────────────
# 收盘报告（PDF + 邮件）
0  17 * * 1-5 cd /home/lyc/.openclaw/workspace/stock_trend && \
  ~/.venv/bin/python closing_report.py >> ~/stock_reports/closing_report.log 2>&1

# ── 工作日 16:30 ──────────────────────────────────
# 信号验证（昨日信号 → T+1 验证）
30 16 * * 1-5 cd /home/lyc/.openclaw/workspace/stock_trend && \
  ~/.venv/bin/python signal_validator.py >> ~/stock_reports/signal_validation.log 2>&1

# ── 每周一 09:00 ──────────────────────────────────
# 反馈追踪 + 评分进化
43 16 * * 1   cd /home/lyc/.openclaw/workspace/stock_trend && \
  ~/.venv/bin/python feedback_tracker.py >> ~/stock_reports/feedback.log 2>&1 && \
  ~/.venv/bin/python score_evolution.py >> ~/stock_reports/evolution.log 2>&1
```

---

## 附：输出文件汇总

| 文件 | 位置 | 脚本 |
|------|------|------|
| `daily_screen_{日期}.txt` | `~/stock_reports/` | `gain_turnover_screen.py` |
| `gain_turnover_backtest_*.csv` | `~/stock_reports/` | `gain_turnover_backtest.py` |
| `gain_turnover_optimize_*.json` | `~/stock_reports/` | `gain_turnover_optimize.py` |
| `signal_validation_{日期}.txt` | `~/stock_reports/` | `signal_validator.py` |
| `feedback_tracker.csv` | `~/stock_reports/` | `feedback_tracker.py` |
| `evolution_history.csv` | `~/stock_reports/` | `score_evolution.py` |
| `evolution_report_{日期}.txt` | `~/stock_reports/` | `score_evolution.py` |
| `closing_report_{日期}.pdf` | `~/stock_reports/` | `closing_report.py` |
| `qfq_daily/*.csv` | `~/.openclaw/workspace/.cache/` | `cache_qfq_daily.py` |
| `fundamental/*.json` | `~/.openclaw/workspace/.cache/` | `cache_fundamental.py` |
