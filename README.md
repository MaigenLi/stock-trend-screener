# stock_trend/ — 股票趋势筛选系统

## 目录结构

```
stock_trend/
├── gain_turnover.py          # ⭐ 核心数据接口（所有策略的数据底座）
├── screen_double.py          # ⭐ 双层过滤策略扫描器（主力）
├── screen_trend.py           # ⭐ 单层趋势筛选器（备选）
├── eval_double.py            # ⭐ screen_double 信号收益评估
├── signal_validator.py       # ⭐ 信号验证（盘中/收盘后）
├── closing_report.py         # ⭐ 每日收盘 PDF 报告
├── sector_hotspot.py         # ⭐ 板块热点识别模块
├── refresh_sector_cache.py  # ⭐ 板块缓存刷新（每日/每周）
├── replenish_sector_map.py   # 补录未映射股票的行业分类
├── cache_qfq_daily.py        # QFQ 前复权日线数据缓存
├── cache_fundamental.py      # 基本面数据缓存
├── gain_turnover_screen.py   # gain_turnover 专属筛选器
├── triple_screen.py          # 三步筛选 v1（蓄势强势股）
├── triple_v2_screen.py       # 三步筛选 v2（启动型强势股）
├── param_optimizer_triple.py # triple_screen 参数优化器
├── screen_momentum_winners.py # 动量精选
├── momentum_screen.py        # 动量筛选器
├── run_daily_pipeline.sh     # 每日流水线 cron 脚本
└── review_screen/            # review_screen 专用子模块
```

---

## 核心文件说明

### ⭐ gain_turnover.py
**数据底座**，所有策略的统一数据接口。

- `load_qfq_history(code, end_date, adjust="qfq")` — 前复权日线
- `get_all_stock_codes()` — 全市场股票代码
- `load_stock_names()` / `get_stock_name()` — 股票名称
- `compute_rsi_scalar()` — RSI（Wilder 平滑）
- `normalize_symbol()` / `get_lock()` — 代码规范化

> **规则**：所有历史 K 线数据必须通过此文件获取，禁止绕过。

---

### ⭐ screen_double.py
**双层过滤策略扫描器**（当前主力）。

- **第一层**（7个条件粗筛）：MA5>MA10>MA20>MA60 多头排列 + 方向向上 + MACD>0 + 5日涨幅 + 换手率门槛 + 数据充足
- **第二层**（6维度精筛评分，满分85分）：RSI健康 / 板块动量 / 偏离MA20 / 换手率质量 / 5日涨幅健康 / 全市场RPS综合
- **输出**：120宽表格 + TXT + JSON

```bash
~/.venv/bin/python screen_double.py --date 2026-04-29 --top-n 50
```

---

### ⭐ screen_trend.py
**单层趋势筛选器**（screen_double 的前身备选）。

- 7个 Layer1 条件，与 screen_double 第一层相同
- 无第二层精筛评分，输出更简洁

```bash
~/.venv/bin/python screen_trend.py --date 2026-04-29
```

---

### ⭐ eval_double.py
**screen_double 信号收益评估**。

- 读取 `screen_double_{signal_date}.txt` 信号文件
- 计算：信号日 T-3 → T+1开盘买入 → T收盘卖出（持有3天）
- 输出：每只股票持有收益 + 胜率/平均收益统计

```bash
python review_screen/eval_double.py --date 2026-04-29
# 卖出日 = 2026-04-29
# 信号日 = 2026-04-24（卖出日前3个交易日）
# 买入日 = 2026-04-27（信号日下一个交易日）
```

---

### ⭐ signal_validator.py
**信号验证器**。

- 解析 `screen_double_*.txt` / `triple_screen_*.txt` / `daily_screen_*.txt` 信号文件
- 用 T+1/T+2 实际行情验证信号质量
- 评分：真实收益 + 止盈 + 止损 + 跳空惩罚
- 输出：`signal_validation_{date}.txt`

```bash
python signal_validator.py --date 2026-04-28
```

---

### ⭐ closing_report.py
**每日收盘 PDF 报告**。

- 读取 `screen_double_{date}.txt` 信号文件
- 生成格式化 PDF（板块分布 / RSI分布 / 评分分布）
- 定时任务自动触发（`run_daily_pipeline.sh`）

---

### ⭐ sector_hotspot.py
**板块热点识别**。

- 数据源：新浪行业板块 API（49个板块）
- 热点定义：当日涨幅前15名板块
- 接口：
  - `get_hot_sectors(date)` — 返回热点板块 dict（name → 涨跌幅）
  - `get_stock_sector(code)` — 股票所属板块

---

### ⭐ refresh_sector_cache.py
**板块缓存刷新**。

```bash
# 每日热点刷新（17:40 cron）
python refresh_sector_cache.py

# 每周重建股票-板块映射（周五20:00 cron）
python refresh_sector_cache.py --refresh-map
```

- `--refresh-map`：重建全量股票→板块映射（约2953只，Sina行业分类）
- 当前覆盖率：~53%（Sina 只覆盖约3000只股票）

---

## 其他文件说明

| 文件 | 说明 |
|------|------|
| `gain_turnover_screen.py` | gain_turnover 专属量价筛选（早期版） |
| `triple_screen.py` | 三步筛选 v1：RPS≥75 蓄势强势股选入 |
| `triple_v2_screen.py` | 三步筛选 v2：当日涨幅[2%,7%] 启动型选入 |
| `param_optimizer_triple.py` | triple_screen 参数优化（遗传算法） |
| `momentum_screen.py` | 动量筛选器 |
| `screen_momentum_winners.py` | 动量精选（与 momentum_screen 有重叠） |
| `replenish_sector_map.py` | 补录腾讯接口未覆盖股票的行业分类 |
| `cache_qfq_daily.py` | QFQ 日线数据 AkShare 缓存管理 |
| `cache_fundamental.py` | 基本面数据缓存 |
| `run_daily_pipeline.sh` | 每日流水线 cron 脚本 |

---

## 数据缓存路径

```
~/.openclaw/workspace/.cache/
├── qfq_daily/           # 前复权日线 CSV（~5196只）
└── sector/
    ├── sector_hotspot.json      # 热点板块（每日更新）
    └── stock_sector_map.json    # 股票→板块映射（每周重建）
```

---

## 定时任务

```cron
# 每日 17:40 — 收盘流水线
0 17 * * 1-5 /home/lyc/.openclaw/workspace/stock_trend/run_daily_pipeline.sh

# 每周五 20:00 — 重建板块映射
0 20 * * 5 /home/lyc/.venv/bin/python /home/lyc/.openclaw/workspace/stock_trend/refresh_sector_cache.py --refresh-map >> /home/lyc/.openclaw/workspace/stock_reports/refresh_sector_cache_weekly.log 2>&1
```
