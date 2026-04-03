# stock_trend - 趋势放量选股工具

> 扫描全市场，筛选趋势向上且放量的股票

## 定位

**选股层** — 全市场扫描工具，从所有股票中筛选出符合条件的候选票。不做策略回测，专注筛选。

```
stock_common  （数据层）
    ↓
stock_trend   （选股层：本项目）
    ↓
quant_system  （策略层：回测分析）
```

## 目录结构

```
stock_trend/
├── core/                              # 核心模块
│   ├── final_full_market_scan.py      # 全市场扫描入口
│   ├── trend_volume_screener_ultimate.py  # 趋势放量筛选器（终极版）
│   ├── stock_sector.py                 # 板块信息获取
│   └── local_stock_names.py            # 本地股票名称库
├── docs/                               # 文档
│   ├── 快速使用指南.md
│   ├── 参数调节指南.md
│   ├── 使用示例.md
│   └── 投资建议报告.md
├── full_scan_auto.py                  # 全市场自动扫描（入口）
├── quick_test.py                       # 快速功能测试
├── run_without_network.py              # 离线运行版
├── extend_stock_cache.py               # 扩展股票名称缓存
└── add_sz_stocks.py                    # 添加深市股票
```

## 核心入口

### 全市场扫描（推荐）

```bash
cd ~/.openclaw/workspace/stock_trend
python full_scan_auto.py
```

### 快速测试

```bash
python quick_test.py
```

## 筛选逻辑

`trend_volume_screener_ultimate.py` 实现了趋势放量双重筛选：

1. **趋势条件**：价格在60日均线上方，均线多头排列
2. **放量条件**：今日成交量超过20日均量的1.5倍
3. **一票否决**：价格在60日线下方、均线空头排列、无量上涨、短期涨幅≥15%

可选叠加：
- 突破平台（今日收盘 > 过去20天最高价）
- 换手率过滤

## 数据来源

- 股票数据：通达信离线 `.day` 文件（`~/stock_data/vipdoc/`）
- 股票名称：本地缓存（`core/local_stock_names.py`）
- 实时数据：腾讯财经 API（需要网络）
- 板块信息：东方财富 API（需要网络）

## 依赖

- Python 3.8+
- `pandas`, `numpy`
- `requests`（获取实时数据/板块）
- `stock_common`（位于 `../stock_common/`）

## 与 quant_system 的分工

| 项目 | 职责 | 输入 |
|------|------|------|
| `stock_trend` | 市场扫描、选股 | 通达信数据 |
| `quant_system` | 策略回测、信号分析 | `stock_trend` 输出的候选股 |
| `stock_common` | 统一数据接口 | — |
