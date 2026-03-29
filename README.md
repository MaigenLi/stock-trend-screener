# Stock Trend Screener
股票趋势放量筛选器

## Git 仓库

这是一个独立的 Git 仓库，包含完整的股票趋势筛选器代码。

**版本信息：** v1.0.0 (初始提交)
**提交哈希：** e30e0a9
**提交时间：** 2026-03-29

## 目录结构
```
stock_trend/
├── README.md                    # 本文件
├── screener.py                  # 主筛选器（可调节参数版）
├── full_scan.py                 # 全市场扫描（符号链接）
├── full_scan_auto.py            # 全市场扫描（自动版，无需交互）
├── quick_test.py                # 快速测试脚本
├── stock_sector.py -> core/stock_sector.py  # 板块信息模块
├── stock_name_cache.json        # 股票名称缓存
├── stock_names.json             # 股票名称数据库
├── core/                        # 核心模块目录
│   ├── final_full_market_scan.py  # 全市场扫描实现
│   ├── stock_sector.py           # 板块信息获取模块
│   ├── trend_volume_screener.py  # 趋势放量筛选器核心
│   └── sector_cache/            # 板块信息缓存目录
├── results/                     # 结果文件目录
│   ├── current/                 # 当前结果
│   └── archive/                 # 历史结果归档
├── docs/                        # 文档目录
│   ├── 参数调节指南.md          # 参数调节指南
│   └── ...                      # 其他文档
└── sector_cache/                # 板块信息缓存（旧目录）
```

## 主要功能

### 1. 趋势放量筛选器 (`screener.py`)
- **功能**：根据可调节参数筛选趋势向上且放量的股票
- **特点**：
  - 可调节的筛选参数（趋势、涨幅、量能、风险等）
  - 支持板块信息显示（包含热度和人气评分）
  - 支持股票名称显示
  - 并行处理，速度快
- **使用**：`python screener.py [参数]`

### 2. 全市场扫描 (`full_scan.py` 或 `full_scan_auto.py`)
- **功能**：扫描全市场股票，使用预设参数组合
- **特点**：
  - 提供严格、中等、宽松三种参数组合
  - 支持自定义参数
  - 自动保存结果
- **使用**：
  - `python full_scan.py`（交互式）
  - `python full_scan_auto.py`（自动版，使用中等参数）

### 3. 快速测试 (`quick_test.py`)
- **功能**：快速验证筛选器功能
- **测试内容**：
  - 股票名称获取
  - 板块信息获取
  - 小规模筛选
- **使用**：`python quick_test.py`

## 核心模块

### 1. `core/trend_volume_screener.py`
- 趋势放量筛选器核心实现
- 包含股票数据读取、趋势分析、评分计算等功能
- 支持股票名称获取（本地数据库+网络获取）

### 2. `core/stock_sector.py`
- 板块信息获取模块
- 支持从东方财富网获取实时板块信息
- 包含热度和人气评分
- 支持本地推断和缓存机制

### 3. `core/final_full_market_scan.py`
- 全市场扫描实现
- 提供多种参数组合
- 支持交互式和自动运行

## 数据文件

### 1. `stock_names.json`
- 股票名称数据库
- 包含常见股票的代码-名称映射
- 用于本地快速获取股票名称

### 2. `stock_name_cache.json`
- 股票名称缓存
- 自动保存网络获取的股票名称
- 减少重复网络请求

### 3. `core/sector_cache/`
- 板块信息缓存目录
- 缓存网络获取的板块信息
- 有效期1天，减少重复请求

## 使用方法

### 基本使用
```bash
# 运行主筛选器（可调节参数）
python screener.py

# 运行全市场扫描（自动版）
python full_scan_auto.py

# 运行快速测试
python quick_test.py
```

### 参数调节
```bash
# 使用自定义参数
python screener.py --min-three-day 5.0 --min-volume-ratio 1.2 --min-score 75.0

# 查看所有参数
python screener.py --help
```

### 全市场扫描参数
```bash
# 交互式选择参数
python full_scan.py

# 自动运行（使用中等参数）
python full_scan_auto.py
```

## 筛选参数说明

### 趋势参数
- `--price-above`：价格在哪个均线之上 (ma5/ma10/ma20)
- `--ma-trend`：是否要求均线多头排列
- `--min-trend-days`：最小上升趋势天数

### 三天表现参数
- `--min-three-day`：最小三天涨幅(%)
- `--max-three-day`：最大三天涨幅(%)
- `--min-up-days`：最小上涨天数(3天内)

### 量能参数
- `--min-volume-ratio`：最小平均量比
- `--consecutive-volume`：是否要求连续三天放量

### 风险参数
- `--max-ten-day`：最大十日涨幅(%)

### 其他参数
- `--min-price`：最小价格(元)
- `--max-price`：最大价格(元)
- `--min-score`：最小综合评分

## 结果文件

筛选结果保存在 `results/current/` 目录，包含：
1. **详细结果文件**：包含所有筛选出的股票详细信息
2. **股票代码列表**：只包含股票代码
3. **板块分类文件**：按板块分类的股票代码

## 注意事项

1. **数据源**：使用通达信日线数据，路径为 `/mnt/d/new_tdx/vipdoc/`
2. **网络请求**：板块信息需要网络连接，失败时会使用本地推断
3. **缓存机制**：网络获取的数据会缓存，减少重复请求
4. **并行处理**：默认使用15个线程并行处理，可调节

## 更新历史

- **2026-03-28**：清理目录结构，优化代码，修复股票名称显示问题
- **2026-03-27**：添加板块信息功能，优化网络请求
- **2026-03-26**：创建趋势放量筛选器基础版本