# review_screen/ — 筛选器与评估工具

---

## 目录结构

```
review_screen/
├── screen_double.py       # ⭐ 主入口（双层过滤策略扫描器，自包含）
├── screen_trend.py        # 单层趋势筛选器（依赖下方模块）
├── eval_double.py         # ⭐ screen_double 信号收益评估
├── indicators.py          # screen_trend.py 依赖：MA/MACD/RSI 计算
├── scorer.py              # screen_trend.py 依赖：波段质量评分
├── filter_rules.py        # screen_trend.py 依赖：Layer1 过滤规则
├── scan_4phase.py         # screen_trend.py 依赖：四相位波段识别
└── analyze_T1_winners.py # T+1 赢家特征分析
```

---

## 文件说明

### ⭐ screen_double.py
**主力双层过滤策略扫描器**（自包含，不依赖同目录其他模块）。

- 第一层 7 条件 + 第二层 6 维度评分（满分 85）
- 自带 `calc_ma / calc_ema / calc_macd / ma_direction / calc_rsi`
- 输出：120 宽表格 + TXT + JSON

```bash
python screen_double.py --date 2026-04-29 --top-n 50
```

---

### ⭐ eval_double.py
**screen_double 信号收益评估**。

```bash
python eval_double.py --date 2026-04-29
```

---

### screen_trend.py
**旧版单层趋势筛选器**（screen_double 的前身，已被 screen_double 替代）。

- 依赖：`indicators.py` / `scorer.py` / `filter_rules.py` / `scan_4phase.py`
- 逻辑：Layer1 7 条件筛选后，按波段质量排序
- 保留原因：波段分析功能比 screen_double 更细

---

### indicators.py
技术指标计算库（`screen_trend.py` 调用）：
- `compute_all()` — 全量指标计算
- `detect_volume_price_wave()` — 量价波段识别
- `calc_ma / calc_ema / calc_macd / calc_rsi`

---

### scorer.py
评分引擎（`screen_trend.py` 调用）：
- `score_stock()` — 综合评分
- `score_wave_quality()` — 波段质量评分
- `score_detail()` — 详细评分
- `classify_phase()` — 波段相位分类

---

### filter_rules.py
Layer1 过滤规则（`screen_trend.py` 调用）：
- `FilterConfig` — 过滤配置
- `check_filters()` — 执行过滤

---

### scan_4phase.py
四相位波段识别（`screen_trend.py` 调用）：
- 识别上涨/下跌标准波段（u1/d2/u3/d4）
- 找最优买入点

---

### analyze_T1_winners.py
T+1 赢家特征分析（独立脚本）。

---

## 调用关系图

```
screen_double.py     ← 自包含，不依赖同目录模块
screen_trend.py     ← 依赖 indicators / scorer / filter_rules / scan_4phase
eval_double.py      ← 独立读取 screen_double 输出评估收益
analyze_T1_winners.py ← 独立分析脚本
```
