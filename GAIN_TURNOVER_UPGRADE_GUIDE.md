# Gain Turnover 升级版使用文档（前复权）

## 1. 这套系统现在包含什么

升级版由 4 个文件组成：

1. `stock_trend/gain_turnover_strategy.py`
   - 公共策略模块
   - 负责前复权数据加载、缓存、指标计算、统一信号评估

2. `stock_trend/screen_by_gain_turnover.py`
   - 升级版筛选器
   - 用于找出当前或指定日期满足条件的股票

3. `stock_trend/backtest_gain_turnover.py`
   - 升级版回测器
   - 用于验证历史表现

4. `stock_trend/optimize_gain_turnover.py`
   - 参数寻优器
   - 用于批量测试参数组合

---

## 2. 核心升级点

相比旧版，升级版做了这些变化：

### 2.1 全部改为前复权
默认：`--adjust qfq`

原因：
- 连续涨幅策略在研究时更适合使用前复权
- 避免除权除息导致历史涨幅和持有收益失真

### 2.2 拆成两层逻辑

#### 信号窗口
最近 `N` 天，每天涨幅必须都在区间内：
- `min_gain <= daily_gain <= max_gain`

#### 质量窗口
最近 `quality_days` 天，额外要求：
- `close > MA20 > MA60`
- MA20 向上
- 20日涨幅 > 0
- 距 MA20 偏离不过热
- 20日平均成交额达标
- 5日平均换手率达标
- RSI 不过热

这意味着：
- **信号窗口** 决定“最近是不是连续温和上涨”
- **质量窗口** 决定“这是不是一个健康趋势里的中继”

### 2.3 筛选与回测统一口径
筛选器和回测器现在共用同一套：
- 数据源
- 指标计算
- 过滤逻辑
- 评分系统

这样参数优化结果才有意义。

### 2.4 回测改成更真实的成交口径
当前回测定义：
- `T` 日收盘后出信号
- `T+1` **开盘价**买入
- 持有 `hold` 个交易日后 **收盘价**卖出

并加入：
- 滑点
- 佣金
- 印花税
- 停牌跳过
- 每个信号日最多买 `K` 只

---

## 3. 统一评分体系（100分）

评分构成：

- 稳定性：20
- 信号强度：10
- 趋势质量：25
- 流动性：15
- 量能配合：15
- K线质量：5
- RSI健康：10

### 3.1 稳定性（20）
看信号窗口内每日涨幅标准差。

含义：
- 越平滑越好
- 波动越乱越扣分

### 3.2 信号强度（10）
看信号窗口内平均涨幅离目标区间中值有多远。

含义：
- 太弱，不好
- 太强，也可能过热
- 接近区间中值最好

### 3.3 趋势质量（25）
包含：
- `close > MA20 > MA60`
- MA20 上行
- 20日涨幅为正
- 距 MA20 偏离合理

这是升级版里最重要的维度之一。

### 3.4 流动性（15）
包含：
- 20日平均成交额
- 5日平均换手率

作用：
- 过滤成交太差、容易失真的标的

### 3.5 量能配合（15）
包含：
- 5日均额相对20日均额的关系
- 上涨日量能是否优于下跌日

### 3.6 K线质量（5）
包含：
- 实体占比
- 上影线是否过长

### 3.7 RSI健康（10）
大致偏好：
- 45~72 最优
- 太低说明弱
- 太高说明热

---

## 4. 参数详细说明

下面按脚本分别说明。

---

# 4A. `screen_by_gain_turnover.py` 参数说明

## 4A.1 基本参数

### `--days`
信号窗口天数。

例子：
- `--days 2`：最近2个交易日每天涨幅都要满足条件
- `--days 3`：最近3个交易日每天都要满足

影响：
- 越小，信号更多、更松
- 越大，信号更少、更严格

推荐：
- 入门：`2`
- 更稳一点：`3`

---

### `--min-gain`
信号窗口内每日涨幅下限。

例子：
- `--min-gain 2`
- 表示每一天至少涨 2%

影响：
- 越高，越偏强势连涨
- 太高容易抓到末端加速

推荐：
- `1.5 ~ 2.5`

---

### `--max-gain`
信号窗口内每日涨幅上限。

例子：
- `--max-gain 6`
- 表示每一天不能超过 6%

影响：
- 控制不要选到过于爆拉的票
- 太低会错过正常强势股
- 太高会混进末端拉升

推荐：
- `5 ~ 6`

---

### `--quality-days`
质量窗口长度。

例子：
- `--quality-days 20`

作用：
- 用这段时间判断趋势、量能、不过热

影响：
- 越小，更敏感
- 越大，更稳但更迟钝

推荐：
- `20`
- 偏稳：`30`

---

### `--turnover`
5日平均换手率下限，单位 `%`。

例子：
- `--turnover 3`

作用：
- 过滤掉换手太差的票

影响：
- 太高会丢失很多慢牛股
- 太低会混入沉寂票

推荐：
- 主板全市场：`1.5 ~ 3`
- 若先粗筛，可设 `0`

注意：
- 这里使用的是**历史换手率**，不是实时 API 了

---

### `--min-volume`
20日平均成交额下限，单位是“元”。

例子：
- `--min-volume 100000000` 表示 1 亿

作用：
- 过滤流动性差的票

推荐：
- 普通全市场：`1e8`
- 更稳：`2e8`
- 小盘也想覆盖：`5e7`

---

### `--score-threshold`
总评分下限。

例子：
- `--score-threshold 60`

作用：
- 低于这个分数直接过滤

推荐：
- 粗筛：`50`
- 常规：`60`
- 严格：`70`

观察：
- 如果 50 和 60 结果差不多，说明这批信号本来就高分
- 这种情况下应重点测 70/80

---

### `--max-extension`
允许价格距离 MA20 的最大正偏离，单位 `%`。

例子：
- `--max-extension 12`

作用：
- 防止买到离均线太远的过热票

推荐：
- 常规：`10 ~ 12`
- 严格：`8`

---

### `--adjust`
复权方式。

可选：
- `qfq`：前复权，默认
- `hfq`：后复权
- `""`：不复权

推荐：
- 研究和回测：`qfq`

---

## 4A.2 结果控制参数

### `--top-n`
返回前 N 只。

推荐：
- 日常看盘：`20 ~ 50`

### `--workers`
并行线程数。

推荐：
- `4 ~ 8`
- 如果网络和机器都稳，可以更高

### `--codes`
指定股票代码，只筛选这些股票。

例子：
```bash
--codes sh600036 sz000858
```

### `--date`
指定截止日期，用于复盘。

例子：
```bash
--date 2026-04-10
```

### `--refresh-cache`
强制刷新前复权缓存。

适用场景：
- 怀疑缓存过旧
- 第一次大规模更新

### `--output`
输出文件路径。

---

# 4B. `backtest_gain_turnover.py` 参数说明

除了和筛选器共用的大部分参数外，还有这些回测专属参数。

### `--hold`
持有交易日数。

例子：
- `--hold 3`
- 表示 T+1 开盘买入后，持有 3 个交易日，最后一天收盘卖出

推荐：
- 短线：`3`
- 中短线：`5`
- 可继续测试：`7`

---

### `--start`
回测开始日期。

### `--end`
回测结束日期。

建议：
- 至少覆盖一个完整牛熊震荡周期
- 不要只测单边行情

---

### `--max-picks-per-day`
每个信号日最多买几只。

例子：
- `--max-picks-per-day 3`

作用：
- 避免默认“同一天几十只全买”的不现实假设

推荐：
- `3` 或 `5`

---

### `--buy-slip`
买入滑点。

默认：`0.001`（0.1%）

### `--sell-slip`
卖出滑点。

默认：`0.001`

### `--commission`
单边佣金。

默认：`0.0003`

### `--tax`
卖出印花税。

默认：`0.001`

说明：
- 这些参数会直接影响策略是否从“纸面正收益”变成“实际没优势”

---

# 4C. `optimize_gain_turnover.py` 参数说明

## 网格搜索参数
这些参数支持用逗号输入多组值：

- `--days`
- `--min-gain`
- `--max-gain`
- `--quality-days`
- `--turnover`
- `--score-threshold`
- `--hold`
- `--max-extension`

例子：
```bash
--days 2,3 \
--min-gain 1.5,2.0 \
--max-gain 5,6 \
--quality-days 20,30 \
--turnover 0,1.5,3 \
--score-threshold 50,60,70 \
--hold 3,5 \
--max-extension 8,10,12
```

---

### `--min-trades`
最小交易数门槛。

作用：
- 防止只靠极少样本刷出看起来很高的结果

默认：`30`

推荐：
- 粗搜：`30`
- 更稳：`50` 或 `100`

---

### `--top-k`
最终输出前多少组参数。

推荐：
- `10 ~ 20`

---

## 5. 使用详解

### 5.1 日常筛选
适合每天盘后或盘中后半段做初筛。

#### 示例 1，标准版
```bash
~/.venv/bin/python stock_trend/screen_by_gain_turnover.py \
  --days 2 \
  --min-gain 2 \
  --max-gain 6 \
  --quality-days 20 \
  --turnover 3 \
  --min-volume 100000000 \
  --score-threshold 60 \
  --top-n 30
```

适合：
- 想找最近两天温和连涨
- 同时要求趋势健康、不过热

---

#### 示例 2，放宽版
```bash
~/.venv/bin/python stock_trend/screen_by_gain_turnover.py \
  --days 2 \
  --min-gain 1.5 \
  --max-gain 6 \
  --quality-days 20 \
  --turnover 0 \
  --score-threshold 50 \
  --top-n 50
```

适合：
- 想先粗筛更多股票
- 再人工复核

---

#### 示例 3，复盘指定日期
```bash
~/.venv/bin/python stock_trend/screen_by_gain_turnover.py \
  --date 2026-04-10 \
  --days 3 \
  --min-gain 2 \
  --max-gain 6 \
  --quality-days 20 \
  --turnover 3 \
  --score-threshold 60
```

适合：
- 复盘当时如果按这套策略会选出什么

---

### 5.2 历史回测

#### 示例 1，标准回测
```bash
~/.venv/bin/python stock_trend/backtest_gain_turnover.py \
  --days 2 \
  --min-gain 2 \
  --max-gain 6 \
  --quality-days 20 \
  --turnover 3 \
  --min-volume 100000000 \
  --score-threshold 60 \
  --hold 5 \
  --start 2024-01-01 \
  --end 2025-12-31 \
  --max-picks-per-day 3
```

---

#### 示例 2，测试短持有
```bash
~/.venv/bin/python stock_trend/backtest_gain_turnover.py \
  --days 2 \
  --min-gain 2 \
  --max-gain 6 \
  --quality-days 20 \
  --turnover 1.5 \
  --score-threshold 60 \
  --hold 3 \
  --start 2024-01-01 \
  --end 2025-12-31
```

---

### 5.3 参数寻优

#### 示例 1，粗搜
```bash
~/.venv/bin/python stock_trend/optimize_gain_turnover.py \
  --start 2024-01-01 \
  --end 2025-12-31 \
  --days 2,3 \
  --min-gain 1.5,2.0 \
  --max-gain 5,6 \
  --quality-days 20 \
  --turnover 0,1.5 \
  --score-threshold 50,60 \
  --hold 3 \
  --max-extension 10 \
  --max-picks-per-day 3 \
  --top-k 10
```

适合：
- 找方向
- 先看哪些参数大致有优势

---

#### 示例 2，精搜
```bash
~/.venv/bin/python stock_trend/optimize_gain_turnover.py \
  --start 2024-01-01 \
  --end 2025-12-31 \
  --days 3 \
  --min-gain 1.8,2.0,2.2 \
  --max-gain 5.5,6.0 \
  --quality-days 20,30 \
  --turnover 0,1.5,3 \
  --score-threshold 60,70,80 \
  --hold 3,5 \
  --max-extension 8,10,12 \
  --max-picks-per-day 3 \
  --min-trades 50 \
  --top-k 20
```

适合：
- 已经知道一个大致方向
- 想进一步缩小范围

---

## 6. 推荐执行步骤

这是我建议的标准流程。

### 第一步，先做粗搜
目标：找方向，不追求一步到位。

推荐粗搜范围：
- `days = 2,3`
- `min_gain = 1.5,2.0`
- `max_gain = 5,6`
- `quality_days = 20`
- `turnover = 0,1.5,3`
- `score_threshold = 50,60,70`
- `hold = 3,5`
- `max_extension = 8,10,12`

你要观察：
1. `days=2` 和 `days=3` 谁更强
2. `max_gain=5` 和 `6` 谁更稳
3. `hold=3` 还是 `5` 更合适
4. `score_threshold` 提高后，平均收益是否变好
5. 换手率过滤是否真的带来提升

---

### 第二步，缩小到精搜区间
粗搜后只保留最有希望的方向。

比如如果粗搜发现：
- `days=3` 明显优于 `days=2`
- `max_gain=6` 明显优于 `5`
- `score_threshold=70` 开始有提升

那精搜就集中在：
- `days=3`
- `min_gain=1.8,2.0,2.2`
- `max_gain=5.5,6.0`
- `score_threshold=60,70,80`
- `hold=3,5`

---

### 第三步，挑 2~3 组候选参数做单独回测
不要只看优化器表格。

对候选参数组分别做：
- 独立回测
- 看交易数
- 看胜率
- 看平均收益
- 看最大亏损
- 看是否过于依赖少数阶段行情

---

### 第四步，用最佳参数做当前筛选
当你确认某组参数比较稳，就用它跑当天筛选。

---

## 7. 我当前最推荐的起步参数

### 推荐起步版 A，平衡型
```bash
--days 2 \
--min-gain 2.0 \
--max-gain 6.0 \
--quality-days 20 \
--turnover 1.5 \
--min-volume 100000000 \
--score-threshold 60 \
--hold 3 \
--max-extension 10
```

特点：
- 不算太松
- 也不算太严
- 适合作为第一组标准参数

### 推荐起步版 B，更稳一些
```bash
--days 3 \
--min-gain 2.0 \
--max-gain 6.0 \
--quality-days 20 \
--turnover 1.5 \
--min-volume 100000000 \
--score-threshold 70 \
--hold 5 \
--max-extension 8
```

特点：
- 更偏中继趋势
- 信号更少，但质量可能更高

---

## 8. 注意事项

### 8.1 第一次会慢
原因：
- 要拉取前复权数据
- 要建立缓存

后续会快很多。

### 8.2 参数寻优不要只看“最高年化”
更应该综合看：
- 交易数是否足够
- Sharpe 是否稳定
- 最大亏损是否可接受
- 不同年份是否都能工作

### 8.3 不要把优化结果直接当实盘参数
优化结果容易过拟合。

更稳的做法：
- 训练区间找参数
- 验证区间看泛化
- 最后再上实盘观察

---

## 9. 推荐的一条完整执行链

### 先粗搜
```bash
~/.venv/bin/python stock_trend/optimize_gain_turnover.py \
  --start 2024-01-01 \
  --end 2025-12-31 \
  --days 2,3 \
  --min-gain 1.5,2.0 \
  --max-gain 5,6 \
  --quality-days 20 \
  --turnover 0,1.5,3 \
  --score-threshold 50,60,70 \
  --hold 3,5 \
  --max-extension 8,10,12 \
  --max-picks-per-day 3 \
  --top-k 10
```

### 再精搜
```bash
~/.venv/bin/python stock_trend/optimize_gain_turnover.py \
  --start 2024-01-01 \
  --end 2025-12-31 \
  --days 3 \
  --min-gain 1.8,2.0,2.2 \
  --max-gain 5.5,6.0 \
  --quality-days 20,30 \
  --turnover 0,1.5 \
  --score-threshold 60,70,80 \
  --hold 3,5 \
  --max-extension 8,10 \
  --max-picks-per-day 3 \
  --min-trades 50 \
  --top-k 20
```

### 最后用胜出的参数做筛选
```bash
~/.venv/bin/python stock_trend/screen_by_gain_turnover.py \
  --days 3 \
  --min-gain 2.0 \
  --max-gain 6.0 \
  --quality-days 20 \
  --turnover 1.5 \
  --min-volume 100000000 \
  --score-threshold 70 \
  --top-n 30
```

---

## 10. 结果文件位置

默认输出：

- 筛选结果：
  - `~/stock_reports/gain_screen_upgrade_YYYY-MM-DD.txt`

- 回测交易明细：
  - `~/stock_reports/gain_turnover_backtest_upgrade_START_END.csv`

- 参数寻优结果：
  - `~/stock_reports/gain_turnover_optimize_START_END.json`

---

## 11. 一句话建议

如果你现在就要开始，我建议：

1. 先跑一轮**粗搜**
2. 看 `days=3 / max_gain=6 / hold=3或5` 是否稳定更优
3. 再做**精搜**
4. 最后固定 2~3 套候选参数，做每日筛选和持续跟踪
