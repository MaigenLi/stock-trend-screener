"""
波段结构分析器

提供：
1. Wave 统一数据结构
2. analyze_structure() — 最近涨跌段对比分析
3. 二级启动、强势结构、主升浪判断
"""

from dataclasses import dataclass


@dataclass
class Wave:
    """统一波段数据结构"""
    start: int
    end: int
    direction: str      # "up" / "down"
    pct: float          # 涨跌幅（+10 表示 +10%）
    days: int           # 持续天数
    avg_volume: float   # 均量
    volume_power: float # 量能爆发力：max(seg) / avg(prev5)
    wave_high: float    # 波段内最高价
    wave_low: float     # 波段内最低价

    @property
    def speed(self) -> float:
        """每日涨跌幅（速度），%/天"""
        return self.pct / max(self.days, 1)


@dataclass
class StructureResult:
    """结构分析结果"""

    # 强势结构：涨得多 > 跌得多 × 1.2 且 速度更快
    is_strong: bool

    # 主升浪特征：涨幅>15%、天数≤15、跌幅>-10%
    is_main_trend: bool

    # 二次启动：上一个涨段高点接近+回调可控+有洗盘周期
    is_second_break: bool

    # 结构评分（用于排序）
    score: float

    # 文字说明
    reason: str

    # 指标明细
    strength_ratio: float       # 涨跌幅比值
    up_speed: float             # 涨段速度 (%/天)
    down_speed: float           # 跌段速度 (%/天)


def analyze_structure(
    up_waves: list[Wave],
    down_waves: list[Wave],
) -> StructureResult | None:
    """
    分析最近涨段 vs 最近跌段的结构特征

    Args:
        up_waves: 所有上涨波段（从老到新）
        down_waves: 所有下跌波段（从老到新）

    Returns:
        StructureResult 或 None（数据不足时）
    """
    if not up_waves or not down_waves:
        return None

    last_up = up_waves[-1]
    last_down = down_waves[-1]
    up_pct = last_up.pct
    down_pct = abs(last_down.pct)

    # ── 1. 强势结构（核心） ─────────────────────────────────
    # 涨跌幅压制：涨幅 > 跌幅 × 1.2
    # 速度压制：单位时间涨得比跌得快
    strength_ratio = up_pct / max(down_pct, 0.01)
    is_strong = (
        up_pct > down_pct * 1.2
        and last_up.speed > last_down.speed
    )

    # ── 2. 主升浪 ─────────────────────────────────────────
    is_main_trend = (
        up_pct > 15
        and last_up.days <= 15
        and down_pct < 10
    )

    # ── 3. 二次启动（洗盘结束） ───────────────────────────
    is_second_break = False
    if len(up_waves) >= 2:
        prev_up = up_waves[-2]
        is_second_break = (
            up_pct > prev_up.pct * 0.8   # 新高接近或超越前高
            and down_pct < 12            # 回调不深
            and last_down.days >= 3      # 有洗盘周期
        )

    # ── 4. 结构评分 ──────────────────────────────────────
    score = 0.0
    score += up_pct * 1.5           # 涨幅贡献
    score -= down_pct * 1.2         # 下跌惩罚
    score += last_up.speed * 10     # 速度奖励
    if is_main_trend:
        score += 20
    if is_second_break:
        score += 15
    if is_strong:
        score += 10

    # ── 5. 可解释输出 ────────────────────────────────────
    reasons = []
    if is_strong:
        reasons.append("上涨强于下跌")
    if is_main_trend:
        reasons.append("主升浪结构")
    if is_second_break:
        reasons.append("二次启动")
    if not reasons:
        reasons.append("结构一般")

    return StructureResult(
        is_strong=is_strong,
        is_main_trend=is_main_trend,
        is_second_break=is_second_break,
        score=round(score, 2),
        reason=" | ".join(reasons),
        strength_ratio=round(strength_ratio, 2),
        up_speed=round(last_up.speed, 2),
        down_speed=round(last_down.speed, 2),
    )
