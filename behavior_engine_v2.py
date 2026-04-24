# behavior_engine_v2.py

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class BehaviorResult:
    phase: str            # accumulation / wash / markup / distribution
    can_buy: bool
    confidence: float     # 0-1
    score: float
    reason: str


def analyze_behavior_v2(waves, structure_result):

    if not waves or not structure_result:
        return BehaviorResult(
            phase="unknown",
            can_buy=False,
            confidence=0.0,
            score=0.0,
            reason="数据不足"
        )

    last = waves[-1]
    prev = waves[-2] if len(waves) >= 2 else None

    up_waves = [w for w in waves if w.direction == "up"]
    down_waves = [w for w in waves if w.direction == "down"]

    last_up = up_waves[-1] if up_waves else None
    last_down = down_waves[-1] if down_waves else None

    score = 0
    reason = []

    # =========================
    # 1️⃣ 拉升期（最重要）
    # =========================
    if (
        structure_result.is_main_trend and
        structure_result.is_strong and
        last.direction == "up" and
        last_up and last_down and
        last_up.avg_volume > last_down.avg_volume and
        last_up.speed > last_down.speed
    ):
        score += 80
        reason.append("主升浪 + 放量上涨 + 速度优势")

        return BehaviorResult(
            phase="markup",
            can_buy=True,
            confidence=0.9,
            score=score,
            reason=" | ".join(reason)
        )

    # =========================
    # 2️⃣ 二次启动（黄金买点）
    # =========================
    if (
        structure_result.is_second_break and
        last.direction == "up" and
        last_up and last_down and
        last_down.pct > -12 and
        last_up.avg_volume >= last_down.avg_volume * 0.9
    ):
        score += 70
        reason.append("洗盘结束 + 二次启动")

        return BehaviorResult(
            phase="markup",
            can_buy=True,
            confidence=0.85,
            score=score,
            reason=" | ".join(reason)
        )

    # =========================
    # 3️⃣ 洗盘期
    # =========================
    if (
        last.direction == "down" and
        last_down and last_up and
        last_down.pct > -10 and
        last_down.avg_volume < last_up.avg_volume and
        last_down.days >= 3
    ):
        score += 40
        reason.append("缩量回调（洗盘）")

        return BehaviorResult(
            phase="wash",
            can_buy=False,
            confidence=0.6,
            score=score,
            reason=" | ".join(reason)
        )

    # =========================
    # 4️⃣ 吸筹期
    # =========================
    if (
        len(waves) >= 4 and
        all(abs(w.pct) < 8 for w in waves[-4:]) and
        not structure_result.is_strong
    ):
        score += 30
        reason.append("低波动震荡（吸筹）")

        return BehaviorResult(
            phase="accumulation",
            can_buy=False,
            confidence=0.5,
            score=score,
            reason=" | ".join(reason)
        )

    # =========================
    # 5️⃣ 出货期（一定要避开）
    # =========================
    if (
        last.direction == "down" and
        last_down and last_up and
        last_down.pct < -15 and
        last_down.avg_volume > last_up.avg_volume
    ):
        score += 90
        reason.append("放量大跌（出货）")

        return BehaviorResult(
            phase="distribution",
            can_buy=False,
            confidence=0.95,
            score=score,
            reason=" | ".join(reason)
        )

    # =========================
    # 默认：不清晰
    # =========================
    return BehaviorResult(
        phase="unclear",
        can_buy=False,
        confidence=0.3,
        score=10,
        reason="结构不清晰"
    )
