"""제한시간 + 초읽기(byo-yomi) 시간 관리 모듈.

기본 규칙 (요청 사양):
    제한시간 1분(main_time=60초) + 초읽기 30초 x 3회(byoyomi_time=30, byoyomi_periods=3)

사용법:
    tc = TimeControl()
    clock = tc.new_clock_pair()          # 흑/백 각각의 PlayerClock 생성
    clock[Color.BLACK].start_turn()
    ... (AI/사람이 생각) ...
    clock[Color.BLACK].stop_turn()       # 소비 시간 차감, 초읽기 자동 전환/소진 판정
    if clock[Color.BLACK].is_flagged():  # 시간패
        ...
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional

from .board import Color


@dataclass
class TimeControl:
    main_time: float = 60.0       # 제한시간 (초)
    byoyomi_time: float = 30.0    # 초읽기 1회당 시간 (초)
    byoyomi_periods: int = 3      # 초읽기 횟수

    def new_clock_pair(self) -> Dict[Color, "PlayerClock"]:
        return {
            Color.BLACK: PlayerClock(self),
            Color.WHITE: PlayerClock(self),
        }


@dataclass
class PlayerClock:
    control: TimeControl
    main_time_left: float = field(init=False)
    byoyomi_periods_left: int = field(init=False)
    in_byoyomi: bool = field(init=False, default=False)
    flagged: bool = field(init=False, default=False)
    _turn_start: Optional[float] = field(init=False, default=None)
    last_think_time: float = field(init=False, default=0.0)

    def __post_init__(self):
        self.main_time_left = self.control.main_time
        self.byoyomi_periods_left = self.control.byoyomi_periods

    # ------------------------------------------------------------------ #
    def start_turn(self) -> None:
        self._turn_start = time.monotonic()

    def time_left_for_budget(self) -> float:
        """이번 수를 두는 데 "안전하게" 쓸 수 있는 권장 시간(초)을 돌려준다.

        - 아직 본시간이 남아 있으면 본시간을 잘게 나눠 쓰도록 권장치를 준다
          (남은 시간을 통째로 한 수에 쓰지 않도록 보수적으로 분배).
        - 초읽기 상태면 초읽기 한 턴 시간(여유 약간 두고)을 권장치로 준다.
        """
        if self.flagged:
            return 0.0
        if not self.in_byoyomi and self.main_time_left > 0:
            # 남은 수가 대략 얼마나 될지 모르므로 보수적으로 1/20 정도만 사용 권장,
            # 다만 최소/최대 한도를 둔다.
            budget = max(1.0, min(self.main_time_left * 0.08, self.main_time_left))
            return budget
        # 초읽기: 설정 시간에서 안전 마진(0.8초)을 뺀 값
        return max(0.5, self.control.byoyomi_time - 0.8)

    def stop_turn(self) -> float:
        """생각을 멈추고 소비한 시간을 차감한다. 소비 시간을 반환한다."""
        if self._turn_start is None:
            return 0.0
        elapsed = time.monotonic() - self._turn_start
        self._turn_start = None
        self.last_think_time = elapsed
        self._consume(elapsed)
        return elapsed

    def _consume(self, elapsed: float) -> None:
        if self.flagged:
            return
        if not self.in_byoyomi:
            if elapsed <= self.main_time_left:
                self.main_time_left -= elapsed
                return
            # 본시간 초과분은 초읽기로 넘어간다.
            overflow = elapsed - self.main_time_left
            self.main_time_left = 0.0
            self.in_byoyomi = True
            self._consume_byoyomi(overflow)
        else:
            self._consume_byoyomi(elapsed)

    def _consume_byoyomi(self, elapsed: float) -> None:
        remaining = elapsed
        while remaining > 0:
            if self.byoyomi_periods_left <= 0:
                self.flagged = True
                return
            if remaining <= self.control.byoyomi_time:
                # 이번 초읽기 한 턴 내에 뒀음 -> 다음 수를 위해 다시 꽉 찬 초읽기 유지
                return
            # 이번 초읽기를 다 써버림 -> 한 회 소진, 남은 시간을 다음 회에서 계속 소비
            remaining -= self.control.byoyomi_time
            self.byoyomi_periods_left -= 1

    def is_flagged(self) -> bool:
        return self.flagged

    def status_string(self) -> str:
        if self.flagged:
            return "시간패"
        if not self.in_byoyomi:
            return f"{self.main_time_left:.0f}s"
        return f"초읽기 {self.control.byoyomi_time:.0f}s x {self.byoyomi_periods_left}"

    def to_dict(self) -> dict:
        return {
            "main_time_left": round(self.main_time_left, 1),
            "in_byoyomi": self.in_byoyomi,
            "byoyomi_periods_left": self.byoyomi_periods_left,
            "byoyomi_time": self.control.byoyomi_time,
            "flagged": self.flagged,
            "last_think_time": round(self.last_think_time, 2),
        }
