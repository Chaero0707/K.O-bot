"""넘기기(pass) / 기권(resign) / 떡수(blunder) 판단 로직.

AI는 매 수마다 아래와 같은 "근거(evidence)"를 남기고, 그 근거에 기반해
PLAY / PASS / RESIGN 중 하나를 결정한다. 사용자는 이 판단이 부당하다고
생각하면 "아니다"라고 교정할 수 있고, 그 교정은 CorrectionMemory에 저장되어
이후 똑같은(또는 유사한) 상황에서 같은 실수를 반복하지 않도록 즉시 반영되며,
추후 학습(trainer)에도 사용된다.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from .board import Color, GoBoard, Move, MoveKind, N_POINTS
from .mcts import MCTSResult, PASS_INDEX

DEFAULT_CORRECTIONS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "corrections", "corrections.jsonl"
)


class ActionKind(str, Enum):
    PLAY = "play"
    PASS = "pass"
    RESIGN = "resign"


@dataclass
class Decision:
    action: ActionKind
    point_index: Optional[int]  # PLAY일 때만 의미 있음
    reason: str                 # 사람이 읽을 수 있는 한국어 근거 설명
    evidence: Dict             # 수치적 근거 (win_prob, visit_count 등)
    position_hash: int          # 교정 매칭을 위한 현재 국면 해시
    move_number: int

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["action"] = self.action.value
        return d


@dataclass
class Correction:
    """사용자가 "아니다"라고 되돌린 기록 한 건."""

    position_hash: int
    color: int  # Color.value
    ai_action: str
    ai_point_index: Optional[int]
    move_number: int
    user_comment: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return asdict(self)


class CorrectionMemory:
    """사용자의 "아니다" 교정 이력을 저장/조회하는 저장소.

    - 즉시 반영: 같은 국면 해시 + 같은 색 + 같은 행동 조합이 다시 나타나면
      해당 행동(pass/resign/그 떡수 자리)을 다시 선택하지 않도록 회피한다.
    - 학습 반영: trainer.py가 이 파일을 읽어 지도학습 손실에 페널티 항으로 사용한다.
    """

    def __init__(self, path: str = DEFAULT_CORRECTIONS_PATH):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._index: Dict[tuple, List[Correction]] = {}
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    corr = Correction(**d)
                    self._index.setdefault(self._key(corr.position_hash, corr.color, corr.ai_action), []).append(corr)
                except Exception:
                    continue

    @staticmethod
    def _key(position_hash: int, color: int, ai_action: str) -> tuple:
        return (position_hash, color, ai_action)

    def add(self, correction: Correction) -> None:
        self._index.setdefault(
            self._key(correction.position_hash, correction.color, correction.ai_action), []
        ).append(correction)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(correction.to_dict(), ensure_ascii=False) + "\n")

    def was_marked_wrong(self, position_hash: int, color: int, ai_action: str) -> bool:
        return len(self._index.get(self._key(position_hash, color, ai_action), [])) > 0

    def all_corrections(self) -> List[Correction]:
        out = []
        for lst in self._index.values():
            out.extend(lst)
        return out

    def count(self) -> int:
        return sum(len(v) for v in self._index.values())


@dataclass
class JudgeConfig:
    resign_threshold: float = -0.90       # root_value가 이보다 낮으면 기권 후보
    resign_min_move_number: int = 40      # 너무 초반에는 기권하지 않음 (실수 방지)
    resign_consecutive_required: int = 3  # 연속 N수 동안 계속 절망적이어야 기권
    pass_value_margin: float = 0.02       # "더 둬도 이득이 거의 없다" 판단 여유값
    blunder_drop_threshold: float = 0.30  # 수를 둔 뒤 승률이 이만큼 급락하면 떡수로 기록


class Judge:
    """MCTS 결과 + 보드 상태를 바탕으로 최종 행동(수/넘기기/기권)을 결정한다."""

    def __init__(self, config: Optional[JudgeConfig] = None, memory: Optional[CorrectionMemory] = None):
        self.config = config or JudgeConfig()
        self.memory = memory or CorrectionMemory()
        self._despair_streak: Dict[int, int] = {Color.BLACK: 0, Color.WHITE: 0}

    def reset_game_state(self) -> None:
        self._despair_streak = {Color.BLACK: 0, Color.WHITE: 0}

    def decide(self, board: GoBoard, color: Color, mcts_result: MCTSResult) -> Decision:
        cfg = self.config
        move_number = len(board.history) + 1
        position_hash = board._current_hash()

        total_visits = sum(mcts_result.visit_counts.values()) or 1
        best_idx = mcts_result.move_index
        best_visits = mcts_result.visit_counts.get(best_idx, 0)
        root_value = mcts_result.root_value  # 현재 둘 색(color) 관점의 승률 추정 [-1,1]

        evidence = {
            "root_value": round(root_value, 4),
            "total_visits": total_visits,
            "best_move_visits": best_visits,
            "best_move_visit_ratio": round(best_visits / total_visits, 4),
            "move_number": move_number,
        }

        # ---------------- 기권 판단 ---------------- #
        if root_value <= cfg.resign_threshold and move_number >= cfg.resign_min_move_number:
            self._despair_streak[color] += 1
        else:
            self._despair_streak[color] = 0
        evidence["despair_streak"] = self._despair_streak[color]

        if self._despair_streak[color] >= cfg.resign_consecutive_required:
            if not self.memory.was_marked_wrong(position_hash, int(color), ActionKind.RESIGN.value):
                reason = (
                    f"{cfg.resign_consecutive_required}수 연속으로 승률 추정치가 "
                    f"{cfg.resign_threshold*100:.0f}% 이하({root_value*100:.1f}%)로 나와 "
                    f"더 두어도 역전이 어렵다고 판단해 기권합니다."
                )
                return Decision(ActionKind.RESIGN, None, reason, evidence, position_hash, move_number)
            else:
                evidence["resign_suppressed_by_user_correction"] = True

        # ---------------- 넘기기(패스) 판단 ---------------- #
        if best_idx == PASS_INDEX:
            if not self.memory.was_marked_wrong(position_hash, int(color), ActionKind.PASS.value):
                reason = (
                    "탐색 결과 보드 위에 더 이상 이득이 되는 수가 없고 "
                    f"패스가 가장 높은 방문 수({best_visits}/{total_visits})를 받아 넘깁니다."
                )
                return Decision(ActionKind.PASS, None, reason, evidence, position_hash, move_number)
            else:
                # 패스가 틀렸다고 교정된 적 있는 국면 -> 패스를 제외하고 다음으로 방문 수 높은 실제 수를 선택
                ranked = sorted(mcts_result.visit_counts.items(), key=lambda kv: -kv[1])
                fallback = next((i for i, _ in ranked if i != PASS_INDEX), None)
                evidence["pass_suppressed_by_user_correction"] = True
                if fallback is not None:
                    reason = (
                        "원래는 패스가 최선으로 나왔지만, 이전에 사용자가 같은 상황에서 "
                        "패스가 틀렸다고 교정했으므로 대신 차선의 실제 수를 둡니다."
                    )
                    return Decision(ActionKind.PLAY, fallback, reason, evidence, position_hash, move_number)

        # ---------------- 일반 착수 ---------------- #
        if not self.memory.was_marked_wrong(position_hash, int(color), f"play:{best_idx}"):
            reason = (
                f"MCTS 탐색 {total_visits}회 중 해당 수가 {best_visits}회로 가장 많이 방문되어 "
                f"선택합니다. (추정 승률 {((root_value+1)/2)*100:.1f}%)"
            )
            return Decision(ActionKind.PLAY, best_idx, reason, evidence, position_hash, move_number)
        else:
            ranked = sorted(mcts_result.visit_counts.items(), key=lambda kv: -kv[1])
            fallback = next(
                (i for i, _ in ranked
                 if i != best_idx and not self.memory.was_marked_wrong(position_hash, int(color), f"play:{i}")),
                None,
            )
            evidence["play_suppressed_by_user_correction"] = True
            if fallback is not None:
                reason = (
                    "원래 최선수가 이전에 사용자로부터 '떡수'라고 교정받은 적이 있어 "
                    "차선의 수를 대신 둡니다."
                )
                return Decision(ActionKind.PLAY, fallback, reason, evidence, position_hash, move_number)
            reason = "대안이 없어 원래 최선수를 그대로 둡니다."
            return Decision(ActionKind.PLAY, best_idx, reason, evidence, position_hash, move_number)

    # ---------------------------------------------------------------- #
    def detect_blunder(self, value_before: float, value_after_opp_view: float) -> bool:
        """내가 수를 둔 직후, 상대 관점 가치가 과도하게 좋아졌다면(=내 승률 급락) 떡수로 판단.

        value_before: 내가 두기 전, 내 관점 승률 추정치
        value_after_opp_view: 내가 둔 후, 상대 관점 승률 추정치 (parent 기준 부호 반전 필요시 호출부에서 처리)
        """
        my_value_after = -value_after_opp_view
        drop = value_before - my_value_after
        return drop >= self.config.blunder_drop_threshold

    def record_user_correction(
        self,
        decision: Decision,
        color: Color,
        user_comment: str = "",
    ) -> Correction:
        ai_action = decision.action.value
        if decision.action == ActionKind.PLAY:
            ai_action = f"play:{decision.point_index}"
        corr = Correction(
            position_hash=decision.position_hash,
            color=int(color),
            ai_action=ai_action,
            ai_point_index=decision.point_index,
            move_number=decision.move_number,
            user_comment=user_comment or "사용자가 '아니다'라고 교정함",
        )
        self.memory.add(corr)
        return corr
