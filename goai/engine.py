"""goai 최상위 엔진: 규칙 엔진 + MCTS + 시간관리 + 판단(Judge) + 교정을 통합한다.

GameSession 하나가 "사람 vs AI" 한 판의 생애주기를 관리한다.
- human_move(...) : 사람의 수/패스/기권 적용
- ai_move(...)    : AI가 시간 예산 내에서 탐색 후 수/패스/기권 결정 및 적용
- reject_last_ai_decision(...) : 사용자가 "아니다"라고 했을 때
  방금 둔 AI의 수(패스/기권/떡수)를 되돌리고 교정을 저장, 즉시 재결정
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

from .board import Color, GoBoard, Move, Point, to_point
from .judge import ActionKind, CorrectionMemory, Decision, Judge
from .mcts import MCTS
from .model import NetworkWrapper
from .sgf import game_to_sgf, record_moves_from_board_history
from .timecontrol import TimeControl

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "models", "latest.pt"
)


@dataclass
class MoveLogEntry:
    move_number: int
    color: int
    decision: Decision
    think_time: float
    was_corrected: bool = False


@dataclass
class GameSession:
    komi: float = 6.5
    handicap_points: List[Point] = field(default_factory=list)
    time_control: TimeControl = field(default_factory=TimeControl)
    mcts_simulations: int = 160
    network_channels: int = 32
    network_blocks: int = 4
    model_path: Optional[str] = None
    black_player: str = "Human"
    white_player: str = "GoAI"

    def __post_init__(self):
        self.board = GoBoard(komi=self.komi, handicap_points=self.handicap_points or None)
        self._board_snapshots: List[GoBoard] = [self.board.copy()]

        model_path = self.model_path or (DEFAULT_MODEL_PATH if os.path.exists(DEFAULT_MODEL_PATH) else None)
        self.network = NetworkWrapper(
            channels=self.network_channels, blocks=self.network_blocks, checkpoint_path=model_path
        )
        self.mcts = MCTS(self.network)
        self.memory = CorrectionMemory()
        self.judge = Judge(memory=self.memory)

        self.clocks = self.time_control.new_clock_pair()
        self.log: List[MoveLogEntry] = []
        self.last_ai_decision: Optional[Decision] = None

    @property
    def to_move(self) -> Color:
        return self.board.to_move

    def is_over(self) -> bool:
        return self.board.is_game_over()

    # ------------------------------------------------------------------ #
    def human_move(self, point: Optional[Point]) -> None:
        color = self.board.to_move
        if point is None:
            self.board.play(Move.pass_(color))
        else:
            self.board.play(Move.play(color, point))
        self._board_snapshots.append(self.board.copy())

    def human_resign(self) -> None:
        color = self.board.to_move
        self.board.play(Move.resign(color))
        self._board_snapshots.append(self.board.copy())

    # ------------------------------------------------------------------ #
    def ai_move(self) -> Decision:
        color = self.board.to_move
        clock = self.clocks[color]
        clock.start_turn()

        time_budget = clock.time_left_for_budget()
        result = self.mcts.run(
            self.board, color, n_simulations=self.mcts_simulations,
            time_budget=time_budget, add_root_noise=False,
        )
        decision = self.judge.decide(self.board, color, result)

        think_time = clock.stop_turn()

        if decision.action == ActionKind.RESIGN:
            self.board.play(Move.resign(color))
        elif decision.action == ActionKind.PASS:
            self.board.play(Move.pass_(color))
        else:
            p = to_point(decision.point_index)
            self.board.play(Move.play(color, p))

        self._board_snapshots.append(self.board.copy())
        self.last_ai_decision = decision
        self.log.append(MoveLogEntry(
            move_number=decision.move_number, color=int(color), decision=decision, think_time=think_time,
        ))
        return decision

    # ------------------------------------------------------------------ #
    def reject_last_ai_decision(self, comment: str = "") -> Decision:
        """방금 AI가 둔 수(패스/기권/떡수)를 사용자가 부당하다고 판단했을 때 호출."""
        if self.last_ai_decision is None or not self.log:
            raise ValueError("되돌릴 AI의 수가 없습니다.")

        last_entry = self.log[-1]
        color = Color(last_entry.color)
        decision = last_entry.decision

        self.judge.record_user_correction(decision, color, comment)

        self._board_snapshots.pop()
        self.board = self._board_snapshots[-1].copy()
        self.log.pop()
        self.judge._despair_streak[color] = 0

        new_decision = self.ai_move()
        if self.log:
            self.log[-1].was_corrected = True
        return new_decision

    # ------------------------------------------------------------------ #
    def to_sgf(self, result: Optional[str] = None) -> str:
        moves = record_moves_from_board_history(self.board.history)
        res = result if result is not None else (self.board.result_string() if self.is_over() else "")
        return game_to_sgf(
            moves, komi=self.komi, result=res,
            black_player=self.black_player, white_player=self.white_player,
            handicap_points=self.handicap_points,
        )

    def save_sgf(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_sgf())

    def status(self) -> dict:
        return {
            "to_move": self.board.to_move.name,
            "move_count": len(self.board.history),
            "is_over": self.is_over(),
            "result": self.board.result_string() if self.is_over() else None,
            "black_clock": self.clocks[Color.BLACK].to_dict(),
            "white_clock": self.clocks[Color.WHITE].to_dict(),
            "score": self.board.score(),
        }
