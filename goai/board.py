"""19x19 바둑판 규칙 엔진 (고성능 버전).

- 돌 놓기 / 따내기 / 자살수 금지 / 패(ko) 금지 (상황적 초과수 - situational superko,
  조브리스트 해시 기반으로 빠르게 판정)
- 집 계산 (중국식 면적 계산법: 돌 수 + 둘러싼 집 수, 덤 포함)
- 기권/넘기기 처리
- 핸디캡(접바둑) 돌 배치
- MCTS에서 초당 수백~수천 회 호출해도 버틸 수 있도록 deepcopy 대신
  플랫 배열 + 조브리스트 해시 + 최소 변경 복원(undo) 방식을 사용한다.

이 모듈은 신경망/탐색과 독립적으로 동작하는 순수 규칙 레이어입니다.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

BOARD_SIZE = 19  # 본 엔진은 19x19 전용
N_POINTS = BOARD_SIZE * BOARD_SIZE
PASS_INDEX = N_POINTS  # 정책 벡터 등에서 "패스"를 가리키는 인덱스로 사용

Point = Tuple[int, int]  # (row, col), 0-indexed


class Color(IntEnum):
    EMPTY = 0
    BLACK = 1
    WHITE = 2

    @property
    def opponent(self) -> "Color":
        if self is Color.BLACK:
            return Color.WHITE
        if self is Color.WHITE:
            return Color.BLACK
        return Color.EMPTY


class MoveKind(IntEnum):
    PLAY = 0
    PASS = 1
    RESIGN = 2


@dataclass(frozen=True)
class Move:
    kind: MoveKind
    color: Color
    point: Optional[Point] = None  # PLAY일 때만 사용

    @staticmethod
    def play(color: Color, point: Point) -> "Move":
        return Move(MoveKind.PLAY, color, point)

    @staticmethod
    def pass_(color: Color) -> "Move":
        return Move(MoveKind.PASS, color, None)

    @staticmethod
    def resign(color: Color) -> "Move":
        return Move(MoveKind.RESIGN, color, None)

    def __str__(self) -> str:
        if self.kind == MoveKind.PASS:
            return f"{self.color.name} PASS"
        if self.kind == MoveKind.RESIGN:
            return f"{self.color.name} RESIGN"
        r, c = self.point
        return f"{self.color.name} {coord_to_gtp((r, c))}"


GTP_COLS = "ABCDEFGHJKLMNOPQRST"  # I는 관례상 건너뜀


def coord_to_gtp(point: Point) -> str:
    r, c = point
    return f"{GTP_COLS[c]}{BOARD_SIZE - r}"


def gtp_to_coord(s: str) -> Point:
    s = s.strip().upper()
    col_letter = s[0]
    row_num = int(s[1:])
    c = GTP_COLS.index(col_letter)
    r = BOARD_SIZE - row_num
    return (r, c)


def idx(p: Point) -> int:
    r, c = p
    return r * BOARD_SIZE + c


def to_point(i: int) -> Point:
    return divmod(i, BOARD_SIZE)


class IllegalMoveError(Exception):
    pass


@dataclass
class MoveRecord:
    """재현/기보 작성을 위한 한 수의 기록."""

    move: Move
    captured: FrozenSet[Point] = field(default_factory=frozenset)
    was_ko_capture_point: Optional[Point] = None
    comment: str = ""


# ---------------------------------------------------------------------- #
# 전역 사전 계산 테이블 (이웃, 조브리스트 해시)
# ---------------------------------------------------------------------- #

def _build_neighbors() -> List[Tuple[int, ...]]:
    table: List[Tuple[int, ...]] = []
    for i in range(N_POINTS):
        r, c = divmod(i, BOARD_SIZE)
        cand = []
        if r > 0:
            cand.append(i - BOARD_SIZE)
        if r < BOARD_SIZE - 1:
            cand.append(i + BOARD_SIZE)
        if c > 0:
            cand.append(i - 1)
        if c < BOARD_SIZE - 1:
            cand.append(i + 1)
        table.append(tuple(cand))
    return table


NEIGHBORS: List[Tuple[int, ...]] = _build_neighbors()

_rng = random.Random(0xC0FFEE19)
# zobrist[point][color] : color는 1(흑) 또는 2(백)
ZOBRIST: List[Tuple[int, int, int]] = [
    (0, _rng.getrandbits(64), _rng.getrandbits(64)) for _ in range(N_POINTS)
]
ZOBRIST_TURN = _rng.getrandbits(64)  # 차례(흑/백) 구분용 비트


class GoBoard:
    """19x19 바둑판 상태와 규칙을 담당하는 클래스 (성능 최적화 버전)."""

    __slots__ = (
        "size",
        "komi",
        "board",
        "to_move",
        "history",
        "_position_hashes",
        "_zobrist_hash",
        "ko_point",
        "consecutive_passes",
        "resigned_by",
        "handicap_points",
        "captures",
        "last_move_point",
    )

    def __init__(self, komi: float = 6.5, handicap_points: Optional[List[Point]] = None):
        self.size = BOARD_SIZE
        self.komi = komi
        self.board: List[int] = [0] * N_POINTS  # 0 empty, 1 black, 2 white
        self.to_move: Color = Color.BLACK
        self.history: List[MoveRecord] = []
        self._zobrist_hash = 0
        self._position_hashes: Set[int] = set()
        self.ko_point: Optional[int] = None
        self.consecutive_passes = 0
        self.resigned_by: Optional[Color] = None
        self.handicap_points: List[Point] = []
        self.captures: Dict[Color, int] = {Color.BLACK: 0, Color.WHITE: 0}
        self.last_move_point: Optional[int] = None

        if handicap_points:
            self.place_handicap(handicap_points)
            self.to_move = Color.WHITE  # 접바둑은 백이 먼저 둠

        self._position_hashes.add(self._current_hash())

    # ------------------------------------------------------------------ #
    # 기본 유틸
    # ------------------------------------------------------------------ #
    def copy(self) -> "GoBoard":
        new = GoBoard.__new__(GoBoard)
        new.size = self.size
        new.komi = self.komi
        new.board = self.board[:]  # 얕은 복사로 충분 (int 리스트)
        new.to_move = self.to_move
        new.history = list(self.history)
        new._zobrist_hash = self._zobrist_hash
        new._position_hashes = set(self._position_hashes)
        new.ko_point = self.ko_point
        new.consecutive_passes = self.consecutive_passes
        new.resigned_by = self.resigned_by
        new.handicap_points = list(self.handicap_points)
        new.captures = dict(self.captures)
        new.last_move_point = self.last_move_point
        return new

    def in_bounds(self, p: Point) -> bool:
        r, c = p
        return 0 <= r < self.size and 0 <= c < self.size

    def neighbors(self, p: Point) -> List[Point]:
        return [to_point(n) for n in NEIGHBORS[idx(p)]]

    def get(self, p: Point) -> Color:
        return Color(self.board[idx(p)])

    def _get_i(self, i: int) -> int:
        return self.board[i]

    def _current_hash(self) -> int:
        return self._zobrist_hash ^ (ZOBRIST_TURN if self.to_move == Color.WHITE else 0)

    def place_handicap(self, points: List[Point]) -> None:
        for p in points:
            i = idx(p)
            self.board[i] = Color.BLACK
            self._zobrist_hash ^= ZOBRIST[i][Color.BLACK]
        self.handicap_points = list(points)

    # ------------------------------------------------------------------ #
    # 그룹 / 활로(기) 계산 (인덱스 기반, 빠른 BFS)
    # ------------------------------------------------------------------ #
    def _group_and_liberties_i(self, i: int) -> Tuple[Set[int], Set[int]]:
        color = self.board[i]
        assert color != Color.EMPTY
        stack = [i]
        visited: Set[int] = {i}
        liberties: Set[int] = set()
        board = self.board
        while stack:
            cur = stack.pop()
            for n in NEIGHBORS[cur]:
                nc = board[n]
                if nc == 0:
                    liberties.add(n)
                elif nc == color and n not in visited:
                    visited.add(n)
                    stack.append(n)
        return visited, liberties

    def group_at(self, p: Point) -> Tuple[Set[Point], Set[Point]]:
        """p가 속한 돌 그룹과, 그 그룹의 활로(빈 이웃점) 집합을 반환."""
        grp, libs = self._group_and_liberties_i(idx(p))
        return {to_point(x) for x in grp}, {to_point(x) for x in libs}

    # ------------------------------------------------------------------ #
    # 합법성 검사 (deepcopy 없이, 바뀐 칸만 되돌리는 방식)
    # ------------------------------------------------------------------ #
    def is_legal_play(self, color: Color, p: Point) -> bool:
        return self._legal_i(int(color), idx(p)) is not None

    def _legal_i(self, color: int, i: int) -> Optional[Tuple[Set[int], Optional[int]]]:
        """합법이면 (잡힌 돌 인덱스 집합, ko 후보 인덱스)를 반환, 아니면 None."""
        board = self.board
        if board[i] != 0:
            return None
        if self.ko_point is not None and i == self.ko_point:
            return None

        opp = 2 if color == 1 else 1
        board[i] = color  # 임시로 둬 본다
        h = self._zobrist_hash ^ ZOBRIST[i][color]

        captured: Set[int] = set()
        for n in NEIGHBORS[i]:
            if board[n] == opp:
                grp, libs = self._group_and_liberties_i(n)
                if not libs:
                    captured |= grp

        for c_i in captured:
            board[c_i] = 0
            h ^= ZOBRIST[c_i][opp]

        my_grp, my_libs = self._group_and_liberties_i(i)
        suicide = len(my_libs) == 0

        new_ko_point = None
        if len(captured) == 1 and len(my_grp) == 1:
            new_ko_point = next(iter(captured))

        would_repeat = False
        if not suicide:
            next_turn_hash = h ^ (ZOBRIST_TURN if opp == 2 else 0)
            # 다음 차례는 opp 이므로, to_move==WHITE(2)일 때 TURN 비트 적용
            next_turn_hash = h ^ (ZOBRIST_TURN if opp == Color.WHITE else 0)
            if next_turn_hash in self._position_hashes:
                would_repeat = True

        # 되돌리기
        for c_i in captured:
            board[c_i] = opp
        board[i] = 0

        if suicide or would_repeat:
            return None
        return captured, new_ko_point

    def legal_moves(self, color: Color) -> List[Point]:
        moves = []
        col = int(color)
        board = self.board
        for i in range(N_POINTS):
            if board[i] == 0 and self._legal_i(col, i) is not None:
                moves.append(to_point(i))
        return moves

    def legal_move_mask(self, color: Color) -> List[bool]:
        """길이 362(361점 + 패스)의 합법 여부 마스크. 패스는 항상 True."""
        col = int(color)
        board = self.board
        mask = [False] * (N_POINTS + 1)
        for i in range(N_POINTS):
            if board[i] == 0 and self._legal_i(col, i) is not None:
                mask[i] = True
        mask[PASS_INDEX] = True
        return mask

    # ------------------------------------------------------------------ #
    # 수 적용
    # ------------------------------------------------------------------ #
    def play(self, move: Move) -> MoveRecord:
        if move.color != self.to_move:
            raise IllegalMoveError(f"지금은 {self.to_move.name} 차례입니다.")

        if move.kind == MoveKind.RESIGN:
            self.resigned_by = move.color
            rec = MoveRecord(move=move)
            self.history.append(rec)
            return rec

        if move.kind == MoveKind.PASS:
            self.consecutive_passes += 1
            self.to_move = self.to_move.opponent
            self.last_move_point = None
            rec = MoveRecord(move=move)
            self.history.append(rec)
            self._position_hashes.add(self._current_hash())
            return rec

        p = move.point
        i = idx(p)
        col = int(move.color)
        result = self._legal_i(col, i)
        if result is None:
            raise IllegalMoveError(f"{coord_to_gtp(p)}은(는) 둘 수 없는 자리입니다.")
        captured, ko_candidate = result

        board = self.board
        board[i] = col
        self._zobrist_hash ^= ZOBRIST[i][col]
        opp = int(move.color.opponent)
        for c_i in captured:
            board[c_i] = 0
            self._zobrist_hash ^= ZOBRIST[c_i][opp]
        if captured:
            self.captures[move.color] += len(captured)

        self.ko_point = ko_candidate
        self.consecutive_passes = 0
        self.to_move = self.to_move.opponent
        self.last_move_point = i
        self._position_hashes.add(self._current_hash())

        captured_points = frozenset(to_point(x) for x in captured)
        ko_point_p = to_point(ko_candidate) if ko_candidate is not None else None
        rec = MoveRecord(move=move, captured=captured_points, was_ko_capture_point=ko_point_p)
        self.history.append(rec)
        return rec

    def is_game_over(self) -> bool:
        if self.resigned_by is not None:
            return True
        return self.consecutive_passes >= 2

    # ------------------------------------------------------------------ #
    # 집 계산 (중국식 면적 계산)
    # ------------------------------------------------------------------ #
    def score(self) -> Dict[str, float]:
        """중국식(area scoring) 계가: 돌 + 자기 집, 덤은 백에게 가산."""
        territory = {Color.BLACK: 0, Color.WHITE: 0}
        stones = {Color.BLACK: 0, Color.WHITE: 0}
        visited: Set[int] = set()
        board = self.board

        for i in range(N_POINTS):
            col = board[i]
            if col != 0:
                stones[Color(col)] += 1
                continue
            if i in visited:
                continue
            region = {i}
            stack = [i]
            border_colors: Set[int] = set()
            while stack:
                cur = stack.pop()
                for n in NEIGHBORS[cur]:
                    if board[n] == 0:
                        if n not in region:
                            region.add(n)
                            stack.append(n)
                    else:
                        border_colors.add(board[n])
            visited |= region
            if border_colors == {1}:
                territory[Color.BLACK] += len(region)
            elif border_colors == {2}:
                territory[Color.WHITE] += len(region)

        black_total = stones[Color.BLACK] + territory[Color.BLACK]
        white_total = stones[Color.WHITE] + territory[Color.WHITE] + self.komi

        return {
            "black_stones": stones[Color.BLACK],
            "white_stones": stones[Color.WHITE],
            "black_territory": territory[Color.BLACK],
            "white_territory": territory[Color.WHITE],
            "komi": self.komi,
            "black_total": black_total,
            "white_total": white_total,
            "black_captures": self.captures[Color.BLACK],
            "white_captures": self.captures[Color.WHITE],
        }

    def result_string(self) -> str:
        if self.resigned_by is not None:
            winner = self.resigned_by.opponent
            return f"{'B' if winner == Color.BLACK else 'W'}+R"
        s = self.score()
        diff = s["black_total"] - s["white_total"]
        if diff > 0:
            return f"B+{diff:g}"
        elif diff < 0:
            return f"W+{-diff:g}"
        return "Draw"

    # ------------------------------------------------------------------ #
    # 표시용
    # ------------------------------------------------------------------ #
    def __str__(self) -> str:
        sym = {0: ".", 1: "X", 2: "O"}
        lines = []
        header = "   " + " ".join(GTP_COLS)
        lines.append(header)
        for r in range(self.size):
            row_label = f"{self.size - r:2d} "
            row_syms = " ".join(sym[self.board[r * self.size + c]] for c in range(self.size))
            lines.append(row_label + row_syms)
        return "\n".join(lines)
