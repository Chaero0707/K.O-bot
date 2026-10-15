"""SGF(Smart Game Format) 기보 읽기/쓰기 모듈 (19x19 바둑 전용, FF[4] 부분 호환).

- parse_sgf(text)  -> GameRecord
- load_sgf(path)   -> GameRecord
- game_to_sgf(...) -> str
- save_sgf(path, ...)

SGF 좌표는 'aa' ~ 'ss' (19x19, a=0 ~ s=18) 형식을 사용한다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .board import BOARD_SIZE, Color, Move, MoveKind, Point

SGF_LETTERS = "abcdefghijklmnopqrs"  # 19개, 인덱스 0~18


def point_to_sgf(p: Point) -> str:
    r, c = p
    return f"{SGF_LETTERS[c]}{SGF_LETTERS[r]}"


def sgf_to_point(s: str) -> Optional[Point]:
    if len(s) != 2:
        return None
    c = SGF_LETTERS.index(s[0])
    r = SGF_LETTERS.index(s[1])
    return (r, c)


@dataclass
class SgfMove:
    color: Color
    point: Optional[Point]  # None이면 패스
    comment: str = ""


@dataclass
class GameRecord:
    """SGF 한 판 분량의 메타데이터 + 수순."""

    size: int = BOARD_SIZE
    komi: float = 6.5
    handicap_points: List[Point] = field(default_factory=list)
    result: str = ""
    black_player: str = ""
    white_player: str = ""
    event: str = ""
    date: str = ""
    moves: List[SgfMove] = field(default_factory=list)
    extra: dict = field(default_factory=dict)  # 기타 SGF 속성 보존


_PROP_RE = re.compile(r"([A-Z]{1,2})((?:\[[^\]]*\])+)")
_VALUE_RE = re.compile(r"\[([^\]]*)\]")


def _unescape(v: str) -> str:
    return v.replace("\\]", "]").replace("\\\\", "\\")


def parse_sgf(text: str) -> List[GameRecord]:
    """SGF 텍스트를 파싱해 GameRecord 리스트로 반환한다.

    간단한 선형(분기 없는 주 수순만 취급) 파서: 괄호 안 ';' 로 구분된 노드들을
    순서대로 읽는다. 변화도(variation)는 무시하고 주 라인만 사용한다.
    """
    games: List[GameRecord] = []
    pos = 0
    n = len(text)
    while True:
        start = text.find("(", pos)
        if start == -1:
            break
        # 괄호 균형 맞춰서 한 게임 트리 범위를 찾는다.
        depth = 0
        end = start
        for i in range(start, n):
            if text[i] == "(":
                depth += 1
            elif text[i] == ")":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        game_text = text[start + 1:end]
        games.append(_parse_single_game(game_text))
        pos = end + 1
    return games


def _parse_single_game(body: str) -> GameRecord:
    rec = GameRecord()
    # 첫 변화만 사용 (서브 변화 '(' 가 나오면 그 이전까지만 주수순으로 취급)
    sub_start = body.find("(")
    main_body = body if sub_start == -1 else body[:sub_start]

    nodes = main_body.split(";")
    first_node = True
    for node in nodes:
        node = node.strip()
        if not node:
            continue
        props = _PROP_RE.findall(node)
        is_root_like = False
        for key, raw_values in props:
            values = [_unescape(v) for v in _VALUE_RE.findall(raw_values)]
            if key == "SZ":
                rec.size = int(values[0])
                is_root_like = True
            elif key == "KM":
                rec.komi = float(values[0])
                is_root_like = True
            elif key == "RE":
                rec.result = values[0]
                is_root_like = True
            elif key == "PB":
                rec.black_player = values[0]
                is_root_like = True
            elif key == "PW":
                rec.white_player = values[0]
                is_root_like = True
            elif key == "EV":
                rec.event = values[0]
                is_root_like = True
            elif key == "DT":
                rec.date = values[0]
                is_root_like = True
            elif key == "HA":
                is_root_like = True
            elif key == "AB":
                for v in values:
                    p = sgf_to_point(v)
                    if p:
                        rec.handicap_points.append(p)
                is_root_like = True
            elif key == "AW":
                is_root_like = True
            elif key in ("B", "W"):
                color = Color.BLACK if key == "B" else Color.WHITE
                v = values[0] if values else ""
                point = sgf_to_point(v) if v else None
                # 빈 문자열 또는 'tt'(구버전 패스 표기)는 패스로 취급
                if v == "" or v.lower() == "tt":
                    point = None
                rec.moves.append(SgfMove(color=color, point=point))
            elif key == "C":
                if rec.moves:
                    rec.moves[-1].comment = values[0]
            else:
                rec.extra.setdefault(key, []).extend(values)
        first_node = False
    return rec


def load_sgf(path: str) -> List[GameRecord]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return parse_sgf(text)


def _escape(v: str) -> str:
    return v.replace("\\", "\\\\").replace("]", "\\]")


def game_to_sgf(
    moves: List[Tuple[Color, Optional[Point]]],
    komi: float = 6.5,
    result: str = "",
    black_player: str = "AI",
    white_player: str = "Human",
    handicap_points: Optional[List[Point]] = None,
    event: str = "goai self-play",
    date: str = "",
    comments: Optional[List[str]] = None,
) -> str:
    """수순 리스트를 SGF 문자열로 직렬화한다.

    moves: [(color, point_or_None), ...]  point가 None이면 패스.
    """
    parts = ["(;FF[4]GM[1]CA[UTF-8]"]
    parts.append(f"SZ[{BOARD_SIZE}]")
    parts.append(f"KM[{komi:g}]")
    if result:
        parts.append(f"RE[{_escape(result)}]")
    if black_player:
        parts.append(f"PB[{_escape(black_player)}]")
    if white_player:
        parts.append(f"PW[{_escape(white_player)}]")
    if event:
        parts.append(f"EV[{_escape(event)}]")
    if date:
        parts.append(f"DT[{_escape(date)}]")
    if handicap_points:
        parts.append(f"HA[{len(handicap_points)}]")
        ab = "".join(f"[{point_to_sgf(p)}]" for p in handicap_points)
        parts.append(f"AB{ab}")

    for i, (color, point) in enumerate(moves):
        key = "B" if color == Color.BLACK else "W"
        val = point_to_sgf(point) if point is not None else ""
        parts.append(f";{key}[{val}]")
        if comments and i < len(comments) and comments[i]:
            parts.append(f"C[{_escape(comments[i])}]")

    parts.append(")")
    return "".join(parts)


def save_sgf(path: str, *args, **kwargs) -> None:
    sgf_text = game_to_sgf(*args, **kwargs)
    with open(path, "w", encoding="utf-8") as f:
        f.write(sgf_text)


def record_moves_from_board_history(history) -> List[Tuple[Color, Optional[Point]]]:
    """GoBoard.history (MoveRecord 리스트) -> SGF용 (color, point) 리스트.

    RESIGN은 SGF 수순에는 포함하지 않고 RE[] 결과에만 반영한다.
    """
    out: List[Tuple[Color, Optional[Point]]] = []
    for rec in history:
        if rec.move.kind == MoveKind.RESIGN:
            continue
        if rec.move.kind == MoveKind.PASS:
            out.append((rec.move.color, None))
        else:
            out.append((rec.move.color, rec.move.point))
    return out
