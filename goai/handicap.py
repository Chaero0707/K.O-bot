"""19x19 접바둑(핸디캡) 돌 배치 표준 좌표."""

from __future__ import annotations

from typing import Dict, List

from .board import Point

# 0-indexed (row, col). 화점 기준: 3, 9, 15 (1-indexed로는 4, 10, 16)
_STAR = [3, 9, 15]
_TL, _TM, _TR = _STAR[0], _STAR[1], _STAR[2]

HANDICAP_POINTS: Dict[int, List[Point]] = {
    2: [(_TL, _TR), (_TR, _TL)],
    3: [(_TL, _TR), (_TR, _TL), (_TL, _TL)],
    4: [(_TL, _TL), (_TL, _TR), (_TR, _TL), (_TR, _TR)],
    5: [(_TL, _TL), (_TL, _TR), (_TR, _TL), (_TR, _TR), (_TM, _TM)],
    6: [(_TL, _TL), (_TL, _TR), (_TR, _TL), (_TR, _TR), (_TL, _TM), (_TR, _TM)],
    7: [(_TL, _TL), (_TL, _TR), (_TR, _TL), (_TR, _TR), (_TL, _TM), (_TR, _TM), (_TM, _TM)],
    8: [(_TL, _TL), (_TL, _TR), (_TR, _TL), (_TR, _TR), (_TL, _TM), (_TR, _TM), (_TM, _TL), (_TM, _TR)],
    9: [(_TL, _TL), (_TL, _TR), (_TR, _TL), (_TR, _TR), (_TL, _TM), (_TR, _TM), (_TM, _TL), (_TM, _TR), (_TM, _TM)],
}


def get_handicap_points(n: int) -> List[Point]:
    if n not in HANDICAP_POINTS:
        raise ValueError(f"지원하지 않는 접바둑 점수입니다: {n} (2~9만 지원)")
    return list(HANDICAP_POINTS[n])
