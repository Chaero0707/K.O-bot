"""goai.board 규칙 엔진 정확성 테스트."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from goai.board import GoBoard, Color, Move, IllegalMoveError


def test_simple_capture():
    b = GoBoard()
    # 백 돌 하나를 흑이 4방향으로 포위해서 따낸다
    # 백: (1,1), 흑: (0,1),(1,0),(1,2),(2,1)
    b.play(Move.play(Color.BLACK, (0, 1)))
    b.play(Move.play(Color.WHITE, (1, 1)))
    b.play(Move.play(Color.BLACK, (1, 0)))
    b.play(Move.play(Color.WHITE, (5, 5)))  # 상관없는 수
    b.play(Move.play(Color.BLACK, (1, 2)))
    b.play(Move.play(Color.WHITE, (6, 6)))  # 상관없는 수
    rec = b.play(Move.play(Color.BLACK, (2, 1)))
    assert (1, 1) in rec.captured
    assert b.get((1, 1)) == Color.EMPTY
    assert b.captures[Color.BLACK] == 1


def test_suicide_illegal():
    b = GoBoard()
    # 흑이 백 돌 하나를 둘러싸서 자살수가 되는 상황을 만든다.
    b.play(Move.play(Color.BLACK, (0, 1)))
    b.play(Move.play(Color.WHITE, (10, 10)))
    b.play(Move.play(Color.BLACK, (1, 0)))
    b.play(Move.play(Color.WHITE, (10, 11)))
    b.play(Move.play(Color.BLACK, (1, 2)))
    b.play(Move.play(Color.WHITE, (10, 12)))
    # 이제 (1,1)에 백이 두면 자살수 (흑에게 둘러싸임, 따낼 돌 없음)
    try:
        b.play(Move.play(Color.WHITE, (1, 1)))
        assert False, "자살수가 허용되면 안 됩니다."
    except IllegalMoveError:
        pass


def test_ko_rule():
    # 표준 단순패(simple ko) 모양을 만든다.
    #   좌표: (0,1)=B (1,0)=B (2,1)=B  (0,2)=W (1,3)=W (2,2)=W  (1,2)=B
    #   이 상태에서 백이 (1,1)에 두면 흑 (1,2) 한 점을 따내며 단수(1liberty)의
    #   백 돌 하나가 (1,1)에 남는 전형적인 패 모양이 된다.
    b = GoBoard()
    seq = [
        (Color.BLACK, (0, 1)),
        (Color.WHITE, (0, 2)),
        (Color.BLACK, (1, 0)),
        (Color.WHITE, (1, 3)),
        (Color.BLACK, (2, 1)),
        (Color.WHITE, (2, 2)),
        (Color.BLACK, (1, 2)),
    ]
    for color, p in seq:
        b.play(Move.play(color, p))

    rec = b.play(Move.play(Color.WHITE, (1, 1)))  # 백이 흑 (1,2)를 따내며 패 발생
    assert (1, 2) in rec.captured
    assert b.get((1, 2)) == Color.EMPTY
    assert b.get((1, 1)) == Color.WHITE
    assert b.ko_point is not None

    # 흑이 바로 그 자리(1,2)에 되따내는 것은 동일 반상 재현을 일으키므로 금지되어야 한다.
    try:
        b.play(Move.play(Color.BLACK, (1, 2)))
        assert False, "패 규칙 위반: 즉시 되따냄은 금지되어야 합니다."
    except IllegalMoveError:
        pass

    # 흑이 다른 곳에 한 번 두고(패 해소), 백도 다른 곳에 두면
    assert b.is_legal_play(Color.BLACK, (1, 2)) is False
    b.play(Move.play(Color.BLACK, (10, 10)))
    b.play(Move.play(Color.WHITE, (10, 11)))
    # 이제는 흑이 (1,2)에 두어 백 (1,1) 하나를 다시 따낼 수 있다 (패 해소됨).
    assert b.is_legal_play(Color.BLACK, (1, 2))
    rec2 = b.play(Move.play(Color.BLACK, (1, 2)))
    assert (1, 1) in rec2.captured


def test_pass_and_game_over():
    b = GoBoard()
    b.play(Move.pass_(Color.BLACK))
    assert not b.is_game_over()
    b.play(Move.pass_(Color.WHITE))
    assert b.is_game_over()


def test_resign():
    b = GoBoard()
    b.play(Move.resign(Color.BLACK))
    assert b.is_game_over()
    assert b.result_string() == "W+R"


def test_scoring_simple_territory():
    b = GoBoard(komi=0.5)
    # 흑이 좌상단 2x2를 완전히 둘러싼 집을 만든다고 가정 (아주 단순화된 형태)
    # 간단히: 바둑판 전체가 비어있으면 영역 전체가 데임(중립) 처리됨을 확인
    s = b.score()
    assert s["black_total"] == 0
    assert s["white_total"] == 0.5  # komi만 반영, 영역 전체가 접촉없어 중립 아님? -> 빈 보드는 경계색 없음(neutral)
