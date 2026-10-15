"""신경망 입력 특징 평면(feature plane) 인코딩.

AlphaZero류 설계를 단순화한 버전. 19x19 보드, 채널 수를 적게 유지해
메모리가 매우 제한적인 환경(≈1GB RAM)에서도 학습/추론이 가능하도록 한다.

채널 구성 (총 N_CHANNELS):
    0: 현재 차례 색 돌 위치 (1/0)
    1: 상대 색 돌 위치 (1/0)
    2: 빈 점 (1/0)
    3: 직전 수 위치 (1/0) - 없으면 전부 0
    4: 직전전 수 위치 (1/0)
    5: 내 돌 그룹의 활로가 1인 곳(단수) 표시
    6: 상대 돌 그룹의 활로가 1인 곳(단수) 표시
    7: 유효수 마스크 (합법이면 1)
    8: 현재 차례가 흑이면 전부 1, 백이면 전부 0 (색 정보, 대칭 깨기용)
"""

from __future__ import annotations

from typing import List

import numpy as np

from .board import BOARD_SIZE, N_POINTS, PASS_INDEX, Color, GoBoard, idx, to_point

N_CHANNELS = 9
POLICY_SIZE = N_POINTS + 1  # 361개 점 + 패스


def move_to_index(point) -> int:
    """(row, col) -> 정책 벡터 인덱스 (0~360)."""
    return idx(point)


def index_to_move(index: int):
    """정책 벡터 인덱스 -> (row, col). PASS_INDEX는 호출 전에 체크할 것."""
    return to_point(index)


def pass_index() -> int:
    return PASS_INDEX


def policy_size() -> int:
    return POLICY_SIZE


def encode_board(board: GoBoard, to_move: Color) -> np.ndarray:
    """board 상태를 (N_CHANNELS, 19, 19) float32 텐서로 인코딩한다."""
    size = BOARD_SIZE
    planes = np.zeros((N_CHANNELS, size, size), dtype=np.float32)

    me = int(to_move)
    opp = int(to_move.opponent)
    b = np.array(board.board, dtype=np.int32).reshape(size, size)

    planes[0] = (b == me).astype(np.float32)
    planes[1] = (b == opp).astype(np.float32)
    planes[2] = (b == 0).astype(np.float32)

    if board.last_move_point is not None:
        r, c = to_point(board.last_move_point)
        planes[3, r, c] = 1.0
    if len(board.history) >= 2:
        prev_rec = board.history[-2]
        if prev_rec.move.point is not None:
            r, c = prev_rec.move.point
            planes[4, r, c] = 1.0

    # 단수(활로 1) 표시 - 전체 스캔은 비용이 있으나 19x19 규모에서는 충분히 빠르다.
    visited = set()
    for i in range(N_POINTS):
        if i in visited or b.flat[i] == 0:
            continue
        grp, libs = board._group_and_liberties_i(i)
        visited |= grp
        if len(libs) == 1:
            color_here = b.flat[i]
            ch = 5 if color_here == me else 6
            for g in grp:
                r, c = to_point(g)
                planes[ch, r, c] = 1.0

    mask = board.legal_move_mask(to_move)
    for i in range(N_POINTS):
        if mask[i]:
            r, c = to_point(i)
            planes[7, r, c] = 1.0

    if to_move == Color.BLACK:
        planes[8] = 1.0

    return planes


def encode_batch(boards_and_colors) -> np.ndarray:
    return np.stack([encode_board(b, c) for b, c in boards_and_colors], axis=0)
