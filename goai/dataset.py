"""기보(SGF) 폴더 -> 학습용 통합 데이터셋 변환.

`기보/` 폴더 안의 모든 *.sgf 파일을 재귀적으로 찾아 하나의
NPZ(압축) 파일로 병합한다. 각 샘플은 다음을 담는다:
    features : (N_CHANNELS, 19, 19)  - 그 수를 두기 직전 국면
    policy   : (362,) one-hot (실제 둔 자리=1, 나머지=0)  -- 지도학습 타깃
    value    : float, [-1, 1] -- 최종 승패 결과 (해당 수를 둔 색 관점)

대용량이 되면 메모리 문제가 생길 수 있어, 기본적으로 샘플을 셔플한 뒤
설정 가능한 최대 개수로 서브샘플링한다 (이 환경은 RAM이 매우 제한적이므로 중요).
"""

from __future__ import annotations

import glob
import os
import random
from typing import Iterator, List, Optional, Tuple

import numpy as np

from .board import Color, GoBoard, Move, N_POINTS
from .features import encode_board
from .sgf import GameRecord, load_sgf

DEFAULT_SGF_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "기보")
DEFAULT_DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "dataset.npz")

PASS_INDEX = N_POINTS


def find_sgf_files(sgf_dir: str = DEFAULT_SGF_DIR) -> List[str]:
    pattern = os.path.join(sgf_dir, "**", "*.sgf")
    return sorted(glob.glob(pattern, recursive=True))


def _result_winner(result: str) -> Optional[Color]:
    if not result:
        return None
    r = result.strip().upper()
    if r.startswith("B+"):
        return Color.BLACK
    if r.startswith("W+"):
        return Color.WHITE
    return None


def iter_game_samples(record: GameRecord) -> Iterator[Tuple[np.ndarray, int, float]]:
    """한 GameRecord에서 (features, policy_index, value) 샘플들을 생성한다.

    규칙 위반(비합법수) 수가 섞여 있으면 그 지점에서 해당 기보 처리를 중단한다
    (손상된 기보 방어). 결과를 알 수 없으면 value=0.0으로 둔다.
    """
    winner = _result_winner(record.result)
    board = GoBoard(komi=record.komi, handicap_points=record.handicap_points or None)

    buffered: List[Tuple[np.ndarray, int, Color]] = []

    for sgf_move in record.moves:
        color = sgf_move.color
        if color != board.to_move:
            break
        features = encode_board(board, color)
        if sgf_move.point is None:
            policy_index = PASS_INDEX
            move = Move.pass_(color)
        else:
            policy_index = sgf_move.point[0] * 19 + sgf_move.point[1]
            move = Move.play(color, sgf_move.point)

        try:
            board.play(move)
        except Exception:
            break

        buffered.append((features, policy_index, color))

    for features, policy_index, color in buffered:
        value = 0.0 if winner is None else (1.0 if winner == color else -1.0)
        yield features, policy_index, value


def build_dataset(
    sgf_dir: str = DEFAULT_SGF_DIR,
    output_path: str = DEFAULT_DATASET_PATH,
    max_samples: Optional[int] = 200_000,
    seed: int = 0,
) -> dict:
    """기보 폴더의 모든 SGF를 모아 하나의 npz 데이터셋으로 저장한다."""
    files = find_sgf_files(sgf_dir)
    all_features: List[np.ndarray] = []
    all_policy: List[int] = []
    all_value: List[float] = []

    n_games = 0
    n_parse_errors = 0

    for path in files:
        try:
            games = load_sgf(path)
        except Exception:
            n_parse_errors += 1
            continue
        for record in games:
            if record.size != 19:
                continue
            n_games += 1
            for feat, pidx, val in iter_game_samples(record):
                all_features.append(feat)
                all_policy.append(pidx)
                all_value.append(val)

    n_total = len(all_features)
    rng = random.Random(seed)
    indices = list(range(n_total))
    rng.shuffle(indices)
    if max_samples is not None and n_total > max_samples:
        indices = indices[:max_samples]

    if indices:
        X = np.stack([all_features[i] for i in indices], axis=0).astype(np.float32)
        P = np.array([all_policy[i] for i in indices], dtype=np.int64)
        V = np.array([all_value[i] for i in indices], dtype=np.float32)
    else:
        X = np.zeros((0, 9, 19, 19), dtype=np.float32)
        P = np.zeros((0,), dtype=np.int64)
        V = np.zeros((0,), dtype=np.float32)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.savez_compressed(output_path, features=X, policy=P, value=V)

    return {
        "sgf_files_found": len(files),
        "sgf_parse_errors": n_parse_errors,
        "games_used": n_games,
        "total_samples_before_subsample": n_total,
        "samples_saved": int(X.shape[0]),
        "output_path": output_path,
    }


# 하위 호환 별칭
build_combined_dataset = build_dataset
DEFAULT_COMBINED_PATH = DEFAULT_DATASET_PATH


if __name__ == "__main__":
    import json
    stats = build_dataset()
    print(json.dumps(stats, ensure_ascii=False, indent=2))
