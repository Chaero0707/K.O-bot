"""셀프플레이: 현재 모델끼리 자동으로 대국해 SGF 기보를 생성한다.

생성된 기보는 `기보/selfplay/` 폴더에 저장되며, 이후 goai.dataset으로
병합 -> goai.trainer로 학습하는 강화학습 루프의 한 축을 이룬다.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .board import Color, GoBoard, Move, to_point
from .mcts import MCTS, PASS_INDEX
from .model import NetworkWrapper
from .sgf import game_to_sgf, record_moves_from_board_history

DEFAULT_SELFPLAY_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "기보", "selfplay"
)
DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "models", "latest.pt"
)


@dataclass
class SelfPlayConfig:
    n_simulations: int = 100
    max_moves: int = 500
    temperature_moves: int = 20
    resign_threshold: float = -0.95
    resign_min_move: int = 40


def _sample_move_from_visits(visit_counts: dict, temperature: float) -> int:
    items = list(visit_counts.items())
    indices = [i for i, _ in items]
    counts = np.array([c for _, c in items], dtype=np.float64)
    if temperature <= 1e-6:
        return indices[int(np.argmax(counts))]
    counts = np.power(counts, 1.0 / temperature)
    probs = counts / counts.sum()
    return int(np.random.choice(indices, p=probs))


def play_one_game(
    network: NetworkWrapper,
    config: Optional[SelfPlayConfig] = None,
    komi: float = 6.5,
) -> dict:
    cfg = config or SelfPlayConfig()
    mcts = MCTS(network)
    board = GoBoard(komi=komi)

    despair_streak = {Color.BLACK: 0, Color.WHITE: 0}
    move_count = 0
    resigned_by: Optional[Color] = None

    while not board.is_game_over() and move_count < cfg.max_moves:
        color = board.to_move
        result = mcts.run(board, color, n_simulations=cfg.n_simulations, add_root_noise=True)

        if result.root_value <= cfg.resign_threshold and move_count >= cfg.resign_min_move:
            despair_streak[color] += 1
        else:
            despair_streak[color] = 0

        if despair_streak[color] >= 3:
            resigned_by = color
            board.play(Move.resign(color))
            break

        temperature = 1.0 if move_count < cfg.temperature_moves else 0.0
        move_index = _sample_move_from_visits(result.visit_counts, temperature)

        if move_index == PASS_INDEX:
            board.play(Move.pass_(color))
        else:
            board.play(Move.play(color, to_point(move_index)))
        move_count += 1

    result_str = board.result_string()
    return {
        "board": board,
        "result": result_str,
        "move_count": move_count,
        "resigned": resigned_by is not None,
    }


def run_selfplay_session(
    n_games: int = 5,
    output_dir: str = DEFAULT_SELFPLAY_DIR,
    model_path: str = DEFAULT_MODEL_PATH,
    channels: int = 32,
    blocks: int = 4,
    n_simulations: int = 100,
    max_seconds: Optional[float] = None,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    network = NetworkWrapper(
        channels=channels, blocks=blocks,
        checkpoint_path=model_path if os.path.exists(model_path) else None,
    )
    cfg = SelfPlayConfig(n_simulations=n_simulations)

    games_played = 0
    start = time.time()
    saved_paths: List[str] = []

    for g in range(n_games):
        if max_seconds is not None and (time.time() - start) > max_seconds:
            break
        game_result = play_one_game(network, cfg)
        board = game_result["board"]
        moves = record_moves_from_board_history(board.history)
        sgf_text = game_to_sgf(
            moves, komi=board.komi, result=game_result["result"],
            black_player="GoAI-selfplay", white_player="GoAI-selfplay",
            event="selfplay",
        )
        ts = int(time.time() * 1000)
        path = os.path.join(output_dir, f"selfplay_{ts}_{g}.sgf")
        with open(path, "w", encoding="utf-8") as f:
            f.write(sgf_text)
        saved_paths.append(path)
        games_played += 1
        print(f"[selfplay {games_played}/{n_games}] result={game_result['result']} "
              f"moves={game_result['move_count']} -> {path}")

    return {
        "games_played": games_played,
        "output_dir": output_dir,
        "saved_paths": saved_paths,
        "elapsed_seconds": time.time() - start,
    }


def main():
    parser = argparse.ArgumentParser(description="goai 셀프플레이 기보 생성 스크립트")
    parser.add_argument("--games", type=int, default=5)
    parser.add_argument("--output-dir", default=DEFAULT_SELFPLAY_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument("--max-seconds", type=float, default=None)
    args = parser.parse_args()

    result = run_selfplay_session(
        n_games=args.games, output_dir=args.output_dir, model_path=args.model,
        channels=args.channels, blocks=args.blocks, n_simulations=args.simulations,
        max_seconds=args.max_seconds,
    )
    print("셀프플레이 완료:", result["games_played"], "게임,", f"{result['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
