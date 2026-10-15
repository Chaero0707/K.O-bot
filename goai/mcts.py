"""PUCT 기반 몬테카를로 트리 탐색 (AlphaZero 스타일, 단일 스레드 CPU용).

- 정책망의 사전 확률 + 가치망의 리프 평가를 사용
- 패스도 하나의 "수"로 트리에 포함시켜 AI가 스스로 넘기기를 판단할 수 있게 한다.
- 탐색 시간/횟수 제한은 TimeControl과 연동해 바깥에서 제어한다.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .board import BOARD_SIZE, Color, GoBoard, Move, MoveKind, N_POINTS, to_point
from .features import encode_board
from .model import NetworkWrapper

PASS_INDEX = N_POINTS

C_PUCT = 1.5


class MCTSNode:
    __slots__ = (
        "parent", "move_index", "color_to_move", "children", "P", "N", "W", "Q",
        "is_expanded", "legal_indices",
    )

    def __init__(self, parent: Optional["MCTSNode"], move_index: Optional[int], color_to_move: Color):
        self.parent = parent
        self.move_index = move_index
        self.color_to_move = color_to_move
        self.children: Dict[int, "MCTSNode"] = {}
        self.P: Dict[int, float] = {}
        self.N: Dict[int, int] = {}
        self.W: Dict[int, float] = {}
        self.Q: Dict[int, float] = {}
        self.is_expanded = False
        self.legal_indices: List[int] = []

    def total_visits(self) -> int:
        return sum(self.N.values())


@dataclass
class MCTSResult:
    move_index: int
    visit_counts: Dict[int, int]
    root_value: float
    pv: List[int] = field(default_factory=list)


class MCTS:
    def __init__(self, network: NetworkWrapper, c_puct: float = C_PUCT,
                 dirichlet_alpha: float = 0.03, dirichlet_eps: float = 0.25):
        self.network = network
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps

    def _evaluate(self, board: GoBoard, color: Color) -> Tuple[np.ndarray, float, List[int]]:
        features = encode_board(board, color)
        probs, value = self.network.infer(features)
        mask = board.legal_move_mask(color)
        legal_indices = [i for i, ok in enumerate(mask) if ok]
        return probs, value, legal_indices

    def _expand(self, node: MCTSNode, board: GoBoard, add_noise: bool = False) -> float:
        probs, value, legal_indices = self._evaluate(board, node.color_to_move)
        node.legal_indices = legal_indices

        legal_probs = np.array([probs[i] for i in legal_indices], dtype=np.float64)
        s = legal_probs.sum()
        if s <= 1e-8:
            legal_probs = np.ones_like(legal_probs) / max(1, len(legal_probs))
        else:
            legal_probs = legal_probs / s

        if add_noise and len(legal_indices) > 0:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal_indices))
            legal_probs = (1 - self.dirichlet_eps) * legal_probs + self.dirichlet_eps * noise

        for i, p in zip(legal_indices, legal_probs):
            node.P[i] = float(p)
            node.N[i] = 0
            node.W[i] = 0.0
            node.Q[i] = 0.0
        node.is_expanded = True
        return value

    def _select_child_index(self, node: MCTSNode) -> int:
        total_N = max(1, node.total_visits())
        sqrt_total = math.sqrt(total_N)
        best_score = -1e18
        best_i = node.legal_indices[0]
        for i in node.legal_indices:
            u = self.c_puct * node.P[i] * sqrt_total / (1 + node.N[i])
            score = node.Q[i] + u
            if score > best_score:
                best_score = score
                best_i = i
        return best_i

    def _apply_move_index(self, board: GoBoard, color: Color, move_index: int) -> None:
        if move_index == PASS_INDEX:
            board.play(Move.pass_(color))
        else:
            p = to_point(move_index)
            board.play(Move.play(color, p))

    def run(
        self,
        root_board: GoBoard,
        color: Color,
        n_simulations: int = 200,
        time_budget: Optional[float] = None,
        add_root_noise: bool = False,
    ) -> MCTSResult:
        root = MCTSNode(parent=None, move_index=None, color_to_move=color)
        root_value = self._expand(root, root_board, add_noise=add_root_noise)

        start = time.monotonic()
        sims_done = 0
        while sims_done < n_simulations:
            if time_budget is not None and (time.monotonic() - start) >= time_budget:
                break
            self._simulate(root, root_board)
            sims_done += 1

        visit_counts = dict(root.N)
        if visit_counts:
            best_move = max(visit_counts.items(), key=lambda kv: kv[1])[0]
        else:
            best_move = PASS_INDEX
        return MCTSResult(move_index=best_move, visit_counts=visit_counts, root_value=root_value)

    def _simulate(self, root: MCTSNode, root_board: GoBoard) -> None:
        board = root_board.copy()
        node = root
        path: List[Tuple[MCTSNode, int]] = []

        while True:
            if board.is_game_over():
                result = self._terminal_value(board, node.color_to_move)
                self._backup(path, result)
                return

            if not node.is_expanded:
                leaf_value = self._expand(node, board)
                self._backup(path, leaf_value)
                return

            if not node.legal_indices:
                self._backup(path, 0.0)
                return

            move_i = self._select_child_index(node)
            path.append((node, move_i))

            next_color = node.color_to_move.opponent
            self._apply_move_index(board, node.color_to_move, move_i)

            if move_i not in node.children:
                child = MCTSNode(parent=node, move_index=move_i, color_to_move=next_color)
                node.children[move_i] = child
            node = node.children[move_i]

    def _terminal_value(self, board: GoBoard, perspective_color: Color) -> float:
        if board.resigned_by is not None:
            winner = board.resigned_by.opponent
        else:
            s = board.score()
            diff = s["black_total"] - s["white_total"]
            winner = Color.BLACK if diff > 0 else (Color.WHITE if diff < 0 else None)
        if winner is None:
            return 0.0
        return 1.0 if winner == perspective_color else -1.0

    def _backup(self, path: List[Tuple[MCTSNode, int]], leaf_value: float) -> None:
        value = leaf_value
        for node, move_i in reversed(path):
            node.N[move_i] += 1
            node.W[move_i] += value
            node.Q[move_i] = node.W[move_i] / node.N[move_i]
            value = -value
