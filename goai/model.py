"""정책망 + 가치망 (경량 ResNet, PyTorch CPU 전용 환경 고려).

메모리가 매우 제한적인 샌드박스(≈1GB RAM, GPU 없음)에서도 학습/추론이
가능하도록 채널 수와 블록 수를 작게 유지한 AlphaZero 스타일 네트워크.

출력:
    policy_logits : (N_POINTS + 1,)  # 361개 점 + 패스
    value         : (1,)             # [-1, 1], 현재 차례 기준 승률 추정치
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .board import N_POINTS
from .features import N_CHANNELS

PASS_INDEX = N_POINTS


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        return F.relu(out)


class GoNet(nn.Module):
    """경량 정책+가치 네트워크.

    channels=32, blocks=4 정도가 CPU/저메모리 환경에서 현실적인 기본값.
    필요 시 생성자 인자로 조절 가능.
    """

    def __init__(self, channels: int = 32, blocks: int = 4, board_size: int = 19):
        super().__init__()
        self.board_size = board_size
        n_points = board_size * board_size

        self.input_conv = nn.Conv2d(N_CHANNELS, channels, 3, padding=1, bias=False)
        self.input_bn = nn.BatchNorm2d(channels)
        self.res_blocks = nn.ModuleList([ResidualBlock(channels) for _ in range(blocks)])

        # 정책 헤드
        self.policy_conv = nn.Conv2d(channels, 2, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * n_points, n_points + 1)  # +1 = 패스

        # 가치 헤드
        self.value_conv = nn.Conv2d(channels, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(n_points, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = F.relu(self.input_bn(self.input_conv(x)))
        for block in self.res_blocks:
            out = block(out)

        p = F.relu(self.policy_bn(self.policy_conv(out)))
        p = p.flatten(1)
        policy_logits = self.policy_fc(p)

        v = F.relu(self.value_bn(self.value_conv(out)))
        v = v.flatten(1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy_logits, value.squeeze(-1)


class NetworkWrapper:
    """추론/저장/로드를 캡슐화한 래퍼. MCTS와 학습 스크립트에서 공용으로 사용."""

    def __init__(self, channels: int = 32, blocks: int = 4, device: str = "cpu", checkpoint_path: Optional[str] = None):
        self.device = torch.device(device)
        self.net = GoNet(channels=channels, blocks=blocks).to(self.device)
        self.net.eval()
        self.channels = channels
        self.blocks = blocks
        if checkpoint_path and os.path.exists(checkpoint_path):
            self.load(checkpoint_path)

    @torch.no_grad()
    def infer(self, features: np.ndarray) -> Tuple[np.ndarray, float]:
        """단일 보드 추론. features: (N_CHANNELS, 19, 19) -> (policy_probs(362,), value)"""
        x = torch.from_numpy(features).unsqueeze(0).to(self.device)
        logits, value = self.net(x)
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
        return probs, float(value.cpu().numpy()[0])

    @torch.no_grad()
    def infer_batch(self, features_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x = torch.from_numpy(features_batch).to(self.device)
        logits, value = self.net(x)
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        return probs, value.cpu().numpy()

    def save(self, path: str, extra: Optional[dict] = None) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "state_dict": self.net.state_dict(),
            "channels": self.channels,
            "blocks": self.blocks,
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    def load(self, path: str) -> dict:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.net.load_state_dict(payload["state_dict"])
        self.net.eval()
        return payload

    def train_mode(self) -> None:
        self.net.train()

    def eval_mode(self) -> None:
        self.net.eval()


def count_parameters(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters())
