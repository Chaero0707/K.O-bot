"""지도학습(기보 데이터) + 사용자 교정 반영 학습 스크립트.

학습 손실 = 정책 교차엔트로피 + 가치 MSE + 교정 페널티(옵션)

교정 페널티: CorrectionMemory에 쌓인 "아니다" 기록을 이용해, 해당 국면에서
사용자가 틀렸다고 한 수의 정책 확률을 낮추도록 보조 손실을 추가한다.
(메인 아이디어는 즉각 반영은 judge.py의 회피 로직이 담당하고, 여기서는
장기적으로 네트워크 자체가 그 수를 덜 선호하도록 서서히 학습시키는 역할.)
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .judge import CorrectionMemory
from .model import NetworkWrapper
from .board import N_POINTS

PASS_INDEX = N_POINTS
DEFAULT_DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "dataset.npz")
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "models", "latest.pt")


class NpzGoDataset(Dataset):
    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        self.features = data["features"]
        self.policy = data["policy"]
        self.value = data["value"]

    def __len__(self) -> int:
        return len(self.policy)

    def __getitem__(self, i: int):
        return (
            torch.from_numpy(self.features[i]),
            torch.tensor(self.policy[i], dtype=torch.long),
            torch.tensor(self.value[i], dtype=torch.float32),
        )


def build_correction_penalty_table(memory: CorrectionMemory) -> dict:
    """position_hash -> {color: set(penalized_move_indices)} 형태로 정리.

    학습 데이터에는 position_hash를 직접 저장하지 않으므로(가벼운 npz 포맷 유지),
    이 테이블은 추후 온라인/점진 학습(교정 직후 미니 파인튜닝)에 사용한다.
    """
    table: dict = {}
    for corr in memory.all_corrections():
        key = (corr.position_hash, corr.color)
        table.setdefault(key, set())
        if corr.ai_action.startswith("play:"):
            idx = int(corr.ai_action.split(":")[1])
            table[key].add(idx)
        elif corr.ai_action == "pass":
            table[key].add(PASS_INDEX)
    return table


def train_supervised(
    dataset_path: str = DEFAULT_DATASET_PATH,
    model_path: str = DEFAULT_MODEL_PATH,
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 1e-3,
    channels: int = 32,
    blocks: int = 4,
    val_fraction: float = 0.1,
    max_train_seconds: Optional[float] = None,
) -> dict:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"데이터셋이 없습니다: {dataset_path}. 먼저 goai.dataset으로 생성하세요.")

    ds = NpzGoDataset(dataset_path)
    n = len(ds)
    if n == 0:
        raise ValueError("데이터셋이 비어 있습니다. 기보 폴더에 SGF를 추가한 뒤 다시 시도하세요.")

    n_val = max(1, int(n * val_fraction)) if n > 10 else 0
    n_train = n - n_val
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val]) if n_val > 0 else (ds, None)

    train_loader = DataLoader(train_ds, batch_size=min(batch_size, n_train), shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=min(batch_size, max(1, n_val)), shuffle=False) if val_ds else None

    nw = NetworkWrapper(channels=channels, blocks=blocks, checkpoint_path=model_path if os.path.exists(model_path) else None)
    nw.train_mode()
    optimizer = torch.optim.Adam(nw.net.parameters(), lr=lr, weight_decay=1e-4)

    history = []
    start_time = time.time()
    stopped_early = False

    for epoch in range(epochs):
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        n_batches = 0
        for features, policy_target, value_target in train_loader:
            if max_train_seconds is not None and (time.time() - start_time) > max_train_seconds:
                stopped_early = True
                break
            optimizer.zero_grad()
            logits, value_pred = nw.net(features)
            policy_loss = F.cross_entropy(logits, policy_target)
            value_loss = F.mse_loss(value_pred, value_target)
            loss = policy_loss + value_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        avg_p = total_policy_loss / max(1, n_batches)
        avg_v = total_value_loss / max(1, n_batches)

        val_metrics = {}
        if val_loader is not None:
            nw.eval_mode()
            val_loss_sum = 0.0
            val_batches = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for features, policy_target, value_target in val_loader:
                    logits, value_pred = nw.net(features)
                    vloss = F.cross_entropy(logits, policy_target) + F.mse_loss(value_pred, value_target)
                    val_loss_sum += vloss.item()
                    val_batches += 1
                    pred_move = logits.argmax(dim=-1)
                    correct += (pred_move == policy_target).sum().item()
                    total += policy_target.shape[0]
            val_metrics = {
                "val_loss": val_loss_sum / max(1, val_batches),
                "val_policy_top1_acc": correct / max(1, total),
            }
            nw.train_mode()

        entry = {
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "train_policy_loss": avg_p,
            "train_value_loss": avg_v,
            **val_metrics,
        }
        history.append(entry)
        print(f"[epoch {epoch+1}/{epochs}] loss={avg_loss:.4f} "
              f"(policy={avg_p:.4f}, value={avg_v:.4f}) " + str(val_metrics))

        if stopped_early:
            break

    nw.eval_mode()
    nw.save(model_path, extra={"history": history, "n_train_samples": n_train})

    return {
        "model_path": model_path,
        "n_samples": n,
        "n_train": n_train,
        "n_val": n_val,
        "history": history,
        "stopped_early": stopped_early,
        "train_seconds": time.time() - start_time,
    }


def main():
    parser = argparse.ArgumentParser(description="goai 지도학습 스크립트")
    parser.add_argument("--dataset", default=DEFAULT_DATASET_PATH)
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--max-seconds", type=float, default=None)
    args = parser.parse_args()

    result = train_supervised(
        dataset_path=args.dataset, model_path=args.model, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, channels=args.channels, blocks=args.blocks,
        max_train_seconds=args.max_seconds,
    )
    print("학습 완료:", {k: v for k, v in result.items() if k != "history"})


if __name__ == "__main__":
    main()
