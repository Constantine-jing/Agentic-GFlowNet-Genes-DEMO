"""
train.py — GFlowNet training loop with Trajectory Balance loss + replay buffer.

Key improvements over naive TB:
1. **Replay buffer**: stores the top-K highest-reward (actions, reward) pairs.
   Each epoch trains on a mix of fresh samples AND replayed high-reward
   trajectories. The replay trajectories get their log_pf recomputed under
   the *current* policy, so the loss signal is always fresh.

2. **Temperature annealing**: starts warm (explore) and cools (exploit).
   This helps the policy discover good subsets early, then concentrate.

3. **Reward shaping**: reward^exponent amplifies differences between good
   and bad subsets, giving sharper gradient signal.
"""
from __future__ import annotations

import heapq
import time
from collections import Counter
from typing import Callable

import pandas as pd
import torch

from src import config
from src.env.reward import evaluate_subset
from src.gflownet.policy import GFNPolicy
from src.gflownet.sampler import sample_trajectories, compute_log_pf_for_actions


def _gene_id_lookup() -> list[str]:
    cfg = config.active()
    expr = pd.read_csv(cfg["expr"], index_col=0)
    return expr.index.tolist()


class ReplayBuffer:
    """Keep top-K trajectories by reward. Each entry = (actions_tensor, reward)."""
    def __init__(self, capacity: int = 64):
        self.capacity = capacity
        self.heap: list[tuple[float, int, torch.Tensor]] = []
        self._counter = 0  # tiebreaker for heap

    def add(self, actions: torch.Tensor, reward: float):
        """actions: (subset_size,) int tensor."""
        self._counter += 1
        if len(self.heap) < self.capacity:
            heapq.heappush(self.heap, (reward, self._counter, actions.detach().cpu()))
        elif reward > self.heap[0][0]:
            heapq.heapreplace(self.heap, (reward, self._counter, actions.detach().cpu()))

    def sample(self, n: int, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
        """Return (actions, rewards) tensors, sampled uniformly from buffer."""
        if len(self.heap) == 0:
            return None, None
        n = min(n, len(self.heap))
        idxs = torch.randint(0, len(self.heap), (n,))
        actions = torch.stack([self.heap[i][2] for i in idxs]).to(device)
        rewards = torch.tensor([self.heap[i][0] for i in idxs], device=device)
        return actions, rewards

    def __len__(self):
        return len(self.heap)


def train(
    n_epochs: int = 300,
    batch_size: int = 16,
    replay_batch: int = 16,
    subset_size: int = 15,
    lr: float = 5e-3,
    log_z_lr: float = 1e-0,
    reward_exponent: float = 8.0,
    temp_start: float = 2.0,
    temp_end: float = 0.5,
    replay_capacity: int = 128,
    seed: int = 42,
    device: str = "cpu",
    log_every: int = 10,
) -> dict:
    """
    Train a GFlowNet to sample high-reward gene subsets.
    Returns dict with policy, history, sample_counter, reward_cache, gene_ids.
    """
    torch.manual_seed(seed)

    gene_ids = _gene_id_lookup()
    n_genes = len(gene_ids)
    print(f"[train] dataset={config.DATASET}  n_genes={n_genes}  subset_size={subset_size}")
    print(f"[train] epochs={n_epochs}  batch={batch_size}  replay={replay_batch}  "
          f"temp={temp_start}→{temp_end}  R^{reward_exponent}")

    policy = GFNPolicy(n_genes=n_genes).to(device)
    optimizer = torch.optim.Adam([
        {"params": [p for n, p in policy.named_parameters() if n != "log_Z"], "lr": lr},
        {"params": [policy.log_Z], "lr": log_z_lr},
    ])

    replay = ReplayBuffer(capacity=replay_capacity)
    reward_cache: dict[tuple, float] = {}
    sample_counter: Counter = Counter()
    history = {
        "epoch": [], "mean_reward": [], "max_reward": [],
        "loss": [], "n_unique": [], "temp": [], "best_ever": [],
    }
    best_ever = 0.0

    t0 = time.time()
    for epoch in range(n_epochs):
        # Temperature annealing: linear decay
        frac = min(epoch / max(n_epochs * 0.7, 1), 1.0)
        temp = temp_start + (temp_end - temp_start) * frac

        # --- fresh samples ---
        traj = sample_trajectories(policy, batch_size, subset_size,
                                   temperature=temp, device=device)
        final_state = traj["final_state"]
        log_pf = traj["log_pf"]
        actions = traj["actions"]

        rewards = torch.zeros(batch_size, device=device)
        for b in range(batch_size):
            chosen_idx = torch.nonzero(final_state[b], as_tuple=False).flatten().tolist()
            chosen_genes = tuple(sorted(gene_ids[i] for i in chosen_idx))
            sample_counter[chosen_genes] += 1
            if chosen_genes in reward_cache:
                r = reward_cache[chosen_genes]
            else:
                result = evaluate_subset(list(chosen_genes))
                r = float(result["score"]) + 1e-3
                reward_cache[chosen_genes] = r
            rewards[b] = r
            replay.add(actions[b], r)

        best_ever = max(best_ever, rewards.max().item())

        # Shaped reward
        shaped = rewards.clamp(min=1e-6).pow(reward_exponent)
        log_R = shaped.log()
        loss_fresh = ((policy.log_Z + log_pf - log_R) ** 2).mean()

        # --- replay samples ---
        loss_replay = torch.tensor(0.0, device=device)
        if len(replay) >= replay_batch:
            r_actions, r_rewards = replay.sample(replay_batch, device=device)
            r_log_pf = compute_log_pf_for_actions(policy, r_actions, device=device)
            r_shaped = r_rewards.clamp(min=1e-6).pow(reward_exponent)
            r_log_R = r_shaped.log()
            loss_replay = ((policy.log_Z + r_log_pf - r_log_R) ** 2).mean()

        loss = loss_fresh + loss_replay

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=5.0)
        optimizer.step()

        history["epoch"].append(epoch)
        history["mean_reward"].append(rewards.mean().item())
        history["max_reward"].append(rewards.max().item())
        history["loss"].append(loss.item())
        history["n_unique"].append(len(reward_cache))
        history["temp"].append(temp)
        history["best_ever"].append(best_ever)

        if epoch % log_every == 0 or epoch == n_epochs - 1:
            elapsed = time.time() - t0
            print(
                f"epoch {epoch:3d}  loss={loss.item():8.2f}  "
                f"mean_R={rewards.mean().item():.3f}  "
                f"max_R={rewards.max().item():.3f}  "
                f"best={best_ever:.3f}  "
                f"unique={len(reward_cache):4d}  "
                f"temp={temp:.2f}  "
                f"logZ={policy.log_Z.item():.1f}  "
                f"({elapsed:.1f}s)"
            )

    return {
        "policy": policy,
        "history": history,
        "sample_counter": sample_counter,
        "reward_cache": reward_cache,
        "gene_ids": gene_ids,
    }
