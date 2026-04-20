"""
sampler.py — Build gene subsets gene-by-gene using the policy.

Supports temperature control:
  - temp > 1  → more exploration (softer distribution)
  - temp = 1  → default
  - temp < 1  → more exploitation (sharper, greedier)
  - temp → 0  → argmax (greedy)

Also supports computing log P_F of a *given* subset under the current policy
(needed for replay buffer training).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .policy import GFNPolicy


def sample_trajectories(
    policy: GFNPolicy,
    batch_size: int,
    subset_size: int,
    temperature: float = 1.0,
    device: str = "cpu",
) -> dict:
    n_genes = policy.n_genes
    state = torch.zeros(batch_size, n_genes, device=device)
    log_pf_total = torch.zeros(batch_size, device=device)
    actions_history = []

    for _ in range(subset_size):
        logits = policy(state)
        logits = logits.masked_fill(state.bool(), float("-inf"))
        # Apply temperature
        logits = logits / max(temperature, 1e-6)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()

        action = torch.multinomial(probs, num_samples=1).squeeze(-1)
        log_pf_step = log_probs.gather(1, action.unsqueeze(-1)).squeeze(-1)

        log_pf_total = log_pf_total + log_pf_step
        actions_history.append(action)
        state = state.scatter(1, action.unsqueeze(-1), 1.0)

    actions = torch.stack(actions_history, dim=1)
    return {
        "final_state": state,
        "actions": actions,
        "log_pf": log_pf_total,
    }


def compute_log_pf_for_actions(
    policy: GFNPolicy,
    actions: torch.Tensor,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Given a batch of action sequences (batch, subset_size), compute
    log P_F under the current policy for each trajectory.

    This is used by the replay buffer: we already know which subset was good,
    now we compute how likely the *current* policy thinks that trajectory is.
    """
    batch_size, subset_size = actions.shape
    n_genes = policy.n_genes
    state = torch.zeros(batch_size, n_genes, device=device)
    log_pf_total = torch.zeros(batch_size, device=device)

    for t in range(subset_size):
        logits = policy(state)
        logits = logits.masked_fill(state.bool(), float("-inf"))
        log_probs = F.log_softmax(logits, dim=-1)

        action = actions[:, t]
        log_pf_step = log_probs.gather(1, action.unsqueeze(-1)).squeeze(-1)
        log_pf_total = log_pf_total + log_pf_step

        state = state.scatter(1, action.unsqueeze(-1), 1.0)

    return log_pf_total
