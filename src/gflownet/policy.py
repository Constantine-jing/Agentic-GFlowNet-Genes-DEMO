"""
policy.py — MLP policy for the GFlowNet.

State: binary vector (n_genes,) — 1 = gene already in subset.
Output: logits over n_genes for which gene to add next.
Also: learnable log_Z (partition function estimate for TB loss).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class GFNPolicy(nn.Module):
    def __init__(self, n_genes: int, hidden: int = 256):
        super().__init__()
        self.n_genes = n_genes
        self.body = nn.Sequential(
            nn.Linear(n_genes, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.forward_head = nn.Linear(hidden, n_genes)
        self.log_Z = nn.Parameter(torch.zeros(1))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """state: (batch, n_genes) float 0/1. Returns logits (batch, n_genes)."""
        h = self.body(state)
        return self.forward_head(h)
