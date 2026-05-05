"""
refiner.py — Diffusion Refiner for Gene Subset Polishing.

Takes high-reward subsets discovered by the GFlowNet and "refines" them
via a learned denoising process on binary gene vectors.

Core idea:
  1. Collect the top-N subsets from the GFlowNet as "clean" training data
  2. Train a small denoising model: given a noisy (corrupted) binary vector,
     predict the clean version
  3. At inference: take a GFlowNet output, add slight noise, denoise it
     → the denoised version may swap out weak genes for stronger ones
  4. Re-evaluate the refined subset — if it scores higher, keep it

This is essentially a Bernoulli diffusion process (discrete diffusion)
on binary vectors, simplified for the demo:
  - Forward process: randomly flip k bits (gene inclusions)
  - Reverse process: neural network predicts which bits to flip back

References:
  - Austin et al. (2021) "Structured Denoising Diffusion Models in Discrete
    State-Spaces" — D3PM, the theoretical basis
  - Hoogeboom et al. (2021) "Argmax Flows and Multinomial Diffusion"

Run from repo root:
    python -m scripts.run_refiner
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src import config
from src.env.reward import evaluate_subset

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"


class BinaryDenoiser(nn.Module):
    """
    Given a noisy binary gene vector, predict the clean version.
    Input:  (batch, n_genes) binary vector + (batch, 1) noise level
    Output: (batch, n_genes) probability each gene should be ON in the clean version
    """
    def __init__(self, n_genes: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_genes + 1, hidden),  # +1 for noise level
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_genes),
        )

    def forward(self, noisy: torch.Tensor, noise_level: torch.Tensor) -> torch.Tensor:
        """Returns logits (not probabilities)."""
        x = torch.cat([noisy, noise_level], dim=-1)
        return self.net(x)


def _corrupt(clean: torch.Tensor, flip_rate: float) -> torch.Tensor:
    """Corrupt a binary vector by flipping each bit with probability flip_rate."""
    mask = torch.rand_like(clean) < flip_rate
    return torch.where(mask, 1.0 - clean, clean)


def train_denoiser(
    top_subsets: list[tuple[str, ...]],
    gene_ids: list[str],
    n_epochs: int = 500,
    batch_size: int = 32,
    lr: float = 1e-3,
    max_flip_rate: float = 0.3,
    seed: int = 0,
) -> BinaryDenoiser:
    """
    Train the denoiser on the top GFlowNet subsets.

    top_subsets: list of tuples of gene names (the "clean" data)
    gene_ids: full list of gene names (defines the vector ordering)
    """
    torch.manual_seed(seed)
    n_genes = len(gene_ids)
    gene_to_idx = {g: i for i, g in enumerate(gene_ids)}

    # Convert top subsets to binary vectors
    clean_vectors = []
    for subset in top_subsets:
        vec = torch.zeros(n_genes)
        for g in subset:
            if g in gene_to_idx:
                vec[gene_to_idx[g]] = 1.0
        clean_vectors.append(vec)
    clean_data = torch.stack(clean_vectors)  # (N, n_genes)
    n_data = len(clean_data)
    print(f"[refiner] training on {n_data} clean subsets, {n_genes} genes")

    model = BinaryDenoiser(n_genes)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        # Sample a batch from the clean data (with replacement)
        idx = torch.randint(0, n_data, (batch_size,))
        clean = clean_data[idx]

        # Sample noise level uniformly
        flip_rate = torch.rand(batch_size, 1) * max_flip_rate
        # Corrupt
        noisy = torch.stack([_corrupt(clean[i], flip_rate[i].item()) for i in range(batch_size)])

        # Predict clean from noisy
        logits = model(noisy, flip_rate)
        loss = F.binary_cross_entropy_with_logits(logits, clean)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0 or epoch == n_epochs - 1:
            print(f"  epoch {epoch:4d}  loss={loss.item():.4f}")

    return model


def refine_subset(
    model: BinaryDenoiser,
    subset: tuple[str, ...],
    gene_ids: list[str],
    subset_size: int = 15,
    n_candidates: int = 20,
    noise_levels: list[float] = [0.05, 0.10, 0.15, 0.20],
) -> list[dict]:
    """
    Refine a single subset by:
    1. Encode as binary vector
    2. For each noise level, add noise and denoise multiple times
    3. Threshold the denoised output to get exactly subset_size genes
    4. Evaluate each candidate
    5. Return sorted by reward

    Returns list of {genes, score, noise_level, improvement} dicts.
    """
    n_genes = len(gene_ids)
    gene_to_idx = {g: i for i, g in enumerate(gene_ids)}

    # Original vector
    original = torch.zeros(n_genes)
    for g in subset:
        if g in gene_to_idx:
            original[gene_to_idx[g]] = 1.0

    original_result = evaluate_subset(list(subset))
    original_score = original_result["score"]

    candidates = []
    model.eval()
    with torch.no_grad():
        for nl in noise_levels:
            for _ in range(n_candidates // len(noise_levels)):
                # Add noise
                noisy = _corrupt(original.unsqueeze(0), nl)
                noise_t = torch.tensor([[nl]])

                # Denoise
                logits = model(noisy, noise_t)
                probs = torch.sigmoid(logits).squeeze(0)

                # Select top-k genes by probability
                top_k_idx = torch.argsort(probs, descending=True)[:subset_size]
                refined_genes = tuple(sorted(gene_ids[i] for i in top_k_idx.tolist()))

                # Evaluate
                result = evaluate_subset(list(refined_genes))
                candidates.append({
                    "genes": refined_genes,
                    "score": result["score"],
                    "n_sig": result["n_sig"],
                    "noise_level": nl,
                    "improvement": result["score"] - original_score,
                })

    # Deduplicate and sort by score
    seen = set()
    unique_candidates = []
    for c in sorted(candidates, key=lambda x: x["score"], reverse=True):
        if c["genes"] not in seen:
            seen.add(c["genes"])
            unique_candidates.append(c)

    return unique_candidates


def main():
    """
    Full refiner pipeline:
    1. Load GFlowNet top subsets
    2. Train denoiser on them
    3. Refine each top subset
    4. Report improvements
    """
    RESULTS_DIR.mkdir(exist_ok=True)

    # Load GFlowNet results
    top_path = RESULTS_DIR / "outer_loop_top_subsets.csv"
    if not top_path.exists():
        top_path = RESULTS_DIR / "top_subsets.csv"
    if not top_path.exists():
        print("ERROR: No GFlowNet results found. Run the outer loop first.")
        print("  python -m src.loop.outer_loop")
        return

    cfg = config.active()
    expr = pd.read_csv(cfg["expr"], index_col=0)
    gene_ids = expr.index.tolist()

    top_df = pd.read_csv(top_path)
    top_subsets = [tuple(row["genes"].split(";")) for _, row in top_df.iterrows()]
    top_rewards = top_df["reward"].tolist()

    print("=" * 60)
    print("  DIFFUSION REFINER")
    print(f"  dataset: {config.DATASET}")
    print(f"  input: {len(top_subsets)} GFlowNet subsets")
    print(f"  best input reward: {max(top_rewards):.4f}")
    print("=" * 60)

    t0 = time.time()

    # ---- 1. Train denoiser ----
    print("\n[Phase 1] Training denoiser...")
    model = train_denoiser(
        top_subsets=top_subsets[:50],  # train on top-50
        gene_ids=gene_ids,
        n_epochs=500,
        batch_size=32,
    )

    # ---- 2. Refine top subsets ----
    print("\n[Phase 2] Refining top subsets...")
    all_refined = []
    improved_count = 0

    for i, (subset, orig_reward) in enumerate(zip(top_subsets[:10], top_rewards[:10])):
        candidates = refine_subset(model, subset, gene_ids, n_candidates=20)
        best = candidates[0] if candidates else None

        if best and best["score"] > orig_reward:
            improved_count += 1
            tag = f"  ✓ IMPROVED +{best['improvement']:.3f}"
        else:
            tag = "  = no improvement"

        if best:
            all_refined.append({
                "original_rank": i + 1,
                "original_score": orig_reward,
                "refined_score": best["score"],
                "improvement": best["improvement"],
                "noise_level": best["noise_level"],
                "refined_genes": ";".join(best["genes"]),
            })
            print(f"  subset {i+1}: {orig_reward:.3f} -> {best['score']:.3f}{tag}")

    elapsed = time.time() - t0

    # ---- 3. Summary ----
    print(f"\n{'=' * 60}")
    print(f"  REFINER RESULTS")
    print(f"{'=' * 60}")
    print(f"  Subsets refined: {len(all_refined)}")
    print(f"  Improved: {improved_count}/{len(all_refined)}")

    if all_refined:
        improvements = [r["improvement"] for r in all_refined]
        print(f"  Mean improvement: {np.mean(improvements):+.4f}")
        print(f"  Max improvement:  {max(improvements):+.4f}")
        print(f"  Best refined score: {max(r['refined_score'] for r in all_refined):.4f}")
        print(f"  Best original score: {max(top_rewards[:10]):.4f}")

    print(f"  Time: {elapsed:.1f}s")

    # Save
    if all_refined:
        pd.DataFrame(all_refined).to_csv(RESULTS_DIR / "refiner_results.csv", index=False)
        print(f"\n[saved] {RESULTS_DIR / 'refiner_results.csv'}")

    # ---- 4. Gene-level analysis ----
    if all_refined:
        print(f"\n  Genes ADDED by refiner (not in original, appear in refined):")
        from collections import Counter
        added_genes = Counter()
        removed_genes = Counter()
        for r in all_refined:
            orig_idx = r["original_rank"] - 1
            orig_set = set(top_subsets[orig_idx])
            refined_set = set(r["refined_genes"].split(";"))
            for g in refined_set - orig_set:
                added_genes[g] += 1
            for g in orig_set - refined_set:
                removed_genes[g] += 1

        if added_genes:
            for g, c in added_genes.most_common(10):
                print(f"    + {g:15s} added {c}x")
        if removed_genes:
            print(f"\n  Genes REMOVED by refiner:")
            for g, c in removed_genes.most_common(10):
                print(f"    - {g:15s} removed {c}x")

    print(f"\n✅ refiner complete.")


if __name__ == "__main__":
    main()
