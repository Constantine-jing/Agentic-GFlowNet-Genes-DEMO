"""
baselines.py — Compare GFlowNet against standard methods on the same task.

Baselines:
  1. Top-K Differential Expression (naive limma/scipy top-15 by p-value)
  2. LASSO logistic regression (L1 feature selection)
  3. Random Forest feature importance (top-15 by Gini importance)
  4. REINFORCE (policy gradient RL — same reward, no diversity)
  5. Random subsets (lower bound)

All baselines select 15 genes and are scored by the SAME reward function
as the GFlowNet, so the comparison is apples-to-apples.

Run from repo root:
    python -m scripts.baselines
"""
from __future__ import annotations

import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src import config
from src.env.reward import evaluate_subset, _load_python_backend

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"

SUBSET_SIZE = 15
N_RANDOM_TRIALS = 100


def _load_data():
    cfg = config.active()
    expr = pd.read_csv(cfg["expr"], index_col=0)
    labels = pd.read_csv(cfg["labels"])
    gene_ids = expr.index.tolist()
    return expr, labels, gene_ids, cfg


# ============================================================
#  Baseline 1: Top-K Differential Expression
# ============================================================
def baseline_topk_de(k: int = SUBSET_SIZE) -> dict:
    """Select the top-k genes by raw differential expression (t-test p-value)."""
    print("\n[Baseline 1] Top-K Differential Expression")
    cache = _load_python_backend()
    gene_ids = cache["gene_ids"]
    p_vals = cache["p_vals"]

    # Sort by p-value, take top k
    top_idx = np.argsort(p_vals)[:k]
    top_genes = [gene_ids[i] for i in top_idx]

    result = evaluate_subset(top_genes)
    print(f"  genes: {top_genes[:5]}...")
    print(f"  score: {result['score']:.4f}  n_sig: {result['n_sig']}/{k}")
    return {"method": "Top-K DE", "genes": top_genes, **result}


# ============================================================
#  Baseline 2: LASSO (L1 logistic regression)
# ============================================================
def baseline_lasso(k: int = SUBSET_SIZE) -> dict:
    """Use L1-penalized logistic regression for feature selection."""
    print("\n[Baseline 2] LASSO Logistic Regression")
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    expr, labels, gene_ids, cfg = _load_data()
    ctrl_label, trt_label = cfg["groups"]

    X = expr.values.T  # (samples, genes)
    y = (labels["group"] == trt_label).astype(int).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Search for C that gives ~k nonzero coefficients
    best_genes = []
    for C in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]:
        lr = LogisticRegression(penalty="l1", C=C, solver="saga",
                                max_iter=5000, random_state=0)
        lr.fit(X_scaled, y)
        nonzero = np.where(lr.coef_[0] != 0)[0]
        if len(nonzero) >= k:
            # Take top-k by absolute coefficient
            abs_coef = np.abs(lr.coef_[0][nonzero])
            top_k_idx = nonzero[np.argsort(abs_coef)[-k:]]
            best_genes = [gene_ids[i] for i in top_k_idx]
            break

    if len(best_genes) < k:
        # Fallback: take top-k by absolute coefficient regardless
        abs_coef = np.abs(lr.coef_[0])
        top_k_idx = np.argsort(abs_coef)[-k:]
        best_genes = [gene_ids[i] for i in top_k_idx]

    result = evaluate_subset(best_genes)
    print(f"  genes: {best_genes[:5]}...")
    print(f"  score: {result['score']:.4f}  n_sig: {result['n_sig']}/{k}")
    return {"method": "LASSO", "genes": best_genes, **result}


# ============================================================
#  Baseline 3: Random Forest Feature Importance
# ============================================================
def baseline_rf(k: int = SUBSET_SIZE) -> dict:
    """Select top-k genes by Random Forest Gini importance."""
    print("\n[Baseline 3] Random Forest Feature Importance")
    from sklearn.ensemble import RandomForestClassifier

    expr, labels, gene_ids, cfg = _load_data()
    ctrl_label, trt_label = cfg["groups"]

    X = expr.values.T
    y = (labels["group"] == trt_label).astype(int).values

    rf = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)
    rf.fit(X, y)

    top_idx = np.argsort(rf.feature_importances_)[-k:]
    top_genes = [gene_ids[i] for i in top_idx]

    result = evaluate_subset(top_genes)
    print(f"  genes: {top_genes[:5]}...")
    print(f"  score: {result['score']:.4f}  n_sig: {result['n_sig']}/{k}")
    return {"method": "Random Forest", "genes": top_genes, **result}


# ============================================================
#  Baseline 4: REINFORCE (Policy Gradient RL)
# ============================================================
def baseline_reinforce(
    k: int = SUBSET_SIZE,
    n_epochs: int = 300,
    batch_size: int = 16,
    lr: float = 5e-3,
) -> dict:
    """
    Standard REINFORCE policy gradient on the same reward.
    Key difference from GFlowNet: RL maximizes expected reward,
    so it converges to a SINGLE best subset. No diversity.
    """
    print("\n[Baseline 4] REINFORCE (Policy Gradient RL)")
    expr, labels, gene_ids, cfg = _load_data()
    n_genes = len(gene_ids)

    # Same architecture as GFlowNet policy
    policy = nn.Sequential(
        nn.Linear(n_genes, 256), nn.ReLU(),
        nn.Linear(256, 256), nn.ReLU(),
        nn.Linear(256, n_genes),
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    reward_cache = {}
    best_reward = 0.0
    best_subset = None
    sample_counter = Counter()

    for epoch in range(n_epochs):
        # Sample batch of subsets
        states = torch.zeros(batch_size, n_genes)
        log_probs_total = torch.zeros(batch_size)

        for _ in range(k):
            logits = policy(states)
            logits = logits.masked_fill(states.bool(), float("-inf"))
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_probs_total += dist.log_prob(action)
            states = states.scatter(1, action.unsqueeze(-1), 1.0)

        # Evaluate rewards
        rewards = torch.zeros(batch_size)
        for b in range(batch_size):
            chosen_idx = torch.nonzero(states[b]).flatten().tolist()
            chosen_genes = tuple(sorted(gene_ids[i] for i in chosen_idx))
            sample_counter[chosen_genes] += 1
            if chosen_genes in reward_cache:
                r = reward_cache[chosen_genes]
            else:
                result = evaluate_subset(list(chosen_genes))
                r = float(result["score"])
                reward_cache[chosen_genes] = r
            rewards[b] = r

        # REINFORCE loss: -E[R * log pi]
        baseline_r = rewards.mean()
        loss = -((rewards - baseline_r) * log_probs_total).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
        optimizer.step()

        batch_best = rewards.max().item()
        if batch_best > best_reward:
            best_reward = batch_best
            best_idx = rewards.argmax().item()
            chosen_idx = torch.nonzero(states[best_idx]).flatten().tolist()
            best_subset = [gene_ids[i] for i in chosen_idx]

        if epoch % 50 == 0 or epoch == n_epochs - 1:
            print(f"  epoch {epoch:3d}  mean_R={rewards.mean():.3f}  "
                  f"max_R={batch_best:.3f}  best_ever={best_reward:.3f}  "
                  f"unique={len(reward_cache)}")

    # Measure diversity: how many distinct subsets in top-10?
    top10 = sorted(reward_cache.items(), key=lambda kv: kv[1], reverse=True)[:10]
    # Check overlap between top subsets
    top_sets = [set(s) for s, _ in top10]
    avg_jaccard = 0.0
    count = 0
    for i in range(len(top_sets)):
        for j in range(i+1, len(top_sets)):
            jaccard = len(top_sets[i] & top_sets[j]) / len(top_sets[i] | top_sets[j])
            avg_jaccard += jaccard
            count += 1
    avg_jaccard /= max(count, 1)

    result = evaluate_subset(best_subset)
    print(f"  best genes: {best_subset[:5]}...")
    print(f"  best score: {result['score']:.4f}")
    print(f"  avg Jaccard similarity (top-10): {avg_jaccard:.3f}")
    print(f"  (1.0 = identical subsets = no diversity, 0.0 = completely different)")

    return {
        "method": "REINFORCE",
        "genes": best_subset,
        "n_unique": len(reward_cache),
        "avg_jaccard_top10": round(avg_jaccard, 3),
        **result,
    }


# ============================================================
#  Baseline 5: Random Subsets (lower bound)
# ============================================================
def baseline_random(k: int = SUBSET_SIZE, n_trials: int = N_RANDOM_TRIALS) -> dict:
    """Random subsets — the lower bound any method should beat."""
    print(f"\n[Baseline 5] Random Subsets ({n_trials} trials)")
    expr, labels, gene_ids, cfg = _load_data()

    scores = []
    best_score = 0.0
    best_genes = None
    for i in range(n_trials):
        subset = random.sample(gene_ids, k)
        result = evaluate_subset(subset)
        scores.append(result["score"])
        if result["score"] > best_score:
            best_score = result["score"]
            best_genes = subset

    result = evaluate_subset(best_genes)
    print(f"  mean score: {np.mean(scores):.4f} +/- {np.std(scores):.4f}")
    print(f"  best score: {best_score:.4f}")
    return {
        "method": "Random",
        "genes": best_genes,
        "mean_score": round(np.mean(scores), 4),
        "std_score": round(np.std(scores), 4),
        **result,
    }


# ============================================================
#  Main: run all baselines + load GFlowNet results for comparison
# ============================================================
def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    print("=" * 60)
    print(f"  BASELINE COMPARISON — {config.DATASET}")
    print("=" * 60)

    t0 = time.time()

    results = []
    results.append(baseline_topk_de())
    results.append(baseline_lasso())
    results.append(baseline_rf())
    results.append(baseline_reinforce(n_epochs=300))
    results.append(baseline_random())

    # Load GFlowNet result if available
    gfn_path = RESULTS_DIR / "outer_loop_top_subsets.csv"
    if gfn_path.exists():
        gfn_top = pd.read_csv(gfn_path)
        if len(gfn_top) > 0:
            best_gfn = gfn_top.iloc[0]
            gfn_genes = best_gfn["genes"].split(";")
            gfn_result = evaluate_subset(gfn_genes)

            # Compute GFlowNet diversity (Jaccard on top-10)
            top10_genes = [row["genes"].split(";") for _, row in gfn_top.head(10).iterrows()]
            top10_sets = [set(g) for g in top10_genes]
            jaccard_sum, count = 0.0, 0
            for i in range(len(top10_sets)):
                for j in range(i+1, len(top10_sets)):
                    jaccard_sum += len(top10_sets[i] & top10_sets[j]) / len(top10_sets[i] | top10_sets[j])
                    count += 1
            gfn_jaccard = jaccard_sum / max(count, 1)

            results.append({
                "method": "GFlowNet (ours)",
                "genes": gfn_genes,
                "avg_jaccard_top10": round(gfn_jaccard, 3),
                **gfn_result,
            })

    elapsed = time.time() - t0

    # ---- Summary table ----
    print(f"\n{'=' * 60}")
    print(f"  COMPARISON TABLE")
    print(f"{'=' * 60}")
    print(f"{'Method':<22} {'Score':>8} {'n_sig':>6} {'|logFC|':>8} {'Diversity':>10}")
    print("-" * 58)
    for r in results:
        diversity = ""
        if "avg_jaccard_top10" in r:
            # Lower Jaccard = more diverse
            diversity = f"J={r['avg_jaccard_top10']:.3f}"
        elif r["method"] == "Random":
            diversity = "N/A"
        else:
            diversity = "single"
        print(f"{r['method']:<22} {r['score']:>8.4f} {r['n_sig']:>6} "
              f"{r['mean_abs_logfc']:>8.4f} {diversity:>10}")

    print(f"\n  Diversity: Jaccard similarity of top-10 subsets")
    print(f"    J=1.0 means all identical (no diversity)")
    print(f"    J=0.0 means completely different (max diversity)")
    print(f"    'single' means the method only produces one answer")
    print(f"\n  Total time: {elapsed:.1f}s")

    # ---- Save ----
    rows = []
    for r in results:
        rows.append({
            "method": r["method"],
            "score": r["score"],
            "n_sig": r["n_sig"],
            "mean_abs_logfc": r["mean_abs_logfc"],
            "diversity_jaccard": r.get("avg_jaccard_top10", None),
            "genes": ";".join(r.get("genes", [])),
        })
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "baselines_comparison.csv", index=False)
    print(f"\n[saved] {RESULTS_DIR / 'baselines_comparison.csv'}")

    # ---- Key insight ----
    gfn_entry = next((r for r in results if r["method"] == "GFlowNet (ours)"), None)
    rl_entry = next((r for r in results if r["method"] == "REINFORCE"), None)
    if gfn_entry and rl_entry:
        print(f"\n  KEY COMPARISON: GFlowNet vs REINFORCE")
        print(f"    Score:     GFN={gfn_entry['score']:.3f}  RL={rl_entry['score']:.3f}")
        print(f"    Diversity: GFN Jaccard={gfn_entry.get('avg_jaccard_top10', 'N/A')}  "
              f"RL Jaccard={rl_entry.get('avg_jaccard_top10', 'N/A')}")
        if gfn_entry.get("avg_jaccard_top10", 1) < rl_entry.get("avg_jaccard_top10", 1):
            print(f"    → GFlowNet produces MORE DIVERSE subsets (lower Jaccard)")
        print(f"    → This is the core value proposition: similar quality, more diversity")

    print(f"\n✅ baselines complete.")


if __name__ == "__main__":
    main()
