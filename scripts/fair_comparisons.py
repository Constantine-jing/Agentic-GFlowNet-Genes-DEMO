"""
fair_comparisons.py — Three experiments for fair apples-to-apples comparisons.

A. Rerun mock agent with reward_exponent=1.0 (matching what Claude chose)
   → Fair Claude vs Mock comparison (same hyperparams)

B. Rerun REINFORCE with 7200 evaluation budget (matching GFlowNet's budget)
   → Fair RL vs GFlowNet comparison (same compute)

C. Spot-validate top-50 subsets with R/limma
   → Strengthens "Python backend is identical" claim beyond top-10

Run from repo root:
    python -m scripts.fair_comparisons
"""
from __future__ import annotations

import json
import random
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src import config
from src.env.reward import evaluate_subset, _load_python_backend

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"


# ============================================================
#  EXPERIMENT A: Mock agents with reward_exponent = 1.0
# ============================================================

def experiment_a():
    """
    Rerun the outer loop with mock agents BUT reward_exponent=1.0,
    matching what Claude's Manager chose. This isolates the agent
    intelligence from the hyperparameter difference.
    """
    print("=" * 60)
    print("  EXPERIMENT A: Mock agents @ reward_exponent=1.0")
    print("  (fair comparison vs Claude run)")
    print("=" * 60)

    # We need to import and call the training directly with controlled params,
    # not through the outer loop (which reads from Manager suggestions).
    from src.gflownet.train import train

    all_reward_caches = {}
    all_histories = []

    for round_num in range(3):
        print(f"\n  --- Round {round_num + 1} / 3 ---")

        # Match Claude's settings exactly:
        # Round 1: temp 2.0→0.5, R^1.0
        # Round 2+: temp 1.0→0.5, R^1.0
        out = train(
            n_epochs=150,
            batch_size=16,
            replay_batch=16,
            subset_size=15,
            lr=5e-3,
            log_z_lr=1.0,
            reward_exponent=1.0,  # ← THE KEY CHANGE: matches Claude's choice
            temp_start=2.0 if round_num == 0 else 1.0,
            temp_end=0.5,
            replay_capacity=128,
            seed=42 + round_num,
            log_every=50,
        )
        all_reward_caches.update(out["reward_cache"])
        all_histories.append({
            "round": round_num + 1,
            "best_reward": max(out["history"]["best_ever"]),
            "final_mean_reward": out["history"]["mean_reward"][-1],
            "unique_explored": out["history"]["n_unique"][-1],
        })

    # Top subsets
    top_by_reward = sorted(all_reward_caches.items(), key=lambda kv: kv[1], reverse=True)[:50]

    # Gene frequency
    gene_freq = Counter()
    for s, _ in top_by_reward[:50]:
        for g in s:
            gene_freq[g] += 1

    # Diversity (Jaccard on top-10)
    top10_sets = [set(s) for s, _ in top_by_reward[:10]]
    jaccard_sum, count = 0.0, 0
    for i in range(len(top10_sets)):
        for j in range(i + 1, len(top10_sets)):
            jaccard_sum += len(top10_sets[i] & top10_sets[j]) / len(top10_sets[i] | top10_sets[j])
            count += 1
    jaccard = jaccard_sum / max(count, 1)

    best_reward = top_by_reward[0][1] if top_by_reward else 0

    print(f"\n  EXPERIMENT A RESULTS")
    print(f"  Best reward: {best_reward:.4f}")
    print(f"  Total unique: {len(all_reward_caches)}")
    print(f"  Top-10 Jaccard: {jaccard:.3f}")
    print(f"  Top 10 genes:")
    for g, freq in gene_freq.most_common(10):
        print(f"    {g:15s}  {freq}/50")

    # Save
    result = {
        "experiment": "A_mock_r1.0",
        "reward_exponent": 1.0,
        "best_reward": best_reward,
        "total_unique": len(all_reward_caches),
        "jaccard_top10": round(jaccard, 4),
        "rounds": all_histories,
        "top_genes": [(g, f) for g, f in gene_freq.most_common(15)],
    }
    with open(RESULTS_DIR / "exp_a_mock_r1.json", "w") as f:
        json.dump(result, f, indent=2)

    top_rows = [
        {"rank": i + 1, "reward": r, "genes": ";".join(s)}
        for i, (s, r) in enumerate(top_by_reward[:50])
    ]
    pd.DataFrame(top_rows).to_csv(RESULTS_DIR / "exp_a_mock_r1_top_subsets.csv", index=False)

    print(f"\n  [saved] {RESULTS_DIR / 'exp_a_mock_r1.json'}")
    print(f"  [saved] {RESULTS_DIR / 'exp_a_mock_r1_top_subsets.csv'}")

    return result


# ============================================================
#  EXPERIMENT B: REINFORCE with 7200 evaluation budget
# ============================================================

def experiment_b():
    """
    Run REINFORCE with the same total evaluation budget as GFlowNet
    (7200 unique evaluations = 3 rounds × 2400 unique subsets).
    Original baseline only ran 300 epochs × 16 batch = 4800 samples
    but with many duplicates (only 664 unique).
    """
    print(f"\n{'=' * 60}")
    print("  EXPERIMENT B: REINFORCE with 7200 evaluation budget")
    print("  (fair compute comparison vs GFlowNet)")
    print("=" * 60)

    cfg = config.active()
    expr = pd.read_csv(cfg["expr"], index_col=0)
    gene_ids = expr.index.tolist()
    n_genes = len(gene_ids)
    k = 15

    torch.manual_seed(42)

    policy = nn.Sequential(
        nn.Linear(n_genes, 256), nn.ReLU(),
        nn.Linear(256, 256), nn.ReLU(),
        nn.Linear(256, n_genes),
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=5e-3)

    reward_cache = {}
    best_reward = 0.0
    best_subset = None
    sample_counter = Counter()

    # Run until we hit 7200 unique evaluations or max epochs
    max_epochs = 2000  # safety cap
    batch_size = 16
    target_unique = 7200

    t0 = time.time()
    epoch = 0

    while len(reward_cache) < target_unique and epoch < max_epochs:
        states = torch.zeros(batch_size, n_genes)
        log_probs_total = torch.zeros(batch_size)

        for _ in range(k):
            logits = policy(states)
            logits = logits.masked_fill(states.bool(), float("-inf"))
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_probs_total += dist.log_prob(action)
            states = states.scatter(1, action.unsqueeze(-1), 1.0)

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

        if epoch % 100 == 0:
            elapsed = time.time() - t0
            print(f"  epoch {epoch:4d}  mean_R={rewards.mean():.3f}  "
                  f"best_ever={best_reward:.3f}  "
                  f"unique={len(reward_cache):5d}/{target_unique}  "
                  f"({elapsed:.1f}s)")

        epoch += 1

    elapsed = time.time() - t0

    # Diversity
    top10 = sorted(reward_cache.items(), key=lambda kv: kv[1], reverse=True)[:10]
    top10_sets = [set(s) for s, _ in top10]
    jaccard_sum, count = 0.0, 0
    for i in range(len(top10_sets)):
        for j in range(i + 1, len(top10_sets)):
            jaccard_sum += len(top10_sets[i] & top10_sets[j]) / len(top10_sets[i] | top10_sets[j])
            count += 1
    jaccard = jaccard_sum / max(count, 1)

    print(f"\n  EXPERIMENT B RESULTS")
    print(f"  Epochs: {epoch}")
    print(f"  Best reward: {best_reward:.4f}")
    print(f"  Unique evaluations: {len(reward_cache)}")
    print(f"  Top-10 Jaccard: {jaccard:.3f}")
    print(f"  Time: {elapsed:.1f}s")

    if best_subset:
        print(f"  Best genes: {best_subset[:5]}...")

    result = {
        "experiment": "B_reinforce_7200",
        "epochs": epoch,
        "best_reward": best_reward,
        "total_unique": len(reward_cache),
        "jaccard_top10": round(jaccard, 4),
        "time_seconds": round(elapsed, 1),
    }
    with open(RESULTS_DIR / "exp_b_reinforce_7200.json", "w") as f:
        json.dump(result, f, indent=2)

    top_rows = [
        {"rank": i + 1, "reward": r, "genes": ";".join(s)}
        for i, (s, r) in enumerate(sorted(reward_cache.items(), key=lambda kv: kv[1], reverse=True)[:50])
    ]
    pd.DataFrame(top_rows).to_csv(RESULTS_DIR / "exp_b_reinforce_7200_top_subsets.csv", index=False)

    print(f"\n  [saved] {RESULTS_DIR / 'exp_b_reinforce_7200.json'}")
    print(f"  [saved] {RESULTS_DIR / 'exp_b_reinforce_7200_top_subsets.csv'}")

    return result


# ============================================================
#  EXPERIMENT C: Spot-validate top-50 with R/limma
# ============================================================

def experiment_c():
    """
    Re-score the top-50 GFlowNet subsets with R/limma to extend
    the validation beyond top-10.
    """
    print(f"\n{'=' * 60}")
    print("  EXPERIMENT C: Limma spot-validation of top-50 subsets")
    print("=" * 60)

    # Load the best available top subsets
    for fname in ["claude_top_subsets.csv", "outer_loop_top_subsets.csv", "top_subsets.csv"]:
        path = RESULTS_DIR / fname
        if path.exists():
            print(f"  Using: {path.name}")
            break
    else:
        print("  ERROR: No top subsets file found.")
        return None

    top_df = pd.read_csv(path)
    n_compare = min(50, len(top_df))

    print(f"  Re-scoring {n_compare} subsets with both backends...")

    # Python scores
    config.REWARD_BACKEND = "python"
    import src.env.reward as rm
    rm._py_cache = None  # force reload

    python_scores = []
    for i in range(n_compare):
        genes = top_df.iloc[i]["genes"].split(";")
        result = evaluate_subset(genes)
        python_scores.append(result["score"])

    # R/limma scores
    config.REWARD_BACKEND = "r"
    from src.env.reward import _evaluate_r

    limma_scores = []
    t0 = time.time()
    for i in range(n_compare):
        genes = top_df.iloc[i]["genes"].split(";")
        result = _evaluate_r(genes)
        limma_scores.append(result["score"])
        if (i + 1) % 10 == 0:
            print(f"    scored {i + 1}/{n_compare}  ({time.time() - t0:.1f}s)")

    elapsed = time.time() - t0
    config.REWARD_BACKEND = "python"

    python_scores = np.array(python_scores)
    limma_scores = np.array(limma_scores)

    from scipy.stats import spearmanr
    pearson_r = float(np.corrcoef(python_scores, limma_scores)[0, 1])
    spearman_r, spearman_p = spearmanr(python_scores, limma_scores)
    mad = float(np.mean(np.abs(python_scores - limma_scores)))

    # Rank overlap at various k
    py_rank = np.argsort(-python_scores)
    lm_rank = np.argsort(-limma_scores)
    top5_overlap = len(set(py_rank[:5]) & set(lm_rank[:5]))
    top10_overlap = len(set(py_rank[:10]) & set(lm_rank[:10]))
    top20_overlap = len(set(py_rank[:20]) & set(lm_rank[:20]))

    print(f"\n  EXPERIMENT C RESULTS (n={n_compare})")
    print(f"  Pearson r:           {pearson_r:.6f}")
    print(f"  Spearman r:          {spearman_r:.6f} (p={spearman_p:.2e})")
    print(f"  Mean absolute diff:  {mad:.6f}")
    print(f"  Top-5 rank overlap:  {top5_overlap}/5")
    print(f"  Top-10 rank overlap: {top10_overlap}/10")
    print(f"  Top-20 rank overlap: {top20_overlap}/20")
    print(f"  R/limma eval time:   {elapsed:.1f}s")

    if pearson_r > 0.999:
        print(f"\n  ✅ PERFECT agreement across {n_compare} subsets.")
    elif pearson_r > 0.95:
        print(f"\n  ✅ STRONG agreement across {n_compare} subsets.")
    else:
        print(f"\n  ⚠️  Agreement weaker than expected on {n_compare} subsets.")

    result = {
        "experiment": "C_limma_top50",
        "n_subsets": n_compare,
        "pearson_r": round(pearson_r, 6),
        "spearman_r": round(float(spearman_r), 6),
        "mean_abs_diff": round(mad, 6),
        "top5_overlap": top5_overlap,
        "top10_overlap": top10_overlap,
        "top20_overlap": top20_overlap,
        "time_seconds": round(elapsed, 1),
    }
    with open(RESULTS_DIR / "exp_c_limma_top50.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  [saved] {RESULTS_DIR / 'exp_c_limma_top50.json'}")

    return result


# ============================================================
#  Main
# ============================================================

def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    random.seed(42)
    np.random.seed(42)

    print("\n" + "#" * 60)
    print("#  FAIR COMPARISON EXPERIMENTS")
    print(f"#  Dataset: {config.DATASET}")
    print("#" * 60)

    t0 = time.time()

    res_a = experiment_a()
    res_b = experiment_b()
    res_c = experiment_c()

    elapsed = time.time() - t0

    # ---- Final comparison table ----
    print(f"\n{'=' * 60}")
    print("  UPDATED COMPARISON TABLE (FAIR)")
    print("=" * 60)

    # Load Claude results if available
    claude_path = RESULTS_DIR / "claude_summary.json"
    claude_best = "N/A"
    if claude_path.exists():
        with open(claude_path) as f:
            claude_data = json.load(f)
            claude_best = claude_data.get("best_reward", "N/A")

    print(f"\n  A. Agent comparison (both at R^1.0):")
    print(f"     Claude agents:  best={claude_best}")
    print(f"     Mock agents:    best={res_a['best_reward']:.4f}")
    if isinstance(claude_best, (int, float)):
        diff = res_a["best_reward"] - claude_best
        if abs(diff) < 0.3:
            print(f"     → Similar performance ({diff:+.3f}) — confirms agent effect is minimal on score")
        elif diff > 0.3:
            print(f"     → Mock still higher ({diff:+.3f}) — even with matched exponent")
        else:
            print(f"     → Claude higher ({diff:+.3f}) — agent reasoning helps at matched settings")

    print(f"\n  B. RL comparison (matched 7200 eval budget):")
    print(f"     GFlowNet:   best=4.533  unique=7200  Jaccard=0.114")
    print(f"     REINFORCE:  best={res_b['best_reward']:.3f}  "
          f"unique={res_b['total_unique']}  Jaccard={res_b['jaccard_top10']:.3f}")
    if res_b["jaccard_top10"] > 0.5:
        print(f"     → REINFORCE still collapses (J={res_b['jaccard_top10']:.3f} >> 0.114)")
        print(f"     → GFlowNet diversity advantage holds even with matched budget")

    print(f"\n  C. Limma validation (top-{res_c['n_subsets']}):")
    print(f"     Pearson r={res_c['pearson_r']:.6f}  MAD={res_c['mean_abs_diff']:.6f}")
    print(f"     Top-20 rank overlap: {res_c['top20_overlap']}/20")

    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"\n✅ all fair comparison experiments complete.")


if __name__ == "__main__":
    main()
