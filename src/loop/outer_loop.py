"""
outer_loop.py — The full agentic loop.

This is the dual-loop system from the proposal:
  Outer loop (this file):
    1. Manager generates plan/constraints
    2. Inner loop (GFlowNet train) runs
    3. Critic analyzes results
    4. If Critic says continue → go to step 1 with updated context
    5. If Critic says stop → output final results

  Inner loop (src/gflownet/train.py):
    For N epochs: sample subsets → evaluate rewards → TB loss → backprop

Run from repo root:
    python -m src.loop.outer_loop
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import pandas as pd

from src import config
from src.agents.manager import generate_plan
from src.agents.critic import analyze_results
from src.gflownet.train import train

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"


def _dataset_info() -> dict:
    cfg = config.active()
    expr = pd.read_csv(cfg["expr"], index_col=0)
    return {
        "dataset_name": config.DATASET,
        "n_genes": expr.shape[0],
        "n_samples": expr.shape[1],
        "groups": list(cfg["groups"]),
    }


def run(
    max_rounds: int = 3,
    epochs_per_round: int = 150,
    batch_size: int = 16,
    subset_size: int = 15,
) -> dict:
    """
    Run the full agentic outer loop.

    Each round:
      Manager → GFlowNet training → Critic → decide continue/stop
    """
    RESULTS_DIR.mkdir(exist_ok=True)
    ds_info = _dataset_info()
    all_rounds = []
    cumulative_reward_cache: dict[tuple, float] = {}
    cumulative_sample_counter: Counter = Counter()

    print("=" * 60)
    print("  AGENTIC GFLOWNET — OUTER LOOP")
    print(f"  dataset: {config.DATASET}")
    print(f"  max_rounds: {max_rounds}  epochs/round: {epochs_per_round}")
    print("=" * 60)

    prior_results = None

    for round_num in range(max_rounds):
        t0 = time.time()
        print(f"\n{'─' * 60}")
        print(f"  ROUND {round_num + 1} / {max_rounds}")
        print(f"{'─' * 60}")

        # ---- 1. MANAGER: generate plan ----
        print("\n[Manager] generating plan...")
        plan = generate_plan(ds_info, prior_results, round_num)
        print(f"  strategy: {plan['strategy'][:100]}...")
        if plan.get("constraints"):
            print(f"  constraints: {json.dumps(plan['constraints'])}")

        # Extract hyperparams from Manager's suggestions (with safe defaults)
        hp = plan.get("hyperparams", {})
        lr = hp.get("lr", 5e-3)
        temp_start = hp.get("temp_start", 2.0 if round_num == 0 else 1.0)
        temp_end = hp.get("temp_end", 0.5)
        reward_exp = hp.get("reward_exponent", 8.0)

        # ---- 2. INNER LOOP: GFlowNet training ----
        print(f"\n[GFlowNet] training for {epochs_per_round} epochs...")
        out = train(
            n_epochs=epochs_per_round,
            batch_size=batch_size,
            replay_batch=batch_size,
            subset_size=subset_size,
            lr=lr,
            log_z_lr=1.0,
            reward_exponent=reward_exp,
            temp_start=temp_start,
            temp_end=temp_end,
            replay_capacity=128,
            seed=42 + round_num,
            log_every=50,
        )

        # Merge into cumulative caches
        cumulative_reward_cache.update(out["reward_cache"])
        cumulative_sample_counter.update(out["sample_counter"])

        history = out["history"]
        best_reward = max(history["best_ever"])
        final_mean = history["mean_reward"][-1]
        n_unique = history["n_unique"][-1]

        # Top genes across this round's top subsets
        top_by_reward = sorted(
            out["reward_cache"].items(), key=lambda kv: kv[1], reverse=True
        )[:20]
        gene_freq = Counter()
        for s, _ in top_by_reward:
            for g in s:
                gene_freq[g] += 1
        top_genes = [g for g, _ in gene_freq.most_common(10)]

        prior_results = {
            "best_reward": best_reward,
            "mean_reward": final_mean,
            "n_unique": n_unique,
            "top_genes": top_genes,
        }

        # Build top_subsets list for Critic
        top_subsets_for_critic = []
        for rank, (s, r) in enumerate(top_by_reward[:10], 1):
            top_subsets_for_critic.append({
                "rank": rank,
                "reward": round(r, 3),
                "genes": ";".join(s),
            })

        # ---- 3. CRITIC: analyze results ----
        print(f"\n[Critic] analyzing round {round_num + 1} results...")
        feedback = analyze_results(history, top_subsets_for_critic, ds_info, round_num)
        print(f"  assessment: {feedback['assessment'][:120]}...")
        for obs in feedback.get("observations", [])[:3]:
            print(f"  • {obs}")
        for sug in feedback.get("suggestions", [])[:3]:
            print(f"  → {sug}")

        elapsed = time.time() - t0
        round_summary = {
            "round": round_num + 1,
            "best_reward": best_reward,
            "final_mean_reward": final_mean,
            "unique_explored": n_unique,
            "top_genes": top_genes,
            "critic_continue": feedback.get("continue_training", False),
            "elapsed_seconds": round(elapsed, 1),
        }
        all_rounds.append(round_summary)
        print(f"\n  round {round_num+1} complete  best={best_reward:.3f}  ({elapsed:.0f}s)")

        # ---- 4. DECIDE: continue or stop ----
        if not feedback.get("continue_training", False):
            print(f"\n[Critic] says STOP after round {round_num + 1}.")
            break
        else:
            print(f"\n[Critic] says CONTINUE to round {round_num + 2}.")

    # ---- FINAL REPORT ----
    print(f"\n{'=' * 60}")
    print("  FINAL REPORT")
    print(f"{'=' * 60}")
    print(f"  Rounds completed: {len(all_rounds)}")
    print(f"  Total unique subsets explored: {len(cumulative_reward_cache)}")

    # Overall top subsets
    overall_top = sorted(
        cumulative_reward_cache.items(), key=lambda kv: kv[1], reverse=True
    )[:10]
    print(f"\n  Top 10 subsets across all rounds:")
    for rank, (s, r) in enumerate(overall_top, 1):
        preview = ", ".join(s[:5]) + ("..." if len(s) > 5 else "")
        print(f"  {rank:2d}. R={r:.3f}  [{preview}]")

    # Gene frequency
    gene_freq_all = Counter()
    for s, _ in sorted(
        cumulative_reward_cache.items(), key=lambda kv: kv[1], reverse=True
    )[:50]:
        for g in s:
            gene_freq_all[g] += 1

    cfg = config.active()
    truth_path = cfg.get("truth")
    truth = set()
    if truth_path and Path(truth_path).exists():
        truth = set(pd.read_csv(truth_path)["gene_id"].tolist())

    print(f"\n  Most frequent genes in top-50 subsets:")
    for g, freq in gene_freq_all.most_common(15):
        tag = " ★ TRUTH" if g in truth else ""
        print(f"    {g:12s}  {freq:2d}/50{tag}")

    if truth:
        best_subset = overall_top[0][0]
        overlap = len(set(best_subset) & truth)
        print(f"\n  Ground-truth recovery (best subset): {overlap}/{len(truth)}")

    # Save
    pd.DataFrame(all_rounds).to_csv(RESULTS_DIR / "outer_loop_rounds.csv", index=False)
    top_rows = [
        {"rank": i+1, "reward": r, "genes": ";".join(s)}
        for i, (s, r) in enumerate(overall_top[:50])
    ]
    pd.DataFrame(top_rows).to_csv(RESULTS_DIR / "outer_loop_top_subsets.csv", index=False)

    with open(RESULTS_DIR / "outer_loop_summary.json", "w") as f:
        json.dump({
            "dataset": config.DATASET,
            "rounds": all_rounds,
            "total_unique": len(cumulative_reward_cache),
            "best_reward": overall_top[0][1] if overall_top else 0,
            "best_overlap": len(set(overall_top[0][0]) & truth) if truth and overall_top else 0,
        }, f, indent=2)

    print(f"\n[saved] {RESULTS_DIR / 'outer_loop_rounds.csv'}")
    print(f"[saved] {RESULTS_DIR / 'outer_loop_top_subsets.csv'}")
    print(f"[saved] {RESULTS_DIR / 'outer_loop_summary.json'}")
    print("\n✅ agentic loop complete.")

    return {
        "rounds": all_rounds,
        "reward_cache": cumulative_reward_cache,
        "sample_counter": cumulative_sample_counter,
    }


if __name__ == "__main__":
    run(max_rounds=3, epochs_per_round=150)
