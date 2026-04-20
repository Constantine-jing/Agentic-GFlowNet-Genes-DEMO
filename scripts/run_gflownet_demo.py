"""
run_gflownet_demo.py — Run the GFlowNet end-to-end on the active dataset.

Run from repo root:
    python -m scripts.run_gflownet_demo
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src import config
from src.gflownet.train import train

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    # ---- train ----
    out = train(
        n_epochs=300,
        batch_size=16,
        replay_batch=16,
        subset_size=15,
        lr=5e-3,
        log_z_lr=1.0,
        reward_exponent=8.0,
        temp_start=2.0,
        temp_end=0.5,
        replay_capacity=128,
        log_every=10,
    )
    history = out["history"]
    reward_cache = out["reward_cache"]
    sample_counter = out["sample_counter"]

    # ---- save history ----
    pd.DataFrame(history).to_csv(RESULTS_DIR / "training_history.csv", index=False)
    print(f"\n[saved] {RESULTS_DIR / 'training_history.csv'}")

    # ---- reward curve plot ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        ax = axes[0]
        ax.plot(history["epoch"], history["mean_reward"], label="mean (per batch)", alpha=0.7)
        ax.plot(history["epoch"], history["best_ever"], label="best ever", color="green")
        ax.set_xlabel("epoch")
        ax.set_ylabel("reward")
        ax.set_title(f"Reward — {config.DATASET}")
        ax.legend()
        ax.grid(alpha=0.3)

        ax = axes[1]
        ax.plot(history["epoch"], history["loss"], color="crimson", alpha=0.7)
        ax.set_xlabel("epoch")
        ax.set_ylabel("TB loss")
        ax.set_title("Trajectory balance loss")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)

        ax = axes[2]
        ax.plot(history["epoch"], history["temp"], color="orange")
        ax.set_xlabel("epoch")
        ax.set_ylabel("temperature")
        ax.set_title("Sampling temperature")
        ax.grid(alpha=0.3)

        fig.tight_layout()
        plot_path = RESULTS_DIR / "reward_curve.png"
        fig.savefig(plot_path, dpi=120)
        print(f"[saved] {plot_path}")
    except ImportError:
        print("[skip] matplotlib not installed — pip install matplotlib for the plot")

    # ---- top subsets by reward ----
    print("\n=== Top 10 highest-reward subsets discovered ===")
    top_by_reward = sorted(reward_cache.items(), key=lambda kv: kv[1], reverse=True)[:10]
    for rank, (subset, r) in enumerate(top_by_reward, 1):
        count = sample_counter[subset]
        preview = ", ".join(subset[:5]) + ("..." if len(subset) > 5 else "")
        print(f"{rank:2d}. reward={r:6.3f}  count={count:3d}  [{preview}]")

    # ---- save top subsets ----
    top_rows = []
    for rank, (subset, r) in enumerate(
        sorted(reward_cache.items(), key=lambda kv: kv[1], reverse=True)[:50], 1
    ):
        top_rows.append({
            "rank": rank,
            "reward": r,
            "sample_count": sample_counter[subset],
            "genes": ";".join(subset),
        })
    pd.DataFrame(top_rows).to_csv(RESULTS_DIR / "top_subsets.csv", index=False)
    print(f"[saved] {RESULTS_DIR / 'top_subsets.csv'}")

    # ---- ground-truth recovery ----
    cfg = config.active()
    truth_path = cfg.get("truth")
    if truth_path and Path(truth_path).exists():
        truth = set(pd.read_csv(truth_path)["gene_id"].tolist())
        print(f"\n=== Ground-truth recovery (n_truth={len(truth)}) ===")

        best_subset, best_r = max(reward_cache.items(), key=lambda kv: kv[1])
        most_sampled_subset, _ = sample_counter.most_common(1)[0]

        def overlap(s):
            return len(set(s) & truth)

        print(f"  highest-reward subset:  {overlap(best_subset)}/{len(truth)} "
              f"truth genes recovered  (R={best_r:.3f})")
        print(f"  most-sampled subset:    {overlap(most_sampled_subset)}/{len(truth)} "
              f"truth genes recovered")

        recovered = sum(1 for s, _ in top_by_reward if overlap(s) >= len(truth) // 2)
        print(f"  top-10 by-reward subsets with ≥{len(truth)//2} truth genes:  "
              f"{recovered}/10")

        # Gene-level frequency in top-50 subsets
        from collections import Counter
        gene_freq = Counter()
        for s, _ in sorted(reward_cache.items(), key=lambda kv: kv[1], reverse=True)[:50]:
            for g in s:
                gene_freq[g] += 1
        print(f"\n  Most frequent genes in top-50 subsets:")
        for g, freq in gene_freq.most_common(20):
            tag = " ★ TRUTH" if g in truth else ""
            print(f"    {g:12s}  appears {freq:2d}/50{tag}")

        summary = {
            "dataset": config.DATASET,
            "n_truth": len(truth),
            "best_subset_overlap": overlap(best_subset),
            "best_subset_reward": best_r,
            "most_sampled_overlap": overlap(most_sampled_subset),
            "top10_majority_recovery": recovered,
        }
        with open(RESULTS_DIR / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n[saved] {RESULTS_DIR / 'summary.json'}")
    else:
        print("\n(no ground truth for this dataset — skipping recovery check)")

    print("\n✅ done.")


if __name__ == "__main__":
    main()
