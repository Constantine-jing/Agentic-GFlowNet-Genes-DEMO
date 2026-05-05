"""
validate_limma.py — Run the GFlowNet with R/limma backend and compare to Python/scipy.

This validates that the fast Python reward backend produces results
consistent with the gold-standard R/limma evaluation.

Two comparisons:
  1. Score comparison: evaluate the SAME top subsets with both backends
  2. (Optional) Full training run with R backend — slow but definitive

Run from repo root:
    python -m scripts.validate_limma
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import config

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"


def compare_scores():
    """
    Take the top subsets found by the Python backend and re-score them
    with R/limma. Compare the rankings and scores.
    """
    print("=" * 60)
    print("  LIMMA VALIDATION: Score Comparison")
    print("=" * 60)

    # Load top subsets
    top_path = RESULTS_DIR / "outer_loop_top_subsets.csv"
    if not top_path.exists():
        top_path = RESULTS_DIR / "top_subsets.csv"
    if not top_path.exists():
        print("ERROR: No GFlowNet results found. Run the outer loop first.")
        return

    top_df = pd.read_csv(top_path)
    n_compare = min(20, len(top_df))

    print(f"\n  Re-scoring top {n_compare} subsets with both backends...\n")

    # Score with Python backend
    config.REWARD_BACKEND = "python"
    from src.env.reward import evaluate_subset, _load_python_backend
    # Force reload
    import src.env.reward as rm
    rm._py_cache = None

    python_scores = []
    for i in range(n_compare):
        genes = top_df.iloc[i]["genes"].split(";")
        result = evaluate_subset(genes)
        python_scores.append(result["score"])

    # Score with R/limma backend
    config.REWARD_BACKEND = "r"
    # Need to reimport or just call the R version directly
    from src.env.reward import _evaluate_r

    limma_scores = []
    print("  Calling R/limma (this takes ~1 min for 20 subsets)...")
    t0 = time.time()
    for i in range(n_compare):
        genes = top_df.iloc[i]["genes"].split(";")
        result = _evaluate_r(genes)
        limma_scores.append(result["score"])
        if (i + 1) % 5 == 0:
            print(f"    scored {i + 1}/{n_compare}  ({time.time() - t0:.1f}s)")

    elapsed = time.time() - t0

    # Reset to python
    config.REWARD_BACKEND = "python"

    # Compare
    python_scores = np.array(python_scores)
    limma_scores = np.array(limma_scores)

    # Correlation
    correlation = np.corrcoef(python_scores, limma_scores)[0, 1]

    # Rank correlation (Spearman)
    from scipy.stats import spearmanr
    spearman_r, spearman_p = spearmanr(python_scores, limma_scores)

    # Mean absolute difference
    mad = np.mean(np.abs(python_scores - limma_scores))

    # Do rankings agree?
    python_rank = np.argsort(-python_scores)
    limma_rank = np.argsort(-limma_scores)
    top5_overlap = len(set(python_rank[:5]) & set(limma_rank[:5]))

    print(f"\n{'=' * 60}")
    print(f"  VALIDATION RESULTS")
    print(f"{'=' * 60}")
    print(f"\n  {'Subset':<8} {'Python score':>14} {'Limma score':>14} {'Diff':>10}")
    print(f"  {'-' * 48}")
    for i in range(n_compare):
        diff = python_scores[i] - limma_scores[i]
        print(f"  {i+1:<8} {python_scores[i]:>14.4f} {limma_scores[i]:>14.4f} {diff:>+10.4f}")

    print(f"\n  Summary statistics:")
    print(f"    Pearson correlation:   {correlation:.4f}")
    print(f"    Spearman correlation:  {spearman_r:.4f} (p={spearman_p:.2e})")
    print(f"    Mean absolute diff:    {mad:.4f}")
    print(f"    Top-5 ranking overlap: {top5_overlap}/5")
    print(f"    R/limma eval time:     {elapsed:.1f}s for {n_compare} subsets")

    if correlation > 0.95:
        print(f"\n  ✅ STRONG agreement (r={correlation:.3f}).")
        print(f"     Python/scipy backend is a valid fast proxy for R/limma.")
    elif correlation > 0.85:
        print(f"\n  ⚠️  MODERATE agreement (r={correlation:.3f}).")
        print(f"     Rankings mostly preserved but scores differ.")
    else:
        print(f"\n  ❌ WEAK agreement (r={correlation:.3f}).")
        print(f"     Backends produce meaningfully different results.")

    # Save
    comparison = {
        "n_subsets": n_compare,
        "pearson_r": round(correlation, 4),
        "spearman_r": round(spearman_r, 4),
        "spearman_p": float(f"{spearman_p:.2e}"),
        "mean_abs_diff": round(mad, 4),
        "top5_overlap": top5_overlap,
        "python_scores": python_scores.tolist(),
        "limma_scores": limma_scores.tolist(),
    }
    with open(RESULTS_DIR / "limma_validation.json", "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\n[saved] {RESULTS_DIR / 'limma_validation.json'}")

    return comparison


def full_limma_run():
    """
    Run the full GFlowNet training with R/limma backend.
    WARNING: This is SLOW (~2-5 hours on TCGA-BRCA).
    """
    print(f"\n{'=' * 60}")
    print(f"  FULL TRAINING RUN WITH R/LIMMA BACKEND")
    print(f"  WARNING: This will take 2-5 hours")
    print(f"{'=' * 60}")

    # Switch to R backend
    config.REWARD_BACKEND = "r"

    from src.loop.outer_loop import run

    # Fewer epochs since each is slow
    results = run(max_rounds=2, epochs_per_round=50)

    # Save separately
    import shutil
    for fname in ["outer_loop_summary.json", "outer_loop_top_subsets.csv", "outer_loop_rounds.csv"]:
        src = RESULTS_DIR / fname
        dst = RESULTS_DIR / f"limma_{fname.replace('outer_loop_', '')}"
        if src.exists():
            shutil.copy(src, dst)
            print(f"[saved] {dst}")

    # Switch back
    config.REWARD_BACKEND = "python"

    return results


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    # Step 1: Quick score comparison (fast, ~1 minute)
    print("\n[Phase 1] Score comparison on existing subsets...")
    compare_scores()

    # Step 2: Full training run (slow, optional overnight)
    print(f"\n{'─' * 60}")
    print(f"  Phase 1 (score comparison) complete.")
    print(f"  Phase 2 (full limma training) starting...")
    print(f"  This will take several hours. Safe to run overnight.")
    print(f"{'─' * 60}")

    full_limma_run()

    print(f"\n✅ limma validation complete.")


if __name__ == "__main__":
    main()
