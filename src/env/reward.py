"""
reward.py — Evaluate gene subsets with switchable backend.

Two backends (set REWARD_BACKEND in src/config.py):
  "python"  — scipy Welch's t-test, ~0.001s per call (dev/iteration)
  "r"       — limma eBayes via Rscript, ~2.5s per call (gold standard)

Both compute:  score = (n_sig / subset_size) * mean |logFC|

On large sample sizes (TCGA ~1200 samples), the rankings are nearly
identical. Use "python" for rapid iteration, "r" for final validation.

Usage:
    from src.env.reward import evaluate_subset
    result = evaluate_subset(["BRCA1", "ESR1", ...])
    print(result["score"])
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src import config

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
R_SCRIPT = REPO_ROOT / "src" / "env" / "limma_eval.R"

# ---- Cached data for the python backend (loaded once, reused) ----
_py_cache: dict | None = None


def _load_python_backend():
    """Load and precompute per-gene stats for the fast python backend."""
    global _py_cache
    if _py_cache is not None and _py_cache["dataset"] == config.DATASET:
        return _py_cache

    cfg = config.active()
    expr = pd.read_csv(cfg["expr"], index_col=0)
    labels = pd.read_csv(cfg["labels"])
    ctrl_label, trt_label = cfg["groups"]

    ctrl_ids = labels[labels["group"] == ctrl_label]["sample_id"].tolist()
    trt_ids = labels[labels["group"] == trt_label]["sample_id"].tolist()

    ctrl_mat = expr[ctrl_ids].values.astype(np.float64)
    trt_mat = expr[trt_ids].values.astype(np.float64)

    n_ctrl = ctrl_mat.shape[1]
    n_trt = trt_mat.shape[1]

    # Precompute per-gene stats
    ctrl_mean = ctrl_mat.mean(axis=1)
    trt_mean = trt_mat.mean(axis=1)
    ctrl_var = ctrl_mat.var(axis=1, ddof=1)
    trt_var = trt_mat.var(axis=1, ddof=1)

    # Welch's t-test (unequal variances)
    se = np.sqrt(ctrl_var / n_ctrl + trt_var / n_trt + 1e-12)
    t_stat = (trt_mean - ctrl_mean) / se
    logfc = trt_mean - ctrl_mean  # already log-scale

    # Welch-Satterthwaite degrees of freedom
    num = (ctrl_var / n_ctrl + trt_var / n_trt) ** 2
    denom = (ctrl_var / n_ctrl) ** 2 / (n_ctrl - 1) + (trt_var / n_trt) ** 2 / (n_trt - 1)
    df = num / (denom + 1e-12)

    # Two-sided p-values
    from scipy import stats
    p_vals = 2.0 * stats.t.sf(np.abs(t_stat), df)

    # BH multiple testing correction (per-gene, applied at subset level later)
    gene_ids = expr.index.tolist()
    gene_to_idx = {g: i for i, g in enumerate(gene_ids)}

    _py_cache = {
        "dataset": config.DATASET,
        "gene_ids": gene_ids,
        "gene_to_idx": gene_to_idx,
        "logfc": logfc,
        "p_vals": p_vals,
        "t_stat": t_stat,
    }
    return _py_cache


def _evaluate_python(gene_ids: list[str]) -> dict:
    """Fast python-native evaluation using precomputed t-test stats."""
    cache = _load_python_backend()
    g2i = cache["gene_to_idx"]
    idxs = [g2i[g] for g in gene_ids if g in g2i]

    if len(idxs) < 2:
        return {"n_sig": 0, "mean_abs_logfc": 0.0, "score": 0.0}

    pvals = cache["p_vals"][idxs]
    logfcs = cache["logfc"][idxs]

    # BH correction on just this subset
    n = len(pvals)
    sorted_idx = np.argsort(pvals)
    sorted_pvals = pvals[sorted_idx]
    bh_thresh = np.arange(1, n + 1) / n * 0.05
    adj_sig = np.zeros(n, dtype=bool)
    # Find the largest k where p_(k) <= k/n * 0.05
    reject = sorted_pvals <= bh_thresh
    if reject.any():
        max_k = np.max(np.where(reject)[0])
        adj_sig[sorted_idx[:max_k + 1]] = True

    n_sig = int(adj_sig.sum())
    mean_abs_logfc = float(np.abs(logfcs).mean())
    score = (n_sig / len(idxs)) * mean_abs_logfc

    return {"n_sig": n_sig, "mean_abs_logfc": mean_abs_logfc, "score": score}


def _evaluate_r(gene_ids: list[str], verbose: bool = False) -> dict:
    """R/limma backend — calls Rscript, reads back result."""
    cfg = config.active()
    expr_path = cfg["expr"]
    labels_path = cfg["labels"]
    ctrl, trt = cfg["groups"]

    for p in (expr_path, labels_path):
        if not Path(p).exists():
            raise FileNotFoundError(
                f"Dataset file missing: {p}\n"
                f"Active dataset = {config.DATASET}. "
                f"Did you run the right data-prep step?"
            )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    subset_path = DATA_DIR / "_tmp_subset.csv"
    out_path = DATA_DIR / "_tmp_score.csv"
    pd.DataFrame({"gene_id": gene_ids}).to_csv(subset_path, index=False)

    proc = subprocess.run(
        ["Rscript", str(R_SCRIPT), str(expr_path), str(labels_path),
         str(subset_path), str(out_path), f"{ctrl},{trt}"],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Rscript failed (exit {proc.returncode})\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    if verbose:
        print(proc.stdout.strip())

    row = pd.read_csv(out_path).iloc[0]
    return {
        "n_sig": int(row["n_sig"]),
        "mean_abs_logfc": float(row["mean_abs_logfc"]),
        "score": float(row["score"]),
    }


def evaluate_subset(gene_ids: Iterable[str], verbose: bool = False) -> dict:
    """Evaluate a gene subset using the configured backend."""
    subset = list(gene_ids)
    if len(subset) == 0:
        return {"n_sig": 0, "mean_abs_logfc": 0.0, "score": 0.0}

    if config.REWARD_BACKEND == "python":
        return _evaluate_python(subset)
    elif config.REWARD_BACKEND == "r":
        return _evaluate_r(subset, verbose=verbose)
    else:
        raise ValueError(f"Unknown REWARD_BACKEND: {config.REWARD_BACKEND!r}")


if __name__ == "__main__":
    import random
    import time

    cfg = config.active()
    print(f"Active dataset: {config.DATASET}")
    print(f"Reward backend: {config.REWARD_BACKEND}")

    expr = pd.read_csv(cfg["expr"], index_col=0)
    all_genes = expr.index.tolist()
    print(f"  n_genes={len(all_genes)}  n_samples={expr.shape[1]}")

    random.seed(0)
    k = 15
    rand_subset = random.sample(all_genes, k=k)

    t0 = time.time()
    print(f"\n>>> Evaluating RANDOM subset of {k} genes")
    r1 = evaluate_subset(rand_subset, verbose=True)
    t1 = time.time()
    print(f"   result: {r1}  ({t1 - t0:.3f}s)")

    if cfg.get("truth") and Path(cfg["truth"]).exists():
        truth = pd.read_csv(cfg["truth"])["gene_id"].tolist()
        print(f"\n>>> Evaluating GROUND-TRUTH subset of {len(truth)} genes")
        t0 = time.time()
        r2 = evaluate_subset(truth, verbose=True)
        t1 = time.time()
        print(f"   result: {r2}  ({t1 - t0:.3f}s)")
        print(f"\n>>> Sanity check:")
        print(f"   truth score ({r2['score']:.4f}) should be >> "
              f"random score ({r1['score']:.4f})")
        if r2["score"] > r1["score"]:
            print("   ✅ bridge works.")
        else:
            print("   ⚠️  unexpected — investigate.")
    else:
        print("\n(no ground-truth gene list — skipping comparison)")

    # Speed benchmark
    print(f"\n>>> Speed benchmark: 100 random evaluations...")
    t0 = time.time()
    for _ in range(100):
        s = random.sample(all_genes, k=k)
        evaluate_subset(s)
    t1 = time.time()
    print(f"   100 calls in {t1 - t0:.2f}s  ({(t1 - t0) / 100 * 1000:.1f}ms per call)")
