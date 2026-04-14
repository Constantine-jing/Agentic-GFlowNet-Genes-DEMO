"""
reward.py — Python wrapper around limma_eval.R

Reads the active dataset's file paths from src/config.py and passes them
to the R script as command-line arguments. To switch datasets, edit
DATASET in src/config.py — no changes needed here.

Usage:
    from src.env.reward import evaluate_subset
    result = evaluate_subset(["gene_0031", ...])
    print(result["score"])
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable

import pandas as pd

from src import config

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
R_SCRIPT = REPO_ROOT / "src" / "env" / "limma_eval.R"


def evaluate_subset(gene_ids: Iterable[str], verbose: bool = False) -> dict:
    """Evaluate a gene subset on the currently active dataset."""
    subset = list(gene_ids)
    if len(subset) == 0:
        return {"n_sig": 0, "mean_abs_logfc": 0.0, "score": 0.0}

    cfg = config.active()
    expr_path = cfg["expr"]
    labels_path = cfg["labels"]
    ctrl, trt = cfg["groups"]

    # Guard: data files must exist.
    for p in (expr_path, labels_path):
        if not Path(p).exists():
            raise FileNotFoundError(
                f"Dataset file missing: {p}\n"
                f"Active dataset = {config.DATASET}. "
                f"Did you run the right data-prep step?"
            )

    # 1. write subset for R to read
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    subset_path = DATA_DIR / "_tmp_subset.csv"
    out_path = DATA_DIR / "_tmp_score.csv"
    pd.DataFrame({"gene_id": subset}).to_csv(subset_path, index=False)

    # 2. call Rscript with explicit paths
    proc = subprocess.run(
        [
            "Rscript",
            str(R_SCRIPT),
            str(expr_path),
            str(labels_path),
            str(subset_path),
            str(out_path),
            f"{ctrl},{trt}",
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Rscript failed (exit {proc.returncode})\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    if verbose:
        print(proc.stdout.strip())

    # 3. read score back
    row = pd.read_csv(out_path).iloc[0]
    return {
        "n_sig": int(row["n_sig"]),
        "mean_abs_logfc": float(row["mean_abs_logfc"]),
        "score": float(row["score"]),
    }


if __name__ == "__main__":
    # Smoke test: random vs truth on the active dataset.
    import random

    cfg = config.active()
    print(f"Active dataset: {config.DATASET}")

    expr = pd.read_csv(cfg["expr"], index_col=0)
    all_genes = expr.index.tolist()
    print(f"  n_genes={len(all_genes)}  n_samples={expr.shape[1]}")

    random.seed(0)
    k = 15
    rand_subset = random.sample(all_genes, k=k)

    print(f"\n>>> Evaluating RANDOM subset of {k} genes")
    r1 = evaluate_subset(rand_subset, verbose=True)
    print("   result:", r1)

    if cfg.get("truth") and Path(cfg["truth"]).exists():
        truth = pd.read_csv(cfg["truth"])["gene_id"].tolist()
        print(f"\n>>> Evaluating GROUND-TRUTH subset of {len(truth)} genes")
        r2 = evaluate_subset(truth, verbose=True)
        print("   result:", r2)
        print("\n>>> Sanity check:")
        print(f"   truth score ({r2['score']:.4f}) should be >> "
              f"random score ({r1['score']:.4f})")
        if r2["score"] > r1["score"]:
            print("   ✅ bridge works — R is returning meaningful signal to Python.")
        else:
            print("   ⚠️  unexpected — investigate before proceeding.")
    else:
        print("\n(no ground-truth gene list for this dataset — skipping comparison)")
