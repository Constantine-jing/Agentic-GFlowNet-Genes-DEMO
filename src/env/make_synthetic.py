"""
Generate a tiny synthetic RNA-seq dataset for the demo.

Structure:
- N samples (half 'control', half 'treatment')
- G genes
- A small hidden set of `signal` genes are genuinely differentially expressed;
  the rest are noise. This lets us later verify the GFlowNet can find them.

Output: data/synthetic_rnaseq.csv
  rows   = genes  (first column: gene_id)
  cols   = samples
  plus a sidecar data/sample_labels.csv with group labels
  plus data/truth_signal_genes.csv listing the planted signal genes
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def generate(
    n_genes: int = 200,
    n_samples_per_group: int = 10,
    n_signal: int = 15,
    effect_size: float = 2.5,
    seed: int = 0,
) -> None:
    rng = np.random.default_rng(seed)
    n_samples = 2 * n_samples_per_group

    # Baseline log-expression ~ N(6, 1) for every gene/sample.
    expr = rng.normal(loc=6.0, scale=1.0, size=(n_genes, n_samples))

    # Plant signal: pick n_signal genes, bump treatment half by effect_size.
    signal_idx = rng.choice(n_genes, size=n_signal, replace=False)
    expr[signal_idx, n_samples_per_group:] += effect_size

    gene_ids = [f"gene_{i:04d}" for i in range(n_genes)]
    sample_ids = (
        [f"ctrl_{i:02d}" for i in range(n_samples_per_group)]
        + [f"trt_{i:02d}" for i in range(n_samples_per_group)]
    )
    labels = ["control"] * n_samples_per_group + ["treatment"] * n_samples_per_group

    expr_df = pd.DataFrame(expr, index=gene_ids, columns=sample_ids)
    expr_df.index.name = "gene_id"
    labels_df = pd.DataFrame({"sample_id": sample_ids, "group": labels})
    truth_df = pd.DataFrame({"gene_id": [gene_ids[i] for i in sorted(signal_idx)]})

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    expr_df.to_csv(DATA_DIR / "synthetic_rnaseq.csv")
    labels_df.to_csv(DATA_DIR / "sample_labels.csv", index=False)
    truth_df.to_csv(DATA_DIR / "truth_signal_genes.csv", index=False)

    print(f"[ok] wrote {DATA_DIR/'synthetic_rnaseq.csv'}   shape={expr_df.shape}")
    print(f"[ok] wrote {DATA_DIR/'sample_labels.csv'}     n={len(labels_df)}")
    print(f"[ok] wrote {DATA_DIR/'truth_signal_genes.csv'} n_signal={n_signal}")


if __name__ == "__main__":
    generate()
