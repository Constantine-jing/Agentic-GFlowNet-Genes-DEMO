"""
config.py — Single source of truth for dataset selection.

To switch which dataset the whole pipeline uses, change DATASET below.
Everything else (reward, GFlowNet, agents, loop) reads from here.

Each dataset entry specifies:
    expr     : path to gene × sample expression matrix (CSV, log-scale)
    labels   : path to sample_id, group table (CSV)
    groups   : (control_label, treatment_label) tuple — order matters,
               limma computes treatment-minus-control logFC.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

# --- change this line to switch datasets ---
DATASET = "tcga_brca"
# -------------------------------------------

# --- reward backend ---
# "python" = scipy t-test, fast (~0.001s/call), good for dev & iteration
# "r"      = limma eBayes, slow (~2.5s/call), gold-standard for final results
# Both produce the same score formula. On large sample sizes (TCGA ~1200)
# the rankings are nearly identical.
REWARD_BACKEND = "python"
# ----------------------

DATASETS = {
    "synthetic": {
        "expr":   DATA_DIR / "synthetic_rnaseq.csv",
        "labels": DATA_DIR / "sample_labels.csv",
        "groups": ("control", "treatment"),
        "truth":  DATA_DIR / "truth_signal_genes.csv",  # only synthetic has this
    },
    "tcga_brca": {
        "expr":   DATA_DIR / "tcga_brca_rnaseq.csv",
        "labels": DATA_DIR / "tcga_brca_labels.csv",
        "groups": ("normal", "tumor"),
        "truth":  None,  # no ground truth for real data
    },
    "tcga_lihc": {
        "expr":   DATA_DIR / "tcga_lihc_rnaseq.csv",
        "labels": DATA_DIR / "tcga_lihc_labels.csv",
        "groups": ("normal", "tumor"),
        "truth":  None,
    },
    "tcga_luad": {
        "expr":   DATA_DIR / "tcga_luad_rnaseq.csv",
        "labels": DATA_DIR / "tcga_luad_labels.csv",
        "groups": ("normal", "tumor"),
        "truth":  None,
    },
}


def active() -> dict:
    """Return the config dict for the currently selected DATASET."""
    if DATASET not in DATASETS:
        raise ValueError(
            f"DATASET={DATASET!r} not in DATASETS. "
            f"Options: {list(DATASETS)}"
        )
    return DATASETS[DATASET]


if __name__ == "__main__":
    cfg = active()
    print(f"Active dataset: {DATASET}")
    for k, v in cfg.items():
        exists = "✅" if (isinstance(v, Path) and v.exists()) else \
                 ("—" if v is None else "❌ missing")
        print(f"  {k:8s} {v}  {exists}")
