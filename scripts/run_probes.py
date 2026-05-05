"""
probes.py — LLM Probing: Does the Critic understand gene subset biology?

Implements linear probing (Alain & Bengio, 2016) with selectivity controls
(Belinkov, 2022) on a local LLM (Phi-3-mini via Ollama).

The experiment:
  1. Generate text descriptions of gene subsets (high-reward and low-reward)
  2. Feed them into Phi-3-mini, extract hidden layer embeddings
  3. Train a linear probe: can we predict subset quality from the LLM's
     internal representations?
  4. Control: repeat with shuffled labels (Belinkov's selectivity test)
  5. If real accuracy >> control accuracy → the LLM encodes biological signal

References:
  - Alain & Bengio (2016) "Understanding intermediate layers using linear
    classifier probes"
  - Belinkov (2022) "Probing Classifiers: Promises, Shortcomings, and Advances"
  - Gurnee & Tegmark (2024) "Language Models Represent Space and Time"

Requirements:
  - Ollama running locally with phi3:mini pulled
  - pip install requests scikit-learn

Run from repo root:
    python -m scripts.run_probes
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from src import config

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"

OLLAMA_URL = "http://localhost:11434"
MODEL = "phi3:mini"


# ============================================================
#  Step 1: Generate probe dataset (text descriptions of subsets)
# ============================================================

def generate_probe_texts(n_high: int = 50, n_low: int = 50) -> tuple[list[str], list[int]]:
    """
    Create text descriptions of gene subsets with binary labels:
      1 = high-reward subset (from GFlowNet top results)
      0 = low-reward subset (random genes)

    Returns (texts, labels).
    """
    cfg = config.active()
    expr = pd.read_csv(cfg["expr"], index_col=0)
    gene_ids = expr.index.tolist()

    # Load GFlowNet top subsets as "high quality"
    top_path = RESULTS_DIR / "outer_loop_top_subsets.csv"
    if not top_path.exists():
        top_path = RESULTS_DIR / "top_subsets.csv"
    if not top_path.exists():
        raise FileNotFoundError("No GFlowNet results found. Run the outer loop first.")

    top_df = pd.read_csv(top_path)
    top_subsets = [row["genes"].split(";") for _, row in top_df.iterrows()]

    texts = []
    labels = []

    # High-reward subsets
    for i in range(min(n_high, len(top_subsets))):
        genes = top_subsets[i]
        text = _subset_to_prompt(genes, quality="unknown")
        texts.append(text)
        labels.append(1)

    # Pad with slight variations if we need more high samples
    while len(texts) < n_high and len(top_subsets) > 0:
        base = random.choice(top_subsets)
        # Swap 1-2 genes to create a variant
        variant = list(base)
        n_swap = random.randint(1, 2)
        for _ in range(n_swap):
            idx = random.randint(0, len(variant) - 1)
            variant[idx] = random.choice(gene_ids)
        texts.append(_subset_to_prompt(variant, quality="unknown"))
        labels.append(1)

    # Low-reward subsets (random)
    for _ in range(n_low):
        random_genes = random.sample(gene_ids, k=15)
        texts.append(_subset_to_prompt(random_genes, quality="unknown"))
        labels.append(0)

    return texts, labels


def _subset_to_prompt(genes: list[str], quality: str = "unknown") -> str:
    """Convert a gene list into a natural language description for the LLM."""
    gene_str = ", ".join(genes[:15])
    return (
        f"The following gene subset was selected from a breast cancer RNA-seq dataset "
        f"for differential expression analysis between tumor and normal tissue samples: "
        f"{gene_str}. "
        f"These genes were evaluated as a group for their ability to distinguish "
        f"tumor from normal breast tissue."
    )


# ============================================================
#  Step 2: Extract hidden states from Ollama
# ============================================================

def get_embeddings(texts: list[str], layer: str = "last") -> np.ndarray:
    """
    Send texts to Ollama's Phi-3-mini and extract embeddings.

    Ollama's /api/embeddings endpoint returns the model's embedding
    for the input text. We use this as a proxy for hidden state
    representations.

    Returns: (n_texts, embed_dim) numpy array
    """
    import urllib.request

    # Warm up: first call loads the model into memory (~30-60s)
    print("  warming up model (first call loads weights, may take 60s)...")
    warmup_body = json.dumps({"model": MODEL, "prompt": "hello"}).encode()
    warmup_req = urllib.request.Request(
        f"{OLLAMA_URL}/api/embeddings",
        data=warmup_body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(warmup_req, timeout=120) as resp:
            json.loads(resp.read())
            print("  model loaded ✓")
    except Exception as e:
        print(f"  warmup failed: {e}")
        print(f"  make sure Ollama is running and phi3:mini is pulled")
        raise

    embeddings = []
    total = len(texts)

    for i, text in enumerate(texts):
        body = json.dumps({
            "model": MODEL,
            "prompt": text,
        }).encode()

        # Retry up to 3 times
        for attempt in range(3):
            try:
                req = urllib.request.Request(
                    f"{OLLAMA_URL}/api/embeddings",
                    data=body,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = json.loads(resp.read())
                    emb = data["embedding"]
                    embeddings.append(emb)
                    break
            except Exception as e:
                if attempt < 2:
                    print(f"  [retry {attempt+1}] text {i}: {e}")
                    import time; time.sleep(2)
                else:
                    print(f"  [warning] failed on text {i} after 3 attempts: {e}")
                    if embeddings:
                        embeddings.append([0.0] * len(embeddings[0]))
                    else:
                        raise

        if (i + 1) % 10 == 0 or i == total - 1:
            print(f"  embeddings: {i + 1}/{total}")

    return np.array(embeddings)


# ============================================================
#  Step 3: Linear probing with Belinkov controls
# ============================================================

def run_probing_experiment(
    X: np.ndarray,
    y: np.ndarray,
    n_random_controls: int = 5,
    cv_folds: int = 5,
) -> dict:
    """
    Train a linear probe and compare against random-label controls.

    Returns dict with real_accuracy, control_accuracy, selectivity.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Real probe
    probe = LogisticRegression(max_iter=1000, random_state=0)
    real_scores = cross_val_score(probe, X_scaled, y, cv=cv_folds, scoring="accuracy")
    real_acc = real_scores.mean()
    real_std = real_scores.std()

    print(f"\n  Real probe accuracy: {real_acc:.3f} +/- {real_std:.3f}")

    # Control probes (shuffled labels — Belinkov's selectivity test)
    control_accs = []
    for i in range(n_random_controls):
        y_shuffled = np.random.permutation(y)
        control_scores = cross_val_score(probe, X_scaled, y_shuffled, cv=cv_folds, scoring="accuracy")
        control_accs.append(control_scores.mean())

    control_mean = np.mean(control_accs)
    control_std = np.std(control_accs)

    print(f"  Control accuracy:   {control_mean:.3f} +/- {control_std:.3f}")

    # Selectivity = real - control (Belinkov 2022)
    selectivity = real_acc - control_mean
    print(f"  Selectivity:        {selectivity:+.3f}")

    if selectivity > 0.1:
        print(f"  → The LLM's representations encode biological signal (selectivity > 0.1)")
    elif selectivity > 0.05:
        print(f"  → Weak signal detected (0.05 < selectivity < 0.1)")
    else:
        print(f"  → No meaningful signal (selectivity <= 0.05)")

    return {
        "real_accuracy": round(real_acc, 4),
        "real_std": round(real_std, 4),
        "control_accuracy": round(control_mean, 4),
        "control_std": round(control_std, 4),
        "selectivity": round(selectivity, 4),
    }


# ============================================================
#  Step 4: Probe different prompt framings
# ============================================================

def generate_biological_prompts(genes: list[str]) -> str:
    """Frame subset with biological context — tests if LLM engages biology knowledge."""
    gene_str = ", ".join(genes[:15])
    return (
        f"In a study of breast cancer gene expression, researchers identified "
        f"the following subset of genes as potentially important for distinguishing "
        f"malignant tumors from healthy tissue: {gene_str}. "
        f"Consider the known functions of these genes in cell proliferation, "
        f"immune response, and metabolic pathways."
    )


def generate_neutral_prompts(genes: list[str]) -> str:
    """Frame subset without biological context — control for domain knowledge."""
    gene_str = ", ".join(genes[:15])
    return (
        f"The following items were selected from a dataset: {gene_str}. "
        f"These items were evaluated as a group."
    )


# ============================================================
#  Main experiment
# ============================================================

def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    random.seed(42)
    np.random.seed(42)

    print("=" * 60)
    print("  LLM PROBING EXPERIMENT")
    print(f"  model: {MODEL}")
    print(f"  dataset: {config.DATASET}")
    print("=" * 60)

    # Check Ollama is running
    print("\n[Step 0] Checking Ollama connection...")
    try:
        import urllib.request
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            models = json.loads(resp.read())
            available = [m["name"] for m in models.get("models", [])]
            if any(MODEL.split(":")[0] in m for m in available):
                print(f"  ✓ Ollama running, {MODEL} available")
            else:
                print(f"  ✗ {MODEL} not found. Run: ollama pull {MODEL}")
                return
    except Exception as e:
        print(f"  ✗ Cannot connect to Ollama at {OLLAMA_URL}")
        print(f"    Error: {e}")
        print(f"    Make sure Ollama is running: ollama serve")
        return

    t0 = time.time()

    # ---- Experiment 1: Biological framing ----
    print("\n" + "=" * 60)
    print("  EXPERIMENT 1: Biological prompt framing")
    print("=" * 60)

    print("\n[Step 1] Generating probe dataset...")
    cfg = config.active()
    expr = pd.read_csv(cfg["expr"], index_col=0)
    gene_ids = expr.index.tolist()

    top_path = RESULTS_DIR / "outer_loop_top_subsets.csv"
    if not top_path.exists():
        top_path = RESULTS_DIR / "top_subsets.csv"
    top_df = pd.read_csv(top_path)
    top_subsets = [row["genes"].split(";") for _, row in top_df.iterrows()]

    bio_texts = []
    bio_labels = []

    # High-reward (from GFlowNet)
    for i in range(min(40, len(top_subsets))):
        bio_texts.append(generate_biological_prompts(top_subsets[i]))
        bio_labels.append(1)

    # Pad high with variants
    while len(bio_texts) < 50:
        base = random.choice(top_subsets[:10])
        variant = list(base)
        idx = random.randint(0, len(variant) - 1)
        variant[idx] = random.choice(gene_ids)
        bio_texts.append(generate_biological_prompts(variant))
        bio_labels.append(1)

    # Low-reward (random)
    for _ in range(50):
        random_genes = random.sample(gene_ids, k=15)
        bio_texts.append(generate_biological_prompts(random_genes))
        bio_labels.append(0)

    print(f"  {len(bio_texts)} prompts ({sum(bio_labels)} high, {len(bio_labels) - sum(bio_labels)} low)")

    print("\n[Step 2] Extracting embeddings (biological framing)...")
    X_bio = get_embeddings(bio_texts)
    y_bio = np.array(bio_labels)

    print("\n[Step 3] Running linear probe (biological framing)...")
    bio_results = run_probing_experiment(X_bio, y_bio)

    # ---- Experiment 2: Neutral framing (control) ----
    print("\n" + "=" * 60)
    print("  EXPERIMENT 2: Neutral prompt framing (control)")
    print("=" * 60)

    neutral_texts = []
    neutral_labels = []

    for i in range(min(40, len(top_subsets))):
        neutral_texts.append(generate_neutral_prompts(top_subsets[i]))
        neutral_labels.append(1)

    while len(neutral_texts) < 50:
        base = random.choice(top_subsets[:10])
        variant = list(base)
        idx = random.randint(0, len(variant) - 1)
        variant[idx] = random.choice(gene_ids)
        neutral_texts.append(generate_neutral_prompts(variant))
        neutral_labels.append(1)

    for _ in range(50):
        random_genes = random.sample(gene_ids, k=15)
        neutral_texts.append(generate_neutral_prompts(random_genes))
        neutral_labels.append(0)

    print(f"\n[Step 4] Extracting embeddings (neutral framing)...")
    X_neutral = get_embeddings(neutral_texts)
    y_neutral = np.array(neutral_labels)

    print("\n[Step 5] Running linear probe (neutral framing)...")
    neutral_results = run_probing_experiment(X_neutral, y_neutral)

    elapsed = time.time() - t0

    # ---- Summary ----
    print(f"\n{'=' * 60}")
    print(f"  PROBING RESULTS SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Model: {MODEL}")
    print(f"  Samples: 50 high-reward + 50 low-reward = 100 total")
    print(f"  Probe: logistic regression, 5-fold CV")
    print(f"  Control: Belinkov selectivity (shuffled labels x5)")
    print()
    print(f"  {'Experiment':<25} {'Accuracy':>10} {'Control':>10} {'Selectivity':>12}")
    print(f"  {'-'*57}")
    print(f"  {'Biological framing':<25} {bio_results['real_accuracy']:>10.3f} "
          f"{bio_results['control_accuracy']:>10.3f} "
          f"{bio_results['selectivity']:>+12.3f}")
    print(f"  {'Neutral framing':<25} {neutral_results['real_accuracy']:>10.3f} "
          f"{neutral_results['control_accuracy']:>10.3f} "
          f"{neutral_results['selectivity']:>+12.3f}")

    print(f"\n  Interpretation:")
    if bio_results["selectivity"] > neutral_results["selectivity"] + 0.05:
        print(f"  → Biological framing produces HIGHER selectivity than neutral.")
        print(f"    This suggests the LLM activates domain-specific biological")
        print(f"    knowledge when given cancer-related context, enabling better")
        print(f"    discrimination between high and low quality gene subsets.")
    elif abs(bio_results["selectivity"] - neutral_results["selectivity"]) <= 0.05:
        print(f"  → Both framings show SIMILAR selectivity.")
        print(f"    The LLM's discrimination comes from gene name patterns")
        print(f"    rather than activated biological knowledge.")
    else:
        print(f"  → Neutral framing shows HIGHER selectivity.")
        print(f"    The biological context may add noise rather than signal.")

    if max(bio_results["selectivity"], neutral_results["selectivity"]) > 0.1:
        print(f"\n  → CONCLUSION: The LLM's internal representations DO encode")
        print(f"    information relevant to gene subset quality. The Critic's")
        print(f"    evaluations are grounded in meaningful representations,")
        print(f"    not purely surface-level pattern matching.")
    else:
        print(f"\n  → CONCLUSION: Limited evidence that the LLM's representations")
        print(f"    encode gene subset quality. The Critic may rely more on")
        print(f"    surface patterns than deep biological understanding.")

    print(f"\n  Time: {elapsed:.1f}s")

    # Save
    all_results = {
        "model": MODEL,
        "dataset": config.DATASET,
        "n_samples": len(bio_texts),
        "biological_framing": bio_results,
        "neutral_framing": neutral_results,
        "elapsed_seconds": round(elapsed, 1),
    }
    with open(RESULTS_DIR / "probing_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[saved] {RESULTS_DIR / 'probing_results.json'}")

    print(f"\n✅ probing experiment complete.")


if __name__ == "__main__":
    main()
