# agentic-gflownet-genes

A personal demo walkthrough of the **Agentic GFlowNet for Gene Subset Discovery** project.
Goal: wire the full pipeline end-to-end on synthetic data before the team builds the real thing.

## Stack
- **Python** — GFlowNet, agents, outer loop, orchestration
- **R** — statistical evaluation via `limma` (differential expression → reward signal)
- **Bridge** — Python calls `Rscript` via `subprocess`, data passed as CSV. Simple, robust, no compile step.
- **LLM** — deferred. `llm_client.py` exposes a `chat()` function backed by a **mock** for now; swap in Anthropic/OpenAI/Ollama later by editing one file.

## Setup

### Python
```bash
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### R
Install R (https://cran.r-project.org/), then in an R console:
```r
if (!require("BiocManager", quietly = TRUE)) install.packages("BiocManager")
BiocManager::install("limma")
```

Verify both are visible from your shell:
```bash
python --version
Rscript --version
```

## Repo layout
```
src/
  agents/      manager.py, critic.py      # LLM-driven planning & feedback
  gflownet/    model.py, train.py         # diverse subset sampler
  env/         reward.py, limma_eval.R    # R-backed evaluation
  loop/        outer_loop.py              # Manager → GFN → Env → Critic
  llm_client.py                           # mock now, real API later
data/          synthetic_rnaseq.csv       # generated, tiny
notebooks/     demo.ipynb                 # runs the whole demo
tests/         test_smoke.py              # "does it run?"
```

## Milestones (each must visibly run)
1. **Scaffold + synthetic data** ← *you are here*
2. R/limma reward callable from Python
3. Minimal GFlowNet samples random subsets
4. Mock LLM Manager + Critic, full outer loop runs
5. Swap mock → real LLM API
6. Demo notebook: reward curve + diverse subsets

## Run milestone 1
```bash
python -m src.env.make_synthetic
```
Produces `data/synthetic_rnaseq.csv` and prints its shape.
