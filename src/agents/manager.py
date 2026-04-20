"""
manager.py — LLM Manager agent.

The Manager sits at the top of the agentic loop. Before each GFlowNet
training round, it:
  1. Receives context about the dataset and current state
  2. Generates constraints, strategy guidance, and hyperparameter suggestions
  3. Returns a structured dict that the outer loop can feed into training

With the mock backend, responses are deterministic and plausible.
With a real LLM, the Manager can reason about biological priors,
suggest pathway-specific constraints, and adapt strategy based on
prior round results.
"""
from __future__ import annotations

import json

from src.llm_client import chat

SYSTEM_PROMPT = """You are a bioinformatics research manager overseeing a GFlowNet-based
gene subset discovery system. Your role is to provide strategic guidance
for each training round.

You will receive:
- Dataset info (n_genes, n_samples, groups)
- Results from prior rounds (if any)

You must respond with a JSON object containing:
- "constraints": dict of constraints for subset selection
- "strategy": text description of the recommended approach
- "hyperparams": suggested training hyperparameters
- "focus_areas": list of specific biological aspects to prioritize

Respond ONLY with valid JSON, no other text."""


def generate_plan(
    dataset_info: dict,
    prior_results: dict | None = None,
    round_num: int = 0,
) -> dict:
    """
    Ask the Manager LLM to generate a plan for the next GFlowNet round.

    Args:
        dataset_info: {n_genes, n_samples, groups, dataset_name}
        prior_results: results dict from previous round (None if first round)
        round_num: which round of the outer loop we're on

    Returns:
        Parsed dict with constraints, strategy, hyperparams, focus_areas
    """
    user_msg = f"""Round {round_num} planning.

Dataset: {json.dumps(dataset_info, indent=2)}
"""
    if prior_results:
        user_msg += f"""
Prior round results:
- Best reward: {prior_results.get('best_reward', 'N/A')}
- Mean reward (final epoch): {prior_results.get('mean_reward', 'N/A')}
- Unique subsets explored: {prior_results.get('n_unique', 'N/A')}
- Top genes: {prior_results.get('top_genes', 'N/A')}

Based on these results, generate an updated plan.
"""
    else:
        user_msg += "\nThis is the first round. Generate an initial exploration plan."

    response = chat(SYSTEM_PROMPT, user_msg)

    try:
        plan = json.loads(response)
    except json.JSONDecodeError:
        # If the LLM doesn't return valid JSON, wrap it
        plan = {
            "constraints": {},
            "strategy": response,
            "hyperparams": {},
            "focus_areas": [],
        }

    # Ensure all expected keys exist
    plan.setdefault("constraints", {})
    plan.setdefault("strategy", "Default exploration strategy.")
    plan.setdefault("hyperparams", {})
    plan.setdefault("focus_areas", [])

    return plan
