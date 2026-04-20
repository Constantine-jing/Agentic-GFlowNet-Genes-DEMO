"""
critic.py — LLM Critic agent.

The Critic sits at the bottom of the agentic loop. After each GFlowNet
training round, it:
  1. Receives the training results (reward curve, top subsets, stats)
  2. Analyzes whether the round was successful
  3. Suggests adjustments for the next round
  4. Decides whether to continue training or stop

With the mock backend, it returns plausible feedback.
With a real LLM, the Critic can reason about biological plausibility
of discovered gene subsets, check against known pathways, and provide
substantive scientific feedback.
"""
from __future__ import annotations

import json

from src.llm_client import chat

SYSTEM_PROMPT = """You are a bioinformatics quality control critic reviewing results
from a GFlowNet gene subset discovery system. Your role is to:

1. Assess the quality of discovered gene subsets
2. Evaluate whether training has converged
3. Suggest specific improvements for the next round
4. Decide whether to continue or stop

You will receive training metrics and top discovered subsets.

Respond ONLY with a JSON object containing:
- "assessment": overall text assessment
- "observations": list of specific observations
- "suggestions": list of actionable suggestions
- "continue_training": boolean
- "suggested_adjustments": dict of parameter changes

Respond ONLY with valid JSON, no other text."""


def analyze_results(
    training_history: dict,
    top_subsets: list[dict],
    dataset_info: dict,
    round_num: int = 0,
) -> dict:
    """
    Ask the Critic LLM to analyze training results and provide feedback.

    Args:
        training_history: {epoch, mean_reward, max_reward, loss, ...} lists
        top_subsets: list of {rank, reward, genes} dicts
        dataset_info: {n_genes, n_samples, groups, dataset_name}
        round_num: which round of the outer loop

    Returns:
        Parsed dict with assessment, observations, suggestions, continue flag
    """
    # Summarize history (don't send all 300 epochs to the LLM)
    n = len(training_history.get("epoch", []))
    if n > 0:
        summary = {
            "total_epochs": n,
            "initial_mean_reward": training_history["mean_reward"][0],
            "final_mean_reward": training_history["mean_reward"][-1],
            "best_ever_reward": max(training_history.get("best_ever",
                                    training_history["max_reward"])),
            "initial_loss": training_history["loss"][0],
            "final_loss": training_history["loss"][-1],
            "unique_subsets_explored": training_history["n_unique"][-1],
        }
    else:
        summary = {"error": "no training data"}

    # Top-5 subsets for the LLM to inspect
    top5 = []
    for s in top_subsets[:5]:
        top5.append({
            "rank": s["rank"],
            "reward": s["reward"],
            "genes": s["genes"][:80] + "..." if len(s["genes"]) > 80 else s["genes"],
        })

    user_msg = f"""Round {round_num} results analysis.

Dataset: {json.dumps(dataset_info)}

Training summary:
{json.dumps(summary, indent=2)}

Top 5 discovered subsets:
{json.dumps(top5, indent=2)}

Provide your analysis and recommendations.
"""

    response = chat(SYSTEM_PROMPT, user_msg)

    try:
        feedback = json.loads(response)
    except json.JSONDecodeError:
        feedback = {
            "assessment": response,
            "observations": [],
            "suggestions": [],
            "continue_training": False,
            "suggested_adjustments": {},
        }

    feedback.setdefault("assessment", "No assessment provided.")
    feedback.setdefault("observations", [])
    feedback.setdefault("suggestions", [])
    feedback.setdefault("continue_training", False)
    feedback.setdefault("suggested_adjustments", {})

    return feedback
