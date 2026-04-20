"""
llm_client.py — Unified LLM interface.

Currently uses a MOCK backend (hardcoded responses).
To swap in a real API later, change BACKEND below and implement the
corresponding function. The rest of the codebase calls `chat()` and
doesn't care what's behind it.

Supported backends:
  "mock"       — deterministic, no API key needed (default)
  "anthropic"  — Claude API (needs ANTHROPIC_API_KEY env var)
  "openai"     — OpenAI API (needs OPENAI_API_KEY env var)
  "ollama"     — local Ollama server (needs ollama running)
"""
from __future__ import annotations

import json
import os

# --- change this to swap backends ---
BACKEND = "anthropic"
# -------------------------------------


def chat(system: str, user: str) -> str:
    """
    Send a message to the LLM and return the response text.

    Args:
        system: system prompt (role/instructions)
        user:   user message (the actual query)

    Returns:
        The LLM's response as a string.
    """
    if BACKEND == "mock":
        return _mock_chat(system, user)
    elif BACKEND == "anthropic":
        return _anthropic_chat(system, user)
    elif BACKEND == "openai":
        return _openai_chat(system, user)
    elif BACKEND == "ollama":
        return _ollama_chat(system, user)
    else:
        raise ValueError(f"Unknown BACKEND: {BACKEND!r}")


# ---- Mock backend ----

def _mock_chat(system: str, user: str) -> str:
    """
    Return plausible hardcoded responses based on keywords in the prompt.
    Good enough to test the full agentic loop without any API.
    """
    lower = user.lower()

    # Manager-style: generate constraints/strategy
    if "constraint" in lower or "strategy" in lower or "plan" in lower:
        return json.dumps({
            "constraints": {
                "min_subset_size": 10,
                "max_subset_size": 20,
                "prefer_genes_with_high_variance": True,
                "avoid_correlated_genes": True,
            },
            "strategy": "Focus on discovering diverse subsets that capture "
                        "different biological pathways. Prioritize genes with "
                        "high variance across samples and penalize redundant "
                        "gene selections.",
            "reward_adjustments": {
                "diversity_bonus": 0.1,
                "correlation_penalty": 0.05,
            },
        }, indent=2)

    # Critic-style: analyze results and suggest improvements
    if "analyze" in lower or "critic" in lower or "feedback" in lower or "results" in lower:
        return json.dumps({
            "assessment": "The GFlowNet is successfully discovering high-reward "
                         "subsets with good recovery of differentially expressed "
                         "genes. Diversity across top subsets is adequate.",
            "observations": [
                "Top subsets share a core of ~10 genes, varying in 3-5 slots",
                "Reward distribution is right-skewed, indicating the policy "
                "has learned to avoid low-quality regions",
                "Some non-signal genes appear frequently — possible false positives",
            ],
            "suggestions": [
                "Increase temperature slightly to improve diversity",
                "Add a pairwise distance penalty to encourage more distinct subsets",
                "Consider running longer to explore the tail of the distribution",
            ],
            "continue_training": True,
            "suggested_epochs": 200,
            "suggested_temp_adjustment": 0.1,
        }, indent=2)

    # Fallback
    return json.dumps({
        "response": "Acknowledged. Ready for the next step.",
    })


# ---- Real API backends (implement when needed) ----

def _anthropic_chat(system: str, user: str) -> str:
    """Call the Anthropic Claude API."""
    try:
        import anthropic
    except ImportError:
        raise ImportError("pip install anthropic")

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    message = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return message.content[0].text


def _openai_chat(system: str, user: str) -> str:
    """Call the OpenAI API."""
    try:
        import openai
    except ImportError:
        raise ImportError("pip install openai")

    client = openai.OpenAI()  # reads OPENAI_API_KEY from env
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return response.choices[0].message.content


def _ollama_chat(system: str, user: str) -> str:
    """Call a local Ollama server."""
    import urllib.request

    body = json.dumps({
        "model": "phi3:mini",
        "system": system,
        "prompt": user,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())["response"]
