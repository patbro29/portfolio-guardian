"""
agent.py — The agent loop (provider-agnostic).

This file owns the domain-specific layer of the agent: the system prompt,
which tools are available, and how many turns the loop is allowed to run.
All LLM provider details (API calls, message formats, event streaming) live
in providers.py. The split keeps prompt iteration separate from API plumbing.

Flow: user message → run_llm_agent (providers.py) → stream of typed events
→ translated here into simple dicts → consumed by app.py for the UI.
"""

from typing import Any, Generator

from providers import run_llm_agent, TextEvent, ToolCallEvent, ToolResultEvent, StopEvent
from tools import TOOL_SCHEMAS, execute_tool


MAX_AGENT_TURNS = 10  # safety cap to prevent runaway tool-use loops


SYSTEM_PROMPT = """You are Portfolio Intelligence Dashboard, an AI portfolio analyst for an Indian retail investor.

Tools:
- get_prices: fetch current market prices. Use .NS suffix for NSE stocks, ^NSEI for NIFTY 50.
- analyze_portfolio: compute P&L, weights, sector exposure, and risk flags. ALWAYS call this after fetching prices.

Workflow (follow this exactly for portfolio reviews):
1. Call get_prices for ALL holdings + ^NSEI in ONE call.
2. Call analyze_portfolio with the holdings and prices.
3. Write the response in the format below.

Output format — use this structure every time, no exceptions:
**[Total P&L]** ₹X gain/loss (+X%) vs cost basis. NIFTY 50 at X.

**Risks**
1. [Risk name]: [one sentence with the actual weight/number from the data]
2. [Risk name]: [one sentence with the actual weight/number]
(2-3 risks max)

**Recommendations**
- [Ticker]: [one-line action with specific reasoning]
- [Ticker]: [one-line action with specific reasoning]
(2-3 recs max)

Rules:
- Every flag and rec MUST cite a number from the tool output. No vague statements.
- If a ticker failed to fetch, mention it in one parenthetical, then move on.
- No preamble ("Here is your review..."), no closing disclaimer.
- Total response under 180 words. If the user asks a follow-up, you can go longer."""


def run_agent(
    user_message: str,
    holdings: list[dict[str, Any]],
) -> Generator[dict[str, Any], None, None]:
    """
    Run the agent and yield UI-friendly event dicts.

    We translate the provider-agnostic events (TextEvent, ToolCallEvent, etc.)
    into the simple dict format app.py expects. This extra hop keeps app.py
    ignorant of the dataclasses in providers.py — loose coupling.
    """
    for event in run_llm_agent(
        user_message=user_message,
        holdings=holdings,
        system_prompt=SYSTEM_PROMPT,
        tool_schemas=TOOL_SCHEMAS,
        tool_executor=execute_tool,
        max_turns=MAX_AGENT_TURNS,
    ):
        if isinstance(event, TextEvent):
            yield {"type": "text", "content": event.content}
        elif isinstance(event, ToolCallEvent):
            yield {"type": "tool_call", "name": event.name, "input": event.input}
        elif isinstance(event, ToolResultEvent):
            yield {"type": "tool_result", "name": event.name, "result": event.result}
        elif isinstance(event, StopEvent):
            if event.reason == "max_tokens":
                yield {
                    "type": "text",
                    "content": f"\n\n_[Agent stopped after {MAX_AGENT_TURNS} turns — safety cap.]_",
                }
            # end_turn is the normal case; no event needed
