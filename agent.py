"""
agent.py — The agent loop (provider-agnostic).

WHAT CHANGED FROM V1:
Previously this file knew about Anthropic's specific API shape — it called
client.messages.create directly and handled Anthropic's content-block format.

Now all provider-specific code lives in providers.py. This file just:
  1. Defines the system prompt (policy — what the agent SHOULD do)
  2. Calls run_llm_agent (mechanism — how the loop actually runs)
  3. Passes events up to the UI

THE AGENT LOOP IS STILL THE SAME IDEA:
    user msg → LLM decides → calls tool OR replies → loop until done
The loop logic itself has moved into providers.py because it differs slightly
between Gemini and Claude (different SDKs, different message formats).
What STAYS here is the portfolio-domain-specific stuff: prompt, tools, limits.

WHY SPLIT IT THIS WAY?
Because the prompt and tools are "your product" — they define what Portfolio
Guardian does. The API plumbing is "your infrastructure" — replaceable.
Keeping them in different files means when you iterate on the prompt (which
you will, many times), you're not touching provider code.
"""

from typing import Any, Generator

from providers import run_llm_agent, TextEvent, ToolCallEvent, ToolResultEvent, StopEvent
from tools import TOOL_SCHEMAS, execute_tool


MAX_AGENT_TURNS = 10  # safety cap to prevent runaway tool-use loops


SYSTEM_PROMPT = """You are Portfolio Guardian, an AI portfolio analyst for an Indian retail investor.

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
