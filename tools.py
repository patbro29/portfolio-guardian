"""
tools.py — The tools our agent can use.

DESIGN PHILOSOPHY:
An "agent" in this codebase = an LLM that can decide to call functions.
Those functions are defined HERE. The LLM doesn't execute them; OUR code does.
The LLM just says "please call get_prices with tickers=[...]" and we run it.

WHY THIS SEPARATION MATTERS:
- Tools are the ONLY way the agent touches the outside world.
- If you want the agent to do something new (e.g. compute Sharpe ratio),
  you add a tool here. You do NOT retrain the model or tweak prompts endlessly.
- This is the clean version of "agency" — the LLM reasons, tools act.

Each tool has two parts:
1. A TOOL SCHEMA (dict): tells Claude what the tool does, what args it takes.
   Claude reads this and decides when to call the tool.
2. An IMPLEMENTATION (function): the actual Python that runs when Claude calls it.
"""

import json
from typing import Any

import yfinance as yf


# ============================================================================
# TOOL 1: get_prices
# ============================================================================
# Fetches current prices for a list of tickers from Yahoo Finance.
# Yahoo is flaky for NSE tickers but works ~90% of the time for liquid ones.
# In V1.5 we swap this implementation for Kite Connect; the SCHEMA stays the same.
# That's the whole point of tool abstraction — the agent doesn't care who fetches.

GET_PRICES_SCHEMA: dict[str, Any] = {
    "name": "get_prices",
    "description": (
        "Fetches the current market price for one or more stock tickers. "
        "Use this when you need to know current prices to calculate P&L, "
        "compute weights, or assess performance. "
        "For Indian NSE stocks, tickers end in .NS (e.g. HDFCBANK.NS). "
        "For the NIFTY 50 index itself, use ^NSEI."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "tickers": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of Yahoo Finance ticker symbols to fetch prices for.",
            }
        },
        "required": ["tickers"],
    },
}


def get_prices(tickers: list[str]) -> dict[str, Any]:
    """
    Fetch current prices via yfinance.
    Returns a dict mapping ticker -> {price, currency, as_of} or {error}.

    yfinance's .history() returns a DataFrame; we take the last Close.
    Using 1-day period + 1-minute interval would be more "real-time" but
    rate-limits faster. Daily close is sufficient for a portfolio dashboard.
    """
    results: dict[str, Any] = {}
    for ticker in tickers:
        try:
            tkr = yf.Ticker(ticker)
            # period="5d" protects against market holidays when today has no data
            hist = tkr.history(period="5d")
            if hist.empty:
                results[ticker] = {"error": f"No data returned for {ticker}"}
                continue
            latest_close = float(hist["Close"].iloc[-1])
            results[ticker] = {
                "price": round(latest_close, 2),
                "currency": getattr(tkr.fast_info, "currency", "INR"),
                "as_of": str(hist.index[-1].date()),
            }
        except Exception as e:
            # We never raise — we return the error IN the tool result.
            # Why? Because the AGENT sees the error text and can decide what to do
            # (retry with different ticker, tell the user, etc.). If we raised,
            # the agent loop would crash and the user would see a 500.
            results[ticker] = {"error": str(e)}
    return results


# ============================================================================
# TOOL 2: analyze_portfolio
# ============================================================================
# Pure Python — no external API. Computes portfolio-level risk metrics.
# WHY THIS IS A SEPARATE TOOL:
# - We COULD put this math in the system prompt and let Claude do arithmetic.
# - We DON'T because LLMs are bad at multi-step arithmetic and this is deterministic.
# - Rule of thumb: if it's math, make it a tool. If it's judgment, let the LLM do it.

ANALYZE_PORTFOLIO_SCHEMA: dict[str, Any] = {
    "name": "analyze_portfolio",
    "description": (
        "Computes portfolio-level metrics: total market value, P&L, "
        "individual position weights, sector concentration, and top overweights. "
        "Call this AFTER get_prices so you have current prices to work with. "
        "Pass the holdings and a dict of current prices."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "holdings": {
                "type": "array",
                "description": "List of holdings with ticker, quantity, avg_buy_price, sector.",
                "items": {
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string"},
                        "name": {"type": "string"},
                        "quantity": {"type": "number"},
                        "avg_buy_price": {"type": "number"},
                        "sector": {"type": "string"},
                    },
                    "required": ["ticker", "quantity", "avg_buy_price", "sector"],
                },
            },
            "current_prices": {
                "type": "object",
                "description": "Dict mapping ticker symbol (string) to current price (float). E.g. {\"HDFCBANK.NS\": 1650.5, \"TCS.NS\": 3400.0}",
                # NOTE: "additionalProperties" is valid JSON Schema but Gemini's API
                # rejects it (it uses a strict subset). Removed — the description above
                # conveys the same intent to the model.
            },
        },
        "required": ["holdings", "current_prices"],
    },
}


def analyze_portfolio(
    holdings: list[dict[str, Any]],
    current_prices: dict[str, float],
) -> dict[str, Any]:
    """
    Compute position-level and portfolio-level metrics.

    Returns a structured dict the LLM can read and reason over. We deliberately
    include BOTH raw numbers AND pre-computed summaries (top overweight, max
    sector concentration). The LLM will typically quote these directly rather
    than re-deriving — and when it re-derives, it gets it wrong. Make the
    right answer easy to reach.
    """
    positions = []
    total_cost = 0.0
    total_value = 0.0
    sector_values: dict[str, float] = {}

    for h in holdings:
        ticker = h["ticker"]
        qty = h["quantity"]
        avg = h["avg_buy_price"]
        sector = h.get("sector", "Unknown")

        # If price is missing (Yahoo failed), we skip it with a note.
        # The agent will see the note and can flag it to the user.
        price = current_prices.get(ticker)
        if price is None:
            positions.append({
                "ticker": ticker,
                "name": h.get("name", ticker),
                "error": "No current price available",
            })
            continue

        cost = qty * avg
        value = qty * price
        pnl_abs = value - cost
        pnl_pct = (pnl_abs / cost) * 100 if cost > 0 else 0.0

        positions.append({
            "ticker": ticker,
            "name": h.get("name", ticker),
            "sector": sector,
            "quantity": qty,
            "avg_buy_price": round(avg, 2),
            "current_price": round(price, 2),
            "cost_basis": round(cost, 2),
            "market_value": round(value, 2),
            "pnl_absolute": round(pnl_abs, 2),
            "pnl_pct": round(pnl_pct, 2),
        })

        total_cost += cost
        total_value += value
        sector_values[sector] = sector_values.get(sector, 0.0) + value

    # Second pass: now that we know total_value, compute each position's weight.
    # Why two passes? Because we need the denominator (total_value) before we
    # can compute weights. Classic dependency — data needs to settle first.
    for p in positions:
        if "market_value" in p and total_value > 0:
            p["weight_pct"] = round((p["market_value"] / total_value) * 100, 2)

    # Sector concentration — sorted so the biggest exposure is first.
    sector_weights = [
        {"sector": s, "value": round(v, 2), "weight_pct": round((v / total_value) * 100, 2)}
        for s, v in sorted(sector_values.items(), key=lambda x: -x[1])
        if total_value > 0
    ]

    # Summary flags — these are judgment calls baked into the tool.
    # We're saying: if any single stock is >30% of portfolio, flag it.
    # If any single sector is >40%, flag it. These are reasonable defaults
    # for a retail Indian equity portfolio. Real PMs use tighter limits.
    valid_positions = [p for p in positions if "weight_pct" in p]
    top_position = max(valid_positions, key=lambda p: p["weight_pct"]) if valid_positions else None
    top_sector = sector_weights[0] if sector_weights else None

    flags = []
    if top_position and top_position["weight_pct"] > 30:
        flags.append(f"Concentration risk: {top_position['ticker']} is {top_position['weight_pct']}% of portfolio (>30% threshold).")
    if top_sector and top_sector["weight_pct"] > 40:
        flags.append(f"Sector concentration: {top_sector['sector']} is {top_sector['weight_pct']}% of portfolio (>40% threshold).")

    total_pnl_abs = total_value - total_cost
    total_pnl_pct = (total_pnl_abs / total_cost) * 100 if total_cost > 0 else 0.0

    return {
        "total_cost_basis": round(total_cost, 2),
        "total_market_value": round(total_value, 2),
        "total_pnl_absolute": round(total_pnl_abs, 2),
        "total_pnl_pct": round(total_pnl_pct, 2),
        "positions": positions,
        "sector_exposure": sector_weights,
        "top_position": top_position,
        "top_sector": top_sector,
        "risk_flags": flags,
    }


# ============================================================================
# TOOL 3: search_news — uses Anthropic's NATIVE web_search tool
# ============================================================================
# SPECIAL CASE: web_search is a "server tool" — Anthropic runs it, not us.
# We don't implement the Python function for this one. We just include the
# tool SCHEMA in our API call (with a special type) and Anthropic handles it.
#
# Advantage: no extra API key, no signup, billed with your Claude usage.
# Disadvantage: costs per search (~$10 per 1K searches as of 2026).
#
# We expose this as its own named tool to Claude so the system prompt can
# instruct "use search_news for news, use get_prices for prices" — clear lanes.
# But under the hood, Claude sees it as the standard web_search primitive.

WEB_SEARCH_TOOL: dict[str, Any] = {
    "type": "web_search_20260209",  # 2026 version: dynamic filtering reduces token cost
    "name": "web_search",
    "max_uses": 3,  # hard cap — prevents runaway costs if the agent loops
}


# ============================================================================
# Tool registry — maps tool names to their Python implementations.
# ============================================================================
# The agent loop (in agent.py) looks up the name Claude sent and calls the
# matching function. Note web_search isn't here — Anthropic handles it.

TOOL_IMPLEMENTATIONS = {
    "get_prices": get_prices,
    "analyze_portfolio": analyze_portfolio,
}

# Schemas we pass to Claude so it knows what tools exist and how to call them.
TOOL_SCHEMAS = [
    GET_PRICES_SCHEMA,
    ANALYZE_PORTFOLIO_SCHEMA,
    WEB_SEARCH_TOOL,
]


def execute_tool(name: str, tool_input: dict[str, Any]) -> str:
    """
    Dispatch a tool call from Claude to our Python implementation.
    Returns a JSON string because Anthropic's tool_result content blocks
    expect string content (or a list of content blocks — we keep it simple).
    """
    if name not in TOOL_IMPLEMENTATIONS:
        return json.dumps({"error": f"Unknown tool: {name}"})

    try:
        result = TOOL_IMPLEMENTATIONS[name](**tool_input)
        return json.dumps(result, default=str)  # default=str handles dates/etc
    except TypeError as e:
        # Claude passed wrong args. Return the error so it can try again.
        return json.dumps({"error": f"Bad arguments for {name}: {e}"})
    except Exception as e:
        return json.dumps({"error": f"Tool {name} failed: {e}"})
