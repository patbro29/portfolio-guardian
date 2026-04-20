"""
tools.py — The tools the agent can use.

Each tool has two parts:
1. A schema (dict) that describes the tool to the LLM — its name, purpose, and
   expected arguments. The LLM reads these at inference time to decide which tool
   to call and with what inputs.
2. A Python implementation that runs when the agent issues a tool call.

The LLM never executes code directly; it emits structured tool calls and our
agent loop (providers.py) dispatches them to the functions defined here.
Adding a new capability means adding a schema + a function — no model changes.
"""

import json
from typing import Any

import yfinance as yf


# ============================================================================
# TOOL 1: get_prices
# ============================================================================
# Fetches current prices for a list of tickers from Yahoo Finance.
# Yahoo Finance works reliably for liquid NSE tickers; the schema is designed
# to remain stable if the underlying data source is swapped out in the future.

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
            # Errors are returned as structured data rather than raised so the agent
            # can read them and decide how to proceed (flag to user, skip, etc.).
            results[ticker] = {"error": str(e)}
    return results


# ============================================================================
# TOOL 2: analyze_portfolio
# ============================================================================
# Pure Python — no external API. Computes portfolio-level P&L and risk metrics.
# Keeping deterministic calculations in code (rather than relying on the LLM to
# do arithmetic) produces consistent, auditable results.

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

    Returns a structured dict with both raw numbers and pre-computed summaries
    (top overweight, sector concentration). Surfacing derived values directly
    makes it easier for the LLM to cite specific figures without re-computing.
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

    # Second pass: compute weights now that total_value is known.
    for p in positions:
        if "market_value" in p and total_value > 0:
            p["weight_pct"] = round((p["market_value"] / total_value) * 100, 2)

    # Sector concentration — sorted so the biggest exposure is first.
    sector_weights = [
        {"sector": s, "value": round(v, 2), "weight_pct": round((v / total_value) * 100, 2)}
        for s, v in sorted(sector_values.items(), key=lambda x: -x[1])
        if total_value > 0
    ]

    # Flag concentration thresholds: >30% in a single stock, >40% in a single sector.
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
# TOOL 3: web_search — Anthropic's native server-side search tool
# ============================================================================
# This is a server tool: Anthropic executes the search, not our code. We include
# the schema in the API request and the results come back as part of the response.
# No separate API key needed; usage is billed with Claude API calls.
# Only active on the Anthropic provider path — Gemini handles web search differently.

WEB_SEARCH_TOOL: dict[str, Any] = {
    "type": "web_search_20260209",  # 2026 version: dynamic filtering reduces token cost
    "name": "web_search",
    "max_uses": 3,  # hard cap — prevents runaway costs if the agent loops
}


# ============================================================================
# Tool registry
# ============================================================================
# Maps tool names to Python implementations. The agent loop dispatches tool
# calls by looking up the name here. web_search is intentionally excluded —
# it's a server tool handled by Anthropic, not a local function.

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
    Dispatch a tool call to its Python implementation and return a JSON string.
    Both provider implementations (Gemini and Anthropic) expect string output
    from tool calls, so we serialize the result here before returning.
    """
    if name not in TOOL_IMPLEMENTATIONS:
        return json.dumps({"error": f"Unknown tool: {name}"})

    try:
        result = TOOL_IMPLEMENTATIONS[name](**tool_input)
        return json.dumps(result, default=str)  # default=str handles dates/etc
    except TypeError as e:
        return json.dumps({"error": f"Bad arguments for {name}: {e}"})
    except Exception as e:
        return json.dumps({"error": f"Tool {name} failed: {e}"})
