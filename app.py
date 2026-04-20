"""
app.py — Streamlit UI for Portfolio Guardian.

Provider-agnostic: reads the current LLM provider from providers.py and
shows the appropriate API key status. The rest is just the usual Streamlit
top-to-bottom script with session_state for persistence across reruns.
"""

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from agent import run_agent
from providers import PROVIDER
from tools import get_prices


# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Portfolio Guardian",
    page_icon="📊",
    layout="wide",
)


# ----------------------------------------------------------------------------
# Load mock portfolio (default) — user can override via CSV upload
# ----------------------------------------------------------------------------
@st.cache_data
def load_mock_portfolio() -> dict:
    path = Path(__file__).parent / "portfolio.json"
    return json.loads(path.read_text())


def parse_uploaded_csv(file) -> list[dict]:
    df = pd.read_csv(file)
    required = {"ticker", "quantity", "avg_buy_price"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    if "name" not in df.columns:
        df["name"] = df["ticker"]
    if "sector" not in df.columns:
        df["sector"] = "Unknown"
    return df.to_dict(orient="records")


# ----------------------------------------------------------------------------
# Session state init
# ----------------------------------------------------------------------------
if "holdings" not in st.session_state:
    st.session_state.holdings = load_mock_portfolio()["holdings"]
if "messages" not in st.session_state:
    st.session_state.messages = []


# ----------------------------------------------------------------------------
# API key resolution — checks the right env var / secret based on PROVIDER
# ----------------------------------------------------------------------------
def get_api_key_status() -> tuple[bool, str]:
    """Returns (is_set, display_name) for the current provider's API key."""
    key_name = {"gemini": "GEMINI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}[PROVIDER]
    # Check Streamlit secrets first (for deployed app), then env var
    value = st.secrets.get(key_name, os.environ.get(key_name))
    return (bool(value), key_name)


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.title("📊 Portfolio Guardian")
    st.caption(f"AI portfolio analyst • Running on **{PROVIDER.title()}**")

    uploaded = st.file_uploader("Upload your holdings CSV", type=["csv"])
    if uploaded is not None:
        try:
            st.session_state.holdings = parse_uploaded_csv(uploaded)
            st.success(f"Loaded {len(st.session_state.holdings)} holdings from CSV")
        except Exception as e:
            st.error(f"Upload failed: {e}")

    if st.button("Reset to mock portfolio"):
        st.session_state.holdings = load_mock_portfolio()["holdings"]
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption(
        "**CSV format:** `ticker,name,quantity,avg_buy_price,sector`\n\n"
        "Use NSE tickers with `.NS` suffix (e.g. `HDFCBANK.NS`)."
    )

    key_set, key_name = get_api_key_status()
    if not key_set:
        st.error(f"No `{key_name}` in secrets. Agent disabled.")
        st.caption(
            f"Get a free Gemini key: https://aistudio.google.com/apikey"
            if PROVIDER == "gemini"
            else "Get an Anthropic key: https://console.anthropic.com"
        )
    else:
        st.success(f"{key_name} loaded ✓")


# ----------------------------------------------------------------------------
# Main layout
# ----------------------------------------------------------------------------
col_portfolio, col_chat = st.columns([1, 1])


# ===== LEFT: Holdings + live P&L =====
with col_portfolio:

    @st.cache_data(ttl=60)
    def fetch_prices_cached(tickers_tuple: tuple[str, ...]) -> dict:
        return get_prices(list(tickers_tuple))

    tickers = tuple(h["ticker"] for h in st.session_state.holdings) + ("^NSEI",)
    with st.spinner("Fetching live prices..."):
        prices = fetch_prices_cached(tickers)

    rows = []
    total_cost = 0.0
    total_value = 0.0
    sector_values: dict[str, float] = {}

    for h in st.session_state.holdings:
        p = prices.get(h["ticker"], {})
        price = p.get("price")
        sector = h.get("sector", "Unknown")
        if price is None:
            rows.append({
                "Ticker": h["ticker"],
                "Qty": h["quantity"],
                "Avg Buy": f"₹{h['avg_buy_price']:,.0f}",
                "Current": "—",
                "P&L %": float("nan"),  # NaN so Styler can detect and show "—"
                "Value": "—",
            })
            continue
        cost = h["quantity"] * h["avg_buy_price"]
        value = h["quantity"] * price
        pnl_pct = ((value - cost) / cost) * 100
        total_cost += cost
        total_value += value
        sector_values[sector] = sector_values.get(sector, 0.0) + value
        rows.append({
            "Ticker": h["ticker"],
            "Qty": h["quantity"],
            "Avg Buy": f"₹{h['avg_buy_price']:,.0f}",
            "Current": f"₹{price:,.0f}",
            "P&L %": round(pnl_pct, 2),  # float — needed for Styler coloring
            "Value": f"₹{value:,.0f}",
        })

    # --- Summary metrics sit ABOVE the table ---
    st.subheader("Holdings")
    if total_value > 0:
        total_pnl = total_value - total_cost
        total_pnl_pct = (total_pnl / total_cost) * 100
        c1, c2, c3 = st.columns(3)
        c1.metric("Portfolio Value", f"₹{total_value:,.0f}")
        c2.metric("Total P&L", f"₹{total_pnl:,.0f}", f"{total_pnl_pct:+.2f}%")
        nifty_price = prices.get("^NSEI", {}).get("price")
        if nifty_price:
            c3.metric("NIFTY 50", f"{nifty_price:,.0f}")

    # --- Holdings table with color-coded P&L % ---
    df = pd.DataFrame(rows)

    def _pnl_color(val):
        """CSS color string for positive/negative P&L. Called per-cell by Styler."""
        try:
            v = float(val)
            if v > 0:
                return "color: #00cc88; font-weight: 600"
            elif v < 0:
                return "color: #ff4b4b; font-weight: 600"
        except (TypeError, ValueError):
            pass
        return ""

    styled = (
        df.style
        .map(_pnl_color, subset=["P&L %"])
        .format({"P&L %": lambda v: f"{v:+.2f}%" if pd.notna(v) else "—"})
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # --- Sector allocation bar chart ---
    if sector_values and total_value > 0:
        st.caption("Sector allocation")
        sector_df = pd.DataFrame([
            {"Sector": s, "Weight (%)": round(v / total_value * 100, 1)}
            for s, v in sorted(sector_values.items(), key=lambda x: -x[1])
        ]).set_index("Sector")
        st.bar_chart(sector_df, height=160)


# ===== RIGHT: Chat with agent =====
with col_chat:
    st.subheader("💬 Ask the Guardian")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("e.g. 'Review my portfolio and flag risks'")
    if prompt:
        key_set, key_name = get_api_key_status()
        if not key_set:
            st.error(f"Set `{key_name}` in secrets to use the agent.")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            event_area = st.container()
            final_text_parts: list[str] = []

            for event in run_agent(prompt, st.session_state.holdings):
                if event["type"] == "tool_call":
                    event_area.caption(f"🔧 Calling `{event['name']}`...")
                elif event["type"] == "tool_result":
                    pass  # quiet by default; uncomment below for debugging
                    # with event_area.expander(f"Result from {event['name']}"):
                    #     st.code(event["result"], language="json")
                elif event["type"] == "text":
                    final_text_parts.append(event["content"])
                    event_area.markdown("".join(final_text_parts))

            full_response = "".join(final_text_parts)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
