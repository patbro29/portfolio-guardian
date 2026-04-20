# 📊 Portfolio Intelligence Dashboard

An AI portfolio analyst for Indian retail investors. Upload your NSE holdings, get live P&L vs NIFTY 50, and chat with an agent that fetches prices, scans news, and flags risks.

**Live demo:** _[add your Streamlit Community Cloud URL here after deploy]_

## Architecture

Single-model, multi-tool agent with a **pluggable provider layer**:

```
User message
    │
    ▼
[providers.py]  ◄── PROVIDER = "gemini" (free) | "anthropic" (paid)
    │
    ▼
LLM ──┬── tool: get_prices (yfinance → NSE)
      ├── tool: analyze_portfolio (pure Python risk metrics)
      └── tool: web_search (Anthropic path only — native server tool)
    │
    ▼
Streaming text response
```

The agent runs a standard tool-use loop: LLM decides what to call, we execute, feed results back, repeat until the LLM produces a final answer.

**Default provider:** Google Gemini 2.5 Flash (free tier). Swap to Claude Sonnet 4.6 by changing one line in `providers.py` or setting `LLM_PROVIDER=anthropic`.

## Stack

- **Python 3.10+** + **Streamlit** for the UI
- **`google-genai`** SDK with Gemini 2.5 Flash (free) — default
- **`anthropic`** SDK with Claude Sonnet 4.6 — optional (paid, higher-quality)
- **yfinance** for NSE market data
- Deployed on **Streamlit Community Cloud** (free tier)

## Local setup

```bash
git clone https://github.com/YOUR_USERNAME/portfolio-intelligence-dashboard.git
cd portfolio-intelligence-dashboard
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Configure your API key — free Gemini key from https://aistudio.google.com/apikey
mkdir -p .streamlit
cp secrets.toml.example .streamlit/secrets.toml
# Edit .streamlit/secrets.toml and paste your Gemini API key

streamlit run app.py
```

## Swapping providers

To run with Claude instead of Gemini:

```bash
export LLM_PROVIDER=anthropic
# add ANTHROPIC_API_KEY to .streamlit/secrets.toml
pip install anthropic
streamlit run app.py
```

Or edit the `PROVIDER` constant at the top of `providers.py`.

## Public demo data

The repo ships with a mock `portfolio.json` — a realistic Indian equity portfolio (HDFCBANK, RELIANCE, TCS, INFY, NIFTYBEES) worth ~₹8L at current prices. This is what the public demo shows. Real holdings stay local and are never committed — load them via CSV upload in the sidebar.

## CSV format

```csv
ticker,name,quantity,avg_buy_price,sector
HDFCBANK.NS,HDFC Bank,250,720,Banking
RELIANCE.NS,Reliance Industries,150,1200,Energy
```

Use `.NS` suffix for NSE tickers. `^NSEI` is the NIFTY 50 index.

## Roadmap

- **v1.0 (current):** Gemini-powered, mock data, yfinance, 3 tools, public Streamlit demo
- **v1.5:** Kite Connect integration for real holdings (local only — daily-use tool)
- **v2.0:** FastAPI microservice for quant math (Monte Carlo VaR, factor regressions) called as a new agent tool

## Disclaimer

Not SEBI-registered investment advice. This is a personal project and portfolio analysis tool. Any recommendations the agent produces are for informational purposes only.
