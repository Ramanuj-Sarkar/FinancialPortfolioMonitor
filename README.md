# Financial Portfolio Monitor

Multi-agent system that analyses a stock portfolio in real time.
Built with **LangGraph + MCP + FastAPI**.

## Architecture

```
User
 │
 ▼ POST /portfolio/analyze
FastAPI  ──background task──►  LangGraph Orchestrator
                                      │
                                      ▼
                              Market Data Agent
                                      │
                              Price Data MCP (subprocess, stdio)
                                      │
                                  yfinance
                                      │
                                      ▼
                              Sentiment Agent
                                      │
                              Sentiment MCP (subprocess, stdio)
                                      │
                               NewsAPI + TextBlob
                                      │
                                      ▼
                               Risk Agent (LLM only)
                                      │
                                      ▼
                              Report Agent (LLM only)
                                      │
                                      ▼
                               Portfolio Report
```

**Execution order:** market data → sentiment → risk → report (sequential)

Agents 1 & 2 spawn MCP servers as subprocesses (stdio transport) to call live data APIs.
Agents 3 & 4 reason directly over the collected data using `claude-sonnet-4-20250514` — no MCP needed.

## Prerequisites

- Python 3.11+
- [Anthropic API key](https://console.anthropic.com)
- [NewsAPI key](https://newsapi.org/register) — free tier: 100 requests/day

## Setup

```bash
# 1. Clone and enter project
git clone <your-repo>
cd FinancialPortfolioMonitor

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
# FinBERT (~440 MB) downloads automatically from HuggingFace on first run

# 4. Configure API keys
cp .env.example .env
# Edit .env and fill in ANTHROPIC_API_KEY and NEWS_API_KEY

# 5. Start the API server
python run.py
```

> **Why `python run.py` instead of `uvicorn` directly?**
> `run.py` sets `PYTHONPATH` to the project root before starting uvicorn, which is required so
> uvicorn's reloader worker processes can resolve imports like `from agents.orchestrator import …`.

## Usage

### CLI (recommended)

```bash
# Default demo — AAPL / MSFT / NVDA
python cli.py

# Custom holdings (SYMBOL:QUANTITY)
python cli.py AAPL:10 TSLA:5 GOOG:3

# Against a remote server
python cli.py --url http://myserver:8000 AAPL:10
```

The CLI submits the job, polls for completion, and renders the Markdown report in the terminal.

### Raw HTTP

```bash
# Submit a portfolio for analysis (returns immediately with a job_id)
curl -X POST http://localhost:8000/portfolio/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "holdings": [
      {"symbol": "AAPL", "quantity": 10},
      {"symbol": "MSFT", "quantity": 5},
      {"symbol": "NVDA", "quantity": 8}
    ]
  }'
# → {"job_id": "abc-123", "status": "running", "created_at": "..."}

# Poll for completion
curl http://localhost:8000/portfolio/status/abc-123

# Retrieve the Markdown report once complete
curl http://localhost:8000/portfolio/report/abc-123

# Health check
curl http://localhost:8000/health
```

## Run the orchestrator directly (no API)

```bash
python agents/orchestrator.py
```

Runs a hardcoded AAPL / MSFT / NVDA example and prints the report to stdout.

## Project structure

```
FinancialPortfolioMonitor/
├── agents/
│   └── orchestrator.py      LangGraph StateGraph with all 4 agent nodes
├── api/
│   └── main.py              FastAPI app with async background-task job queue
├── mcp_servers/
│   ├── price_data_mcp.py    yfinance MCP — get_stock_price, get_historical_data,
│   │                        get_portfolio_value
│   └── sentiment_mcp.py     NewsAPI + FinBERT MCP — get_stock_news,
│                            analyze_sentiment, get_market_sentiment
├── .env.example
├── cli.py                   Rich terminal CLI — submit, poll, and render reports
├── config.py                Pydantic settings (reads .env, ignores empty env vars)
├── run.py                   Server entry point (sets PYTHONPATH, then starts uvicorn)
└── requirements.txt
```