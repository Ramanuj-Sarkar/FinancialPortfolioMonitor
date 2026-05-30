"""LangGraph orchestrator for the financial portfolio monitor.

Execution order:
    market_data ──► sentiment ──► risk ──► report ──► END

Agents 1 & 2 (market_data, sentiment) open MCP server subprocesses and
call real external tools. Agents 3 & 4 (risk, report) reason over the
collected data using the LLM directly — no MCP needed.

Tip: to run agents 1 & 2 in parallel, replace the sequential edges with
asyncio.gather() inside a single combined node. See README for details.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import TypedDict

# Ensure the project root is on sys.path regardless of how this file is invoked
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langchain.agents import create_agent
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools

from config import settings

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

model = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    anthropic_api_key=settings.anthropic_api_key,
    temperature=0,
)

# ---------------------------------------------------------------------------
# MCP server launch configs (stdio transport)
# ---------------------------------------------------------------------------

_VENV_PYTHON = str(Path(__file__).parent.parent / ".venv" / "bin" / "python")

PRICE_MCP = StdioServerParameters(
    command=_VENV_PYTHON,
    args=["mcp_servers/price_data_mcp.py"],
)

SENTIMENT_MCP = StdioServerParameters(
    command=_VENV_PYTHON,
    args=["mcp_servers/sentiment_mcp.py"],
)

# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------


class PortfolioState(TypedDict):
    portfolio: list[dict]    # [{"symbol": "AAPL", "quantity": 10}, ...]
    market_data: str         # JSON summary from market_data_node
    sentiment_data: str      # JSON summary from sentiment_node
    risk_assessment: str     # Risk analysis from risk_node
    report: str              # Final Markdown report from report_node


# ---------------------------------------------------------------------------
# Agent nodes
# ---------------------------------------------------------------------------


async def market_data_node(state: PortfolioState) -> dict:
    """Fetch live prices, OHLCV history, and portfolio value via Price Data MCP."""
    async with stdio_client(PRICE_MCP) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            agent = create_agent(model, tools)

            portfolio_json = json.dumps(state["portfolio"])
            result = await agent.ainvoke({
                "messages": [HumanMessage(content=f"""
You have stock market tools available. Analyse this portfolio:
{portfolio_json}

Steps:
1. Call get_stock_price for every symbol to get today's price and % change.
2. Call get_historical_data for each symbol with period="1mo".
3. Call get_portfolio_value with the full holdings list to get total value and weights.

After all tool calls, return a JSON summary containing:
- total_portfolio_value (USD)
- per_symbol: price, change_pct, 1-month_high, 1-month_low
- top_performer (symbol + change_pct)
- worst_performer (symbol + change_pct)
""")]
            })

            return {"market_data": result["messages"][-1].content}


async def sentiment_node(state: PortfolioState) -> dict:
    """Fetch news and compute aggregate sentiment for each holding via Sentiment MCP."""
    async with stdio_client(SENTIMENT_MCP) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            agent = create_agent(model, tools)

            symbols = [h["symbol"] for h in state["portfolio"]]
            result = await agent.ainvoke({
                "messages": [HumanMessage(content=f"""
You have news and sentiment analysis tools available.
Analyse market sentiment for these stock symbols: {symbols}

Steps:
1. For each symbol, call get_market_sentiment to get an aggregate sentiment score.
2. For any symbol with a strong signal (score > 0.2 or < -0.2), call get_stock_news
   to fetch top headlines (days_back=7, max_articles=5).

After all tool calls, return a JSON summary containing:
- per_symbol: average_sentiment score, signal (bullish/bearish/neutral)
- overall_portfolio_sentiment: weighted average across all symbols
- top_3_themes: key news topics or risks to watch
""")]
            })

            return {"sentiment_data": result["messages"][-1].content}


async def risk_node(state: PortfolioState) -> dict:
    """Score portfolio risk using LLM reasoning over collected market and sentiment data."""
    result = await model.ainvoke([
        HumanMessage(content=f"""
You are a portfolio risk analyst. Assess risk for this portfolio:

Holdings: {json.dumps(state["portfolio"])}

Market Data:
{state.get("market_data", "Not available")}

Sentiment Data:
{state.get("sentiment_data", "Not available")}

Return a JSON object with exactly this structure:
{{
  "overall_risk_score": <integer 1–10, 10 = highest risk>,
  "risk_breakdown": {{
    "volatility_risk": "<low|medium|high>",
    "concentration_risk": "<low|medium|high>",
    "sentiment_risk": "<low|medium|high>",
    "liquidity_risk": "<low|medium|high>"
  }},
  "top_risks": ["<risk description>", ...],
  "alerts": ["<urgent flag if any>"],
  "recommended_actions": ["<specific action>", ...]
}}
""")
    ])
    return {"risk_assessment": result.content}


async def report_node(state: PortfolioState) -> dict:
    """Synthesize all agent outputs into a final Markdown portfolio report."""
    from datetime import datetime, UTC

    result = await model.ainvoke([
        HumanMessage(content=f"""
You are a portfolio report generator. Produce a clear, concise Markdown report.

Date: {datetime.now(UTC).strftime("%B %d, %Y — %H:%M UTC")}
Holdings: {json.dumps(state["portfolio"])}

Market Analysis:
{state.get("market_data", "Not available")}

Sentiment Analysis:
{state.get("sentiment_data", "Not available")}

Risk Assessment:
{state.get("risk_assessment", "Not available")}

Format your response exactly as:

# Portfolio Monitor Report
**Generated:** <date>

## Executive Summary
<2–3 sentences capturing the single most important insight>

## Market Performance
<Key metrics: total value, best/worst performer, notable price moves>

## Sentiment Signals
<Bullish/bearish indicators per symbol, key news themes>

## Risk Assessment
**Risk Score: X/10** — <one-sentence explanation>
<Breakdown of volatility, concentration, and sentiment risk>

## Recommended Actions
<3–5 numbered, specific, actionable items>
""")
    ])
    return {"report": result.content}


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------


def build_graph():
    workflow = StateGraph(PortfolioState)

    workflow.add_node("market_data", market_data_node)
    workflow.add_node("sentiment", sentiment_node)
    workflow.add_node("risk", risk_node)
    workflow.add_node("report", report_node)

    workflow.set_entry_point("market_data")
    workflow.add_edge("market_data", "sentiment")
    workflow.add_edge("sentiment", "risk")
    workflow.add_edge("risk", "report")
    workflow.add_edge("report", END)

    return workflow.compile()


graph = build_graph()


# ---------------------------------------------------------------------------
# CLI test entry point
# ---------------------------------------------------------------------------


async def run_example():
    portfolio = [
        {"symbol": "AAPL", "quantity": 10},
        {"symbol": "MSFT", "quantity": 5},
        {"symbol": "NVDA", "quantity": 8},
    ]

    initial_state: PortfolioState = {
        "portfolio": portfolio,
        "market_data": "",
        "sentiment_data": "",
        "risk_assessment": "",
        "report": "",
    }

    print("Running portfolio analysis…\n")
    final_state = await graph.ainvoke(initial_state)
    print(final_state["report"])


if __name__ == "__main__":
    asyncio.run(run_example())