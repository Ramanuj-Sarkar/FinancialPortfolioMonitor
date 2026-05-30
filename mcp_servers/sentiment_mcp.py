#!/usr/bin/env python3
"""MCP server — news & sentiment via NewsAPI + TextBlob.

Tools exposed:
  get_stock_news(query, days_back, max_articles)  → recent news articles
  analyze_sentiment(text)                         → polarity + label
  get_market_sentiment(symbol, company_name)      → aggregate sentiment score
"""

import json
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta, UTC

import requests
from mcp.server.fastmcp import FastMCP
from transformers import pipeline

load_dotenv()

mcp = FastMCP("sentiment-mcp")
NEWS_API_KEY = os.environ["NEWS_API_KEY"]
NEWS_API_BASE = "https://newsapi.org/v2/everything"

# ---------------------------------------------------------------------------
# Load FinBERT once at startup — reused for every tool call
# ---------------------------------------------------------------------------

_finbert = pipeline(
    "text-classification",
    model="ProsusAI/finbert",
    top_k=None,  # return scores for all three classes
    truncation=True,
    max_length=512,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scores(text: str) -> dict:
    """Run FinBERT and return {positive, negative, neutral} confidence scores."""
    raw = _finbert(text[:2000])  # pre-truncate chars before tokenisation
    return {r["label"]: round(r["score"], 4) for r in raw[0]}


def _polarity(scores: dict) -> float:
    """Derive a signed polarity from FinBERT scores: positive − negative ∈ [-1, 1]."""
    return round(scores.get("positive", 0.0) - scores.get("negative", 0.0), 4)


def _label(polarity: float) -> str:
    if polarity > 0.1:
        return "positive"
    if polarity < -0.1:
        return "negative"
    return "neutral"


# ---------------------------------------------------------------------------
# MCP tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def get_stock_news(query: str, days_back: int = 7, max_articles: int = 10) -> str:
    """Fetch recent news articles for a company or stock symbol.

    Args:
        query: Search term (company name or ticker)
        days_back: How many days of history to search (default 7)
        max_articles: Max articles to return (default 10)

    Returns:
        JSON with list of articles (title, description, source, url, published_at)
    """
    from_date = (datetime.now(UTC) - timedelta(days=days_back)).strftime("%Y-%m-%d")
    params = {
        "q": query,
        "from": from_date,
        "sortBy": "relevancy",
        "language": "en",
        "apiKey": NEWS_API_KEY,
        "pageSize": max_articles,
    }

    resp = requests.get(NEWS_API_BASE, params=params, timeout=10)
    resp.raise_for_status()

    articles = [
        {
            "title": a.get("title", ""),
            "description": a.get("description", ""),
            "source": a.get("source", {}).get("name", ""),
            "published_at": a.get("publishedAt", ""),
            "url": a.get("url", ""),
        }
        for a in resp.json().get("articles", [])
        if a.get("title")
    ]

    return json.dumps({"query": query, "article_count": len(articles), "articles": articles})


@mcp.tool()
async def analyze_sentiment(text: str) -> str:
    """Analyse the sentiment of a piece of financial text using FinBERT.

    Args:
        text: Any financial text string to score

    Returns:
        JSON with polarity (-1 to 1), label (positive/negative/neutral),
        and raw FinBERT confidence scores for all three classes
    """
    scores = _scores(text)
    pol = _polarity(scores)

    return json.dumps({
        "polarity": pol,
        "label": _label(pol),
        "finbert_scores": scores,
    })


@mcp.tool()
async def get_market_sentiment(symbol: str, company_name: str = "") -> str:
    """Compute aggregate market sentiment from recent news for a stock using FinBERT.

    Args:
        symbol: Stock ticker symbol, e.g. 'AAPL'
        company_name: Full company name for better news matching (optional)

    Returns:
        JSON with average_sentiment, signal (bullish/bearish/neutral),
        article_count, confidence breakdown, and top article sentiments
    """
    query = company_name if company_name else symbol
    from_date = (datetime.now(UTC) - timedelta(days=7)).strftime("%Y-%m-%d")
    params = {
        "q": query,
        "from": from_date,
        "sortBy": "relevancy",
        "language": "en",
        "apiKey": NEWS_API_KEY,
        "pageSize": 15,
    }

    resp = requests.get(NEWS_API_BASE, params=params, timeout=10)
    resp.raise_for_status()
    articles = resp.json().get("articles", [])

    polarities = []
    all_positive = []
    all_negative = []
    all_neutral = []
    article_sentiments = []

    for a in articles:
        text = f"{a.get('title', '')} {a.get('description', '') or ''}".strip()
        if not text:
            continue

        scores = _scores(text)
        pol = _polarity(scores)
        polarities.append(pol)
        all_positive.append(scores.get("positive", 0))
        all_negative.append(scores.get("negative", 0))
        all_neutral.append(scores.get("neutral", 0))

        article_sentiments.append({
            "title": a.get("title", ""),
            "polarity": pol,
            "label": _label(pol),
            "finbert_scores": scores,
        })

    avg_polarity = round(sum(polarities) / len(polarities), 4) if polarities else 0.0
    n = len(polarities)

    return json.dumps({
        "symbol": symbol.upper(),
        "average_sentiment": avg_polarity,
        "article_count": n,
        "signal": "bullish" if avg_polarity > 0.1 else "bearish" if avg_polarity < -0.1 else "neutral",
        "confidence_breakdown": {
            "avg_positive": round(sum(all_positive) / n, 4) if n else 0,
            "avg_negative": round(sum(all_negative) / n, 4) if n else 0,
            "avg_neutral": round(sum(all_neutral) / n, 4) if n else 0,
        },
        "top_articles": article_sentiments[:5],
    })


if __name__ == "__main__":
    mcp.run(transport="stdio")
