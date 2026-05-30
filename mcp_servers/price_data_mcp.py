#!/usr/bin/env python3
"""MCP server — stock price data via yfinance.

Tools exposed:
  get_stock_price(symbol)               → current price + daily change
  get_historical_data(symbol, period)   → OHLCV records
  get_portfolio_value(holdings_json)    → total value + per-holding breakdown
"""

import json
from datetime import datetime, UTC

import yfinance as yf
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("price-data-mcp")


@mcp.tool()
async def get_stock_price(symbol: str) -> str:
    """Get current stock price and daily stats for a ticker symbol.

    Args:
        symbol: Stock ticker, e.g. 'AAPL' or 'MSFT'

    Returns:
        JSON with price, change, change_pct, volume, market_cap
    """
    ticker = yf.Ticker(symbol.upper())
    info = ticker.fast_info
    prev = info.previous_close
    price = info.last_price

    return json.dumps({
        "symbol": symbol.upper(),
        "price": round(price, 2),
        "previous_close": round(prev, 2),
        "change": round(price - prev, 2),
        "change_pct": round(((price - prev) / prev) * 100, 2),
        "volume": getattr(info, "three_month_average_volume", None),
        "market_cap": getattr(info, "market_cap", None),
        "timestamp": datetime.now(UTC).isoformat(),
    })


@mcp.tool()
async def get_historical_data(symbol: str, period: str = "1mo") -> str:
    """Get historical OHLCV data for a stock.

    Args:
        symbol: Stock ticker symbol
        period: '1d' | '5d' | '1mo' | '3mo' | '6mo' | '1y' | '2y' | '5y'

    Returns:
        JSON with list of OHLCV records
    """
    ticker = yf.Ticker(symbol.upper())
    hist = ticker.history(period=period)

    records = [
        {
            "date": str(date.date()),
            "open": round(row["Open"], 2),
            "high": round(row["High"], 2),
            "low": round(row["Low"], 2),
            "close": round(row["Close"], 2),
            "volume": int(row["Volume"]),
        }
        for date, row in hist.iterrows()
    ]

    return json.dumps({"symbol": symbol.upper(), "period": period, "data": records})


@mcp.tool()
async def get_portfolio_value(holdings_json: str) -> str:
    """Calculate current market value of a portfolio.

    Args:
        holdings_json: JSON array of holdings, e.g.
            '[{"symbol": "AAPL", "quantity": 10}, {"symbol": "MSFT", "quantity": 5}]'

    Returns:
        JSON with total_value and per-holding breakdown including portfolio weight %
    """
    holdings = json.loads(holdings_json)
    total = 0.0
    breakdown = []

    for h in holdings:
        ticker = yf.Ticker(h["symbol"].upper())
        price = ticker.fast_info.last_price
        value = round(price * h["quantity"], 2)
        total += value
        breakdown.append({
            "symbol": h["symbol"].upper(),
            "quantity": h["quantity"],
            "current_price": round(price, 2),
            "market_value": value,
            "weight_pct": 0.0,  # filled below
        })

    for item in breakdown:
        item["weight_pct"] = round((item["market_value"] / total) * 100, 2)

    return json.dumps({
        "total_value": round(total, 2),
        "currency": "USD",
        "holdings": breakdown,
        "timestamp": datetime.now(UTC).isoformat(),
    })


if __name__ == "__main__":
    mcp.run(transport="stdio")
