#!/usr/bin/env python3
"""Portfolio Monitor CLI — submit a job, wait, and pretty-print the report.

Usage:
  python cli.py                                  # default AAPL/MSFT/NVDA demo
  python cli.py AAPL:10 TSLA:5 GOOG:3           # custom holdings
  python cli.py --url http://myserver:8000 AAPL:10

Arguments:
  SYMBOL:QUANTITY   e.g. AAPL:10 MSFT:5
  --url             API base URL (default: http://localhost:8000)
  --poll-interval   Seconds between status polls (default: 5)
"""

import sys
import time
import argparse

import requests
from rich.console import Console
from rich.markdown import Markdown
from rich.rule import Rule
from rich.spinner import Spinner
from rich.live import Live
from rich.text import Text

console = Console()

DEFAULT_HOLDINGS = [
    {"symbol": "AAPL", "quantity": 10},
    {"symbol": "MSFT", "quantity": 5},
    {"symbol": "NVDA", "quantity": 8},
]


def parse_holdings(args: list[str]) -> list[dict]:
    holdings = []
    for item in args:
        try:
            symbol, qty = item.split(":")
            holdings.append({"symbol": symbol.upper(), "quantity": float(qty)})
        except ValueError:
            console.print(f"[red]Invalid holding '{item}' — expected format SYMBOL:QUANTITY (e.g. AAPL:10)[/red]")
            sys.exit(1)
    return holdings


def submit_job(base_url: str, holdings: list[dict]) -> str:
    resp = requests.post(
        f"{base_url}/portfolio/analyze",
        json={"holdings": holdings},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()["job_id"]


def poll_until_done(base_url: str, job_id: str, interval: int) -> dict:
    url = f"{base_url}/portfolio/status/{job_id}"
    spinner_text = Text()

    with Live(console=console, refresh_per_second=4) as live:
        elapsed = 0
        while True:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            job = resp.json()

            if job["status"] in ("complete", "error"):
                live.stop()
                return job

            spinner_text = Text(f"  Analysing… {elapsed}s elapsed", style="dim")
            live.update(spinner_text)
            time.sleep(interval)
            elapsed += interval


def main():
    parser = argparse.ArgumentParser(description="Portfolio Monitor CLI")
    parser.add_argument("holdings", nargs="*", help="Holdings as SYMBOL:QUANTITY (e.g. AAPL:10)")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--poll-interval", type=int, default=5, metavar="SECONDS")
    args = parser.parse_args()

    holdings = parse_holdings(args.holdings) if args.holdings else DEFAULT_HOLDINGS

    # ── Submit ────────────────────────────────────────────────────────────────
    console.print(Rule("[bold]Portfolio Monitor[/bold]"))
    console.print(f"\nSubmitting {len(holdings)} holding(s) to [cyan]{args.url}[/cyan]\n")
    for h in holdings:
        console.print(f"  [dim]·[/dim] {h['symbol']:6s} × {h['quantity']}")
    console.print()

    try:
        job_id = submit_job(args.url, holdings)
    except requests.RequestException as e:
        console.print(f"[red]Failed to reach API: {e}[/red]")
        console.print("[dim]Is the server running? Try: uvicorn api.main:app --reload --port 8000[/dim]")
        sys.exit(1)

    console.print(f"[green]Job submitted[/green] → [bold]{job_id}[/bold]\n")

    # ── Poll ─────────────────────────────────────────────────────────────────
    job = poll_until_done(args.url, job_id, args.poll_interval)

    if job["status"] == "error":
        console.print(f"\n[red]Analysis failed:[/red] {job.get('error', 'unknown error')}")
        sys.exit(1)

    # ── Render ───────────────────────────────────────────────────────────────
    created = job.get("created_at", "")[:19].replace("T", " ")
    completed = job.get("completed_at", "")[:19].replace("T", " ")
    duration = ""
    try:
        from datetime import datetime
        fmt = "%Y-%m-%d %H:%M:%S"
        delta = datetime.strptime(completed, fmt) - datetime.strptime(created, fmt)
        duration = f"  [dim]({int(delta.total_seconds())}s)[/dim]"
    except Exception:
        pass

    console.print(Rule(f"[green]Complete[/green]{duration}"))
    console.print()
    console.print(Markdown(job["report"]))
    console.print()
    console.print(Rule())
    console.print(
        f"\n[dim]Job ID: {job_id}[/dim]\n"
        f"[dim]To retrieve again: curl {args.url}/portfolio/report/{job_id}[/dim]\n"
    )


if __name__ == "__main__":
    main()
