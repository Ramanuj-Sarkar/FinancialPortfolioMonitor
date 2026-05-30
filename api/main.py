"""FastAPI app — async job queue for portfolio analysis.

Endpoints:
  POST /portfolio/analyze          → submit analysis, returns job_id (202)
  GET  /portfolio/status/{job_id}  → poll job status
  GET  /portfolio/report/{job_id}  → retrieve completed Markdown report
  GET  /health                     → service health check
"""

import uuid
from datetime import datetime, UTC
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agents.orchestrator import PortfolioState, graph

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Financial Portfolio Monitor",
    description="Multi-agent portfolio analysis — LangGraph + MCP + FastAPI",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job store — replace with Redis for production
jobs: dict[str, dict] = {}

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class Holding(BaseModel):
    symbol: str = Field(..., description="Stock ticker, e.g. 'AAPL'")
    quantity: float = Field(..., gt=0, description="Number of shares held")


class PortfolioRequest(BaseModel):
    holdings: list[Holding]

    model_config = {
        "json_schema_extra": {
            "example": {
                "holdings": [
                    {"symbol": "AAPL", "quantity": 10},
                    {"symbol": "MSFT", "quantity": 5},
                    {"symbol": "NVDA", "quantity": 8},
                ]
            }
        }
    }


class JobResponse(BaseModel):
    job_id: str
    status: str  # "running" | "complete" | "error"
    created_at: str
    completed_at: Optional[str] = None
    report: Optional[str] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Background task
# ---------------------------------------------------------------------------


async def run_analysis(job_id: str, portfolio: list[dict]) -> None:
    """Run the full LangGraph pipeline and write the result back to the job store."""
    try:
        initial_state: PortfolioState = {
            "portfolio": portfolio,
            "market_data": "",
            "sentiment_data": "",
            "risk_assessment": "",
            "report": "",
        }
        final_state = await graph.ainvoke(initial_state)
        jobs[job_id].update({
            "status": "complete",
            "report": final_state["report"],
            "completed_at": datetime.now(UTC).isoformat(),
        })
    except Exception as exc:
        jobs[job_id].update({"status": "error", "error": str(exc)})


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.post("/portfolio/analyze", response_model=JobResponse, status_code=202)
async def analyze_portfolio(request: PortfolioRequest, background_tasks: BackgroundTasks):
    """Submit a portfolio for async analysis. Poll /portfolio/status/{job_id} for results."""
    job_id = str(uuid.uuid4())
    portfolio = [{"symbol": h.symbol.upper(), "quantity": h.quantity} for h in request.holdings]

    jobs[job_id] = {
        "status": "running",
        "created_at": datetime.now(UTC).isoformat(),
        "completed_at": None,
        "report": None,
        "error": None,
    }

    background_tasks.add_task(run_analysis, job_id, portfolio)
    return JobResponse(job_id=job_id, **jobs[job_id])


@app.get("/portfolio/status/{job_id}", response_model=JobResponse)
async def get_status(job_id: str):
    """Poll the status of a portfolio analysis job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return JobResponse(job_id=job_id, **jobs[job_id])


@app.get("/portfolio/report/{job_id}")
async def get_report(job_id: str):
    """Retrieve the completed portfolio report (plain text Markdown)."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    job = jobs[job_id]
    if job["status"] == "running":
        raise HTTPException(status_code=202, detail="Analysis still in progress")
    if job["status"] == "error":
        raise HTTPException(status_code=500, detail=job.get("error", "Analysis failed"))

    return {"job_id": job_id, "report": job["report"]}


@app.get("/health")
async def health():
    running = sum(1 for j in jobs.values() if j["status"] == "running")
    return {"status": "ok", "active_jobs": running, "total_jobs": len(jobs)}
