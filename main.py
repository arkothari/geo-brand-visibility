import os
import sys

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()

from models import (
    EvaluationRequest, EvaluationResponse,
    SimulationRequest, SimulationResult,
    WatchlistRequest, WatchlistResponse,
    OpportunityRequest, OpportunityResponse,
)
from services.llm_service import LLMService
from services.evaluator import BrandEvaluator
from services.providers import ClaudeProvider, OpenAIProvider, GeminiProvider, PerplexityProvider

app = FastAPI(
    title="GEO Visibility",
    description="Measure your brand's visibility across AI models",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
if not anthropic_key:
    print("WARNING: ANTHROPIC_API_KEY is not set — required for analysis.", file=sys.stderr)

all_providers = [
    ClaudeProvider(api_key=anthropic_key),
    OpenAIProvider(api_key=os.getenv("OPENAI_API_KEY", "")),
    GeminiProvider(api_key=os.getenv("GOOGLE_API_KEY", "")),
    PerplexityProvider(api_key=os.getenv("PERPLEXITY_API_KEY", "")),
]

llm_service = LLMService(api_key=anthropic_key)
evaluator = BrandEvaluator(llm_service=llm_service, providers=all_providers)

_provider_map = {p.name: p for p in all_providers}

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", include_in_schema=False)
async def root():
    return FileResponse("static/index.html")


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "version": "2.0.0",
        "providers": [
            {"name": p.name, "label": p.label, "color": p.color, "configured": p.is_configured}
            for p in all_providers
        ],
    }


@app.get("/api/providers")
async def get_providers():
    return [
        {"name": p.name, "label": p.label, "color": p.color, "configured": p.is_configured}
        for p in all_providers
    ]


@app.post("/api/evaluate", response_model=EvaluationResponse)
async def evaluate_brand(request: EvaluationRequest):
    if not anthropic_key:
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY not configured.")

    selected = [
        _provider_map[name]
        for name in request.providers
        if name in _provider_map and _provider_map[name].is_configured
    ]
    if not selected:
        raise HTTPException(status_code=400, detail="None of the requested providers are configured.")

    try:
        result = await evaluator.evaluate(
            query=request.query,
            brand=request.brand,
            num_samples=request.num_samples,
            providers=selected,
            locale=request.locale,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/simulate", response_model=SimulationResult)
async def simulate_impact(request: SimulationRequest):
    if not anthropic_key:
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY not configured.")

    selected = [
        _provider_map[name]
        for name in request.providers
        if name in _provider_map and _provider_map[name].is_configured
    ]
    if not selected:
        raise HTTPException(status_code=400, detail="None of the requested providers are configured.")

    try:
        result = await evaluator.simulate(
            query=request.query,
            brand=request.brand,
            change_description=request.change_description,
            num_samples=request.num_samples,
            providers=selected,
            locale=request.locale,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/watchlist", response_model=WatchlistResponse)
async def competitor_watchlist(request: WatchlistRequest):
    if not anthropic_key:
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY not configured.")

    selected = [
        _provider_map[name]
        for name in request.providers
        if name in _provider_map and _provider_map[name].is_configured
    ]
    if not selected:
        raise HTTPException(status_code=400, detail="None of the requested providers are configured.")

    clean = [c.strip() for c in request.competitors if c.strip()]
    if not clean:
        raise HTTPException(status_code=400, detail="At least one competitor name is required.")
    if len(clean) > 5:
        raise HTTPException(status_code=400, detail="Maximum 5 competitors allowed.")

    try:
        result = await evaluator.watchlist(
            query=request.query,
            brand=request.brand,
            competitors=clean,
            num_samples=request.num_samples,
            providers=selected,
            locale=request.locale,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/opportunity", response_model=OpportunityResponse)
async def opportunity_map(request: OpportunityRequest):
    if not anthropic_key:
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY not configured.")
    try:
        result = await evaluator.opportunity_map(
            query=request.query,
            brand=request.brand,
            locale=request.locale,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8001))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
