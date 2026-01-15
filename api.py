#!/usr/bin/env python3
"""
FastAPI REST API for Project Recommendations.

Endpoints:
    GET  /health           - Health check
    GET  /recommend        - Get recommendations by query
    GET  /similar/{id}     - Get similar projects
    GET  /domains          - List all domains
    GET  /languages        - List all languages
    GET  /difficulties     - List difficulty levels
    GET  /project/{id}     - Get project details

Usage:
    uvicorn api:app --reload --port 8000
    
    # Or run directly
    python api.py
"""

from typing import Optional, List
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from recommender import ProjectRecommender

# ============================================================
# Pydantic Models
# ============================================================

class ProjectResult(BaseModel):
    """Single project recommendation result."""
    project_id: int
    title: str
    domain: str
    description: str
    language: str
    difficulty: str
    keywords: str
    libraries: str
    build_time: str
    documentation: str
    score: float


class RecommendationResponse(BaseModel):
    """Response for recommendation queries."""
    query: str
    filters: dict
    count: int
    results: List[ProjectResult]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    projects_loaded: int
    engine: str


class MetadataResponse(BaseModel):
    """Response for metadata endpoints."""
    items: List[str]
    count: int


# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(
    title="Project Recommendation API",
    description="Content-based project recommendation system using TF-IDF",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global recommender instance
recommender: Optional[ProjectRecommender] = None


@app.on_event("startup")
async def startup_event():
    """Load recommender on startup."""
    global recommender
    try:
        recommender = ProjectRecommender()
        print("[API] Recommender loaded successfully")
    except FileNotFoundError as e:
        print(f"[API] Error: {e}")
        print("[API] Run 'python build_pipeline.py' first to build artifacts.")


# ============================================================
# Endpoints
# ============================================================

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint."""
    if recommender is None:
        raise HTTPException(
            status_code=503,
            detail="Recommender not loaded. Run 'python build_pipeline.py' first."
        )
    return HealthResponse(
        status="healthy",
        projects_loaded=len(recommender.df),
        engine="TfidfEngine"
    )


@app.get("/recommend", response_model=RecommendationResponse, tags=["Recommendations"])
async def get_recommendations(
    query: str = Query(..., description="Search query (keywords, description, etc.)"),
    top_n: int = Query(10, ge=1, le=100, description="Number of results"),
    domain: Optional[str] = Query(None, description="Filter by domain (partial match)"),
    language: Optional[str] = Query(None, description="Filter by programming language"),
    difficulty: Optional[str] = Query(None, description="Filter by difficulty level"),
    min_score: float = Query(0.0, ge=0, le=1, description="Minimum similarity score")
):
    """
    Get project recommendations based on query.
    
    **Example:**
    ```
    GET /recommend?query=machine learning python&top_n=5
    GET /recommend?query=web app&domain=Web&language=javascript
    ```
    """
    if recommender is None:
        raise HTTPException(status_code=503, detail="Recommender not loaded")
    
    results = recommender.recommend(
        query=query,
        top_n=top_n,
        domain_filter=domain,
        language_filter=language,
        difficulty_filter=difficulty,
        min_score=min_score
    )
    
    return RecommendationResponse(
        query=query,
        filters={
            "domain": domain,
            "language": language,
            "difficulty": difficulty,
            "min_score": min_score
        },
        count=len(results),
        results=[ProjectResult(**r) for r in results]
    )


@app.get("/similar/{project_id}", response_model=RecommendationResponse, tags=["Recommendations"])
async def get_similar_projects(
    project_id: int,
    top_n: int = Query(5, ge=1, le=50, description="Number of similar projects")
):
    """
    Find projects similar to a given project ID.
    
    **Example:**
    ```
    GET /similar/42?top_n=5
    ```
    """
    if recommender is None:
        raise HTTPException(status_code=503, detail="Recommender not loaded")
    
    try:
        results = recommender.recommend_by_project_id(project_id, top_n=top_n)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    return RecommendationResponse(
        query=f"similar_to:{project_id}",
        filters={},
        count=len(results),
        results=[ProjectResult(
            project_id=r['project_id'],
            title=r['title'],
            domain=r['domain'],
            description=r['description'],
            language="",
            difficulty="",
            keywords="",
            libraries="",
            build_time="",
            documentation="",
            score=r['score']
        ) for r in results]
    )


@app.get("/project/{project_id}", response_model=ProjectResult, tags=["Projects"])
async def get_project(project_id: int):
    """
    Get details of a specific project by ID.
    
    **Example:**
    ```
    GET /project/42
    ```
    """
    if recommender is None:
        raise HTTPException(status_code=503, detail="Recommender not loaded")
    
    # Find project
    results = recommender.recommend(query="project", top_n=1000)
    for r in results:
        if r['project_id'] == project_id:
            return ProjectResult(**r)
    
    raise HTTPException(status_code=404, detail=f"Project ID {project_id} not found")


@app.get("/domains", response_model=MetadataResponse, tags=["Metadata"])
async def list_domains():
    """List all available domains."""
    if recommender is None:
        raise HTTPException(status_code=503, detail="Recommender not loaded")
    
    domains = recommender.get_domains()
    return MetadataResponse(items=domains, count=len(domains))


@app.get("/languages", response_model=MetadataResponse, tags=["Metadata"])
async def list_languages():
    """List all available programming languages."""
    if recommender is None:
        raise HTTPException(status_code=503, detail="Recommender not loaded")
    
    languages = recommender.get_languages()
    return MetadataResponse(items=languages, count=len(languages))


@app.get("/difficulties", response_model=MetadataResponse, tags=["Metadata"])
async def list_difficulties():
    """List all difficulty levels."""
    if recommender is None:
        raise HTTPException(status_code=503, detail="Recommender not loaded")
    
    levels = recommender.get_difficulty_levels()
    return MetadataResponse(items=levels, count=len(levels))


# ============================================================
# Run directly
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
