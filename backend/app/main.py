"""
FastAPI Application Entry Point
Main application with all routes configured.
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time

from backend.app.api import auth, documents, chat, agent, metrics
from backend.app.config import config
from backend.app.database.connection import init_database

# Create FastAPI application
app = FastAPI(
    title="Agentic RAG Research Assistant",
    description="Multi-Tenant Agentic RAG Research & Knowledge Assistant API",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request timing middleware
@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(round(process_time, 3))
    return response


# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": str(exc)
        }
    )


# Include routers
app.include_router(auth.router)
app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(agent.router)
app.include_router(metrics.router)


# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": time.time()
    }


# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Agentic RAG Research Assistant API",
        "version": "1.0.0",
        "docs": "/api/docs"
    }


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize database on startup."""
    config.ensure_directories()
    init_database()
    print("Application started successfully!")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print("Application shutting down...")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
