from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import logging
import os
from contextlib import asynccontextmanager

from app.api.routes import router
from app.utils.helpers import setup_logging, validate_environment, get_api_metadata

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("🚀 Starting Credit Risk Scorecard API")

    # Validate environment
    if not validate_environment():
        logger.warning("⚠️ Environment validation failed, but continuing startup")

    # Check for model artifacts
    required_files = [
        "models/credit_risk_model.joblib",
        "models/preprocessing_pipeline.joblib",
        "models/model_metadata.json",
    ]
    for f in required_files:
        if not os.path.exists(f):
            logger.warning(f"⚠️ Missing required artifact: {f}")
        else:
            logger.info(f"✅ Found artifact: {f}")

    # Create static directory if it doesn't exist
    os.makedirs("app/static", exist_ok=True)

    yield

    # Shutdown
    logger.info("🛑 Shutting down Credit Risk Scorecard API")


# Create FastAPI app
app = FastAPI(
    title="Credit Risk Scorecard API",
    description="API for predicting credit risk and generating credit scores",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add trusted host middleware
app.add_middleware(
    TrustedHostMiddleware, allowed_hosts=["*"]  # In production, specify actual hosts
)

# Include API routes
app.include_router(router, prefix="/api/v1")

# Serve static files for UI
app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/")
async def serve_ui():
    """Serve the main UI"""
    ui_path = "app/static/index.html"
    if os.path.exists(ui_path):
        return FileResponse(ui_path)
    else:
        return {
            "message": "Credit Risk Scorecard API",
            "version": "1.0.0",
            "status": "running",
            "ui": "UI files not found. Please check static files.",
            "docs": "/docs",
        }


@app.get("/info")
async def get_info():
    """Get API information"""
    return get_api_metadata()


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests"""
    import time

    start_time = time.time()

    # Log request
    logger.info(f"Request: {request.method} {request.url}")

    # Process request
    response = await call_next(request)

    # Log response
    process_time = time.time() - start_time
    logger.info(f"Response: {response.status_code} - {process_time:.3f}s")

    return response


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return {
        "error": "Internal server error",
        "detail": str(exc)
        if os.getenv("DEBUG", "false").lower() == "true"
        else "An error occurred",
        "status_code": 500,
    }


if __name__ == "__main__":
    import uvicorn

    # Get configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    reload = os.getenv("RELOAD", "false").lower() == "true"

    logger.info(f"Starting server on {host}:{port}")

    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        log_level=log_level,
        reload=reload,
        access_log=True,
    )
