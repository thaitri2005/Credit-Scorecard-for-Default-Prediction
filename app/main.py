import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.utils.helpers import (
    API_VERSION,
    get_api_metadata,
    get_model_instance,
    setup_logging,
)

setup_logging()
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


def _env_list(name: str) -> list[str]:
    """Comma-separated env var -> list; unset or empty means allow everything."""
    values = [v.strip() for v in os.getenv(name, "").split(",") if v.strip()]
    return values or ["*"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup so the first request is not slow and a
    broken artifact shows up in the logs (and in /api/v1/health) immediately."""
    logger.info("Starting Credit Risk Scorecard API")
    try:
        get_model_instance()
    except Exception:
        logger.error(
            "Model failed to load; /api/v1/health will report 503", exc_info=True
        )
    yield
    logger.info("Shutting down Credit Risk Scorecard API")


app = FastAPI(
    title="Credit Risk Scorecard API",
    description="API for predicting credit risk and generating credit scores",
    version=API_VERSION,
    lifespan=lifespan,
)

# The API has no cookies or auth, so credentials are never allowed cross-origin.
# Restrict origins/hosts in production with CORS_ORIGINS / ALLOWED_HOSTS
# (comma-separated).
app.add_middleware(
    CORSMiddleware,
    allow_origins=_env_list("CORS_ORIGINS"),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
# localhost is always allowed so the container HEALTHCHECK keeps working.
allowed_hosts = _env_list("ALLOWED_HOSTS")
if "*" not in allowed_hosts:
    allowed_hosts += ["localhost", "127.0.0.1"]
app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)

app.include_router(router, prefix="/api/v1")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def serve_ui():
    """Serve the main UI"""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/info")
def get_info():
    """Get API information"""
    return get_api_metadata()


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log method, path, status and duration (never the query string or body)."""
    start_time = time.perf_counter()
    response = await call_next(request)
    logger.info(
        f"{request.method} {request.url.path} -> {response.status_code} "
        f"in {time.perf_counter() - start_time:.3f}s"
    )
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """422 without echoing the offending input back: FastAPI's default handler
    crashes when that input is NaN/Infinity, which JSON cannot represent."""
    errors = [
        {"loc": list(err["loc"]), "msg": err["msg"], "type": err["type"]}
        for err in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Return a JSON 500 for anything unhandled, without leaking internals."""
    logger.error(f"Unhandled exception on {request.url.path}", exc_info=exc)
    debug = os.getenv("DEBUG", "false").lower() == "true"
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if debug else "An error occurred",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
        reload=os.getenv("RELOAD", "false").lower() == "true",
    )
