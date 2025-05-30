from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from pathlib import Path

from app.routers import detection, health
from app.utils.logger import get_logger
from config.settings import settings

logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Face Recognition API",
    description="Advanced Face Detection and Recognition API using YOLOv11 and ArcFace",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files for serving processed images/videos
app.mount("/outputs", StaticFiles(directory=settings.output_dir), name="outputs")

# Include routers
app.include_router(health.router)
app.include_router(detection.router)

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    logger.error(f"HTTP {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )

@app.on_event("startup")
async def startup_event():
    """Startup event handler"""
    logger.info("Starting Face Recognition API...")
    logger.info(f"Upload directory: {settings.upload_dir}")
    logger.info(f"Output directory: {settings.output_dir}")
    logger.info(f"Device: {settings.device}")
    
    # Create directories
    for dir_path in [settings.upload_dir, settings.output_dir, settings.log_dir]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    logger.info("Face Recognition API started successfully!")

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler"""
    logger.info("Shutting down Face Recognition API...")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Face Recognition API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health"
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        workers=settings.api_workers
    )