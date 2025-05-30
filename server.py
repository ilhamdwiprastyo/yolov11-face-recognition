import uvicorn
import multiprocessing
from config.settings import settings

def run_server():
    """Run the server with production settings"""
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers if not settings.debug else 1,
        reload=settings.debug,
        access_log=True,
        log_level="info"
    )

if __name__ == "__main__":
    # Set start method for multiprocessing (important for CUDA)
    multiprocessing.set_start_method('spawn', force=True)
    run_server()