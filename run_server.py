"""
FastAPI server startup script
Run this file to start the FastAPI server
"""
import uvicorn
from custom_logger import GLOBAL_LOGGER as logger

if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("Starting AI Agent with RAG FastAPI Server")
    logger.info("=" * 50)
    logger.info("Server will be available at: http://localhost:8000")
    logger.info("=" * 50)

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
