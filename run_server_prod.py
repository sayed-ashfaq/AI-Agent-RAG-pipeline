#!/usr/bin/env python
"""
Production server startup script with multiple workers
For use with gunicorn in production environment
"""
import uvicorn
from custom_logger import GLOBAL_LOGGER as logger

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Starting AI Agent with RAG FastAPI Server (Production)")
    logger.info("=" * 60)
    logger.info("Configuration:")
    logger.info("  - Host: 0.0.0.0")
    logger.info("  - Port: 8000")
    logger.info("  - Workers: 4")
    logger.info("  - Reload: False")
    logger.info("=" * 60)
    logger.info("Server will be available at: http://0.0.0.0:8000")
    logger.info("=" * 60)

    # Production configuration
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        workers=4,  # Multiple workers for production
        reload=False,  # Disable reload in production
        log_level="info",
        access_log=True,
    )
