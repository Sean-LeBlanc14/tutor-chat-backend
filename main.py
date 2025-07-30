""" Optimized FastAPI app for classroom scale (40+ concurrent users) """
import os
import logging
import time
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.exceptions import RequestValidationError

# Import your modules
from models import QuestionRequest, ChatResponse
from query_bot import ask_question, cleanup_cache
from routes import chat, auth, sandbox
from db import db_manager
from error_handler import (
    validation_exception_handler,
    http_exception_handler,
    general_exception_handler,
    CustomHTTPException
)
from security import validate_environment_variables

# Enhanced logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Only console logging to avoid permission issues
    ]
)
logger = logging.getLogger(__name__)

# Validate environment on startup
validate_environment_variables()

# Background task to clean cache periodically
async def cleanup_background_tasks():
    """Background cleanup tasks"""
    while True:
        try:
            cleanup_cache()  # Clean expired cache entries
            await asyncio.sleep(1800)  # Run every 30 minutes
        except Exception as e:
            logger.error(f"Background cleanup error: {e}")
            await asyncio.sleep(300)  # Retry in 5 minutes

@asynccontextmanager
async def lifespan(app: FastAPI):
    """ Enhanced startup and shutdown with background tasks """
    logger.info("🚀 Starting classroom-scale tutor chatbot...")
    
    # Initialize database
    await db_manager.initialize()
    logger.info("✅ Database initialized")
    
    # Start background tasks
    cleanup_task = asyncio.create_task(cleanup_background_tasks())
    logger.info("✅ Background tasks started")
    
    # GPU memory check
    try:
        import torch
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"🎯 GPU: {torch.cuda.get_device_name(0)} ({gpu_memory:.1f}GB)")
        else:
            logger.warning("⚠️  No CUDA GPU detected - using CPU fallback")
    except ImportError:
        logger.info("📊 PyTorch not available for GPU detection")
    
    logger.info("🎓 Application startup complete - Ready for classroom!")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down application...")
    cleanup_task.cancel()
    await db_manager.close()
    logger.info("✅ Application shutdown complete")

# Create FastAPI app with optimized settings
app = FastAPI(
    title="Psychology Tutor Chatbot API",
    description="Classroom-scale psychology tutor with smart RAG and concurrent request handling",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs" if os.getenv("ENVIRONMENT") != "production" else None,
    redoc_url="/redoc" if os.getenv("ENVIRONMENT") != "production" else None,
)

# Environment configuration
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
ALLOWED_HOSTS = os.getenv("ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")

# Trusted hosts for production
if ENVIRONMENT == "production":
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=ALLOWED_HOSTS
    )

# Optimized compression middleware
app.add_middleware(GZipMiddleware, minimum_size=500)  # Lower threshold for better compression

# CORS middleware with optimized settings
if ENVIRONMENT == "production":
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[FRONTEND_URL],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Requested-With"],
        expose_headers=["*"],
        max_age=86400,  # Cache preflight requests for 24 hours
    )
else:
    # Development CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        max_age=3600,
    )

# Exception handlers
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)

# Include API routes
app.include_router(chat.router, prefix="/api", tags=["chat"])
app.include_router(auth.router, prefix="/api", tags=["auth"])
app.include_router(sandbox.router, prefix="/api", tags=["sandbox"])

# Performance monitoring middleware
@app.middleware("http")
async def performance_monitoring(request: Request, call_next):
    """Enhanced request logging with performance metrics"""
    start_time = time.time()
    
    # Log request
    logger.info(
        f"📥 {request.method} {request.url.path} | "
        f"Client: {request.client.host if request.client else 'unknown'}"
    )
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Log response with performance data
        status_emoji = "✅" if response.status_code < 400 else "❌"
        logger.info(
            f"📤 {status_emoji} {response.status_code} | "
            f"⏱️  {process_time:.3f}s | "
            f"Path: {request.url.path}"
        )
        
        # Add performance headers
        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-Server-Version"] = "2.0.0"
        
        return response
        
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(
            f"💥 ERROR after {process_time:.3f}s | "
            f"Path: {request.url.path} | "
            f"Error: {str(e)}"
        )
        raise

# Rate limiting middleware (simple implementation)
from collections import defaultdict
from datetime import datetime, timedelta

# Simple in-memory rate limiting (upgrade to Redis for production)
request_counts = defaultdict(list)
RATE_LIMIT_REQUESTS = 60  # requests per minute per IP
RATE_LIMIT_WINDOW = timedelta(minutes=1)

@app.middleware("http")
async def rate_limiting(request: Request, call_next):
    """Simple rate limiting for classroom protection"""
    if request.url.path.startswith("/api/chat"):
        client_ip = request.client.host if request.client else "unknown"
        now = datetime.now()
        
        # Clean old requests
        request_counts[client_ip] = [
            req_time for req_time in request_counts[client_ip]
            if now - req_time < RATE_LIMIT_WINDOW
        ]
        
        # Check rate limit
        if len(request_counts[client_ip]) >= RATE_LIMIT_REQUESTS:
            logger.warning(f"🚫 Rate limit exceeded for {client_ip}")
            raise HTTPException(
                status_code=429,
                detail="Too many requests. Please wait before trying again."
            )
        
        # Add current request
        request_counts[client_ip].append(now)
    
    return await call_next(request)

# Enhanced health check
@app.get("/api/health")
async def health_check():
    """Comprehensive health check for classroom deployment"""
    try:
        # Database check
        async with db_manager.get_connection() as conn:
            await conn.fetchrow("SELECT 1")
        
        # Model availability check
        model_status = "healthy"
        try:
            # Quick model test
            test_response = await ask_question("test", temperature=0.1)
            if not test_response:
                model_status = "degraded"
        except Exception as e:
            model_status = f"error: {str(e)}"
            logger.error(f"Model health check failed: {e}")
        
        # System resource check
        import psutil
        memory_percent = psutil.virtual_memory().percent
        cpu_percent = psutil.cpu_percent()
        
        # GPU check if available
        gpu_info = {}
        try:
            import torch
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated(0) / torch.cuda.max_memory_allocated(0) * 100
                gpu_info = {
                    "available": True,
                    "memory_usage": f"{gpu_memory:.1f}%",
                    "device_name": torch.cuda.get_device_name(0)
                }
            else:
                gpu_info = {"available": False}
        except ImportError:
            gpu_info = {"available": False, "error": "PyTorch not installed"}
        
        # Overall health status
        overall_status = "healthy"
        if model_status != "healthy":
            overall_status = "degraded"
        if memory_percent > 90 or (gpu_info.get("available") and "memory_usage" in gpu_info and float(gpu_info["memory_usage"].rstrip('%')) > 95):
            overall_status = "critical"
        
        return {
            "status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "environment": ENVIRONMENT,
            "version": "2.0.0",
            "components": {
                "database": "connected",
                "model": model_status,
                "cache": "active"
            },
            "system": {
                "memory_usage": f"{memory_percent:.1f}%",
                "cpu_usage": f"{cpu_percent:.1f}%",
                "gpu": gpu_info
            },
            "features": {
                "smart_rag": True,
                "request_queuing": True,
                "response_caching": True,
                "concurrent_limit": 15
            }
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise CustomHTTPException(
            status_code=503,
            detail="Service unhealthy",
            error_code="HEALTH_CHECK_FAILED"
        ) from e

# System metrics endpoint
@app.get("/api/metrics")
async def system_metrics():
    """Detailed system metrics for monitoring"""
    try:
        import psutil
        
        # Get current metrics from chat router
        from routes.chat import request_metrics
        
        return {
            "timestamp": datetime.now().isoformat(),
            "requests": {
                "total": request_metrics['total_requests'],
                "concurrent": request_metrics['concurrent_requests'],
                "avg_response_time": round(request_metrics['avg_response_time'], 3),
                "cache_hits": request_metrics['cache_hits']
            },
            "system": {
                "memory": {
                    "used_percent": psutil.virtual_memory().percent,
                    "available_gb": round(psutil.virtual_memory().available / 1024**3, 2)
                },
                "cpu": {
                    "percent": psutil.cpu_percent(),
                    "cores": psutil.cpu_count()
                },
                "disk": {
                    "used_percent": psutil.disk_usage('/').percent
                }
            }
        }
    except Exception as e:
        logger.error(f"Metrics collection failed: {e}")
        return {"error": str(e), "timestamp": datetime.now().isoformat()}

@app.get("/")
async def root():
    """Root endpoint with system info"""
    return {
        "message": "🎓 Psychology Tutor Chatbot API - Classroom Ready",
        "version": "2.0.0",
        "status": "operational",
        "features": [
            "Smart RAG (context only when needed)",
            "Concurrent request handling (15+ users)",
            "Response caching",
            "Optimized 3B model",
            "Real-time streaming"
        ],
        "endpoints": {
            "chat": "/api/chat/stream",
            "health": "/api/health",
            "metrics": "/api/metrics",
            "docs": "/docs" if ENVIRONMENT != "production" else "disabled"
        }
    }

# Startup event for additional initialization
@app.on_event("startup")
async def startup_event():
    """Additional startup tasks"""
    logger.info("🎯 Classroom-scale chatbot ready!")
    logger.info("📊 Optimizations: 3B model, Smart RAG, Request queuing, Response caching")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))

    if ENVIRONMENT == "development":
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=port,
            reload=True,  # Hot reload in dev
            log_level="info",
            access_log=True
        )
    else:
        # Production settings optimized for classroom scale
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=port,
            workers=1,  # Single worker for GPU sharing
            worker_class="uvicorn.workers.UvicornWorker",
            log_level="info",
            access_log=True,
            loop="asyncio",  # Optimized event loop
            http="httptools",  # Faster HTTP parsing
            lifespan="on"  # Enable lifespan events
        )
