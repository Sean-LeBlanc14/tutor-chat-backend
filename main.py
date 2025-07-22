""" Module for the main FastAPI app """
import os
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.exceptions import RequestValidationError

# Import your modules
from models import QuestionRequest, ChatResponse
from query_bot import ask_question
from routes import chat, auth, sandbox
from db import db_manager  # Import the global instance instead of creating new one
from error_handler import (
    validation_exception_handler,
    http_exception_handler,
    general_exception_handler,
    CustomHTTPException
)
from security import validate_environment_variables

# Validate environment on startup
validate_environment_variables()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """ Handles the start up and shutdown of the fastAPI app """
    # Startup
    logging.info("Starting up application...")
    await db_manager.initialize()
    logging.info("Application startup complete")
    yield
    # Shutdown
    logging.info("Shutting down application...")
    await db_manager.close()
    logging.info("Application shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="Tutor Chatbot API",
    description="Psychology tutor chatbot with RAG capabilities",
    version="1.0.0",
    lifespan=lifespan
)

# Environment configuration
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
ALLOWED_HOSTS = os.getenv("ALLOWED_HOSTS", "localhost,127.0.0.1").split(",")

# Security middleware
if ENVIRONMENT == "production":
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=ALLOWED_HOSTS
    )

# Compression middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# CORS middleware - Updated for university server
if ENVIRONMENT == "production":
    # Production CORS - specific origins
    allowed_origins = [FRONTEND_URL]
    
    # Also allow the university domain for testing
    if "athena.cs.csubak.edu" not in FRONTEND_URL:
        allowed_origins.append("https://athena.cs.csubak.edu")
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
        expose_headers=["*"],
        max_age=86400,  # Cache preflight for 24 hours
    )
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Exception handlers
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)

# Include routers
app.include_router(chat.router, prefix="/api", tags=["chat"])
app.include_router(auth.router, prefix="/api", tags=["auth"])
app.include_router(sandbox.router, prefix="/api", tags=["sandbox"])

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """ Records requests made (http)"""
    start_time = time.time()

    # Log request
    logging.info("Request: %s %s", request.method, request.url)

    response = await call_next(request)

    # Log response
    process_time = time.time() - start_time
    logging.info(
        "Response: %s | Time: %ss | Path: %s", response.status_code, process_time, request.url.path
    )

    return response

# Enhanced main.py chat endpoint
@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(data: QuestionRequest):
    """
    Enhanced chat endpoint with chat memory support
    """
    try:
        # Get chat history for context (if chat_id provided)
        chat_history = []
        if hasattr(data, 'chat_id') and data.chat_id:
            try:
                # Fetch recent messages from this chat
                history_result = await db_manager.execute_query("""
                    SELECT role, content 
                    FROM chat_logs 
                    WHERE chat_id = $1 AND mode = 'chat'
                    ORDER BY created_at ASC
                    LIMIT 20
                """, data.chat_id)

                chat_history = [
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in history_result
                    if msg["role"] in ['user', 'assistant']  # Updated: using 'assistant' instead of 'bot'
                ]
            except Exception as e:
                logging.warning("Could not fetch chat history: %s", e)

        result = ask_question(
            question=data.question,
            system_prompt=data.system_prompt,
            temperature=data.temperature,
            chat_history=chat_history  # Pass chat history
        )

        if not result or not result.strip():
            raise CustomHTTPException(
                status_code=500,
                detail="Failed to generate response",
                error_code="GENERATION_FAILED"
            )

        return ChatResponse(response=result)

    except CustomHTTPException:
        raise
    except Exception as e:
        logging.error("Chat endpoint error: %s", e)
        raise CustomHTTPException(
            status_code=500,
            detail="Internal server error",
            error_code="INTERNAL_ERROR"
        ) from e

@app.get("/api/health")
async def health_check():
    """Enhanced health check with dependency verification"""
    try:
        # Test database connection
        async with db_manager.get_connection() as conn:
            await conn.fetchrow("SELECT 1")

        # Test API key availability (don't expose the actual key)
        api_key_available = bool(os.getenv("API_KEY"))
        pinecone_available = bool(os.getenv("PINECONE_API_KEY"))

        return {
            "status": "healthy",
            "environment": ENVIRONMENT,
            "database": "connected",
            "external_apis": {
                "mistral": api_key_available,
                "pinecone": pinecone_available
            }
        }
    except Exception as e:
        logging.error("Health check failed: %s", e)
        raise CustomHTTPException(
            status_code=503,
            detail="Service unhealthy",
            error_code="HEALTH_CHECK_FAILED"
        ) from e

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Tutor Chatbot API", "version": "1.0.0"}

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))

    # Different configurations for dev vs prod
    if ENVIRONMENT == "development":
        uvicorn.run(
            "main:app", 
            host="0.0.0.0",
            port=port,
            reload=True,
            log_level="debug"
        )
    else:
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=port,
            workers=4,  # Multiple workers for production
            log_level="info"
        )