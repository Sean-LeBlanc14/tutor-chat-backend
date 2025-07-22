""" Module for comprehensive monitoring """
import os
import time
import logging
from typing import Dict, Any
import asyncio
import psutil
from fastapi import Request, Response
from prometheus_client import Counter, Histogram, Gauge
import requests
from query_bot import index
from db import db_manager

# Metrics
REQUEST_COUNT = Counter(
    'http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
ACTIVE_CONNECTIONS = Gauge('active_database_connections', 'Active database connections')
MEMORY_USAGE = Gauge('memory_usage_bytes', 'Memory usage in bytes')
CPU_USAGE = Gauge('cpu_usage_percent', 'CPU usage percentage')

class MetricsMiddleware:
    """ Monitors requests """
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            start_time = time.time()

            # Wrap send to capture response status
            status_code = 500  # Default fallback

            async def send_wrapper(message):
                nonlocal status_code
                if message["type"] == "http.response.start":
                    status_code = message["status"]
                await send(message)

            await self.app(scope, receive, send_wrapper)

            # Record metrics
            duration = time.time() - start_time
            method = scope["method"]
            path = scope["path"]

            REQUEST_COUNT.labels(method=method, endpoint=path, status=status_code).inc()
            REQUEST_DURATION.observe(duration)
        else:
            await self.app(scope, receive, send)

async def update_system_metrics():
    """Update system metrics periodically"""
    while True:
        try:
            # Memory usage
            memory = psutil.virtual_memory()
            MEMORY_USAGE.set(memory.used)

            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            CPU_USAGE.set(cpu_percent)

            await asyncio.sleep(30)  # Update every 30 seconds
        except psutil.Error as e:
            logging.error("Failed to update system metrics: %s", e)
            await asyncio.sleep(60)  # Wait longer on error

class HealthChecker:
    """ Defines all functions related to health checks """
    def __init__(self):
        self.checks = {}

    def register_check(self, name: str, check_func):
        """Register a health check function"""
        self.checks[name] = check_func

    async def run_checks(self) -> Dict[str, Any]:
        """Run all health checks"""
        results = {
            "status": "healthy",
            "checks": {},
            "timestamp": time.time()
        }

        overall_healthy = True

        for name, check_func in self.checks.items():
            try:
                check_result = await check_func()
                results["checks"][name] = {
                    "status": "healthy" if check_result else "unhealthy",
                    "details": check_result
                }
                if not check_result:
                    overall_healthy = False
            except ImportError as e:
                results["checks"][name] = {
                    "status": "error",
                    "error": str(e)
                }
                overall_healthy = False

        results["status"] = "healthy" if overall_healthy else "unhealthy"
        return results

# Health check functions
async def check_database():
    """Check database connectivity"""
    try:
        async with db_manager.get_connection() as conn:
            result = await conn.fetchrow("SELECT 1 as test")
            return result["test"] == 1
    except (ImportError, AttributeError):
        return False

async def check_pinecone():
    """Check Pinecone connectivity"""
    try:
        stats = index.describe_index_stats()
        return stats is not None
    except (ImportError, KeyError):
        return False

async def check_mistral_api():
    """Check Mistral API availability"""
    try:

        url = "https://api.mistral.ai/v1/models"
        headers = {"Authorization": f"Bearer {os.getenv('API_KEY')}"}

        response = requests.get(url, headers=headers, timeout=10)
        return response.status_code == 200
    except KeyError:
        return False

# Global health checker
health_checker = HealthChecker()
health_checker.register_check("database", check_database)
health_checker.register_check("pinecone", check_pinecone)
health_checker.register_check("mistral_api", check_mistral_api)

# Structured logging
class StructuredLogger:
    """ Provides a structured logging system """
    def __init__(self):
        self.logger = logging.getLogger("tutor_chatbot")

    def log_request(self, request: Request, response: Response, duration: float):
        """Log request details"""
        self.logger.info(
            "REQUEST_COMPLETED",
            extra={
                "method": request.method,
                "url": str(request.url),
                "status_code": response.status_code,
                "duration": duration,
                "user_agent": request.headers.get("user-agent"),
                "ip": request.client.host if request.client else None
            }
        )

    def log_security_event(self, event_type: str, details: Dict[str, Any]):
        """Log security events"""
        self.logger.warning("SECURITY_EVENT_%s", event_type, extra=details)

    def log_error(self, error: Exception, context: Dict[str, Any] = None):
        """Log errors with context"""
        self.logger.error(
            "APPLICATION_ERROR",
            extra={
                "error_type": type(error).__name__,
                "error_message": str(error),
                "context": context or {}
            },
            exc_info=True
        )

# Global logger instance
structured_logger = StructuredLogger()
