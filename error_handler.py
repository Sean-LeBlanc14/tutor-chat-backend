""" Module defines how to handle different errors """
import logging
import traceback
from functools import wraps
from typing import Union
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from security import rate_limiter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class CustomHTTPException(HTTPException):
    """Custom HTTP exception with error codes"""
    def __init__(self, status_code: int, detail: str, error_code: str = None):
        super().__init__(status_code=status_code, detail=detail)
        self.error_code = error_code

async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle Pydantic validation errors"""
    logger.warning("Validation error on %s: %s", request.url, exc.errors())

    # Extract the first error message for user-friendly response
    first_error = exc.errors()[0]
    field = first_error.get('loc', ['unknown'])[-1]
    message = first_error.get('msg', 'Invalid input')

    return JSONResponse(
        status_code=422,
        content={
            "detail": f"Invalid {field}: {message}",
            "error_code": "VALIDATION_ERROR"
        }
    )

async def http_exception_handler(
        request: Request, exc: Union[HTTPException, StarletteHTTPException]):
    """Handle HTTP exceptions"""
    error_code = getattr(exc, 'error_code', None)

    # Log internal server errors
    if exc.status_code >= 500:
        logger.error("Internal error on %s: %s", request.url, exc.detail)
        logger.error(traceback.format_exc())

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "error_code": error_code
        }
    )

async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions"""
    logger.error("Unexpected error on %s: %s", request.url, str(exc))
    logger.error(traceback.format_exc())

    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "error_code": "INTERNAL_ERROR"
        }
    )

def rate_limit(identifier_func):
    """Rate limiting decorator"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract identifier (usually request or user info)
            identifier = identifier_func(*args, **kwargs)

            if rate_limiter.is_rate_limited(identifier):
                rate_limiter.record_attempt(identifier, success=False)
                raise CustomHTTPException(
                    status_code=429,
                    detail="Too many attempts. Please try again later.",
                    error_code="RATE_LIMITED"
                )

            try:
                result = await func(*args, **kwargs)
                rate_limiter.record_attempt(identifier, success=True)
                return result
            except Exception as e:
                rate_limiter.record_attempt(identifier, success=False)
                raise e

        return wrapper
    return decorator

# Usage example for login endpoint:
# @rate_limit(lambda data, response: data.email)
# async def login(data: LoginRequest, response: Response):
#     ...
