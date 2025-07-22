""" This module handles all login and signup authentication for the application """
import os
from datetime import datetime, timedelta
import secrets
import hashlib
import bcrypt
import asyncpg
from fastapi import APIRouter, HTTPException, Depends, Response, Request

# Import from your new modules
from db import db_manager  # Use the new db_manager instead of get_connection
from models import SignupRequest, LoginRequest  # Use models from models.py
from security import log_security_event
from error_handler import rate_limit

router = APIRouter()

# Session configuration
SESSION_EXPIRE_HOURS = 8  # Reduced for security
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-fallback-secret-key")
VALID_COURSE_CODE = os.getenv("VALID_COURSE_CODE")

# Validate required environment variables
if not VALID_COURSE_CODE:
    raise ValueError("VALID_COURSE_CODE environment variable must be set")

# Environment-dependent settings
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
IS_PRODUCTION = ENVIRONMENT == "production"

def create_session_token():
    """Create a secure session token"""
    return secrets.token_urlsafe(64)  # Increased length

def hash_token(token: str) -> str:
    """Hash token for secure storage"""
    return hashlib.pbkdf2_hmac(
        'sha256',
        token.encode('utf-8'),
        SECRET_KEY.encode('utf-8'),
        100000
    ).hex()

async def store_session(user_id: str, token_hash: str):
    """Store session in database with proper connection handling"""
    try:
        expire_time = datetime.utcnow() + timedelta(hours=SESSION_EXPIRE_HOURS)
        await db_manager.execute_command("""
            INSERT INTO user_sessions (user_id, token_hash, expires_at)
            VALUES ($1, $2, $3)
            ON CONFLICT (user_id) 
            DO UPDATE SET token_hash = $2, expires_at = $3
        """, user_id, token_hash, expire_time)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to create session") from e

async def get_current_user(request: Request):
    """Get current user from session token"""
    token = request.cookies.get("session_token")

    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")

    token_hash = hash_token(token)

    try:
        # Get user from session
        result = await db_manager.execute_one("""
            SELECT u.id, u.email, u.user_role, s.expires_at
            FROM users u
            JOIN user_sessions s ON u.id = s.user_id
            WHERE s.token_hash = $1 AND s.expires_at > NOW()
        """, token_hash)

        if not result:
            raise HTTPException(status_code=401, detail="Invalid or expired session")

        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to verify session") from e

def validate_course_code(provided_code: str) -> bool:
    """Validate course code against server-side value"""
    return provided_code.strip() == VALID_COURSE_CODE.strip()

def validate_email_domain(email: str) -> bool:
    """Validate email domain"""
    return email.lower().endswith('@csub.edu')

def set_secure_cookie(response: Response, session_token: str):
    """Set secure cookie with environment-appropriate settings"""
    response.set_cookie(
        key="session_token",
        value=session_token,
        httponly=True,
        secure=IS_PRODUCTION,  # Only secure in production
        samesite="lax",
        max_age=SESSION_EXPIRE_HOURS * 3600
    )

@router.post("/signup")
@rate_limit(lambda data, response: data.email)
async def signup(data: SignupRequest, response: Response):
    """Handles making sure signup information is correct and adding new users to database"""
    try:
        # Validate email domain
        if not validate_email_domain(data.email):
            log_security_event("INVALID_EMAIL_DOMAIN", data.email)
            raise HTTPException(status_code=400, detail="Email must be a @csub.edu address")

        # Validate course code
        if not validate_course_code(data.course_code):
            log_security_event("INVALID_COURSE_CODE", data.email)
            raise HTTPException(status_code=400, detail="Invalid course code")

        # Check if user already exists
        existing_user = await db_manager.execute_one(
            "SELECT * FROM users WHERE email = $1", data.email
        )
        if existing_user:
            raise HTTPException(status_code=400, detail="User already exists")

        # Hash password
        hashed_password = bcrypt.hashpw(
            data.password.encode("utf-8"), bcrypt.gensalt()
        ).decode("utf-8")

        # Insert user
        result = await db_manager.execute_one("""
            INSERT INTO users (email, hashed_password, user_role, course_code)
            VALUES ($1, $2, $3, $4)
            RETURNING id, email, user_role
        """, data.email, hashed_password, data.user_role, data.course_code)

        # Create session
        session_token = create_session_token()
        token_hash = hash_token(session_token)
        await store_session(str(result['id']), token_hash)

        # Set secure cookie
        set_secure_cookie(response, session_token)

        log_security_event("USER_SIGNUP_SUCCESS", data.email)

        return {
            "message": "User created successfully",
            "user": {
                "id": str(result['id']),
                "email": result['email'],
                "user_role": result['user_role']
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        log_security_event("USER_SIGNUP_ERROR", data.email, str(e))
        raise HTTPException(status_code=500, detail="Failed to create user") from e

@router.post("/login")
@rate_limit(lambda data, response: data.email)
async def login(data: LoginRequest, response: Response):
    """Handles making sure login information is correct"""
    try:
        # Validate email domain
        if not validate_email_domain(data.email):
            log_security_event("INVALID_EMAIL_DOMAIN", data.email)
            raise HTTPException(status_code=400, detail="Email must be a @csub.edu address")

        # Validate course code
        if not validate_course_code(data.course_code):
            log_security_event("INVALID_COURSE_CODE", data.email)
            raise HTTPException(status_code=400, detail="Invalid course code")

        # Get user by email
        user = await db_manager.execute_one(
            "SELECT * FROM users WHERE email = $1", data.email
        )
        if not user:
            log_security_event("LOGIN_FAILED_USER_NOT_FOUND", data.email)
            raise HTTPException(status_code=401, detail="Invalid credentials")

        # Check password
        if not bcrypt.checkpw(
            data.password.encode("utf-8"), user['hashed_password'].encode("utf-8")
        ):
            log_security_event("LOGIN_FAILED_WRONG_PASSWORD", data.email)
            raise HTTPException(status_code=401, detail="Invalid credentials")

        # Create session
        session_token = create_session_token()
        token_hash = hash_token(session_token)
        await store_session(str(user['id']), token_hash)

        # Set secure cookie
        set_secure_cookie(response, session_token)

        log_security_event("LOGIN_SUCCESS", data.email)

        return {
            "message": "Login successful",
            "user": {
                "id": str(user['id']),
                "email": user['email'],
                "user_role": user['user_role']
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        log_security_event("LOGIN_ERROR", data.email, str(e))
        raise HTTPException(status_code=500, detail="Login failed") from e

@router.post("/logout")
async def logout(response: Response, current_user = Depends(get_current_user)):
    """Handles logging users out of their current session"""
    try:
        await db_manager.execute_command(
            "DELETE FROM user_sessions WHERE user_id = $1", 
            current_user['id']
        )
        log_security_event("LOGOUT_SUCCESS", current_user['email'])
    except asyncpg.PostgresError as e:
        # Log error but don't fail logout
        log_security_event("LOGOUT_ERROR", current_user['email'], str(e))

    # Clear cookie
    response.delete_cookie(key="session_token")
    return {"message": "Logged out successfully"}

@router.get("/me")
async def get_current_user_info(current_user = Depends(get_current_user)):
    """Returns necessary user information needed"""
    return {
        "id": str(current_user['id']),
        "email": current_user['email'],
        "user_role": current_user['user_role']
    }
