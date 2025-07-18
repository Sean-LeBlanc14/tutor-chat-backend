from fastapi import APIRouter, HTTPException, Depends, Response, Request, Cookie
from pydantic import BaseModel, EmailStr
from db import get_connection
import bcrypt
import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Optional
import os

router = APIRouter()

# Session configuration
SESSION_EXPIRE_HOURS = 24
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-fallback-secret-key")
VALID_COURSE_CODE = os.getenv("VALID_COURSE_CODE")

# Validate required environment variables
if not VALID_COURSE_CODE:
    raise ValueError("VALID_COURSE_CODE environment variable must be set")

# Environment-dependent settings
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
IS_PRODUCTION = ENVIRONMENT == "production"

class SignupRequest(BaseModel):
    email: EmailStr
    password: str
    user_role: Optional[str] = "student"
    course_code: str  # Required for validation

class LoginRequest(BaseModel):
    email: EmailStr
    password: str
    course_code: str  # Required for validation

def create_session_token():
    """Create a secure session token"""
    return secrets.token_urlsafe(32)

def hash_token(token: str) -> str:
    """Hash token for secure storage"""
    return hashlib.sha256((token + SECRET_KEY).encode()).hexdigest()

async def store_session(user_id: str, token_hash: str):
    """Store session in database with proper connection handling"""
    conn = await get_connection()
    try:
        expire_time = datetime.utcnow() + timedelta(hours=SESSION_EXPIRE_HOURS)
        await conn.execute("""
            INSERT INTO user_sessions (user_id, token_hash, expires_at)
            VALUES ($1, $2, $3)
            ON CONFLICT (user_id) 
            DO UPDATE SET token_hash = $2, expires_at = $3
        """, user_id, token_hash, expire_time)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to create session")
    finally:
        await conn.close()

async def get_current_user(request: Request):
    """Get current user from session token"""
    token = request.cookies.get("session_token")
    
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    token_hash = hash_token(token)
    
    conn = await get_connection()
    try:
        # Get user from session
        result = await conn.fetchrow("""
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
        raise HTTPException(status_code=500, detail="Failed to verify session")
    finally:
        await conn.close()

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
async def signup(data: SignupRequest, response: Response):
    # Validate email domain
    if not validate_email_domain(data.email):
        raise HTTPException(status_code=400, detail="Email must be a @csub.edu address")
    
    # Validate course code
    if not validate_course_code(data.course_code):
        raise HTTPException(status_code=400, detail="Invalid course code")
    
    conn = await get_connection()
    try:
        # Check if user already exists
        existing_user = await conn.fetchrow("SELECT * FROM users WHERE email = $1", data.email)
        if existing_user:
            raise HTTPException(status_code=400, detail="User already exists")
        
        # Hash password
        hashed_password = bcrypt.hashpw(data.password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

        # Insert user (course_code stored but not returned for privacy)
        result = await conn.fetchrow("""
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

        return {
            "message": "User created successfully",
            "user": {
                "id": str(result['id']),
                "email": result['email'],
                "user_role": result['user_role']
                # course_code intentionally omitted for privacy
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to create user")
    finally:
        await conn.close()

@router.post("/login")
async def login(data: LoginRequest, response: Response):
    # Validate email domain
    if not validate_email_domain(data.email):
        raise HTTPException(status_code=400, detail="Email must be a @csub.edu address")
    
    # Validate course code
    if not validate_course_code(data.course_code):
        raise HTTPException(status_code=400, detail="Invalid course code")
    
    conn = await get_connection()
    try:
        # Get user by email
        user = await conn.fetchrow("SELECT * FROM users WHERE email = $1", data.email)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        # Check password
        if not bcrypt.checkpw(data.password.encode("utf-8"), user['hashed_password'].encode("utf-8")):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        # Create session
        session_token = create_session_token()
        token_hash = hash_token(session_token)
        await store_session(str(user['id']), token_hash)
        
        # Set secure cookie
        set_secure_cookie(response, session_token)

        return {
            "message": "Login successful",
            "user": {
                "id": str(user['id']),
                "email": user['email'],
                "user_role": user['user_role']
                # course_code intentionally omitted for privacy
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Login failed")
    finally:
        await conn.close()

@router.post("/logout")
async def logout(response: Response, current_user = Depends(get_current_user)):
    # Delete session from database
    conn = await get_connection()
    try:
        await conn.execute("DELETE FROM user_sessions WHERE user_id = $1", current_user['id'])
    except Exception as e:
        # Log error but don't fail logout
        print(f"Failed to delete session: {e}")
    finally:
        await conn.close()
    
    # Clear cookie
    response.delete_cookie(key="session_token")
    return {"message": "Logged out successfully"}

@router.get("/me")
async def get_current_user_info(current_user = Depends(get_current_user)):
    return {
        "id": str(current_user['id']),
        "email": current_user['email'],
        "user_role": current_user['user_role']
        # course_code intentionally omitted for privacy
    }