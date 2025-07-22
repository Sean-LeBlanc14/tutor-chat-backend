""" Handles all security utilities, login attempts, etc """

import os
import secrets
import hashlib
import logging
from datetime import datetime, timedelta
import re

# Security Configuration
class SecurityConfig:
    """ Session settings """
    SESSION_TOKEN_LENGTH = 64  # Increased from 32
    SESSION_EXPIRE_HOURS = 8   # Reduced from 24 for better security
    MAX_LOGIN_ATTEMPTS = 5
    LOCKOUT_DURATION_MINUTES = 15

    # Input validation
    MAX_MESSAGE_LENGTH = 5000
    MAX_CHAT_TITLE_LENGTH = 100
    MAX_ENVIRONMENT_NAME_LENGTH = 50

    # Password requirements
    MIN_PASSWORD_LENGTH = 12  # Increased from implicit minimum
    PASSWORD_PATTERN = re.compile(
        r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]'
    )

def validate_environment_variables():
    """Validate required environment variables are set"""
    required_vars = [
        "API_KEY",
        "PINECONE_API_KEY", 
        "PINECONE_ENVIRONMENT",
        "DATABASE_URL",
        "JWT_SECRET_KEY",
        "VALID_COURSE_CODE"
    ]

    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise ValueError(f"Missing required environment variables: {missing}")

def create_secure_session_token() -> str:
    """Create a cryptographically secure session token"""
    return secrets.token_urlsafe(SecurityConfig.SESSION_TOKEN_LENGTH)

def validate_password(password: str) -> tuple[bool, str]:
    """Validate password meets security requirements"""
    if len(password) < SecurityConfig.MIN_PASSWORD_LENGTH:
        return False, f"Password must be at least {SecurityConfig.MIN_PASSWORD_LENGTH} characters"

    if not SecurityConfig.PASSWORD_PATTERN.search(password):
        return False, "Password must contain uppercase, lowercase, number, and special character"

    return True, "Password is valid"

def sanitize_input(text: str, max_length: int) -> str:
    """Sanitize and validate user input"""
    if not text:
        return ""

    # Strip whitespace and limit length
    sanitized = text.strip()[:max_length]

    # Remove potentially dangerous characters for basic XSS prevention
    dangerous_chars = ['<', '>', '"', "'", '&']
    for char in dangerous_chars:
        sanitized = sanitized.replace(char, '')

    return sanitized

def hash_token_secure(token: str, salt: str = None) -> str:
    """More secure token hashing with salt"""
    if not salt:
        salt = os.getenv("JWT_SECRET_KEY", "fallback-salt")

    return hashlib.pbkdf2_hmac(
        'sha256',
        token.encode('utf-8'),
        salt.encode('utf-8'),
        100000  # iterations
    ).hex()

class RateLimiter:
    """ Rate limiting utilities """
    def __init__(self):
        self.attempts = {}
        self.lockouts = {}

    def is_rate_limited(self, identifier: str) -> bool:
        """Check if identifier is rate limited"""
        now = datetime.utcnow()

        # Check if currently locked out
        if identifier in self.lockouts:
            if now < self.lockouts[identifier]:
                return True
            else:
                del self.lockouts[identifier]

        return False

    def record_attempt(self, identifier: str, success: bool = False):
        """Record login attempt"""
        now = datetime.utcnow()

        if success:
            # Clear attempts on successful login
            if identifier in self.attempts:
                del self.attempts[identifier]
            return

        # Record failed attempt
        if identifier not in self.attempts:
            self.attempts[identifier] = []

        # Remove old attempts (older than 1 hour)
        self.attempts[identifier] = [
            attempt for attempt in self.attempts[identifier]
            if now - attempt < timedelta(hours=1)
        ]

        self.attempts[identifier].append(now)

        # Check if should be locked out
        if len(self.attempts[identifier]) >= SecurityConfig.MAX_LOGIN_ATTEMPTS:
            self.lockouts[identifier] = now + timedelta(
                minutes=SecurityConfig.LOCKOUT_DURATION_MINUTES
            )
            logging.warning("Rate limit lockout for %s", identifier)

# Global rate limiter instance
rate_limiter = RateLimiter()

# Improved auth.py changes
def validate_email_domain_secure(email: str) -> bool:
    """More robust email validation"""
    if not email or len(email) > 254:  # RFC limit
        return False

    email = email.lower().strip()

    # Basic email format validation
    if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
        return False

    return email.endswith('@csub.edu')

def log_security_event(event_type: str, user_identifier: str, details: str = ""):
    """Log security events for monitoring"""
    logging.warning(
        "SECURITY_EVENT: %s | User: %s | Details: %s", event_type, user_identifier, details
    )
