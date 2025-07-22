""" Defines the data models for the application """
import re
from typing import Optional, Dict, Any
from pydantic import BaseModel, EmailStr, Field, field_validator

class SignupRequest(BaseModel):
    """ Defines data model for signing up """
    email: EmailStr = Field(..., description="Valid @csub.edu email address")
    password: str = Field(..., min_length=12, max_length=128)
    user_role: Optional[str] = Field(default="student", pattern="^(student|admin)$")
    course_code: str = Field(..., min_length=1, max_length=20)

    @field_validator('email')
    @classmethod
    def validate_email_domain(cls, v):
        """ Validates emails, checks if csub """
        if not v.lower().endswith('@csub.edu'):
            raise ValueError('Email must be a @csub.edu address')
        return v.lower()

    @field_validator('password')
    @classmethod
    def validate_password_strength(cls, v):
        """ Validates password strength """
        if len(v) < 12:
            raise ValueError('Password must be at least 12 characters')

        pattern = re.compile(r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]')
        if not pattern.search(v):
            raise ValueError(
                'Password must contain uppercase, lowercase, number, and special character')

        return v

class LoginRequest(BaseModel):
    """ Defines data for login """
    email: EmailStr
    password: str = Field(..., min_length=1, max_length=128)
    course_code: str = Field(..., min_length=1, max_length=20)

    @field_validator('email')
    @classmethod
    def validate_email_domain(cls, v):
        """ Validates email addresses """
        if not v.lower().endswith('@csub.edu'):
            raise ValueError('Email must be a @csub.edu address')
        return v.lower()

# Enhanced models.py with chat_id support
class QuestionRequest(BaseModel):
    """ Enhanced question request with optional chat context """
    question: str = Field(..., min_length=1, max_length=5000, description="User's question")
    system_prompt: Optional[str] = Field(
        None, max_length=2000, description="Optional custom prompt")
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0, description="AI temperature")
    chat_id: Optional[str] = Field(None, description="Optional chat ID for context")

    @field_validator('question')
    @classmethod
    def validate_question(cls, v):
        """ Makes sure the question is valid """
        if not v.strip():
            raise ValueError('Question cannot be empty')
        return v.strip()[:5000]

class ChatTitleUpdate(BaseModel):
    """ Defines data for a title rename """
    title: str = Field(..., min_length=1, max_length=100)

    @field_validator('title')
    @classmethod
    def validate_title(cls, v):
        """ Makes sure tite is valid """
        return v.strip()[:100]

class EnvironmentCreate(BaseModel):
    """ Defines data to create an environment """
    name: str = Field(..., min_length=1, max_length=50)
    description: Optional[str] = Field(None, max_length=500)
    system_prompt: str = Field(..., min_length=1, max_length=2000)
    ai_config: Optional[Dict[str, Any]] = Field(default={"temperature": 0.7})

    @field_validator('name', 'description')
    @classmethod
    def validate_strings(cls, v):
        """ Validates that the entries are good """
        if v is not None:
            return v.strip()
        return v

class SessionCreate(BaseModel):
    """ Defines the data for a session within an environment """
    environment_id: str = Field(..., pattern=r'^[0-9a-f-]{36}$')  # UUID format
    session_name: Optional[str] = Field(None, max_length=100)

    @field_validator('session_name')
    @classmethod
    def validate_session_name(cls, v):
        """ Makes sure the name isn't empty """
        if v is not None:
            return v.strip()[:100]
        return v

class SandboxMessage(BaseModel):
    """ Defines data for a message in sandbox """
    role: str = Field(..., pattern="^(user|assistant|system)$")
    content: str = Field(..., min_length=1, max_length=5000)

    @field_validator('content')
    @classmethod
    def validate_content(cls, v):
        """ Validates the message is good """
        return v.strip()[:5000]

# Response models
class UserResponse(BaseModel):
    """ Defines data for the response associated with a user """
    id: str
    email: str
    user_role: str

class ChatResponse(BaseModel):
    """ Defines data for the chatbot's response """
    response: str

class SuccessResponse(BaseModel):
    """ Defines data for a successful response """
    message: str

class ErrorResponse(BaseModel):
    """ Defines data for an unsuccessful response """
    detail: str
    error_code: Optional[str] = None
