"""
This module creates the FastAPI app, sets up api endpoints for chat, auth, sandbox,
and for sending queries to Mistral
"""
import os
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from query_bot import ask_question
from routes import chat, auth, sandbox

app = FastAPI()

# Environment-dependent CORS settings
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

if ENVIRONMENT == "production":
    # Restrict CORS in production
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[FRONTEND_URL],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )
else:
    # Allow all origins in development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(chat.router, prefix="/api")
app.include_router(auth.router, prefix="/api")
app.include_router(sandbox.router, prefix="/api")

class QuestionRequest(BaseModel):
    """ Outlines the basic format for all prompts sent to Mistral LLM """
    question: str
    system_prompt: Optional[str] = None  # Optional custom prompt for sandbox
    temperature: Optional[float] = 0.7   # Optional temperature override

class ChatResponse(BaseModel):
    """ Outlines the format for all responses made by Mistral LLM"""
    response: str

@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(data: QuestionRequest):
    """
    Main chat endpoint that supports both regular chat and sandbox mode.
    For regular chat: Only send 'question' For sandbox mode:
    Send 'question', 'system_prompt' (natural language), and optionally 'temperature'
    """
    try:
        # Validate input
        if not data.question or not data.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")

        # Use the updated ask_question function with optional parameters
        result = ask_question(
            question=data.question,
            system_prompt=data.system_prompt,
            temperature=data.temperature
        )

        if not result:
            raise HTTPException(status_code=500, detail="Failed to generate response")

        return {"response": result}

    except HTTPException:
        raise
    except Exception as e:
        print(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e

@app.get("/api/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "environment": ENVIRONMENT}

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=ENVIRONMENT == "development")
