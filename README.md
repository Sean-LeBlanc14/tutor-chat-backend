# Psychology Tutor Chatbot — Backend

A production-deployed FastAPI backend powering an AI tutoring platform built for Dr. Maruti Mishra's Sensation and Perception lab course at California State University, Bakersfield. Presented as a research poster at AIxHEArt 2025 (25% acceptance rate).

**Frontend Repository:** [tutor-chat-frontend](https://github.com/Sean-LeBlanc14/tutor-chat-frontend)

---

## Overview

An intelligent tutoring system that uses Retrieval-Augmented Generation (RAG) to answer student questions using course-specific materials. The system distinguishes between conceptual questions (answered via Socratic method) and administrative questions (answered directly), routing each to the appropriate response strategy.

---

## Technical Stack

| Category | Technologies |
|---|---|
| Backend | FastAPI, Python 3.11, Pydantic |
| Database | PostgreSQL, asyncpg (connection pooling) |
| AI/ML | vLLM (AsyncLLMEngine), Llama 3.1 8B Instruct, Sentence Transformers (multi-qa-mpnet-base-dot-v1), FAISS |
| Security | JWT (httpOnly cookies), bcrypt, IP-based rate limiting, CORS, TrustedHost middleware |
| DevOps | Docker (multi-stage, non-root), Uvicorn, Gunicorn |
| Monitoring | Prometheus metrics, psutil, structured logging |

---

## Architecture

```
User Question
     |
     v
FastAPI (rate limiting, auth middleware)
     |
     v
Question Classifier (conceptual / administrative / casual / off-topic)
     |
     +---> RAG Pipeline (if needed)
     |          |
     |          v
     |     Sentence Transformers --> FAISS vector search --> course material chunks
     |
     v
Dynamic token budgeting (context window management)
     |
     v
vLLM AsyncLLMEngine --> Llama 3.1 8B (GPU-accelerated)
     |
     v
SSE stream --> Frontend
     |
     v
PostgreSQL (persist chat history via asyncpg)
```

---

## Key Engineering Details

### RAG Pipeline

Questions are classified into types (conceptual, administrative, casual, test question) before retrieval. Administrative questions (office hours, deadlines, grades) trigger direct retrieval from course syllabi. Conceptual questions use semantic search via FAISS with `multi-qa-mpnet-base-dot-v1` embeddings, with adaptive `k` depending on question complexity.

Lab queries use an enhanced retrieval path that appends semantic keywords before embedding to improve chunk recall for course-specific terminology.

### Dynamic Token Budgeting

Each request dynamically allocates the available context window across the system prompt, RAG context, and chat history. Token counts are computed using the model's HuggingFace tokenizer, with binary trimming applied if the prompt exceeds the budget. This prevents context overflow at classroom scale without sacrificing response quality.

### Async Concurrency

The backend uses vLLM's `AsyncLLMEngine` for true async inference — requests are processed concurrently rather than queued behind a blocking model call. A custom `RequestQueue` manages up to 25 concurrent requests with a queue depth of 75, designed for simultaneous classroom use.

### Authentication

JWT tokens are issued on login and stored in httpOnly cookies, preventing JavaScript access and protecting against XSS. All authenticated routes verify the token via a `/api/me` session check on page load.

### Socratic Tutoring System

The system prompt encodes a full decision tree: administrative questions receive direct answers from context, while conceptual questions always use the Socratic method regardless of whether the answer is available. The prompt includes a progression pattern (broad question → focused question → concept connection → hint) and handles student frustration gracefully.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/chat/stream` | Main tutoring interface (SSE streaming) |
| GET | `/api/chats/{user_email}` | Retrieve conversation history |
| POST | `/api/auth/login` | JWT authentication |
| POST | `/api/auth/signup` | User registration |
| GET | `/api/me` | Session verification |
| GET | `/api/health` | Health check (DB, model, GPU, system) |
| GET | `/api/metrics` | Prometheus + system metrics |
| POST | `/api/sandbox/environments` | Create custom AI configurations (admin) |
| GET | `/api/sandbox/sessions/{env_id}` | Manage sandbox sessions (admin) |

---

## Deployment

The backend is containerized using a multi-stage Docker build with a non-root user for security. It runs as a single Uvicorn worker (required for GPU memory sharing) with uvloop and httptools for production performance. FAISS index files (`faiss_index.bin`, `faiss_metadata.pkl`) are built separately via `embed_chunks.py` and copied into the container at build time.

GPU support requires CUDA 12.x. CPU fallback uses a smaller HuggingFace model via Transformers for development environments.

---

## Contact

**Sean LeBlanc-Grappendorf** — CS Student, Cal Poly San Luis Obispo

- Email: seanaugustlg2006@gmail.com
- LinkedIn: [linkedin.com/in/sean-leblanc-grappendorf-6045a8331](https://www.linkedin.com/in/sean-leblanc-grappendorf-6045a8331/)
- Portfolio: [seanlg.com](https://seanlg.com/)
- GitHub: [github.com/Sean-LeBlanc14](https://github.com/Sean-LeBlanc14)
