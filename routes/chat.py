# routes/chat.py - True async concurrency for classroom scale
""" This module handles all database interactions for the regular chat environment """
import asyncio
import logging
import time
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from models import QuestionRequest, ChatResponse
from query_bot import ask_question_stream, ask_question
from error_handler import CustomHTTPException
from db import db_manager

router = APIRouter()

# Performance monitoring
request_metrics = {
    'total_requests': 0,
    'concurrent_requests': 0,
    'avg_response_time': 0,
    'cache_hits': 0
}

@router.get("/chats/{user_email}")
async def get_user_chats(user_email: str):
    """ Retrieve all the users previous chats and displays for persistence """
    try:
        # Optimized query with LIMIT for performance
        chat_logs = await db_manager.execute_query("""
            SELECT chat_id, role, content, created_at, id, title
            FROM chat_logs
            WHERE user_email = $1 AND mode = 'chat'
            ORDER BY created_at DESC
            LIMIT 100
        """, user_email)

        # Group messages by chat_id (optimized)
        chats_dict = {}
        for log in chat_logs:
            chat_id = str(log['chat_id'])
            if chat_id not in chats_dict:
                chats_dict[chat_id] = {
                    'id': chat_id,
                    'title': log['title'] or '',
                    'messages': [],
                    'created_at': log['created_at'].isoformat()
                }

            # Only add non-system messages
            if log['role'] != 'system':
                chats_dict[chat_id]['messages'].append({
                    'id': str(log['id']),
                    'role': log['role'],
                    'content': log['content']
                })

            if log['title']:
                chats_dict[chat_id]['title'] = log['title']

        # Sort by creation time (most recent first)
        chats_list = list(chats_dict.values())
        chats_list.sort(key=lambda x: x['created_at'], reverse=True)

        # Return only last 20 chats for performance
        return chats_list[:20]

    except Exception as e:
        logging.error(f"Failed to fetch chats for {user_email}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch chats: {str(e)}") from e

@router.post("/chats/{chat_id}/messages")
async def add_message(chat_id: str, message: dict):
    """ Handles adding a message into an existing chat """
    try:
        result = await db_manager.execute_one("""
            INSERT INTO chat_logs (chat_id, user_email, role, content, mode)
            VALUES ($1, $2, $3, $4, 'chat')
            RETURNING id, created_at
        """, chat_id, message['user_email'], message['role'], message['content'])

        return {
            "id": str(result['id']),
            "chat_id": chat_id,
            "role": message['role'],
            "content": message['content'],
            "created_at": result['created_at'].isoformat()
        }

    except Exception as e:
        logging.error(f"Failed to add message to chat {chat_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add message: {str(e)}") from e

@router.put("/chats/{chat_id}")
async def update_chat_title(chat_id: str, data: dict):
    """ Handles updating the title in the database when a user renames it """
    try:
        await db_manager.execute_command("""
            UPDATE chat_logs
            SET title = $1
            WHERE chat_id = $2
        """, data['title'], chat_id)

        verification = await db_manager.execute_one("""
            SELECT title FROM chat_logs
            WHERE chat_id = $1
            LIMIT 1
        """, chat_id)

        if not verification:
            raise HTTPException(status_code=404, detail="Chat not found")

        return {
            "message": "Title updated successfully",
            "title": verification['title']
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Failed to update title for chat {chat_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update title: {str(e)}") from e

@router.delete("/chats/{chat_id}")
async def delete_chat(chat_id: str, user_email: str):
    """ Handles deleting a chat from the database """
    try:
        await db_manager.execute_command("""
            DELETE FROM chat_logs
            WHERE chat_id = $1 AND user_email = $2
        """, chat_id, user_email)

        return {"message": "Chat deleted successfully"}

    except Exception as e:
        logging.error(f"Failed to delete chat {chat_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete chat: {str(e)}") from e

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(data: QuestionRequest):
    """
    Non-streaming chat endpoint - now truly async
    """
    start_time = time.time()
    request_metrics['total_requests'] += 1
    request_metrics['concurrent_requests'] += 1

    try:
        # Optimized chat history retrieval
        chat_history = []
        if hasattr(data, 'chat_id') and data.chat_id:
            try:
                # Limit history for performance
                history_result = await db_manager.execute_query("""
                    SELECT role, content
                    FROM chat_logs
                    WHERE chat_id = $1 AND mode = 'chat'
                    ORDER BY created_at DESC
                    LIMIT 10
                """, data.chat_id)

                chat_history = [
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in reversed(history_result)  # Reverse to get chronological order
                    if msg["role"] in ['user', 'assistant']
                ]
            except Exception as e:
                logging.warning(f"Could not fetch chat history for {data.chat_id}: {e}")

        # Use truly async ask_question function
        result = await ask_question(
            data.question,
            data.system_prompt,
            data.temperature,
            chat_history
        )

        if not result or not result.strip():
            raise CustomHTTPException(
                status_code=500,
                detail="Failed to generate response",
                error_code="GENERATION_FAILED"
            )

        # Update metrics
        response_time = time.time() - start_time
        request_metrics['avg_response_time'] = (
            (request_metrics['avg_response_time'] * (request_metrics['total_requests'] - 1) + response_time) /
            request_metrics['total_requests']
        )

        return ChatResponse(response=result)

    except CustomHTTPException:
        raise
    except Exception as e:
        logging.error(f"Chat endpoint error: {e}")
        raise CustomHTTPException(
            status_code=500,
            detail="Internal server error",
            error_code="INTERNAL_ERROR"
        ) from e
    finally:
        request_metrics['concurrent_requests'] -= 1

@router.post("/chat/stream")
async def chat_stream_endpoint(data: QuestionRequest):
    """
    TRUE async streaming endpoint - no blocking!
    """
    import os
    from starlette.responses import StreamingResponse

    start_time = time.time()
    request_metrics['total_requests'] += 1
    request_metrics['concurrent_requests'] += 1

    try:
        # Optimized chat history retrieval
        chat_history = []
        if hasattr(data, 'chat_id') and data.chat_id:
            try:
                # Reduced history for performance
                history_result = await db_manager.execute_query("""
                    SELECT role, content
                    FROM chat_logs
                    WHERE chat_id = $1 AND mode = 'chat'
                    ORDER BY created_at DESC
                    LIMIT 8
                """, data.chat_id)

                chat_history = [
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in reversed(history_result)
                    if msg["role"] in ['user', 'assistant']
                ]
            except Exception as e:
                logging.warning(f"Could not fetch chat history for {data.chat_id}: {e}")

        async def generate_stream():
            """TRUE async streaming generator - no blocking!"""
            try:
                token_count = 0

                # Direct async streaming - no executors, no blocking!
                async for token in ask_question_stream(
                    data.question,
                    data.system_prompt,
                    data.temperature,
                    chat_history
                ):
                    token_count += 1
                    yield f"data: {token}\n\n"

                    # Optional: Add small delay for client processing
                    if token_count % 20 == 0:
                        await asyncio.sleep(0.001)  # 1ms pause

                # Send completion signal
                yield "data: [DONE]\n\n"

            except Exception as e:
                import traceback
                logging.error(f"Streaming error: {str(e)}")
                logging.error(f"Full traceback:\n{traceback.format_exc()}")
                yield f"data: Error: {str(e)}\n\n"

        # Optimized response with proper headers
        response = StreamingResponse(
            generate_stream(),
            media_type="text/event-stream"
        )

        # Add CORS and caching headers
        response.headers["Access-Control-Allow-Origin"] = os.getenv("FRONTEND_URL", "*")
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Cache-Control"] = "no-cache"
        response.headers["Connection"] = "keep-alive"
        response.headers["X-Accel-Buffering"] = "no"  # Disable nginx buffering

        # Update metrics
        response_time = time.time() - start_time
        request_metrics['avg_response_time'] = (
            (request_metrics['avg_response_time'] * (request_metrics['total_requests'] - 1) + response_time) /
            request_metrics['total_requests']
        )

        return response

    except Exception as e:
        logging.error(f"Stream endpoint error: {e}")
        raise CustomHTTPException(
            status_code=500,
            detail="Internal server error",
            error_code="INTERNAL_ERROR"
        ) from e
    finally:
        request_metrics['concurrent_requests'] -= 1

@router.get("/chat/metrics")
async def get_chat_metrics():
    """Get performance metrics for monitoring"""
    return {
        "total_requests": request_metrics['total_requests'],
        "concurrent_requests": request_metrics['concurrent_requests'],
        "avg_response_time": round(request_metrics['avg_response_time'], 3),
        "cache_hits": request_metrics['cache_hits'],
        "system_status": "operational" if request_metrics['concurrent_requests'] < 20 else "high_load"
    }

@router.post("/chat/batch")
async def batch_chat_endpoint(requests: list[QuestionRequest]):
    """
    Batch processing endpoint - now truly concurrent!
    """
    if len(requests) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 10 requests per batch")

    try:
        # Process requests truly concurrently with async
        tasks = []
        for req in requests:
            task = ask_question(
                req.question,
                req.system_prompt,
                req.temperature,
                []  # No history for batch requests
            )
            tasks.append(task)

        # Execute all requests truly concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Format responses
        responses = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                responses.append({
                    "index": i,
                    "response": f"Error: {str(result)}",
                    "status": "error"
                })
            else:
                responses.append({
                    "index": i,
                    "response": result,
                    "status": "success"
                })

        return {"responses": responses}

    except Exception as e:
        logging.error(f"Batch endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")

# Health check for chat system
@router.get("/chat/health")
async def chat_health_check():
    """Health check specific to chat functionality"""
    try:
        # Quick test of the model - now truly async
        test_response = await ask_question("Hi", temperature=0.1)

        return {
            "status": "healthy",
            "model_responsive": bool(test_response),
            "concurrent_requests": request_metrics['concurrent_requests'],
            "total_requests": request_metrics['total_requests'],
            "avg_response_time": round(request_metrics['avg_response_time'], 3),
            "memory_usage": "optimized"  # Could add actual memory checking
        }
    except Exception as e:
        logging.error(f"Chat health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "concurrent_requests": request_metrics['concurrent_requests']
        }


@router.get("/chat/queue-status")
async def get_queue_status_endpoint():
    """Get current queue and capacity status"""
    from query_bot import get_queue_status
    
    status = get_queue_status()
    
    # Add health indicator
    if status["capacity_percentage"] > 90:
        status["health"] = "critical"
    elif status["capacity_percentage"] > 70:
        status["health"] = "high"
    elif status["capacity_percentage"] > 50:
        status["health"] = "moderate"
    else:
        status["health"] = "good"
    
    # Add wait time estimate
    if status["queue_length"] > 0:
        # Estimate ~2 seconds per queued request
        status["estimated_wait_seconds"] = status["queue_length"] * 2
    else:
        status["estimated_wait_seconds"] = 0
    
    return status

