
# routes/chat.py - Fixed version
""" This module handles all database interactions for the regular chat environment """
import asyncio
import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from models import QuestionRequest, ChatResponse
from query_bot import ask_question_stream, ask_question
from error_handler import CustomHTTPException
from db import db_manager

router = APIRouter()

@router.get("/chats/{user_email}")
async def get_user_chats(user_email: str):
    """ Retrieve all the users previous chats and displays for persistence """
    try:
        # Get all chat logs for this user, ordered by creation time
        chat_logs = await db_manager.execute_query("""
            SELECT chat_id, role, content, created_at, id, title
            FROM chat_logs 
            WHERE user_email = $1 AND mode = 'chat'
            ORDER BY created_at ASC
        """, user_email)

        # Group messages by chat_id
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

            # Only add non-system messages to the messages array
            if log['role'] != 'system':
                chats_dict[chat_id]['messages'].append({
                    'id': str(log['id']),
                    'role': log['role'],
                    'content': log['content']
                })

            # Update title if this log has a title
            if log['title']:
                chats_dict[chat_id]['title'] = log['title']

        # Sort chats by creation time (most recent first)
        chats_list = list(chats_dict.values())
        chats_list.sort(key=lambda x: x['created_at'], reverse=True)

        return chats_list

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch chats: {str(e)}") from e

@router.post("/chats/{chat_id}/messages")
async def add_message(chat_id: str, message: dict):
    """ Handles adding a message into an existing chat """
    try:
        # Insert new message
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
        raise HTTPException(status_code=500, detail=f"Failed to add message: {str(e)}") from e

@router.put("/chats/{chat_id}")
async def update_chat_title(chat_id: str, data: dict):
    """ Handles updating the title in the database when a user renames it """
    try:
        # Update title for all messages in this chat
        await db_manager.execute_command("""
            UPDATE chat_logs 
            SET title = $1 
            WHERE chat_id = $2
        """, data['title'], chat_id)

        # Verify the update worked
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
        raise HTTPException(status_code=500, detail=f"Failed to update title: {str(e)}") from e

@router.delete("/chats/{chat_id}")
async def delete_chat(chat_id: str, user_email: str):
    """ Handles deleting a chat from the database """
    try:
        # Delete all messages in this chat
        await db_manager.execute_command("""
            DELETE FROM chat_logs 
            WHERE chat_id = $1 AND user_email = $2
        """, chat_id, user_email)

        return {"message": "Chat deleted successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete chat: {str(e)}") from e


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(data: QuestionRequest):
    """
    Non-streaming chat endpoint for backward compatibility
    """
    try:
        # Get chat history for context (if chat_id provided)
        chat_history = []
        if hasattr(data, 'chat_id') and data.chat_id:
            try:
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
                    if msg["role"] in ['user', 'assistant']
                ]
            except Exception as e:
                logging.warning("Could not fetch chat history: %s", e)

        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            ask_question,
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

@router.post("/chat/stream")
async def chat_stream_endpoint(data: QuestionRequest):
    """
    NEW: Streaming chat endpoint for real-time responses
    """
    try:
        # Get chat history for context (if chat_id provided)
        chat_history = []
        if hasattr(data, 'chat_id') and data.chat_id:
            try:
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
                    if msg["role"] in ['user', 'assistant']
                ]
            except Exception as e:
                logging.warning("Could not fetch chat history: %s", e)

        async def generate_stream():
            try:
                # Stream the response
                loop = asyncio.get_event_loop()
                
                # Create a generator function that yields tokens
                def token_generator():
                    return ask_question_stream(
                        data.question,
                        data.system_prompt,
                        data.temperature,
                        chat_history
                    )
                
                # Run the generator in a thread pool
                generator = await loop.run_in_executor(None, token_generator)
                
                for token in generator:
                    # Format as Server-Sent Events
                    yield f"data: {token}\n\n"
                    
            except Exception as e:
                logging.error("Streaming error: %s", e)
                yield f"data: Error generating response\n\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )

    except Exception as e:
        logging.error("Stream endpoint error: %s", e)
        raise CustomHTTPException(
            status_code=500,
            detail="Internal server error",
            error_code="INTERNAL_ERROR"
        ) from e
