
# routes/chat.py - Fixed version
""" This module handles all database interactions for the regular chat environment """
from fastapi import APIRouter, HTTPException
from db import db_manager  # Use new db_manager

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
