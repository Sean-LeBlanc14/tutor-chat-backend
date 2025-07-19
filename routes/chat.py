""" This module handles all database interacts for the regular chat environment """
import uuid
from fastapi import APIRouter, HTTPException
from db import get_connection

router = APIRouter()

@router.get("/chats/{user_email}")
async def get_user_chats(user_email: str):
    """ Retrieve all the users previous chats a displays for persistence """
    conn = await get_connection()
    try:
        # Get all chat logs for this user, ordered by creation time
        chat_logs = await conn.fetch("""
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
                    # Use the title from database, fallback to empty string
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
            # (in case different messages have different titles)
            if log['title']:
                chats_dict[chat_id]['title'] = log['title']

        # Sort chats by creation time (most recent first)
        chats_list = list(chats_dict.values())
        chats_list.sort(key=lambda x: x['created_at'], reverse=True)

        return chats_list

    finally:
        await conn.close()

@router.post("/chats/{chat_id}/messages")
async def add_message(chat_id: str, message: dict):
    """ Handles adding a message into an existing chat """
    conn = await get_connection()
    try:
        # Insert new message
        result = await conn.fetchrow("""
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

    finally:
        await conn.close()

@router.put("/chats/{chat_id}")
async def update_chat_title(chat_id: str, data: dict):
    """ Handles updating the title in the database when a user renames it """
    conn = await get_connection()
    try:
        print(f"Updating chat {chat_id} with title: {data['title']}")  # Debug log

        # Update title for all messages in this chat
        result = await conn.execute("""
            UPDATE chat_logs 
            SET title = $1 
            WHERE chat_id = $2
        """, data['title'], chat_id)

        print(f"Update result: {result}")  # Debug log

        # Verify the update worked
        verification = await conn.fetchrow("""
            SELECT title FROM chat_logs 
            WHERE chat_id = $1 
            LIMIT 1
        """, chat_id)

        # Debug log
        print(
            f"""Verification - title after update:
            {verification['title'] if verification else 'NOT FOUND'}"""
        )

        if not verification:
            raise HTTPException(status_code=404, detail="Chat not found")

        return {
            "message": "Title updated successfully",
            "title": verification['title']
        }

    finally:
        await conn.close()

@router.delete("/chats/{chat_id}")
async def delete_chat(chat_id: str, user_email: str):
    """ Handles deleting a chat from the database """
    conn = await get_connection()
    try:
        # Delete all messages in this chat
        await conn.execute("""
            DELETE FROM chat_logs 
            WHERE chat_id = $1 AND user_email = $2
        """, chat_id, user_email)

        return {"message": "Chat deleted successfully"}

    finally:
        await conn.close()

@router.post("/chats")
async def create_new_chat(data: dict):
    """ Handles creating a new chat and adding it to the database """
    conn = await get_connection()
    try:
        # Generate new chat ID
        chat_id = str(uuid.uuid4())

        # Create initial system message to mark chat creation
        await conn.execute("""
            INSERT INTO chat_logs (chat_id, user_email, role, content, mode)
            VALUES ($1, $2, 'system', 'New chat started', 'chat')
        """, chat_id, data['user_email'])

        return {"chat_id": chat_id, "message": "New chat created"}

    finally:
        await conn.close()
