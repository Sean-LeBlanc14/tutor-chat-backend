# routes/sandbox.py - Complete fixed version
""" This handles all interactions with the database for the sandbox feature """
import uuid
import json
from fastapi import APIRouter, HTTPException, Depends
from db import db_manager  # Use new db_manager
from models import EnvironmentCreate, SessionCreate, SandboxMessage  # Import from models
from routes.auth import get_current_user
import asyncio
import logging
from fastapi.responses import StreamingResponse
from query_bot import ask_question_stream

router = APIRouter()

# Environment Management Endpoints

@router.get("/sandbox/environments")
async def get_environments(current_user = Depends(get_current_user)):
    """Get all sandbox environments (admin only) - all admins can see all environments"""
    if current_user['user_role'] != 'admin':
        raise HTTPException(status_code=403, detail="Admin access required")

    try:
        environments = await db_manager.execute_query("""
            SELECT e.*, u.email as created_by_email
            FROM sandbox_environments e
            LEFT JOIN users u ON e.created_by = u.id
            ORDER BY e.created_at DESC
        """)

        result = []
        for env in environments:
            env_dict = dict(env)
            # Parse JSON string back to dict
            if env_dict['model_config']:
                env_dict['model_config'] = json.loads(env_dict['model_config'])

            # Add a flag to indicate if current user can delete this environment
            env_dict['can_delete'] = env_dict['created_by'] == current_user['id']

            result.append(env_dict)

        return result
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch environments: {str(e)}") from e

@router.post("/sandbox/environments")
async def create_environment(
    env_data: EnvironmentCreate,
    current_user = Depends(get_current_user)
):
    """Create a new sandbox environment (admin only)"""
    if current_user['user_role'] != 'admin':
        raise HTTPException(status_code=403, detail="Admin access required")

    try:
        result = await db_manager.execute_one("""
            INSERT INTO sandbox_environments
            (name, description, system_prompt, model_config, created_by)
            VALUES ($1, $2, $3, $4, $5)
            RETURNING *
        """, env_data.name, env_data.description, env_data.system_prompt,
            json.dumps(env_data.ai_config), current_user['id'])

        return dict(result)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to create environment: {str(e)}") from e

@router.delete("/sandbox/environments/{environment_id}")
async def delete_environment(
    environment_id: str,
    current_user = Depends(get_current_user)
):
    """Delete a sandbox environment (admin only, creator only)"""
    if current_user['user_role'] != 'admin':
        raise HTTPException(status_code=403, detail="Admin access required")

    try:
        # First check if environment exists at all
        env_exists = await db_manager.execute_one("""
            SELECT created_by, u.email as created_by_email
            FROM sandbox_environments e
            LEFT JOIN users u ON e.created_by = u.id
            WHERE e.id = $1
        """, int(environment_id))

        if not env_exists:
            raise HTTPException(status_code=404, detail="Environment not found")

        # Check if current user created it
        if env_exists['created_by'] != current_user['id']:
            creator_email = env_exists['created_by_email'] or 'Unknown'
            raise HTTPException(
                status_code=403,
                detail=f"Only the creator ({creator_email}) can delete this environment"
            )

        # Delete the environment (cascade will handle sessions and messages)
        await db_manager.execute_command("""
            DELETE FROM sandbox_environments WHERE id = $1
        """, environment_id)

        return {"message": "Environment deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to delete environment: {str(e)}") from e

# Session Management Endpoints

@router.get("/sandbox/sessions/{environment_id}")
async def get_sessions(
    environment_id: str,
    current_user = Depends(get_current_user)
):
    """Get all sessions for an environment"""
    if current_user['user_role'] != 'admin':
        raise HTTPException(status_code=403, detail="Admin access required")

    try:
        # Convert environment_id to int
        env_id = int(environment_id)
        
        # Get sessions for this environment and user
        sessions = await db_manager.execute_query("""
            SELECT s.*,
            COALESCE(msg_count.count, 0) as message_count
            FROM sandbox_sessions s
            LEFT JOIN (
                SELECT sandbox_session_id, COUNT(*) as count
                FROM chat_logs
                WHERE mode = 'sandbox' AND role != 'system'
                GROUP BY sandbox_session_id
            ) msg_count ON s.id = msg_count.sandbox_session_id
            WHERE s.environment_id = $1 AND s.user_id = $2
            ORDER BY s.created_at DESC
        """, env_id, current_user['id'])

        # Get messages for each session
        session_ids = [s['id'] for s in sessions]
        if session_ids:
            messages = await db_manager.execute_query("""
                SELECT sandbox_session_id, id, role, content, created_at
                FROM chat_logs
                WHERE sandbox_session_id = ANY($1::integer[]) AND mode = 'sandbox'
                ORDER BY created_at ASC
            """, session_ids)

            # Group messages by session
            grouped_messages = {}
            for msg in messages:
                session_id = msg['sandbox_session_id']
                if session_id not in grouped_messages:
                    grouped_messages[session_id] = []
                
                # Skip system messages from display
                if msg['role'] != 'system':
                    grouped_messages[session_id].append({
                        'id': str(msg['id']),
                        'role': msg['role'],
                        'content': msg['content'],
                        'created_at': msg['created_at'].isoformat()
                    })

            # Format response
            result = []
            for session in sessions:
                session_id = session['id']
                result.append({
                    'id': str(session_id),
                    'title': session['session_name'] or 'Untitled Session',
                    'messages': grouped_messages.get(session_id, []),
                    'created_at': session['created_at'].isoformat()
                })

            return result
        else:
            return []

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch sessions: {str(e)}") from e

@router.post("/sandbox/sessions")
async def create_session(
    session_data: SessionCreate,
    current_user = Depends(get_current_user)
):
    """Create a new sandbox session"""
    if current_user['user_role'] != 'admin':
        raise HTTPException(status_code=403, detail="Admin access required")

    try:
        # Verify environment exists
        env = await db_manager.execute_one("""
            SELECT * FROM sandbox_environments WHERE id = $1
        """, session_data.environment_id)

        if not env:
            raise HTTPException(status_code=404, detail="Environment not found")

        # Create session
        session_name = session_data.session_name or f"Session {uuid.uuid4().hex[:8]}"

        result = await db_manager.execute_one("""
            INSERT INTO sandbox_sessions (environment_id, user_id, session_name)
            VALUES ($1, $2, $3)
            RETURNING *
        """, session_data.environment_id, current_user['id'], session_name)

        return {
            "session_id": str(result['id']),
            "session_name": result['session_name'],
            "message": "Session created successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}") from e

@router.put("/sandbox/sessions/{session_id}")
async def update_session(
    session_id: int,
    update_data: dict,
    current_user = Depends(get_current_user)
):
    """Update session name"""
    if current_user['user_role'] != 'admin':
        raise HTTPException(status_code=403, detail="Admin access required")

    try:
        await db_manager.execute_command("""
            UPDATE sandbox_sessions
            SET session_name = $1
            WHERE id = $2 AND user_id = $3
        """, update_data.get('session_name'), session_id, current_user['id'])

        return {"message": "Session updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update session: {str(e)}") from e

@router.delete("/sandbox/sessions/{session_id}")
async def delete_session(
    session_id: int,
    current_user = Depends(get_current_user)
):
    """Delete a sandbox session"""
    if current_user['user_role'] != 'admin':
        raise HTTPException(status_code=403, detail="Admin access required")

    try:
        # Delete session and related messages
        await db_manager.execute_command("""
            DELETE FROM sandbox_sessions
            WHERE id = $1 AND user_id = $2
        """, session_id, current_user['id'])

        return {"message": "Session deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete session: {str(e)}") from e

@router.post("/sandbox/sessions/{session_id}/messages")
async def add_sandbox_message(
    session_id: int,
    message: SandboxMessage,
    current_user = Depends(get_current_user)
):
    """Add a message to a sandbox session"""
    if current_user['user_role'] != 'admin':
        raise HTTPException(status_code=403, detail="Admin access required")

    try:
        # Verify session exists and belongs to user
        session = await db_manager.execute_one("""
            SELECT * FROM sandbox_sessions
            WHERE id = $1 AND user_id = $2
        """, session_id, current_user['id'])

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Add message
        result = await db_manager.execute_one("""
            INSERT INTO chat_logs
            (chat_id, sandbox_session_id, user_id, role, content, mode)
            VALUES ($1, $2, $3, $4, $5, 'sandbox')
            RETURNING id, created_at
        """, str(session_id), session_id, current_user['id'], message.role, message.content)

        return {
            "id": str(result['id']),
            "session_id": str(session_id),
            "role": message.role,
            "content": message.content,
            "created_at": result['created_at'].isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add message: {str(e)}") from e

@router.get("/sandbox/environments/{environment_id}/details")
async def get_environment_details(
    environment_id: int,
    current_user = Depends(get_current_user)
):
    """Get detailed information about a specific environment"""
    if current_user['user_role'] != 'admin':
        raise HTTPException(status_code=403, detail="Admin access required")

    try:
        # Get environment details
        env = await db_manager.execute_one("""
            SELECT e.*, u.email as created_by_email
            FROM sandbox_environments e
            LEFT JOIN users u ON e.created_by = u.id
            WHERE e.id = $1
        """, environment_id)

        if not env:
            raise HTTPException(status_code=404, detail="Environment not found")

        env_dict = dict(env)
        if env_dict['model_config']:
            env_dict['model_config'] = json.loads(env_dict['model_config'])

        # Get session count for this environment
        session_count = await db_manager.execute_one("""
            SELECT COUNT(*) as count
            FROM sandbox_sessions
            WHERE environment_id = $1
        """, environment_id)

        env_dict['session_count'] = session_count['count'] if session_count else 0
        env_dict['can_delete'] = env_dict['created_by'] == current_user['id']

        return env_dict
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch environment details: {str(e)}") from e

@router.put("/sandbox/environments/{environment_id}")
async def update_environment(
    environment_id: int,
    env_data: EnvironmentCreate,
    current_user = Depends(get_current_user)
):
    """Update a sandbox environment (admin only, creator only)"""
    if current_user['user_role'] != 'admin':
        raise HTTPException(status_code=403, detail="Admin access required")

    try:
        # Check if environment exists and user owns it
        env_exists = await db_manager.execute_one("""
            SELECT created_by FROM sandbox_environments WHERE id = $1
        """, environment_id)

        if not env_exists:
            raise HTTPException(status_code=404, detail="Environment not found")

        if env_exists['created_by'] != current_user['id']:
            raise HTTPException(
                status_code=403,
                detail="Only the creator can modify this environment"
            )

        # Update environment
        result = await db_manager.execute_one("""
            UPDATE sandbox_environments
            SET name = $1, description = $2, system_prompt = $3,
                model_config = $4, updated_at = CURRENT_TIMESTAMP
            WHERE id = $5
            RETURNING *
        """, env_data.name, env_data.description, env_data.system_prompt,
            json.dumps(env_data.ai_config), environment_id)

        if result and result['model_config']:
            result_dict = dict(result)
            result_dict['model_config'] = json.loads(result_dict['model_config'])
            return result_dict

        return dict(result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to update environment: {str(e)}") from e

@router.post("/sandbox/{session_id}/chat/stream")
async def sandbox_chat_stream(
    session_id: int,
    message: SandboxMessage,
    current_user = Depends(get_current_user)
):
    """Streaming chat endpoint for sandbox sessions"""
    if current_user['user_role'] != 'admin':
        raise HTTPException(status_code=403, detail="Admin access required")

    try:
        # Get session and environment details
        session_data = await db_manager.execute_one("""
            SELECT s.*, e.system_prompt, e.model_config
            FROM sandbox_sessions s
            JOIN sandbox_environments e ON s.environment_id = e.id
            WHERE s.id = $1 AND s.user_id = $2
        """, session_id, current_user['id'])

        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")

        # Save user message first
        await db_manager.execute_command("""
            INSERT INTO chat_logs (chat_id, sandbox_session_id, user_id, role, content, mode)
            VALUES ($1, $2, $3, 'user', $4, 'sandbox')
        """, str(session_id), session_id, current_user['id'], message.content)

        # Get chat history for this session
        chat_history = await db_manager.execute_query("""
            SELECT role, content
            FROM chat_logs
            WHERE sandbox_session_id = $1 AND mode = 'sandbox'
            ORDER BY created_at ASC
            LIMIT 20
        """, session_id)

        chat_history_list = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in chat_history
            if msg["role"] in ['user', 'assistant']
        ]

        async def generate_stream():
            full_response = ""
            try:
                loop = asyncio.get_event_loop()

                def token_generator():
                    return ask_question_stream(
                        message.content,
                        session_data['system_prompt'],  # Use environment's system prompt
                        0.7,  # Could get from model_config
                        chat_history_list
                    )

                generator = await loop.run_in_executor(None, token_generator)

                for token in generator:
                    full_response += token
                    yield f"data: {token}\n\n"

                # Save assistant response
                await db_manager.execute_command("""
                    INSERT INTO chat_logs (chat_id, sandbox_session_id, user_id, role, content, mode)
                    VALUES ($1, $2, $3, 'assistant', $4, 'sandbox')
                """, str(session_id), session_id, current_user['id'], full_response)

                yield "data: [DONE]\n\n"

            except Exception as e:
                logging.error("Sandbox streaming error: %s", e)
                yield f"data: Error generating response\n\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logging.error("Sandbox stream endpoint error: %s", e)
        raise HTTPException(status_code=500, detail="Internal server error") from e
