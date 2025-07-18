from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, Dict, Any
from db import get_connection
from routes.auth import get_current_user
import uuid
import json

router = APIRouter()

class EnvironmentCreate(BaseModel):
    name: str
    description: Optional[str] = None
    system_prompt: str
    model_config: Optional[Dict[str, Any]] = {"temperature": 0.7}

class SessionCreate(BaseModel):
    environment_id: str
    session_name: Optional[str] = None

class SandboxMessage(BaseModel):
    role: str
    content: str

# Environment Management Endpoints

@router.get("/sandbox/environments")
async def get_environments(current_user = Depends(get_current_user)):
    """Get all sandbox environments (admin only) - all admins can see all environments"""
    if current_user['user_role'] != 'admin':
        raise HTTPException(status_code=403, detail="Admin access required")
    
    conn = await get_connection()
    try:
        environments = await conn.fetch("""
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
            env_dict['can_delete'] = (env_dict['created_by'] == current_user['id'])
            
            result.append(env_dict)
        
        return result
    finally:
        await conn.close()

@router.post("/sandbox/environments")
async def create_environment(
    env_data: EnvironmentCreate, 
    current_user = Depends(get_current_user)
):
    """Create a new sandbox environment (admin only)"""
    if current_user['user_role'] != 'admin':
        raise HTTPException(status_code=403, detail="Admin access required")
    
    conn = await get_connection()
    try:
        result = await conn.fetchrow("""
            INSERT INTO sandbox_environments 
            (name, description, system_prompt, model_config, created_by)
            VALUES ($1, $2, $3, $4, $5)
            RETURNING *
        """, env_data.name, env_data.description, env_data.system_prompt, 
            json.dumps(env_data.model_config), current_user['id'])
        
        return dict(result)
    finally:
        await conn.close()

@router.delete("/sandbox/environments/{environment_id}")
async def delete_environment(
    environment_id: str, 
    current_user = Depends(get_current_user)
):
    """Delete a sandbox environment (admin only, creator only)"""
    if current_user['user_role'] != 'admin':
        raise HTTPException(status_code=403, detail="Admin access required")
    
    conn = await get_connection()
    try:
        # First check if environment exists at all
        env_exists = await conn.fetchrow("""
            SELECT created_by, u.email as created_by_email 
            FROM sandbox_environments e
            LEFT JOIN users u ON e.created_by = u.id
            WHERE e.id = $1
        """, environment_id)
        
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
        await conn.execute("""
            DELETE FROM sandbox_environments WHERE id = $1
        """, environment_id)
        
        return {"message": "Environment deleted successfully"}
    finally:
        await conn.close()

# Session Management Endpoints

@router.get("/sandbox/sessions/{environment_id}")
async def get_sessions(
    environment_id: str, 
    current_user = Depends(get_current_user)
):
    """Get all sessions for an environment"""
    if current_user['user_role'] != 'admin':
        raise HTTPException(status_code=403, detail="Admin access required")
    
    conn = await get_connection()
    try:
        # Get sessions for this environment and user
        sessions = await conn.fetch("""
            SELECT s.*, 
                   COALESCE(msg_count.count, 0) as message_count
            FROM sandbox_sessions s
            LEFT JOIN (
                SELECT sandbox_session_id, COUNT(*) as count
                FROM chat_logs 
                WHERE mode = 'sandbox'
                GROUP BY sandbox_session_id
            ) msg_count ON s.id = msg_count.sandbox_session_id
            WHERE s.environment_id = $1 AND s.user_id = $2
            ORDER BY s.created_at DESC
        """, environment_id, current_user['id'])
        
        # Get messages for each session
        session_ids = [str(s['id']) for s in sessions]
        if session_ids:
            messages = await conn.fetch("""
                SELECT chat_id, id, role, content, created_at
                FROM chat_logs 
                WHERE sandbox_session_id = ANY($1::uuid[]) AND mode = 'sandbox'
                ORDER BY created_at ASC
            """, session_ids)
            
            # Group messages by session
            grouped_messages = {}
            for msg in messages:
                session_id = str(msg['chat_id'])
                if session_id not in grouped_messages:
                    grouped_messages[session_id] = []
                grouped_messages[session_id].append({
                    'id': str(msg['id']),
                    'role': msg['role'],
                    'content': msg['content'],
                    'created_at': msg['created_at'].isoformat()
                })
            
            # Format response
            result = []
            for session in sessions:
                session_id = str(session['id'])
                result.append({
                    'id': session_id,
                    'title': session['session_name'] or 'Untitled Session',
                    'messages': grouped_messages.get(session_id, []),
                    'created_at': session['created_at'].isoformat()
                })
            
            return result
        else:
            return []
            
    finally:
        await conn.close()

@router.post("/sandbox/sessions")
async def create_session(
    session_data: SessionCreate, 
    current_user = Depends(get_current_user)
):
    """Create a new sandbox session"""
    if current_user['user_role'] != 'admin':
        raise HTTPException(status_code=403, detail="Admin access required")
    
    conn = await get_connection()
    try:
        # Verify environment exists
        env = await conn.fetchrow("""
            SELECT * FROM sandbox_environments WHERE id = $1
        """, session_data.environment_id)
        
        if not env:
            raise HTTPException(status_code=404, detail="Environment not found")
        
        # Create session
        session_name = session_data.session_name or f"Session {uuid.uuid4().hex[:8]}"
        
        result = await conn.fetchrow("""
            INSERT INTO sandbox_sessions (environment_id, user_id, session_name)
            VALUES ($1, $2, $3)
            RETURNING *
        """, session_data.environment_id, current_user['id'], session_name)
        
        # Create initial system message
        await conn.execute("""
            INSERT INTO chat_logs (chat_id, sandbox_session_id, user_id, role, content, mode)
            VALUES ($1, $1, $2, 'system', 'New sandbox session started', 'sandbox')
        """, result['id'], current_user['id'])
        
        return {
            "session_id": str(result['id']),
            "session_name": result['session_name'],
            "message": "Session created successfully"
        }
    finally:
        await conn.close()

@router.put("/sandbox/sessions/{session_id}")
async def update_session(
    session_id: str, 
    update_data: dict,
    current_user = Depends(get_current_user)
):
    """Update session name"""
    if current_user['user_role'] != 'admin':
        raise HTTPException(status_code=403, detail="Admin access required")
    
    conn = await get_connection()
    try:
        await conn.execute("""
            UPDATE sandbox_sessions 
            SET session_name = $1 
            WHERE id = $2 AND user_id = $3
        """, update_data.get('session_name'), session_id, current_user['id'])
        
        return {"message": "Session updated successfully"}
    finally:
        await conn.close()

@router.delete("/sandbox/sessions/{session_id}")
async def delete_session(
    session_id: str, 
    current_user = Depends(get_current_user)
):
    """Delete a sandbox session"""
    if current_user['user_role'] != 'admin':
        raise HTTPException(status_code=403, detail="Admin access required")
    
    conn = await get_connection()
    try:
        # Delete session and related messages
        await conn.execute("""
            DELETE FROM sandbox_sessions 
            WHERE id = $1 AND user_id = $2
        """, session_id, current_user['id'])
        
        return {"message": "Session deleted successfully"}
    finally:
        await conn.close()

@router.post("/sandbox/sessions/{session_id}/messages")
async def add_sandbox_message(
    session_id: str,
    message: SandboxMessage,
    current_user = Depends(get_current_user)
):
    """Add a message to a sandbox session"""
    if current_user['user_role'] != 'admin':
        raise HTTPException(status_code=403, detail="Admin access required")
    
    conn = await get_connection()
    try:
        # Verify session exists and belongs to user
        session = await conn.fetchrow("""
            SELECT * FROM sandbox_sessions 
            WHERE id = $1 AND user_id = $2
        """, session_id, current_user['id'])
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Add message
        result = await conn.fetchrow("""
            INSERT INTO chat_logs 
            (chat_id, sandbox_session_id, user_id, role, content, mode)
            VALUES ($1, $1, $2, $3, $4, 'sandbox')
            RETURNING id, created_at
        """, session_id, current_user['id'], message.role, message.content)
        
        return {
            "id": str(result['id']),
            "session_id": session_id,
            "role": message.role,
            "content": message.content,
            "created_at": result['created_at'].isoformat()
        }
    finally:
        await conn.close()