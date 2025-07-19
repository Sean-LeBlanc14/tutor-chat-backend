"""
This module connects provides the connection to the postgres db
"""
import os
import asyncpg
from dotenv import load_dotenv

load_dotenv()
DB_URL = os.getenv("DATABASE_URL")

async def get_connection():
    """ This function connects to the database that stores our user information """
    try:
        print(f"🔍 Connecting to: {DB_URL}")
        conn = await asyncpg.connect(DB_URL)
        # Test query to see what database we're actually in
        result = await conn.fetchrow("SELECT current_database()")
        print(f"🔍 Connected to database: {result['current_database']}")
        return conn
    except Exception as e:
        print("❌ DB connection error:", e)
        raise


async def fetch_messages():
    """ This function retrieves chats associated with the user for persistence """
    conn = await get_connection()
    try:
        rows = await conn.fetch("SELECT * FROM chat_logs ORDER BY created_at DESC")
        return rows
    finally:
        await conn.close()
