""" Handles connecting to the postgres database """
import os
import logging
from contextlib import asynccontextmanager
from typing import Optional
import asyncpg
from fastapi import FastAPI
from dotenv import load_dotenv

load_dotenv()

class DatabaseManager:
    """ Manages all database connections an operations """
    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None
        self.db_url = os.getenv("DATABASE_URL")

        if not self.db_url:
            raise ValueError("DATABASE_URL environment variable is required")

    async def initialize(self):
        """Initialize connection pool"""
        try:
            self.pool = await asyncpg.create_pool(
                self.db_url,
                min_size=2,
                max_size=10,
                command_timeout=30,
                server_settings={
                    'application_name': 'tutor_chatbot',
                }
            )
            logging.info("Database pool initialized successfully")
        except Exception as e:
            logging.error("Failed to initialize database pool: %s", e)
            raise

    async def close(self):
        """Close connection pool"""
        if self.pool:
            await self.pool.close()
            logging.info("Database pool closed")

    @asynccontextmanager
    async def get_connection(self):
        """Get database connection from pool"""
        if not self.pool:
            raise RuntimeError("Database pool not initialized")

        async with self.pool.acquire() as connection:
            try:
                yield connection
            except Exception as e:
                logging.error("Database operation failed: %s", e)
                raise

    async def execute_query(self, query: str, *args):
        """Execute a query and return results"""
        async with self.get_connection() as conn:
            return await conn.fetch(query, *args)

    async def execute_one(self, query: str, *args):
        """Execute a query and return one result"""
        async with self.get_connection() as conn:
            return await conn.fetchrow(query, *args)

    async def execute_command(self, query: str, *args):
        """Execute a command (INSERT, UPDATE, DELETE)"""
        async with self.get_connection() as conn:
            return await conn.execute(query, *args)

# Global database manager instance
db_manager = DatabaseManager()

# Updated main.py startup/shutdown events

@asynccontextmanager
async def lifespan(app: FastAPI):
    """ Starts and stops the database """
    # Startup
    await db_manager.initialize()
    yield
    # Shutdown
    await db_manager.close()

# Use in FastAPI app creation:
# app = FastAPI(lifespan=lifespan)

# Helper function for backward compatibility
async def get_connection():
    """Backward compatibility wrapper"""
    return db_manager.get_connection()
